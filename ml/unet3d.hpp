#ifndef UNET3D_HPP
#define UNET3D_HPP
#include "../po.hpp"
#include "cnn3d.hpp"
#include "../numerical/statistics.hpp"
#include "../numerical/basic_op.hpp"
#include "../numerical/transformation.hpp"
#include "../numerical/resampling.hpp"
#include "../filter/gaussian.hpp"
#include "../morphology/morphology.hpp"
#include "../io/interface.hpp"


namespace tipl
{
namespace ml3d
{

inline tipl::shape<3> round_up_size(const tipl::shape<3>& s)
{
    return tipl::shape<3>(int(std::ceil(float(s[0])/32.0f))*32,int(std::ceil(float(s[1])/32.0f))*32,int(std::ceil(float(s[2])/32.0f))*32);
}


inline void preproc_actions(tipl::image<3>& images,
                     const tipl::shape<3>& image_dim,
                     const tipl::vector<3>& image_vs,
                     const tipl::shape<3>& model_dim,
                     const tipl::vector<3>& model_vs,
                     tipl::transformation_matrix<float>& trans,
                     bool match_resolution,
                     bool match_fov)
{
    if(model_dim == image_dim && image_vs == model_vs)
    {
        trans = tipl::transformation_matrix<float>();
        tipl::normalize(images);
        return;
    }
    int in_channel = images.depth()/image_dim[2];
    auto target_vs = match_resolution ? model_vs : image_vs;
    auto target_dim = match_fov ? model_dim :
                            tipl::ml3d::round_up_size(tipl::shape<3>(float(image_dim.width())*image_vs[0]/target_vs[0],
                                float(image_dim.height())*image_vs[1]/target_vs[1],
                                float(image_dim.depth())*image_vs[2]/target_vs[2]));

    tipl::image<3> target_images(target_dim.multiply(tipl::shape<3>::z,in_channel));
    auto shift = tipl::vector<3,int>(target_dim)-tipl::vector<3,int>(image_dim);
    shift[0] /= 2;
    shift[1] /= 2;
    tipl::affine_transform<float> arg;
    trans = tipl::transformation_matrix<float>(arg,target_dim,target_vs,image_dim,image_vs);

    tipl::par_for(in_channel,[&](int c)
    {
        auto image = images.alias(image_dim.size()*c,image_dim);
        auto target_image = target_images.alias(target_dim.size()*c,target_dim);

        if(!match_fov && !match_resolution)
            tipl::draw(image,target_image,shift);
        else
            tipl::resample(image,target_image,trans);

        tipl::normalize(target_image);
    });

    target_images.swap(images);
}
/*
 * calculate the 3d sum and use it to defragment each frame
 */
template<typename image_type>
tipl::image<3> defragment4d(image_type& this_image,float prob_threshold)
{
    tipl::shape<3> dim3d(this_image.shape().begin());
    auto this_image_frames = *(this_image.shape().end()-1);
    tipl::image<3> sum(dim3d);
    if(this_image.empty())
        return sum;

    // hard threshold to make sure prob is between 0 and 1
    tipl::upper_lower_threshold(this_image,0.0f,1.0f);
    // 4d to 3d partial sum
    tipl::sum_partial(this_image,sum);

    auto original_sum = sum;
    {
        tipl::morphology::defragment_by_threshold(sum,prob_threshold);
        tipl::filter::gaussian(sum);
        tipl::filter::gaussian(sum);
        tipl::upper_threshold(sum,1.0f);
    }

    tipl::par_for(this_image_frames,[&](size_t label)
    {
        auto I = this_image.alias(dim3d.size()*label,dim3d);
        for(size_t pos = 0;pos < dim3d.size();++pos)
            if(original_sum[pos] != 0.0f)
                I[pos] *= sum[pos]/original_sum[pos];
    });

    return sum;
}

class unet3d : public network {
    std::deque<std::shared_ptr<network> > encoding,decoding,up;
    std::shared_ptr<layer> output;
private:
    std::shared_ptr<layer> add_layer(layer* l)
    {
        std::shared_ptr<layer> new_layer(l);
        layers.push_back(new_layer);
        return new_layer;
    }
public:
    tipl::vector<3> vs;
    unet3d(int in_channels_v,int out_channels_v,std::string feature_string)
    {
        in_channels_ = in_channels_v;
        std::vector<std::vector<int> > features_down;
        std::vector<std::vector<int> > features_up;
        std::vector<int> kernel_size;
        {
            int input_feature = in_channels_;
            for(auto feature_string_per_level : tipl::split(feature_string,'+'))
            {
                auto level_feature_string = tipl::split(feature_string_per_level,',');
                if(level_feature_string.size() == 2)
                    kernel_size.push_back(std::stoi(level_feature_string.back()));
                else
                    kernel_size.push_back(3);
                features_down.push_back(std::vector<int>({input_feature}));
                for(auto s : tipl::split(feature_string_per_level,'x'))
                    features_down.back().push_back(input_feature = std::stoi(s));
                features_up.push_back(std::vector<int>(features_down.back().rbegin(),features_down.back().rend()-1));
                features_up.back()[0] *= 2; // due to input concatenation
            }
        }

        for(int level=0; level< features_down.size(); level++)
        {
            std::shared_ptr<network> n_en(new network);
            if(level)
                *n_en.get() << add_layer(new max_pool_3d(features_down[level][0]));
            add_conv_block(*n_en.get(),features_down[level],kernel_size[level]);
            encoding.push_back(n_en);
        }
        for(int level=features_down.size()-2; level>=0; level--)
        {
            std::shared_ptr<network> n_up(new network),n_de(new network);
            *n_up.get() << add_layer(new upsample_3d(features_up[level+1].back()));
            add_conv_block(*n_up.get(),{features_up[level+1].back(),features_down[level].back()},kernel_size[level]);
            add_conv_block(*n_de.get(),features_up[level],kernel_size[level]);

            up.push_front(n_up);
            decoding.push_front(n_de);
        }
        output = add_layer(new conv_3d(features_up[0].back(), out_channels_v, 1));
        out_channels_ = out_channels_v;
    }
    void add_conv_block(network& n,const std::vector<int>& rhs,size_t ks)
    {
        int count = 0;
        for(auto next_count : rhs)
        {
            if(count)
            {
                n << add_layer(new conv_3d(count, next_count,ks));
                n << add_layer(new leakyrelu(next_count));
                n << add_layer(new batch_norm_3d(next_count));
            }
            count = next_count;
        }
    }
    virtual void init_image(tipl::shape<3>& dim)
    {
        for(int level=0; level< encoding.size(); level++)
            encoding[level]->init_image(dim);

        for(int level=encoding.size()-2; level>=0; level--)
        {
            up[level]->init_image(dim);
            // create space for concatenation
            {
                auto conv = dynamic_cast<conv_3d*>(encoding[level]->layers[encoding[level]->layers.size()-3].get());
                conv->out.resize(up[level]->out_size()+encoding[level]->out_size());
            }
            decoding[level]->init_image(dim);
        }
        output->init_image(dim);
    }
    virtual float* forward(float* in)
    {
        return forward_with_prog(in,tipl::io::default_prog_type());
    }
    template<typename prog_type = tipl::io::default_prog_type>
    float* forward_with_prog(float* in,prog_type&& prog = prog_type())
    {
        std::vector<float*> buf;
        for(int level=0; level < encoding.size() && prog(int(level),int(encoding.size()*2)); level++)
            buf.push_back(in = encoding[level]->forward(in));
        if(prog.aborted())
            return nullptr;
        for(int level=encoding.size()-2; level>=0 && prog(int(encoding.size()*2)-level,int(encoding.size()*2+1)); level--)
        {
            buf.pop_back();
            auto in2 = up[level]->forward(in);
            std::copy(in2,in2+up[level]->out_size(),buf.back()+encoding[level]->out_size());
            in=decoding[level]->forward(buf.back());
        }
        if(prog.aborted())
            return nullptr;
        return output->forward(in);
    }
public:
    tipl::image<3> out,sum;
    template<typename image_type,typename prog_type = tipl::io::default_prog_type>
    bool forward(const image_type& I,tipl::vector<3> I_vs,prog_type&& prog = prog_type())
    {
        tipl::affine_transform<float> arg;
        // align top so that the T1W bottom is trancated
        arg.translocation[2] = (float(I.depth())*I_vs[2]-float(dim[2])*vs[2])*0.5f;
        tipl::transformation_matrix<float> trans(arg,dim,vs,I.shape(),I_vs);
        tipl::image<3> input_image(dim);
        tipl::resample(I,input_image,trans);
        tipl::normalize(input_image);
        auto ptr = forward_with_prog(&input_image[0],prog);
        if(ptr == nullptr)
            return false;
        trans.inverse();
        out.resize(I.shape().multiply(tipl::shape<3>::z,out_channels_));
        tipl::par_for(out_channels_,[&](int i)
        {
            auto J = out.alias(I.size()*i,I.shape());
            tipl::resample(tipl::make_image(ptr+i*dim.size(),dim),J,trans);
            for(size_t j = 0;j < J.size();++j)
                if(I[j] == 0)
                    J[j] = 0.0f;
        });
        auto J = tipl::make_image(&out[0],I.shape().expand(out_channels_));
        sum = tipl::ml3d::defragment4d(J,0.5f);
        return true;
    }
    template<typename reader>
    static std::shared_ptr<unet3d> load_model(const char* file_name)
    {
        reader in;
        if(!in.load_from_file(file_name))
            return nullptr;
        std::string feature_string;
        std::vector<int> param({1,1});
        if(!in.read("param",param) || !in.read("feature_string",feature_string))
            return nullptr;
        std::shared_ptr<unet3d> un(new unet3d(param[0],param[1],feature_string));
        in.read("voxel_size",un->vs);
        in.read("dimension",un->dim);
        tipl::shape<3> d(un->dim);
        un->init_image(d);
        int id = 0;
        for(auto& param : un->parameters())
            if(!in.read((std::string("tensor")+std::to_string(id++)).c_str(),param.first,param.first+param.second))
                return nullptr;
        return un;
    }
};





}//ml3d
}//tipl

#endif//UNET3D_HPP
