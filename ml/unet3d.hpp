#ifndef UNET3D_HPP
#define UNET3D_HPP
#include "../po.hpp"
#include "cnn3d.hpp"
#include "../numerical/statistics.hpp"
#include "../numerical/basic_op.hpp"
#include "../filter/gaussian.hpp"
#include "../morphology/morphology.hpp"
#include "../io/interface.hpp"

namespace tipl
{
namespace ml3d
{

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
        {
            int input_feature = in_channels_;
            for(auto feature_string_per_level : tipl::split(feature_string,'+'))
            {
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
            add_conv_block(*n_en.get(),features_down[level]);
            encoding.push_back(n_en);
        }
        for(int level=features_down.size()-2; level>=0; level--)
        {
            std::shared_ptr<network> n_up(new network),n_de(new network);
            *n_up.get() << add_layer(new upsample_3d(features_up[level+1].back()));
            add_conv_block(*n_up.get(),{features_up[level+1].back(),features_down[level].back()});
            add_conv_block(*n_de.get(),features_up[level]);

            up.push_front(n_up);
            decoding.push_front(n_de);
        }
        output = add_layer(new conv_3d(features_up[0].back(), out_channels_v, 1));
        out_channels_ = out_channels_v;
    }
    void add_conv_block(network& n,const std::vector<int>& rhs)
    {
        int count = 0;
        for(auto next_count : rhs)
        {
            if(count)
            {
                n << add_layer(new conv_3d(count, next_count));
                n << add_layer(new relu(next_count));
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
    template<typename reader>
    static std::shared_ptr<unet3d> load_model(reader& mat)
    {
        std::string feature_string;
        std::vector<int> param({1,1});
        if(!mat.read("param",param) || !mat.read("feature_string",feature_string))
            return nullptr;
        std::shared_ptr<unet3d> un(new unet3d(param[0],param[1],feature_string));
        mat.read("voxel_size",un->vs);
        mat.read("dimension",un->dim);
        tipl::shape<3> d(un->dim);
        un->init_image(d);
        int id = 0;
        for(auto& param : un->parameters())
            if(!mat.read((std::string("tensor")+std::to_string(id++)).c_str(),param.first,param.first+param.second))
                return nullptr;
        return un;
    }
};



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
    tipl::sum_partial_mt(this_image,sum);

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


}//ml3d
}//tipl

#endif//UNET3D_HPP
