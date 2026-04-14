#ifndef UNET3D_HPP
#define UNET3D_HPP
#include "cnn3d.hpp"
#include "../po.hpp"
#include "../prog.hpp"
#include "../numerical/statistics.hpp"
#include "../numerical/basic_op.hpp"
#include "../numerical/transformation.hpp"
#include "../numerical/resampling.hpp"
#include "../numerical/morphology.hpp"
#include "../numerical/otsu.hpp"
#include "../filter/gaussian.hpp"
#include "../io/interface.hpp"
#include "../utility/basic_image.hpp"
#include "../utility/shape.hpp"
#include "../reg/linear.hpp"


namespace tipl
{
namespace ml3d
{

inline auto round_up_size(const tipl::shape<3>& s)
{
    return tipl::shape<3>((s[0]+31)&~31,(s[1]+31)&~31,(s[2]+31)&~31);
}

inline void preproc_actions(tipl::image<3>& images,
                            const tipl::shape<3>& image_dim,
                            const tipl::vector<3>& image_vs,
                            const tipl::shape<3>& model_dim,
                            const tipl::vector<3>& model_vs,
                            tipl::transformation_matrix<float,3>& trans,
                            bool match_resolution,
                            bool match_fov)
{
    if(model_dim == image_dim && image_vs == model_vs)
    {
        trans = tipl::transformation_matrix<float,3>();
        return;
    }

    int in_channel = images.depth()/image_dim[2];
    auto target_vs = match_resolution ? model_vs : image_vs;



    auto target_dim = match_fov ? model_dim :
        round_up_size(tipl::shape<3>(int(float(image_dim[0])*image_vs[0]/target_vs[0]),
                                int(float(image_dim[1])*image_vs[1]/target_vs[1]),
                                int(float(image_dim[2])*image_vs[2]/target_vs[2])));

    tipl::image<3> target_images(target_dim.multiply(tipl::shape<3>::z,in_channel));
    trans = tipl::transformation_matrix<float,3>(tipl::affine_param<float>(),target_dim,target_vs,image_dim,image_vs);

    for(int c = 0;c < in_channel;++c)
    {
        auto image = images.alias(image_dim.size()*c,image_dim);
        auto target_image = target_images.alias(target_dim.size()*c,target_dim);
        tipl::resample(image,target_image,trans);
    }

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

template<typename U,typename V>
tipl::image<3> postproc_actions(const U& eval_output,const V& raw_image_shape,
                                tipl::transformation_matrix<float,3> trans,size_t model_out_count)
{

    tipl::shape<3> dim_from(eval_output.shape().divide(tipl::shape<3>::z,model_out_count)),
                   dim_to(raw_image_shape);
    tipl::image<3> label_prob(dim_to.multiply(tipl::shape<3>::z,model_out_count));
    trans.inverse();
    tipl::par_for(model_out_count,[&](int i)
    {
        auto from = eval_output.alias(dim_from.size()*i,dim_from);
        auto to = label_prob.alias(dim_to.size()*i,dim_to);
        tipl::resample(from,to,trans);
    },model_out_count);
    return label_prob;
}

template<typename feature_type,typename kernel_type>
void parse_feature_string(const std::string& feature_string,
                          int input_feature,
                          feature_type& features_down,
                          feature_type& features_up,
                          kernel_type& kernel_size)
{
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

template<typename T>
inline auto soft_mask(const T& label)
{
    tipl::image<3> foreground_mask(label.shape());
    tipl::threshold(label,foreground_mask,0,1.0f,0.0f);
    tipl::filter::gaussian(foreground_mask);
    return foreground_mask;
}

class tissue_seg{
public:
    std::shared_ptr<unet3d<tipl::progress>> unet;
    std::string error_msg;
    tipl::vector<3> vs;
    tipl::shape<3> dim;

    template<typename reader>
    bool load_model(const std::string& file_name)
    {
        reader in;
        std::string feature_string;
        std::vector<int> param({1,1});
        if(!in.load_from_file(file_name))
            return error_msg = "cannot open file: " + file_name,false;
        if(!in.read("param",param) || !in.read("feature_string",feature_string))
            return error_msg = "invalid network file format",false;

        std::vector<std::vector<int> > features_down;
        std::vector<std::vector<int> > features_up;
        std::vector<int> kernel_size;
        parse_feature_string(feature_string,param[0],features_down,features_up,kernel_size);
        unet.reset(new unet3d<tipl::progress>(features_down,features_up,kernel_size,param[0],param[1]));

        if(!in.read("dimension",dim) || !in.read("voxel_size",vs))
            return error_msg = "cannot read dimension and voxel size",false;
        unet->init_image(dim); // the dim value will be changed
        dim = unet->dim;
        int id = 0;
        for(auto& param : unet->parameters())
            if(!in.read((std::string("tensor")+std::to_string(id++)).c_str(),param.first,param.first+param.second))
                return error_msg = "tensor structure mismatch",false;
        return true;
    }
public:
    template<typename image_type>
    bool forward(const image_type& raw_image,
                 const tipl::vector<3>& raw_image_vs,
                 tipl::image<3,unsigned char>& label,
                 tipl::progress& prog)
    {
        const bool match_resolution = true;
        const bool match_fov = true;
        const float prob_threshold = 0.5f;
        tipl::transformation_matrix<float,3> trans;
        tipl::image<3> input_image(raw_image);
        tipl::segmentation::normalize_otsu_median(input_image);
        tipl::ml3d::preproc_actions(input_image,input_image.shape(),raw_image_vs,
                                        dim,vs,trans,match_resolution,match_fov);
        if(input_image.shape() != dim)
            return false;
        unet->prog = &prog;
        auto ptr = unet->forward(input_image.data());
        if(ptr == nullptr)
            return false;
        auto evaluate_output = tipl::make_image(ptr,dim.multiply(tipl::shape<3>::z,unet->out_channels_));
        auto label_prob = postproc_actions(evaluate_output,raw_image.shape(),trans,unet->out_channels_);
        auto label_prob_4d = tipl::make_image(label_prob.data(),raw_image.shape().expand(unet->out_channels_));
        auto fg_prob = tipl::ml3d::defragment4d(label_prob_4d,prob_threshold);

        {
            tipl::image<3,unsigned char> I(fg_prob.shape());
            size_t s = fg_prob.size();
            for(size_t pos = 0;pos < s;++pos)
            {
                if(fg_prob[pos] <= prob_threshold)
                    continue;
                float m = label_prob[pos];
                unsigned char max_label = 1;
                for(size_t i = pos+s,label = 2;i < label_prob.size();i += s,++label)
                    if(label_prob[i] > m)
                    {
                        m = label_prob[i];
                        max_label = label;
                    }
                I[pos] = max_label;
            }
            I.swap(label);
        }
        return true;
    }
};





}//ml3d
}//tipl

#endif//UNET3D_HPP
