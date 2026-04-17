#ifndef UNET3D_HPP
#define UNET3D_HPP

#include <algorithm> // added for std::copy
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
#include "../cu.hpp"

namespace tipl
{
namespace ml3d
{

inline auto round_up_size(const tipl::shape<3>& s)
{
    return tipl::shape<3>((s[0]+31)&~31, (s[1]+31)&~31, (s[2]+31)&~31);
}

template<typename image_type>
void preproc_actions(image_type& images,
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

    image_type target_images(target_dim.multiply(tipl::shape<3>::z,in_channel));
    tipl::affine_param<float> arg;
    arg.translocation[2] = (image_dim[2]*image_vs[2]-target_dim[2]*target_vs[2])*0.5f; //align top
    trans = tipl::transformation_matrix<float,3>(arg,target_dim,target_vs,image_dim,image_vs);

    for(int c = 0;c < in_channel;++c)
    {
        auto image = images.alias(image_dim.size()*c,image_dim);
        auto target_image = target_images.alias(target_dim.size()*c,target_dim);
        tipl::resample(image,target_image,trans);
    }
    target_images.swap(images);
}

template<typename U,typename V>
void postproc(const U& eval_output,
              const shape<3>& raw_shape,
              tipl::transformation_matrix<float,3> trans,
              size_t model_out_count,
              bool has_bg_channel,
              float prob_threshold,
              V& label_prob,
              V& fg_prob)
{
    tipl::shape<3> dim_from(eval_output.shape().divide(tipl::shape<3>::z,model_out_count));
    label_prob.resize(raw_shape.multiply(tipl::shape<3>::z,model_out_count));
    trans.inverse();

    tipl::par_for(model_out_count,[&](int i)
    {
        auto from = eval_output.alias(dim_from.size()*i,dim_from);
        auto to = label_prob.alias(raw_shape.size()*i,raw_shape);
        tipl::resample(from,to,trans);
        tipl::upper_lower_threshold(to, 0.0f, 1.0f);
    },model_out_count);


    if(has_bg_channel)
    {
        std::copy(label_prob.begin() + raw_shape.size(), label_prob.end(), label_prob.begin());
        model_out_count -= 1;
        label_prob.resize(raw_shape.multiply(tipl::shape<3>::z, model_out_count));
    }

    auto labels_4d = tipl::make_image(label_prob.data(), raw_shape.expand(model_out_count));

    fg_prob.resize(raw_shape);
    tipl::sum_partial(labels_4d, fg_prob);
    auto original_sum = fg_prob;
    tipl::upper_threshold(fg_prob,1.0f); // normalize label sum to 1
    tipl::morphology::defragment_by_threshold(fg_prob, prob_threshold);
    tipl::filter::gaussian(fg_prob);
    tipl::filter::gaussian(fg_prob);

    tipl::lower_threshold(original_sum,prob_threshold);

    tipl::par_for(model_out_count, [&](size_t label)
    {
        auto I = labels_4d.slice_at(label);
        tipl::multiply(I,fg_prob);
        tipl::divide(I,original_sum);
    });
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
        // Condensed if/else logic using ternary operator
        kernel_size.push_back(level_feature_string.size() == 2 ? std::stoi(level_feature_string.back()) : 3);

        features_down.push_back({input_feature});
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
    tipl::device_vector<float> gpu_memory;
    bool deep_supervision = false;
    std::vector<float> memory;
    std::shared_ptr<network> unet;
    std::string error_msg;
    tipl::vector<3> vs;
    tipl::shape<3> dim;
    int num_tissue_channels = 5;

    template<typename reader>
    bool load_model(const std::string& file_name)
    {
        reader in;
        std::string feature_string;
        std::vector<int> param({1,1});
        if(!in.load_from_file(file_name)) return error_msg = "cannot open file: " + file_name,false;
        if(!in.read("param",param) || !in.read("feature_string",feature_string)) return error_msg = "invalid network file format",false;

        std::vector<std::vector<int> > features_down;
        std::vector<std::vector<int> > features_up;
        std::vector<int> kernel_size;
        parse_feature_string(feature_string,param[0],features_down,features_up,kernel_size);

        if(in.has("errors"))
        {
            unet.reset(new unet3d<unet_version::deep_supervision>(features_down,features_up,kernel_size,param[0],param[1]));
            deep_supervision = true;
        }
        else
            unet.reset(new unet3d<unet_version::standard>(features_down,features_up,kernel_size,param[0],param[1]));

        if(!in.read("dimension",dim) || !in.read("voxel_size",vs))
            return error_msg = "cannot read dimension and voxel size",false;
        unet->init_image(dim);
        unet->allocate_memory(memory);
        dim = unet->dim;

        int id = 0;
        for(auto& p : unet->parameters())
        {
            if(!p.second) continue;

            std::string name = "tensor" + std::to_string(id++);
            if(!in.has(name.c_str())) return error_msg = "tensor structure mismatch (missing " + name + ")", false;

            unsigned int tr, tc;
            if(!in.get_col_row(name.c_str(), tr, tc) || tr * tc != p.second)
                return error_msg = "tensor size mismatch in " + name, false;

            if(!in.read(name.c_str(), p.first, p.first + p.second))
                return error_msg = "error reading tensor structure " + name, false;
        }

        if constexpr(tipl::use_cuda)
        {
            gpu_memory = memory;
            auto ptr = gpu_memory.data();
            unet->allocate(ptr, true /*is gpu*/);
        }
        return true;
    }

    bool forward(tipl::image<3> input_image,
                 const tipl::vector<3>& image_vs,
                 tipl::image<3,unsigned char>& label,
                 tipl::progress& prog)
    {
        const float prob_threshold = 0.5f;
        tipl::transformation_matrix<float,3> trans;
        tipl::shape<3> input_dim(input_image.shape());
        tipl::segmentation::normalize_otsu_median(input_image);

        unet->prog = [&](void){return prog(0,4);};
        const float* ptr = nullptr;
        std::vector<float> buffer;

        // 1. PRE-PROCESSING & INFERENCE
        if constexpr(tipl::use_cuda)
        {
            tipl::device_image<3,float> I = input_image;
            tipl::ml3d::preproc_actions(I,input_dim,image_vs,dim,vs,trans,true,true);
            auto gpu_ptr = unet->forward(I.data());
            if (!gpu_ptr) return false;
            buffer.resize(unet->dim.size()*unet->out_channels_);
            cu_copy_d2h<float,float>(buffer.data(),gpu_ptr,buffer.size());
            ptr = buffer.data();
        }
        else
        {
            tipl::ml3d::preproc_actions(input_image,input_dim,image_vs,dim,vs,trans,true,true);
            ptr = unet->forward(input_image.data());
        }

        if(!ptr) return false;
        prog(1,4);

        auto evaluate_output = tipl::make_image(ptr,unet->dim.multiply(tipl::shape<3>::z,unet->out_channels_));
        tipl::image<3> label_prob, fg_prob;
        postproc(evaluate_output, input_dim, trans, unet->out_channels_, deep_supervision, prob_threshold, label_prob, fg_prob);

        num_tissue_channels = deep_supervision ? (unet->out_channels_ - 1) : unet->out_channels_;
        prog(3,4);

        tipl::image<3,unsigned char> I(fg_prob.shape());
        size_t s = fg_prob.size();
        for(size_t pos = 0; pos < s; ++pos)
        {
            if(fg_prob[pos] <= prob_threshold)
                continue;

            float m = label_prob[pos];
            unsigned char max_label = 1;
            for(size_t i = pos+s, label = 2; i < label_prob.size(); i += s, ++label)
            {
                if(label_prob[i] > m)
                {
                    m = label_prob[i];
                    max_label = label;
                }
            }
            I[pos] = max_label;
        }

        I.swap(label);
        prog(4,4);
        return true;
    }
};

#ifdef __CUDACC__
template void preproc_actions<tipl::device_image<3,float>>(
    tipl::device_image<3,float>& images,
    const tipl::shape<3>& image_dim,
    const tipl::vector<3>& image_vs,
    const tipl::shape<3>& model_dim,
    const tipl::vector<3>& model_vs,
    tipl::transformation_matrix<float, 3>& trans,
    bool match_resolution,
    bool match_fov
);
#endif

}//ml3d
}//tipl

#endif//UNET3D_HPP
