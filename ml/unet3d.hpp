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
void postproc_actions(const std::string& command,
                      float param1,float param2,
                      image_type& this_image,
                      image_type& mask,
                      const tipl::shape<3>& image_dim,
                      char& is_label)
{
    auto out_channel = this_image.depth()/image_dim[2];
    if(this_image.empty())
        return;
    tipl::out() << "run " << command;
    if(command == "argmax")
    {
        this_image = tipl::argmax(this_image,mask > param1);
        is_label = true;
    }

    // per channel operations
    tipl::par_for(out_channel,[&](size_t label)
    {
        auto I = this_image.alias(image_dim.size()*label,image_dim);
        if(command == "upper_threshold")
        {
            float upper_threshold_threshold = param1;
            tipl::upper_threshold(I,upper_threshold_threshold);
            is_label = false;
            return;
        }
        if(command == "lower_threshold")
        {
            float lower_threshold_threshold = param1;
            tipl::lower_threshold(I,lower_threshold_threshold);
            is_label = false;
            return;
        }
        if(command == "minus")
        {
            float minus_value = param1;
            for(size_t i = 0,sz = I.size();i < sz;++i)
                I[i] -= minus_value;
            is_label = false;
            return;
        }

        if(command == "defragment_each")
        {
            float defragment_each_threshold = param1;
            tipl::image<3,char> mask(I.shape()),mask2;
            for(size_t i = 0,sz = I.size();i < sz;++i)
                mask[i] = (I[i] > defragment_each_threshold ? 1:0);
            mask2 = mask;
            tipl::morphology::defragment_by_size_ratio(mask);
            for(size_t i = 0,sz = I.size();i < sz;++i)
                if(!mask[i] && mask2[i])
                I[i] = 0;
            return;
        }
        if(command == "normalize_each")
        {
            tipl::normalize(I);
            is_label = false;
            return;
        }
        if(command == "gaussian_smoothing")
        {
            tipl::filter::gaussian(I);
            is_label = false;
            return;
        }
    });

    tipl::error() << "unknown command " << command << std::endl;
}


template<typename image_type = tipl::image<3>>
struct evalution_set{

private:
    image_type gaussian;
    void create_gaussian(void)
    {
        gaussian.resize(model_dim);
        float sigma_scale = 1.0f/8.0f;
        tipl::vector<3> center(tipl::vector<3>(model_dim)*0.5f);
        float max_val = 0.0f;
        tipl::vector<3> inv_sigmas(1.0f/(model_dim[0]*sigma_scale),1.0f/(model_dim[1]*sigma_scale),1.0f/(model_dim[2]*sigma_scale));
        size_t sz = model_dim.size();
        for(tipl::pixel_index<3> index(model_dim);index < sz;++index)
        {
            tipl::vector<3,float> pos(index);
            pos -= center;
            pos.elem_mul(inv_sigmas);

            float val = std::exp(-0.5f*pos.length2());
            gaussian[index.index()] = val;
            max_val = std::max(max_val,val);
        }

        gaussian /= max_val;
        tipl::lower_threshold(gaussian,1e-6f);
    }
public:
    std::vector<image_type> model_input,model_output;
    std::vector<std::decay_t<decltype(tipl::make_image(std::declval<image_type&>(), 0, std::declval<tipl::shape<3>>()))>> model_output2;

public:
    std::string error_msg;
    std::vector<tipl::transformation_matrix<float>> trans;
    tipl::image<3,unsigned char> mask,label;
    image_type label_prob,fg_prob;

    tipl::shape<3> image_dim,model_dim;
    tipl::vector<3> image_vs,model_vs;
    tipl::matrix<4,4,float> untouched_srow,srow;
    std::vector<char> flip_swap;
    float prob_threshold = 0.5f;

    size_t in_count,out_count;

    template<typename image_type2>
    bool preproc(image_type2& source_image,const tipl::image<3,unsigned char>& mask_)
    {
        tipl::par_for(in_count,[&](int c)
        {
            tipl::segmentation::normalize_otsu_median(source_image.alias(c*image_dim.size(),image_dim));
        });

        if(mask_.empty())
        {
            tipl::threshold(source_image.alias(0,image_dim),mask,0.0f);
            tipl::morphology::defragment(mask);
        }
        else
            mask = mask_;

        tipl::vector<3> from,to;
        if(!tipl::bounding_box(mask,from,to))
            return false;

        tipl::vector<3> mask_center((from+to)*0.5f),image_center(tipl::vector<3>(image_dim)*0.5f),
                        patch_phys(model_dim),bb_phys(to-from);

        mask_center.elem_mul(image_vs);
        image_center.elem_mul(image_vs);
        patch_phys.elem_mul(model_vs);
        bb_phys.elem_mul(image_vs);

        std::vector<float> s[3];
        for(int d=0;d<3;++d)
        {
            if(bb_phys[d]>2.0f*patch_phys[d])
                bb_phys[d] = 2.0f*patch_phys[d];
            if(bb_phys[d]>patch_phys[d])
            {
                float d_shift = (bb_phys[d]-patch_phys[d])*0.5f;
                s[d].push_back(-d_shift);
                s[d].push_back(d_shift);
            }
            else
                s[d].push_back(0.0f);
        }

        std::vector<tipl::vector<3>> active_shifts;
        for(float z:s[2])
            for(float y:s[1])
                for(float x:s[0])
                    active_shifts.push_back(tipl::vector<3>(x,y,z));

        if(active_shifts.size()!=1)
        {
            active_shifts.push_back(tipl::vector<3>(0.0f,0.0f,0.0f));
            create_gaussian();
        }
        tipl::out() << "total of sliding window:" << active_shifts.size();
        for(auto shift:active_shifts)
        {
            shift += mask_center;
            shift -= image_center;

            tipl::affine_param<float> arg;
            arg.translocation[0] = shift[0];
            arg.translocation[1] = shift[1];
            arg.translocation[2] = shift[2];

            auto tran = tipl::transformation_matrix<float,3>(arg,model_dim,model_vs,image_dim,image_vs);
            image_type target_image(model_dim.multiply(tipl::shape<3>::z,in_count));

            for(int c=0;c<in_count;++c)
                tran(make_image(source_image,image_dim.size()*c,image_dim),
                     make_image(target_image,model_dim.size()*c,model_dim));

            model_input.push_back(std::move(target_image));
            trans.push_back(tran);
        }
        return true;
    }

    bool command(const std::string& cmd)
    {
        for(const auto& each : tipl::split(cmd,'+'))
        {
            tipl::out() << "run " << each;
            if(each == "postproc" && postproc())
                continue;
            if(each == "remove_bg_channel" && remove_bg_channel())
                continue;
            if(each == "softmax" && softmax())
                continue;
            if(each == "argmax" && argmax())
                continue;
            if(each == "create_mask" && create_mask())
                continue;
            if(error_msg.empty())
                error_msg = "invalid command: " + cmd;
            return false;
        }
        return true;
    }
    bool softmax(void)
    {
        if(label_prob.empty())
            return error_msg = "empty label probability",false;
        tipl::softmax(label_prob,image_dim.size(),out_count);
        for(int c = 1;c<out_count;++c)
            tipl::preserve(label_prob.alias(image_dim.size()*c,image_dim),mask);
        return true;
    }

    bool postproc(void)
    {
        if(model_output2.empty() && model_output.empty())
            return error_msg = "no output data",false;
        if(model_output2.empty())
            for(auto& each : model_output)
                model_output2.push_back(each.alias());
        auto& outputs = model_output2;
        label_prob.resize(image_dim.multiply(tipl::shape<3>::z,out_count));
        if(outputs.size()==1)
        {
            auto each_trans = trans[0];
            each_trans.inverse();
            for(int i=0;i<out_count;++i)
            {
                auto t = outputs[0].alias(model_dim.size()*i,model_dim);
                auto o = label_prob.alias(image_dim.size()*i,image_dim);
                tipl::resample(t,o,each_trans);
            }
        }
        else
        {
            tipl::image<3,float> weight_map(image_dim);

            tipl::par_for(out_count+1,[&](int c)
            {
                for(int t=0;t < outputs.size();++t)
                {
                    auto each_trans = trans[t];
                    each_trans.inverse();

                    if(c == out_count)
                    {
                        weight_map += each_trans(gaussian,image_dim);
                        continue;
                    }
                    auto w = outputs[t].alias(model_dim.size()*c,model_dim);
                    auto o = label_prob.alias(image_dim.size()*c,image_dim);
                    w *= gaussian;
                    o += each_trans(w,image_dim);
                }
            });

            for(size_t i=0;i<weight_map.size();++i)
                if(weight_map[i] > 1e-6)
                    weight_map[i] = 1.0f/weight_map[i];
            tipl::par_for(out_count,[&](int c)
            {
                label_prob.alias(image_dim.size()*c,image_dim) *= weight_map;
            });
        }
        return true;
    }
    bool remove_bg_channel(void)
    {
        if(out_count <= 1)
            return error_msg = "not enough out channel to remove",false;
        tipl::remove_channel(label_prob,image_dim);
        out_count--;
        return true;
    }
    bool create_mask(void)
    {
        if(label_prob.empty())
            return error_msg = "no label probability",false;
        auto labels_4d = tipl::make_image(label_prob.data(), image_dim.expand(out_count));
        fg_prob.resize(image_dim);
        tipl::sum_partial(labels_4d, fg_prob);
        auto original_sum = fg_prob;
        tipl::morphology::defragment_by_threshold(fg_prob, prob_threshold);
        tipl::lower_threshold(original_sum,prob_threshold);

        tipl::par_for(out_count, [&](size_t label)
        {
            auto I = labels_4d.slice_at(label);
            tipl::multiply(I,fg_prob);
            tipl::divide(I,original_sum);
        });
        return true;
    }
    bool argmax(void)
    {
        if(label_prob.empty())
            return error_msg = "no label probability",false;
        if(fg_prob.empty())
            return error_msg = "no foreground probability",false;
        label = tipl::argmax(label_prob,fg_prob > prob_threshold);
        return true;
    }
    template<typename io_type>
    bool save_to_file(const std::string& file_name)
    {
        io_type out;
        if(!out.open(file_name,std::ios::out))
            return error_msg = out.error_msg,false;
        out << untouched_srow << image_vs;
        out.flip_swap_seq = flip_swap;
        if(tipl::is_label_image(label_prob))
        {
            tipl::image<3,unsigned char> label(label_prob);
            out.apply_flip_swap_seq(label,true);
            if(label_prob.depth() == image_dim[2])
                out << label;
            else
                out << label.alias(0,tipl::shape<4>(image_dim.expand(label_prob.depth()/image_dim[2])));
        }
        else
        {
            tipl::image<3> prob(label_prob);
            out.apply_flip_swap_seq(prob,true);
            if(label_prob.depth() == image_dim[2])
                out << prob;
            else
                out << prob.alias(0,tipl::shape<4>(image_dim.expand(label_prob.depth()/image_dim[2])));
        }
        return true;

    }
};





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

template<typename U>
auto postproc0(const U& eval_output,const shape<3>& image_dim,
              tipl::transformation_matrix<float,3> trans,size_t out_count)
{
    tipl::shape<3> dim_from(eval_output.shape().divide(tipl::shape<3>::z,out_count));
    tipl::image<3> label_prob(image_dim.multiply(tipl::shape<3>::z,out_count));
    trans.inverse();

    tipl::par_for(out_count,[&](int i)
    {
        auto from = eval_output.alias(dim_from.size()*i,dim_from);
        auto to = label_prob.alias(image_dim.size()*i,image_dim);
        tipl::resample(from,to,trans);
    },out_count);
    return label_prob;
}




class tissue_seg{
private:

public:
    tipl::device_vector<float> gpu_memory;
    std::vector<float> memory;
    std::shared_ptr<network> unet;
    bool new_version = false;
public:
    evalution_set<tipl::image<3>>& eval;
    std::string error_msg;

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

        if(in.has("report"))
        {
            tipl::out() << "loading deep supervision unet";
            unet.reset(new unet3d<unet_version::deep_supervision>(features_down,features_up,kernel_size,param[0],param[1]));
            new_version = true;
        }
        else
        {
            tipl::out() << "loading conventional unet";
            unet.reset(new unet3d<unet_version::standard>(features_down,features_up,kernel_size,param[0],param[1]));
        }
        tipl::shape<3> dim;
        if(!in.read("dimension",dim) || !in.read("voxel_size",eval.model_vs))
            return error_msg = "cannot read dimension and voxel size",false;
        unet->init_image(dim);
        unet->allocate_memory(memory);
        eval.model_dim = unet->dim;
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

        /*
        if constexpr(tipl::use_cuda)
        {
            gpu_memory = memory;
            auto ptr = gpu_memory.data();
            unet->allocate(ptr, true);
        }
        */
        return true;
    }

    bool forward(tipl::progress& prog)
    {
        const float prob_threshold = 0.5f;
        prog(1,4);
        tipl::out() << "preprocessing";

        unet->prog = [&](void){return prog(0,4);};

        // 1. PRE-PROCESSING & INFERENCE
        /*
        std::vector<float> buffer;
        if constexpr(tipl::use_cuda)
        {
            tipl::device_image<3,float> I = input_image;
            tipl::ml3d::preproc_actions(I,image_dim,image_vs,dim,vs,trans,true,true);
            auto gpu_ptr = unet->forward(I.data());
            if (!gpu_ptr) return false;
            buffer.resize(unet->dim.size()*unet->out_channels_);
            cu_copy_d2h<float,float>(buffer.data(),gpu_ptr,buffer.size());
            ptr = buffer.data();
        }
        else
        */
        prog(2,4);
        for(auto& each : eval.model_input)
        {
            auto ptr = unet->forward(each.data());
            if(!ptr)
                return false;
            eval.model_output2.push_back(tipl::make_image(ptr,eval.model_dim));
        }

        prog(3,4);
        eval.postproc();
        if(new_version)
        {
            eval.softmax();
            eval.remove_bg_channel();
        }
        else
            tipl::upper_lower_threshold(eval.label_prob, 0.0f, 1.0f);
        eval.create_mask();
        eval.argmax();
        prog(4,4);
        return true;
    }
};

#ifdef __CUDACC__


#endif

}//ml3d

#ifdef __CUDACC__
template void tipl::resample<(tipl::interpolation)2,
                             tipl::pointer_device_image<3, float>,
                             tipl::pointer_device_image<3, float>&,
                             float>(
    const tipl::pointer_device_image<3, float>& from,
    tipl::pointer_device_image<3, float>& to,
    const tipl::transformation_matrix<float, 3>& trans
);
#endif

}//tipl

#endif//UNET3D_HPP
