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
#include "../io/nifti.hpp"
#include "../utility/basic_image.hpp"
#include "../utility/shape.hpp"
#include "../reg/linear.hpp"


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
        tipl::image<3> gau(model_dim);
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
            gau[index.index()] = val;
            max_val = std::max(max_val,val);
        }

        gau /= max_val;
        tipl::lower_threshold(gau,1e-6f);
        gaussian = std::move(gau);
    }
public:
    image_type source_image;
    std::vector<image_type> model_input,model_output;

public:
    evalution_set(void):label_prob(source_image){}
    std::string error_msg;
    std::vector<tipl::transformation_matrix<float>> trans;
    tipl::image<3,unsigned char> mask,label;
    image_type& label_prob;
    image_type fg_prob;

    tipl::shape<3> image_dim,model_dim;
    tipl::vector<3> image_vs,model_vs;
    tipl::matrix<4,4,float> untouched_srow,srow;
    std::vector<char> flip_swap;


    size_t in_count,out_count;

    bool preproc(const std::string& cmd)
    {
        if(source_image.empty())
            return error_msg = "no source image",false;
        tipl::progress prog("preprocessing");
        tipl::out() << "cmd: " << cmd;
        tipl::par_for(in_count,[&](int c)
        {
            auto I = source_image.alias(c*image_dim.size(),image_dim);
            tipl::segmentation::normalize_otsu_median(I);
            for(const auto& each : tipl::split(cmd,'+'))
            {
                if(each == "flip_xy")
                    flip_xy();
            }
        });

        if(mask.empty())
        {
            tipl::threshold(source_image.alias(0,image_dim),mask,0.0f);
            tipl::morphology::defragment(mask);
        }


        tipl::vector<3> from,to;
        if(!tipl::bounding_box(mask,from,to))
            return error_msg = "source image is all zero",false;

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
        model_input.resize(active_shifts.size());
        trans.resize(active_shifts.size());
        tipl::par_for(active_shifts.size(),[&](int i)
        {
            auto shift = active_shifts[i];
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

            model_input[i] = std::move(target_image);
            trans[i] = tran;
        });
        source_image.clear();
        return !prog.aborted();
    }
    bool flip_xy(void)
    {
        if(!label_prob.empty())
            tipl::flip_xy(label_prob);
        if(!fg_prob.empty())
            tipl::flip_xy(fg_prob);
        if(!label.empty())
            tipl::flip_xy(label);
        if(!mask.empty())
            tipl::flip_xy(mask);
        return true;
    }
    bool clamp_prob(void)
    {
        return tipl::upper_lower_threshold(label_prob, 0.0f, 1.0f),true;
    }
    bool smoothing(void)
    {
        if(label_prob.empty())
            return error_msg = "empty label probability",false;
        auto labels_4d = tipl::make_image(label_prob.data(), image_dim.expand(out_count));
        tipl::par_for(out_count, [&](size_t label)
        {
            auto I = labels_4d.slice_at(label);
            tipl::filter::gaussian(I);
        });
        return true;
    }
    std::vector<std::string> param;
    bool command(const std::string& cmd)
    {
        progress prog("postprocessing");
        auto cmds = tipl::split(cmd,'+');
        for(size_t i = 0;prog(i,cmds.size());++i)
        {
            tipl::out() << "run " << cmds[i];
            param = tipl::split(cmds[i],',');
            if(param[0] == "postproc" && postproc())
                continue;
            if(param[0] == "remove_bg_channel" && remove_bg_channel())
                continue;
            if(param[0] == "softmax" && softmax())
                continue;
            if(param[0] == "argmax" && argmax())
                continue;
            if(param[0] == "create_mask" && create_mask())
                continue;
            if(param[0] == "flip_xy" && flip_xy())
                continue;
            if(param[0] == "clamp_prob" && clamp_prob())
                continue;
            if(param[0] == "smoothing" && smoothing())
                continue;
            if(error_msg.empty())
                error_msg = "invalid command: " + cmds[i];
            return false;
        }
        return !prog.aborted();
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
        tipl::progress prog("postprocessing");
        if(model_output.empty())
            return error_msg = "no output data",false;
        label_prob.resize(image_dim.multiply(tipl::shape<3>::z,out_count));
        if(model_output.size()==1)
        {
            auto each_trans = trans[0];
            each_trans.inverse();
            for(int i=0;prog(i,out_count);++i)
            {
                auto t = model_output[0].alias(model_dim.size()*i,model_dim);
                auto o = label_prob.alias(image_dim.size()*i,image_dim);
                tipl::resample(t,o,each_trans);
            }
        }
        else
        {
            tipl::image<3,float> weight_map(image_dim);

            std::atomic<int> p = 0;
            tipl::par_for(out_count+1,[&](int c)
            {
                if(!prog(p++,out_count+2))
                    return;
                for(int t=0;t < model_output.size();++t)
                {
                    auto each_trans = trans[t];
                    each_trans.inverse();

                    if(c == out_count)
                    {
                        weight_map += each_trans(gaussian,image_dim);
                        continue;
                    }
                    auto w = model_output[t].alias(model_dim.size()*c,model_dim);
                    auto o = label_prob.alias(image_dim.size()*c,image_dim);
                    w *= gaussian;
                    o += each_trans(w,image_dim);
                }
            });

            for(size_t i=0;i<weight_map.size();++i)
                if(weight_map[i] > 1e-6)
                    weight_map[i] = 1.0f/weight_map[i];
            p = 0;
            tipl::par_for(out_count,[&](int c)
            {
                if(!prog(p++,out_count+1))
                    return;
                label_prob.alias(image_dim.size()*c,image_dim) *= weight_map;
            });
        }
        return !prog.aborted();
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
        float prob_threshold = 0.5f;
        if(param.size() > 1)
            prob_threshold = std::stof(param[1]);
        if(label_prob.empty())
            return error_msg = "no label probability",false;
        auto labels_4d = tipl::make_image(label_prob.data(), image_dim.expand(out_count));
        fg_prob.resize(image_dim);
        tipl::sum_partial(labels_4d, fg_prob);
        auto original_sum = fg_prob;
        tipl::morphology::defragment_by_threshold(fg_prob, prob_threshold);
        mask = fg_prob > prob_threshold;

        // renormalize
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
        label = tipl::argmax(label_prob,mask);
        return true;
    }
    template<typename io_type>
    bool load_from_file(const std::string& file_name)
    {
        io_type in(file_name,std::ios::in);
        if(!in)
            return error_msg = in.error_msg,false;
        if(in.dim(4) != in_count)
            return error_msg = "input channel mismatch",false;
        in.get_image_transformation(untouched_srow);
        if(!(in >> source_image >> image_vs >> srow))
            return error_msg = "cannot read image data " + in.error_msg,false;

        flip_swap = in.flip_swap_seq;
        image_dim = source_image.shape();
        tipl::out() << "dim: " << source_image.shape() << " vs:" << image_vs << " channel:" << in.dim(4);

        if(in_count > 1)
        {
            source_image.resize(image_dim.multiply(tipl::shape<3>::z,in_count));
            tipl::out() << "handle multiple channels. model channel count:" << in_count;
            for(size_t c = 1;c < in_count;++c)
            {
                auto I = source_image.alias(c*image_dim.size(),image_dim);
                if(!(in >> I))
                    return error_msg = "file corrupted",false;
            }
        }
        return true;
    }
    bool load_from_image(const image_type& raw_image,tipl::vector<3>& vs)
    {
        source_image = raw_image;
        image_dim = source_image.shape();
        image_vs = vs;
        tipl::io::initial_nifti_srow(srow,image_dim,image_vs);
        tipl::io::initial_nifti_srow(untouched_srow,image_dim,image_vs);
        return true;
    }
    bool load_from_image(image_type&& raw_image,tipl::vector<3>& vs)
    {
        source_image = std::move(raw_image);
        image_dim = source_image.shape();
        image_vs = vs;
        tipl::io::initial_nifti_srow(srow,image_dim,image_vs);
        tipl::io::initial_nifti_srow(untouched_srow,image_dim,image_vs);
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
    std::shared_ptr<unet3d> unet;
public:
    evalution_set<tipl::image<3>> eval;
    std::string error_msg,preproc,postproc,version,report;

    template<typename reader>
    bool load_model(const std::string& file_name)
    {
        tipl::progress prog("loading unet model");
        reader in;
        std::vector<int> param({1,1});
        std::string arch;
        tipl::shape<3> dim;
        if(!in.load_from_file(file_name))
            return error_msg = "cannot open file: " + file_name,false;
        if(in.has("feature_string"))
            return error_msg = "cannot read old network format: " + file_name,false;

        if(!in.read("param",param) ||
           !in.read("architecture",arch) ||
           !in.read("postproc",postproc) ||
           !in.read("version",version) ||
           !in.read("report",report) ||
           !in.read_pointer("dimension",eval.model_dim) ||
           !in.read_pointer("voxel_size",eval.model_vs))
            return error_msg = "invalid network file format",false;
        tipl::out() << "dim: " << eval.model_dim << "vs: " << eval.model_vs;
        tipl::out() << "in: " << param[0] << " out:" << param[1];
        tipl::out() << "version: " << version;
        tipl::out() << "report: " << report;
        if(in.read("preproc",preproc))
            tipl::out() << "preproc: " << preproc;
        tipl::out() << "loading unet: " << arch;


        unet.reset(new unet3d(arch,param[0],param[1]));
        auto params = unet->parameters();
        for(int id = 0;prog(id,params.size());++id)
        {
            if(!params[id].second)
                continue;
            std::string name = "tensor" + std::to_string(id);
            if(!in.has(name.c_str()))
                return error_msg = "tensor structure mismatch (missing " + name + ")", false;
            unsigned int tr, tc;
            if(!in.get_col_row(name.c_str(), tr, tc) || tr * tc != params[id].second)
                return error_msg = "tensor size mismatch in " + name, false;

            if(!in.read(name.c_str(), params[id].first, params[id].first + params[id].second))
                return error_msg = "error reading tensor structure " + name, false;
        }
        if(prog.aborted())
            return false;

        if constexpr(tipl::use_cuda)
            if(tipl::has_gpu)
                unet->to_gpu();

        eval.in_count = param[0];
        eval.out_count = param[1];
        return true;
    }

    bool forward(void)
    {
        tipl::progress prog("unet segmentation");
        if(!eval.preproc(preproc))
            return false;

        auto out_shape = eval.model_dim.multiply(tipl::shape<3>::z,eval.out_count);
        {
            for(size_t i = 0;prog(i,eval.model_input.size());++i)
            {
                tipl::progress prog2("forwarding");
                unet->prog = [&](int cur,int total)
                {
                    return prog2(cur,total);
                };
                unet->forward(eval.model_input[i]);
                if constexpr(tipl::use_cuda)
                    if(unet->is_gpu)
                    {
                        eval.model_output.push_back(tipl::image<3>(out_shape));
                        cu_copy_d2h<float,float>(eval.model_output.back().data(),
                                             unet->layers.back()->out,out_shape.size());
                        continue;
                    }
                eval.model_output.push_back(tipl::make_image(unet->layers.back()->out,out_shape));

            }
            if(prog.aborted())
                return false;
        }
        eval.postproc();
        if(!eval.command(postproc))
            return error_msg = eval.error_msg,false;
        prog(4,4);
        return !prog.aborted();
    }
    template<typename image_type>
    bool forward(const image_type& I,tipl::vector<3>& vs)
    {
        eval.load_from_image(I,vs);
        return forward();
    }
    template<typename image_type>
    bool forward(image_type&& I,tipl::vector<3>& vs)
    {
        eval.load_from_image(std::forward<image_type>(I),vs);
        return forward();
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
