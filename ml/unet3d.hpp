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

template<typename T>
auto round_up_size(const T& round_up_multiple,const tipl::shape<3>& s)
{
    return tipl::shape<3>(
        ((s[0]+round_up_multiple[0]-1)/round_up_multiple[0])*round_up_multiple[0],
        ((s[1]+round_up_multiple[1]-1)/round_up_multiple[1])*round_up_multiple[1],
        ((s[2]+round_up_multiple[2]-1)/round_up_multiple[2])*round_up_multiple[2]);
}

template<typename image_type = tipl::image<3>>
struct evalution_set{


public:
    image_type source_image;
    std::vector<image_type> model_io;
    tipl::vector<3,int> round_up_multiple;
public:
    evalution_set(void):label_prob(source_image){}
    std::string error_msg;
    std::vector<tipl::transformation_matrix<float>> trans;
    tipl::image<3,unsigned char> mask,label;
    std::vector<unsigned int> single_component_label;
    image_type& label_prob;
    image_type fg_prob;
public:
    tipl::shape<3> image_dim,model_dim;
    tipl::vector<3> image_vs,model_vs;
    tipl::matrix<4,4,float> untouched_srow,srow;
    std::vector<char> flip_swap;


    size_t in_count = 1,out_count = 1,cur_count = 1;
public:
    tipl::transformation_matrix<float> to_native_space;
    tipl::shape<3> native_space;

    bool handle_orientation(const std::string& cmd,bool reversed = false)
    {
        if(cmd.empty() || model_io.empty())
            return true;
        tipl::progress prog("handle orientation");
        tipl::out() << "run " << cmd;
        auto cmds = tipl::split(cmd,'+');
        if(reversed)
            cmds = std::vector<std::string>(cmds.rbegin(),cmds.rend());
        auto dim = model_io.front().shape().divide(tipl::shape<3>::z,reversed ? out_count:in_count);
        auto new_dim = dim;
        for(const auto& each : cmds)
        {
            if(each == "swap_xy")
                new_dim = tipl::shape<3>(new_dim[1],new_dim[0],new_dim[2]);
            else if(each == "swap_xz")
                new_dim = tipl::shape<3>(new_dim[2],new_dim[1],new_dim[0]);
            else if(each == "swap_yz")
                new_dim = tipl::shape<3>(new_dim[0],new_dim[2],new_dim[1]);
        }

        tipl::par_for(tipl::shape<2>(reversed ? out_count:in_count,model_io.size()),[&](auto index)
        {
            auto I = tipl::make_image(model_io[index.y()].data() + index.x()*dim.size(),dim);
            for(const auto& each : cmds)
            {
                if(each == "swap_xy")
                    tipl::swap_xy(I);
                if(each == "swap_xz")
                    tipl::swap_xz(I);
                if(each == "swap_yz")
                    tipl::swap_yz(I);
                if(each == "flip_xy")
                    tipl::flip_xy(I);
                if(each == "flip_x")
                    tipl::flip_x(I);
                if(each == "flip_y")
                    tipl::flip_y(I);
                if(each == "flip_z")
                    tipl::flip_z(I);
            }
        });
        if(new_dim != dim)
            for(auto& each : model_io)
                each.resize(new_dim.multiply(tipl::shape<3>::z,reversed ? out_count:in_count));
        return true;
    }
    bool run_preproc(const std::string& cmd)
    {
        progress prog("pre-processing");
        if(source_image.empty())
            return error_msg = "no source image",false;

        // universal preprocessing applied
        tipl::par_for(in_count,[&](int c)
        {
            auto I = source_image.alias(c*image_dim.size(),image_dim);
            tipl::segmentation::normalize_otsu_median(I);
        });

        auto cmds = tipl::split(cmd,'+');
        for(size_t i = 0;prog(i,cmds.size());++i)
        {
            param = tipl::split(cmds[i],',');
            progress prog(param[0]);
            if(param[0] == "zscore_nonzero")
            {
                tipl::par_for(in_count,[&](int c)
                {
                    auto I = source_image.alias(c*image_dim.size(),image_dim);
                    double sum = 0.0, sum2 = 0.0;
                    size_t n = 0;
                    for(float v : I)
                        if(v != 0.0f)
                            sum += v, sum2 += double(v)*double(v), ++n;
                    if(n)
                    {
                        double mean = sum / double(n);
                        double var = sum2 / double(n) - mean*mean; // population variance
                        double sd = std::sqrt(std::max(0.0,var));
                        if(sd == 0.0)
                            sd = 1.0;

                        for(float& v : I)
                            if(v != 0.0f)
                                v = float((double(v) - mean) / sd);
                    }
                });
                continue;
            }
        }
        return !prog.aborted();
    }

    bool handle_fov_pre(const std::string& fov_strategy)
    {
        tipl::progress prog("handle fov");
        if(mask.empty())
        {
            tipl::threshold(source_image.alias(0,image_dim),mask,0.0f);
            tipl::morphology::defragment(mask);
        }

        tipl::vector<3> from,to;
        if(!tipl::bounding_box(mask,from,to))
            return error_msg = "source image is all zero",false;

        tipl::vector<3> mask_center((from+to)*0.5f),image_center(tipl::vector<3>(image_dim)*0.5f);
        tipl::vector<3> mask_center_phys(mask_center),image_center_phys(image_center);
        tipl::vector<3> patch_phys(model_dim),bb_phys(to-from),mask_top_phys(to);

        mask_center_phys.elem_mul(image_vs);
        image_center_phys.elem_mul(image_vs);
        patch_phys.elem_mul(model_vs);
        bb_phys.elem_mul(image_vs);
        mask_top_phys.elem_mul(image_vs);

        auto add_view = [&](const tipl::vector<3>& shift,auto& input,auto &t)
        {
            tipl::affine_param<float> arg;
            arg.translocation[0] = shift[0];
            arg.translocation[1] = shift[1];
            arg.translocation[2] = shift[2];

            t = tipl::transformation_matrix<float,3>(arg,model_dim,model_vs,image_dim,image_vs);
            if(model_vs == image_vs)
            {
                t.shift[0] = std::round(t.shift[0]);
                t.shift[1] = std::round(t.shift[1]);
                t.shift[2] = std::round(t.shift[2]);
            }
            image_type target_image(model_dim.multiply(tipl::shape<3>::z,in_count));

            for(int c=0;c<in_count;++c)
                t(make_image(source_image,image_dim.size()*c,image_dim),
                     make_image(target_image,model_dim.size()*c,model_dim));

            input = std::move(target_image);
        };

        tipl::out() << "fov_strategy: " << fov_strategy;
        if(fov_strategy == "sliding_window")
        {
            std::vector<float> s[3];
            for(int d = 0;d < 3;++d)
            {
                if(bb_phys[d] > 2.0f*patch_phys[d])
                    bb_phys[d] = 2.0f*patch_phys[d];
                if(bb_phys[d] > patch_phys[d])
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

            if(active_shifts.size() > 1)
            {
                active_shifts.push_back(tipl::vector<3>(0.0f,0.0f,0.0f));
                tipl::out() << "total of sliding window:" << active_shifts.size();
                model_io.resize(active_shifts.size());
                trans.resize(active_shifts.size());

                tipl::par_for(active_shifts.size(),[&](int i)
                {
                    add_view(active_shifts[i]+mask_center_phys-image_center_phys,model_io[i],trans[i]);
                });
                goto end;
            }
            // if active_shifts == 1 then use default align_center
        }

        if(fov_strategy == "image")
        {
            model_vs[0] = model_vs[1] = model_vs[2] = ((image_vs[0] + image_vs[2]) * 0.5f);
            patch_phys = model_dim = round_up_size(round_up_multiple,
                    tipl::shape<3>(bb_phys[0]/model_vs[0],bb_phys[1]/model_vs[1],bb_phys[2]/model_vs[2]));
            patch_phys.elem_mul(model_vs);

            model_io.resize(1);
            trans.resize(1);
            add_view(mask_center_phys-image_center_phys,model_io[0],trans[0]);
            mask = trans[0].template operator()<tipl::majority>(mask,model_dim);
        }
        else if(fov_strategy == "align_top")
        {
            model_io.resize(1);
            trans.resize(1);

            auto view_center_phys = mask_center_phys;
            view_center_phys[2] = mask_top_phys[2] - patch_phys[2]*0.5f;

            add_view(view_center_phys-image_center_phys,model_io[0],trans[0]);
            mask = trans[0].template operator()<tipl::majority>(mask,model_dim);

        }
        else // default align_center
        {
            model_io.resize(1);
            trans.resize(1);
            add_view(mask_center_phys-image_center_phys,model_io[0],trans[0]);
            mask = trans[0].template operator()<tipl::majority>(mask,model_dim);
        }

        end:
        source_image.clear();
        for(auto& each : trans)
            tipl::out() << "trans:" << each;
        for(auto& each : trans)
        {
            each.inverse();
            tipl::out() << "inv trans:" << each;
        }
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
        tipl::par_for(cur_count, [&](size_t c)
        {
            tipl::filter::gaussian(tipl::make_image(label_prob.data()+mask.size()*c,mask.shape()));
        });
        return true;
    }
    std::vector<std::string> param;
    bool run_postproc(const std::string& cmd)
    {
        progress prog("post-processing");
        auto cmds = tipl::split(cmd,'+');
        for(size_t i = 0;prog(i,cmds.size());++i)
        {
            param = tipl::split(cmds[i],',');
            progress prog(param[0]);
            if(param[0] == "create_bg" && create_bg())
                continue;
            if(param[0] == "softmax" && softmax())
                continue;
            if(param[0] == "argmax" && argmax())
                continue;
            if(param[0] == "create_mask" && create_mask())
                continue;
            if(param[0] == "clamp_prob" && clamp_prob())
                continue;
            if(param[0] == "smoothing" && smoothing())
                continue;
            if(error_msg.empty())
                error_msg = "invalid command: " + cmds[i];
            return false;
        }

        if(model_io.size() == 1 && label_prob.size() == model_dim.size()*cur_count)
        {
            tipl::progress prog("handle fov");
            if(cur_count == 0)
                return error_msg = "invalid channel count",false;

            image_type new_label_prob(image_dim.multiply(tipl::shape<3>::z,cur_count));

            tipl::par_for(cur_count,[&](size_t i)
            {
                auto dim = label_prob.shape().divide(tipl::shape<3>::z,cur_count);
                auto t = label_prob.alias(dim.size()*i,dim);
                auto o = new_label_prob.alias(image_dim.size()*i,image_dim);
                trans[0](t,o);
            });

            new_label_prob.swap(label_prob);

            if(!fg_prob.empty())
                fg_prob = trans[0](fg_prob,image_dim);
            if(!label.empty())
                label = trans[0].template operator()<tipl::majority>(label,image_dim);
            if(!mask.empty())
                mask = trans[0].template operator()<tipl::majority>(mask,image_dim);
        }
        model_io.clear();
        return !prog.aborted();
    }
    bool softmax(void)
    {
        if(label_prob.empty())
            return error_msg = "empty label probability",false;
        tipl::softmax(label_prob,mask.size(),cur_count);
        return true;
    }

    bool handle_fov_post()
    {
        if(model_io.empty())
            return error_msg = "no output data",false;

        if(model_io.size() > 1)
        {
            auto dim = model_io.front().shape().divide(tipl::shape<3>::z,out_count);
            // generate gaussian basis
            tipl::image<3> gaussian(dim);
            {

                float sigma_scale = 1.0f/8.0f;
                tipl::vector<3> center(tipl::vector<3>(dim)*0.5f);
                float max_val = 0.0f;
                tipl::vector<3> inv_sigmas(1.0f/(dim[0]*sigma_scale),1.0f/(dim[1]*sigma_scale),1.0f/(dim[2]*sigma_scale));
                size_t sz = dim.size();
                for(tipl::pixel_index<3> index(dim);index < sz;++index)
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

            tipl::progress prog("handle sliding window");
            label_prob.resize(image_dim.multiply(tipl::shape<3>::z,out_count));
            label_prob = 0.0f;
            tipl::image<3,float> weight_map(image_dim);

            tipl::par_for(out_count+1,[&](int c)
            {
                for(int t=0;t < model_io.size();++t)
                {
                    if(c == out_count)
                    {
                        weight_map += trans[t](gaussian,image_dim);
                        continue;
                    }

                    auto w = model_io[t].alias(dim.size()*c,dim);
                    auto o = label_prob.alias(image_dim.size()*c,image_dim);
                    w *= gaussian;
                    o += trans[t](w,image_dim);
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
        else
            label_prob = std::move(model_io[0]);
        cur_count = out_count;
        return true;
    }
    bool create_bg(void)
    {
        size_t image_size = mask.size();
        label_prob.resize(mask.shape().multiply(tipl::shape<3>::z,cur_count+1));
        size_t total_size = label_prob.size();
        tipl::par_for<sequential>(image_size, [&](size_t pos)
        {
            double sum(0);
            for(size_t offset = total_size - image_size + pos; offset >= image_size; offset -= image_size)
                sum += label_prob[offset] = label_prob[offset-image_size];
            if(sum < 1.0)
                label_prob[pos] = 1.0f-sum;
            else
                label_prob[pos] = 0.0f;
        });
        ++cur_count;
        return true;
    }
    bool create_mask(void)
    {
        float prob_threshold = 0.5f;
        if(param.size() > 1)
            prob_threshold = std::stof(param[1]);
        if(label_prob.empty())
            return error_msg = "no label probability",false;
        fg_prob = label_prob.alias(0,mask.shape());
        tipl::filter::gaussian(fg_prob);
        tipl::filter::gaussian(fg_prob);

        // refine current mask based on model output
        tipl::masking_by_value(mask,fg_prob,prob_threshold);

        tipl::morphology::defragment(mask);
        tipl::morphology::smoothing(mask);
        tipl::morphology::negate(mask);
        tipl::morphology::defragment(mask);
        tipl::morphology::negate(mask);
        fg_prob = mask;
        tipl::filter::gaussian(fg_prob);
        tipl::filter::gaussian(fg_prob);

        size_t image_size = mask.size();
        size_t total_size = label_prob.size();

        // renormalize
        tipl::par_for<sequential>(image_size, [&](size_t pos)
        {
            double sum(0);
            for(size_t offset = pos + image_size; offset < total_size; offset += image_size)
                sum += label_prob[offset];
            label_prob[pos] = 1.0-fg_prob[pos];
            float scale = (sum == 0.0 ? 0.0f : float(fg_prob[pos]/sum));
            for(size_t offset = pos + image_size; offset < total_size; offset += image_size)
                label_prob[offset] *= scale;
        });
        return true;
    }
    bool argmax(void)
    {
        if(label_prob.empty())
            return error_msg = "no label probability",false;
        if(fg_prob.empty() && !create_mask())
            return false;

        const size_t image_size = mask.size();
        auto prob = label_prob.alias(image_size,mask.shape().multiply(tipl::shape<3>::z,cur_count-1));
        label = tipl::argmax<1>(prob,mask.shape(),mask.data());

        if(single_component_label.empty())
            return true;

        for(size_t iter = 0;iter < 5;++iter)
        {
            std::vector<std::pair<unsigned int,size_t> > result;
            for(auto c : single_component_label)
            {
                if(c == 0 || c >= cur_count)
                    return error_msg = "invalid single component label of " + std::to_string(c),false;

                size_t changed = 0;
                tipl::image<3,unsigned char> m(mask.shape()), kept;
                for(size_t i = 0;i < image_size;++i)
                    m[i] = label[i] == c;

                kept = m;
                tipl::morphology::defragment(kept);

                auto p = label_prob.alias(image_size*c,mask.shape());
                for(size_t i = 0;i < image_size;++i)
                    if(m[i] && !kept[i])
                        p[i] = 0.0f,++changed;

                if(changed)
                    result.push_back({c,changed});
            }

            if(result.empty())
                return true;
            label = tipl::argmax<1>(prob,mask.shape(),mask.data());
            std::stringstream out;
            out << "apply single component: ";
            for(auto& each : result)
                out << "label " << each.first << " removed " << each.second << " voxel(s) ";
            tipl::out() << out.str();
        }
        return true;
    }
    template<typename io_type>
    bool load_from_file(const std::filesystem::path& file_name)
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
};



class tissue_seg{
private:

public:
    std::shared_ptr<unet3d> unet;
public:
    evalution_set<tipl::image<3>> eval;
    std::string error_msg;
    std::string preproc,postproc,orientation,fov_strategy,arch;
    std::vector<std::string> labels;

    template<typename reader>
    bool load_model(const std::filesystem::path& file_name)
    {
        tipl::progress prog("loading unet model");
        reader in;
        std::vector<int> channels({1,1});
        if(!in.load_from_file(file_name))
            return error_msg = "cannot open file: " + file_name.u8string(),false;
        if(in.has("feature_string"))
            return error_msg = "cannot read old network format: " + file_name.u8string(),false;

        if(!in.read("channels",channels) ||
           !in.read("architecture",arch) ||
           !in.read_pointer("dimension",eval.model_dim) ||
           !in.read_pointer("voxel_size",eval.model_vs))
            return error_msg = "invalid model format: " + in.error_msg,false;
        if(in.read("fov_strategy",fov_strategy))
            tipl::out() << "fov_strategy: " << fov_strategy;
        if(in.read("postproc",postproc))
            tipl::out() << "postproc: " << postproc;
        if(in.read("orientation",orientation))
            tipl::out() << "orientation: " << orientation;
        if(in.read("preproc",preproc))
            tipl::out() << "preproc: " << preproc;
        labels = tipl::split(in.template read<std::string>("labels"),'\n');

        if(!(eval.single_component_label = in.template read_as_vector<unsigned int>("single_component_label")).empty())
        {
            std::string out;
            for(auto each : eval.single_component_label)
            {
                if(!out.empty())
                    out += ", ";
                if(each == 0 || each-1 >= labels.size())
                    return error_msg = "invalid single component label: " + std::to_string(each), false;
                out += labels[each-1];
            }
            tipl::out() << "single_component_label: " << out;
        }
        tipl::out() << "dim: " << eval.model_dim << " vs: " << eval.model_vs;
        tipl::out() << "in: " << channels[0] << " out:" << channels[1];
        tipl::out() << "loading unet: " << arch;

        try{
            unet.reset(new unet3d(arch,channels[0],channels[1]));
            eval.round_up_multiple = unet->round_up_multiple;
        }
        catch(std::runtime_error& e)
        {
            return error_msg = e.what(),false;
        }

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

        eval.in_count = channels[0];
        eval.out_count = channels[1];
        return true;
    }

    bool forward(void)
    {
        tipl::progress prog("unet segmentation");
        if(!eval.run_preproc(preproc) || !eval.handle_fov_pre(fov_strategy) || !eval.handle_orientation(orientation))
            return false;

        {
            for(size_t i = 0;prog(i,eval.model_io.size());++i)
            {
                tipl::progress prog2("forwarding");
                unet->prog = [&](int cur,int total)
                {
                    return prog2(cur,total);
                };
                unet->forward(eval.model_io[i]);
                auto out_shape = eval.model_io[i].shape().multiply(tipl::shape<3>::z,eval.out_count/eval.in_count);
                if constexpr(tipl::use_cuda)
                    if(unet->is_gpu)
                    {
                        tipl::image<3> out(out_shape);
                        cu_copy_d2h<float,float>(out.data(),
                                             unet->layers.back()->out,out_shape.size());
                        eval.model_io[i] = std::move(out);
                        continue;
                    }
                eval.model_io[i] = tipl::make_image(unet->layers.back()->out,out_shape);
            }
            if(prog.aborted())
                return false;
        }

        if(!eval.handle_orientation(orientation,true) || !eval.handle_fov_post() || !eval.run_postproc(postproc))
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



}//ml3d


}//tipl

#endif//UNET3D_HPP
