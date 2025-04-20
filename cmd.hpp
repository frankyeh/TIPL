#ifndef CMD_HPP
#define CMD_HPP
#include <sstream>
#include "utility/basic_image.hpp"
#include "numerical/basic_op.hpp"
#include "numerical/matrix.hpp"
#include "numerical/resampling.hpp"
#include "numerical/transformation.hpp"
#include "filter/mean.hpp"
#include "filter/sobel.hpp"
#include "filter/gaussian.hpp"
#include "filter/anisotropic_diffusion.hpp"
#include "morphology/morphology.hpp"
#include "segmentation/otsu.hpp"

namespace tipl{

template<typename image_type>
bool command(image_type& data,std::string cmd,std::string param1)
{
    if constexpr (std::is_floating_point<typename image_type::value_type>::value)
    {
        if(cmd.find("morphology") == 0)
        {
            tipl::image<image_type::dimension,char> mask(data.shape());
            for(size_t pos = 0;pos < mask.size();++pos)
                mask[pos] = data[pos] > typename image_type::value_type(0) ? 1 : 0;
            if(!command(mask,cmd,param1))
                return false;
            for(size_t pos = 0;pos < mask.size();++pos)
                data[pos] = typename image_type::value_type(mask[pos]);
            return true;
        }
    }
    if(cmd == "morphology_defragment")
    {
        tipl::morphology::defragment(data);
        return true;
    }
    if(cmd == "morphology_defragment_by_size")
    {
        tipl::morphology::defragment_by_size_ratio(data,param1.empty() ? 0.05f : std::stof(param1));
        return true;
    }
    if(cmd == "morphology_dilation")
    {
        tipl::morphology::for_each_label(data,[](tipl::image<3,char>& mask){tipl::morphology::dilation(mask);});
        return true;
    }
    if(cmd == "morphology_erosion")
    {
        tipl::morphology::for_each_label(data,[](tipl::image<3,char>& mask){tipl::morphology::erosion(mask);});
        return true;
    }
    if(cmd == "morphology_edge")
    {
        tipl::morphology::for_each_label(data,[](tipl::image<3,char>& mask){tipl::morphology::edge(mask);});
        return true;
    }
    if(cmd == "morphology_edge_xy")
    {
        tipl::morphology::for_each_label(data,[](tipl::image<3,char>& mask){tipl::morphology::edge_xy(mask);});
        return true;
    }
    if(cmd == "morphology_edge_xz")
    {
        tipl::morphology::for_each_label(data,[](tipl::image<3,char>& mask){tipl::morphology::edge_xz(mask);});
        return true;
    }
    if(cmd == "morphology_smoothing")
    {
        if(std::any_of(data.begin(), data.end(), [](auto val) { return val != 0 && val != 1; }))
            tipl::morphology::smoothing_multiple_region(data);
        else
            tipl::morphology::smoothing(data);
        return true;
    }
    if(cmd == "sobel_filter")
    {
        tipl::filter::sobel(data);
        return true;
    }
    if(cmd == "gaussian_filter")
    {
        tipl::filter::gaussian(data);
        return true;
    }
    if(cmd == "mean_filter")
    {
        tipl::filter::mean(data);
        return true;
    }
    if(cmd == "smoothing_filter")
    {
        tipl::filter::anisotropic_diffusion(data);
        return true;
    }
    if(cmd == "normalize")
    {
        if constexpr(std::is_integral<typename image_type::value_type>::value)
        {
            tipl::normalize(data,255.0f);
            tipl::upper_threshold(data,255);
        }
        else
            tipl::normalize(data);
        return true;
    }
    if(cmd == "normalize_otsu_median")
    {
        tipl::segmentation::normalize_otsu_median(data);
        return true;
    }
    if(cmd == "flip_x")
    {
        tipl::flip_x(data);
        return true;
    }
    if(cmd == "flip_y")
    {
        tipl::flip_y(data);
        return true;
    }

    if(cmd == "flip_z")
    {
        tipl::flip_z(data);
        return true;
    }

    // need param1
    if(param1.empty())
        return false;

    if(cmd == "select_value")
    {
        typename image_type::value_type v = std::stof(param1);
        for(size_t index = 0;index < data.size();++index)
            data[index] = (data[index] == v ? 1:0);
        return true;
    }
    if(cmd == "add_value")
    {
        add_constant(data,std::stof(param1));
        return true;
    }
    if(cmd == "multiply_value")
    {
        multiply_constant(data,std::stof(param1));
        return true;
    }
    if(cmd == "lower_threshold")
    {
        lower_threshold(data,std::stof(param1));
        return true;
    }
    if(cmd == "upper_threshold")
    {
        upper_threshold(data,std::stof(param1));
        return true;
    }

    if(cmd == "threshold")
    {
        float value = std::stof(param1);
        for(size_t i = 0;i < data.size();++i)
            data[i] = data[i] > value ? 1.0f : 0.0f;
        return true;
    }
    if(cmd == "otsu_threshold")
    {
        float threshold = tipl::segmentation::otsu_threshold(data)*float(std::stof(param1));
        for(size_t i = 0;i < data.size();++i)
            data[i] = data[i] > threshold ? 1.0f : 0.0f;
        return true;
    }
    return false;
}


template<typename out = void,typename image_type>
bool equation(image_type& x,std::string eq,std::string& error_msg);

template<typename out = void,typename image_loader,typename image_type>
bool command(image_type& data,tipl::vector<3>& vs,tipl::matrix<4,4>& T,bool& is_mni,
             std::string cmd,std::string param1,bool interpolation,std::string& error_msg)
{
    if constexpr(!std::is_void_v<out>)out() << std::string(param1.empty() ? cmd : cmd+":"+param1);

    if(cmd == "equation")
        return equation<out>(data,param1,error_msg);
    if(command(data,cmd,param1))
        return true;
    if(cmd == "set_transformation")
    {
        std::istringstream in(param1);
        for(int i = 0;i < 16;++i)
            in >> T[i];
        for(int i = 0;i < 3;++i)
            vs[i] = std::sqrt(T[i]*T[i]+T[i+4]*T[i+4]+T[i+8]*T[i+8]);
        return true;
    }
    if(cmd == "set_translocation")
    {
        std::istringstream in(param1);
        in >> T[3] >> T[7] >> T[11];
        return true;
    }
    if(cmd == "set_mni")
    {
        if(param1.empty())
        {
            error_msg = "invalid value";
            return false;
        }
        is_mni = (param1[0] == '1');
        return true;
    }
    if(cmd == "upsampling")
    {
        tipl::upsampling(data);
        vs *= 0.5f;
        T = T*tipl::matrix<4,4>({0.5f,0.0f,0.0f,0.0f,
                                 0.0f,0.5f,0.0f,0.0f,
                                 0.0f,0.0f,0.5f,0.0f,
                                 0.0f,0.0f,0.0f,1.0f});

        return true;
    }
    if(cmd == "downsampling")
    {
        tipl::downsample_with_padding(data);
        vs *= 2.0f;
        T = T*tipl::matrix<4,4>({2,0,0,0,
                                 0,2,0,0,
                                 0,0,2,0,
                                 0,0,0,1});
        return true;
    }
    if(cmd == "header_flip_x")
    {
        T[3] += T[0]*float(data.width()-1);
        T[7] += T[4]*float(data.width()-1);
        T[11] += T[8]*float(data.width()-1);
        T[0] = -T[0];
        T[4] = -T[4];
        T[8] = -T[8];
        return true;
    }
    if(cmd == "header_flip_y")
    {
        T[3] += T[1]*float(data.height()-1);
        T[7] += T[5]*float(data.height()-1);
        T[11] += T[9]*float(data.height()-1);
        T[1] = -T[1];
        T[5] = -T[5];
        T[9] = -T[9];
        return true;
    }

    if(cmd == "header_flip_z")
    {
        T[3] += T[2]*float(data.depth()-1);
        T[7] += T[6]*float(data.depth()-1);
        T[11] += T[10]*float(data.depth()-1);
        T[2] = -T[2];
        T[6] = -T[6];
        T[10] = -T[10];
        return true;
    }
    if(cmd == "header_swap_xy")
    {
        T = tipl::matrix<4,4>({0,1,0,0,
                               1,0,0,0,
                               0,0,1,0,
                               0,0,0,1})*T;
        std::swap(vs[0],vs[1]);
        return true;
    }
    if(cmd == "header_swap_xz")
    {
        T = tipl::matrix<4,4>({0,0,1,0,
                               0,1,0,0,
                               1,0,0,0,
                               0,0,0,1})*T;
        std::swap(vs[0],vs[2]);
        return true;
    }
    if(cmd == "header_swap_yz")
    {
        T = tipl::matrix<4,4>({1,0,0,0,
                               0,0,1,0,
                               0,1,0,0,
                               0,0,0,1})*T;
        std::swap(vs[1],vs[2]);
        return true;
    }

    if(cmd == "swap_xy")
    {
        tipl::swap_xy(data);
        std::swap(vs[0],vs[1]);
        return true;
    }
    if(cmd == "swap_xz")
    {
        tipl::swap_xz(data);
        std::swap(vs[0],vs[2]);
        return true;
    }
    if(cmd == "swap_yz")
    {
        tipl::swap_yz(data);
        std::swap(vs[1],vs[2]);
        return true;
    }


    // need param1
    if(param1.empty())
    {
        error_msg = "need param1";
        return false;
    }
    if(cmd == "crop_to_fit")
    {
        tipl::vector<3,int> range_min,range_max,margin;
        tipl::bounding_box(data,range_min,range_max,data[0]);
        std::istringstream in(param1);
        in >> margin[0] >> margin[1] >> margin[2];
        range_min[0] = std::max<int>(0,range_min[0]-margin[0]);
        range_min[1] = std::max<int>(0,range_min[1]-margin[1]);
        range_min[2] = std::max<int>(0,range_min[2]-margin[2]);
        range_max[0] = std::min<int>(data.width(),range_max[0]+margin[0]);
        range_max[1] = std::min<int>(data.height(),range_max[1]+margin[1]);
        range_max[2] = std::min<int>(data.depth(),range_max[2]+margin[2]);
        range_max -= range_min;
        typename image_type::buffer_type original_data(data);
        data.resize(tipl::shape<3>(range_max.begin()));
        tipl::draw(original_data,data,range_min);
        return true;
    }
    if(cmd == "transform")
    {
        tipl::matrix<4,4> U((tipl::identity_matrix()));
        std::istringstream in(param1);
        for(int i = 0;i < 12;++i)
        {
            if(!in)
            {
                error_msg = "invalid transformation matrix";
                return false;
            }
            in >> U[i];
        }
        tipl::vector<3> new_vs;
        for(int i = 0;i < 3;++i)
            new_vs[i] = std::sqrt(U[i]*U[i]+U[i+4]*U[i+4]+U[i+8]*U[i+8]);

        // main axis not in diagonal, just use transformation
        if(T[1] != 0.0f || U[1] != 0.0f ||
           T[2] != 0.0f || U[2] != 0.0f ||
           T[4] != 0.0f || U[4] != 0.0f ||
           T[6] != 0.0f || U[6] != 0.0f ||
           T[8] != 0.0f || U[8] != 0.0f ||
           T[9] != 0.0f || U[9] != 0.0f)
        {
            image_type new_data(data.shape());
            if(interpolation)
                tipl::resample<tipl::interpolation::linear>(data,new_data,tipl::transformation_matrix<float>(tipl::from_space(U).to(T)));
            else
                tipl::resample<tipl::interpolation::majority>(data,new_data,tipl::transformation_matrix<float>(tipl::from_space(U).to(T)));
            new_data.swap(data);
            vs = new_vs;
            return true;
        }
        // flip in x y z
        if(T[0]*U[0] < 0)
        {
            command<out,image_loader>(data,vs,T,is_mni,"header_flip_x","",interpolation,error_msg);
            command<out,image_loader>(data,vs,T,is_mni,"flip_x","",interpolation,error_msg);
            return command<out,image_loader>(data,vs,T,is_mni,cmd,param1,interpolation,error_msg);
        }
        if(T[5]*U[5] < 0)
        {
            command<out,image_loader>(data,vs,T,is_mni,"header_flip_y","",interpolation,error_msg);
            command<out,image_loader>(data,vs,T,is_mni,"flip_y","",interpolation,error_msg);
            return command<out,image_loader>(data,vs,T,is_mni,cmd,param1,interpolation,error_msg);
        }
        if(T[10]*U[10] < 0)
        {
            command<out,image_loader>(data,vs,T,is_mni,"header_flip_z","",interpolation,error_msg);
            command<out,image_loader>(data,vs,T,is_mni,"flip_z","",interpolation,error_msg);
            return command<out,image_loader>(data,vs,T,is_mni,cmd,param1,interpolation,error_msg);
        }
        // consider voxel size
        if(T[0] != U[0] || T[5] != U[5] || T[10] != U[10])
        {
            if(!command<out,image_loader>(data,vs,T,is_mni,"regrid",std::to_string(new_vs[0])+" "+std::to_string(new_vs[1])+" "+std::to_string(new_vs[2]),interpolation,error_msg))
                return false;
            T[0] = U[0];
            T[5] = U[5];
            T[10] = U[10];
            vs = new_vs;
            return command<out,image_loader>(data,vs,T,is_mni,cmd,param1,interpolation,error_msg);
        }
        // now translocation
        cmd = "translocate";
        tipl::vector<3> shift((T[3] - U[3])/T[0],(T[7] - U[7])/T[5],(T[11] - U[11])/T[10]);
        if(shift[0] == 0.0f && shift[1] == 0.0f && shift[2] == 0.0f)
            return true;
        param1 = std::to_string(shift[0])+" "+std::to_string(shift[1])+" "+std::to_string(shift[2]);
    }
    if(cmd == "translocate")
    {
        image_type new_data(data.shape());
        std::istringstream in(param1);
        tipl::vector<3> shift;
        in >> shift[0] >> shift[1] >> shift[2];
        tipl::vector<3> ishift(shift);
        ishift.floor();
        if(ishift != shift)
        {
            tipl::transformation_matrix<float> m;
            m.sr[0] = m.sr[4] = m.sr[8] = 1.0f;
            m.shift[0] = shift[0];
            m.shift[1] = shift[1];
            m.shift[2] = shift[2];
            T[3] -= T[0]*m.shift[0];
            T[7] -= T[5]*m.shift[1];
            T[11] -= T[10]*m.shift[2];
            // invert m
            m.shift[0] = -m.shift[0];
            m.shift[1] = -m.shift[1];
            m.shift[2] = -m.shift[2];
            if(interpolation)
                tipl::resample<tipl::interpolation::linear>(data,new_data,m);
            else
                tipl::resample<tipl::interpolation::majority>(data,new_data,m);
        }
        else
        {
            tipl::draw(data,new_data,tipl::vector<3,int>(shift));
            T[3] -= T[0]*shift[0];
            T[7] -= T[5]*shift[1];
            T[11] -= T[10]*shift[2];
        }
        data.swap(new_data);
        return true;
    }

    if(cmd == "resize")
    {
        std::istringstream in(param1);
        int w(0),h(0),d(0);
        in >> w >> h >> d;
        if(!w || !h || !d)
        {
            error_msg = "invalid size";
            return false;
        }
        typename image_type::buffer_type original_data(data);
        data.resize(tipl::shape<3>(w,h,d));
        tipl::draw(original_data,data,tipl::vector<3,int>(0,0,0));
        return true;
    }
    if(cmd == "reshape")
    {
        std::istringstream in(param1);
        int w(0),h(0),d(0);
        in >> w >> h >> d;
        if(!w || !h || !d)
        {
            error_msg = "invalid size";
            return false;
        }
        reshape(data,tipl::shape<3>(w,h,d));
        return true;
    }

    if(cmd == "regrid")
    {
        std::istringstream iss(param1);
        std::vector<float> values((std::istream_iterator<float>(iss)),std::istream_iterator<float>());
        if(values.size() == 1)
        {
            values.push_back(values[0]);
            values.push_back(values[0]);
        }
        if(values.size() != 3)
        {
            error_msg = "invalid resolution";
            return false;
        }
        tipl::vector<3> new_vs(values);
        image_type J(tipl::shape<3>(
                int(std::ceil(float(data.width())*vs[0]/new_vs[0])),
                int(std::ceil(float(data.height())*vs[1]/new_vs[1])),
                int(std::ceil(float(data.depth())*vs[2]/new_vs[2]))));
        if(J.empty())
        {
            error_msg = "invalid image dim";
            return false;
        }
        tipl::transformation_matrix<float> T1;
        tipl::matrix<4,4> nT;
        nT.identity();
        nT[0] = T1.sr[0] = new_vs[0]/vs[0];
        nT[5] = T1.sr[4] = new_vs[1]/vs[1];
        nT[10] = T1.sr[8] = new_vs[2]/vs[2];
        if(interpolation)
            tipl::resample<tipl::interpolation::linear>(data,J,T1);
        else
            tipl::resample<tipl::interpolation::majority>(data,J,T1);
        data.swap(J);
        vs = new_vs;
        T = T*nT;
        return true;
    }


    if(cmd == "load_image" || cmd == "multiply_image" || cmd == "add_image" || cmd == "minus_image" || cmd == "max_image" || cmd == "min_image")
    {
        tipl::image<3,typename image_type::value_type> rhs(data.shape());
        if(!image_loader::load_to_space(param1.c_str(),rhs,T))
        {
            error_msg = "cannot open file:";
            error_msg += param1;
            return false;
        }
        if(cmd == "load_image")
            data = std::move(rhs);
        if(cmd == "multiply_image")
            data *= rhs;
        if(cmd == "add_image")
            data += rhs;
        if(cmd == "minus_image")
            data -= rhs;
        if(cmd == "max_image")
        {
            for(size_t i = 0;i < data.size();++i)
                data[i] = std::max(data[i],rhs[i]);
        }
        if(cmd == "min_image")
        {
            for(size_t i = 0;i < data.size();++i)
                data[i] = std::min(data[i],rhs[i]);
        }
        return true;
    }
    if(cmd == "concatenate_image")
    {
        image_loader nii;
        if(!nii.load_from_file(param1.c_str()))
            return false;
        if(nii.width() != data.width() ||
           nii.height() != data.height())
        {
            error_msg = "inconsistent image width and height";
            return false;
        }
        size_t pos = data.size();
        data.resize(data.shape().add(tipl::shape<3>::z,nii.depth()));
        auto new_space = data.alias(pos,tipl::shape<3>(nii.width(),nii.height(),nii.depth()));
        nii.get_untouched_image(new_space);
        return true;
    }
    if(cmd == "save")
    {
        image_loader nii;
        nii.set_image_transformation(T,is_mni);
        nii.set_voxel_size(vs);
        nii << data;
        if(!nii.save_to_file(param1.c_str()))
        {
            error_msg = nii.error_msg;
            return false;
        }
        return true;
    }
    if(cmd == "open")
    {
        image_loader nii;
        if(!nii.load_from_file(param1.c_str()))
        {
            error_msg = nii.error_msg;
            return false;
        }
        nii.get_image_transformation(T);
        nii.get_voxel_size(vs);
        is_mni = nii.is_mni();
        nii >> data;
        return true;
    }
    error_msg = "unknown command:";
    error_msg += cmd;
    return false;
}




template<typename image_type1,typename image_type2,
         typename std::enable_if<std::is_class<image_type1>::value,bool>::type = true,
         typename std::enable_if<std::is_class<image_type2>::value,bool>::type = true>
void equation(image_type1& lhs,const image_type2& rhs,char op)
{
    switch(op)
    {
        case '+': tipl::add(lhs,rhs);return;
        case '-': tipl::minus(lhs,rhs);return;
        case '*': tipl::multiply(lhs,rhs);return;
        case '/': tipl::divide(lhs,rhs);return;
        case '>': tipl::greater(lhs,rhs);return;
        case '<': tipl::lesser(lhs,rhs);return;
        case '=': tipl::equal(lhs,rhs);return;
    }
}

template<typename image_type1,typename value_type,
         typename std::enable_if<std::is_fundamental<value_type>::value,bool>::type = true>
void equation(image_type1& lhs,value_type rhs,char op)
{
    switch(op)
    {
        case '+': tipl::add_constant(lhs,rhs);return;
        case '-': tipl::minus_constant(lhs,rhs);return;
        case '*': tipl::multiply_constant(lhs,rhs);return;
        case '/': tipl::divide_constant(lhs,rhs);return;
        case '>': tipl::greater_constant(lhs,rhs);return;
        case '<': tipl::lesser_constant(lhs,rhs);return;
        case '=': tipl::equal_constant(lhs,rhs);return;
    }
}

template<typename value_type,typename image_type1,
         typename std::enable_if<std::is_fundamental<value_type>::value,bool>::type = true>
void equation(value_type lhs,image_type1& rhs,char op)
{
    switch(op)
    {
        case '+': tipl::add_constant(rhs,lhs);return;
        case '=': tipl::equal_constant(rhs,lhs);return;
        case '*': tipl::multiply_constant(rhs,lhs);return;
        case '>': tipl::lesser_constant(rhs,lhs);return;
        case '<': tipl::greater_constant(rhs,lhs);return;
        case '/': tipl::divide_by_constant(rhs,lhs);return;
        case '-': tipl::minus_by_constant(rhs,lhs);return;
    }
}

template<typename out,typename image_type>
bool equation(image_type& x,std::string eq,std::string& error_msg)
{
    if(eq == "x" || eq.empty())
        return true;

    using buf_image_type = tipl::image<image_type::dimension,typename image_type::value_type>;

    unsigned char parentheses = 0;
    std::vector<std::string> tokens;
    std::vector<char> op;
    std::string cur_token;

    auto is_number = [](const std::string& str) -> bool
    {
        try {
                std::size_t pos;
                float n = std::stof(str, &pos);
                return pos == str.size();
            } catch (...) {
                return false;
            }
    };

    auto add_op = [&](char ch)
    {
        tokens.push_back(cur_token);
        cur_token.clear();
        op.push_back(ch);
    };

    for (int i = 0;i < eq.size();++i)
    {
        auto c = eq[i];
        if (!parentheses && !cur_token.empty())
        {
            if(c == '(' && (cur_token == "x" || cur_token.back() == ')' || is_number(cur_token)))
                add_op('*');
            if(c == 'x' && (cur_token.back() == ')' || is_number(cur_token)))
                add_op('*');
            if(c == '+' || c == '-' || c == '*' || c == '/' || c == '>' || c == '<' || c == '=')
            {
                add_op(c);
                continue;
            }
        }
        if (c == '(')
            ++parentheses;
        if (c == ')')
            --parentheses;
        cur_token.push_back(c);
    }

    if(op.empty())
    {
        // handle function call
        auto pos = eq.find_first_of('(');
        if(pos && pos != std::string::npos)
        {
            if(eq.back() != ')')
            {
                error_msg = std::string("invalid parentheses:") + eq;
                return false;
            }
            std::string param;
            auto comma_pos = eq.find_last_of(',');
            if(comma_pos != std::string::npos)
            {
                param = std::string(eq.begin()+comma_pos+1,eq.end()-1);
                if(!is_number(param))
                {
                    error_msg = std::string("invalid parameter:") + param + " in " + eq;
                    return false;
                }
            }
            else
                comma_pos = eq.length()-1;

            auto eval = std::string(eq.begin()+pos+1,eq.begin()+comma_pos);
            if(!equation<out>(x,eval,error_msg))
                return false;

            auto function_name = eq.substr(0,pos);
            if constexpr(!std::is_void_v<out>)
                    out() << "call " << function_name << "(" << std::string(param.empty() ? eval : eval + "," + param) << ")";
            if(!command(x,function_name,param))
            {
                error_msg = std::string("unsupported function:") + function_name;
                return false;
            }
            return true;
        }
        error_msg = std::string("illegal operator found in equation:") + eq;
        return false;
    }
    tokens.push_back(cur_token);

    std::vector<buf_image_type> operands(tokens.size());
    std::vector<float> values(tokens.size());
    for(size_t i = 0;i < tokens.size();++i)
    {
        if(is_number(tokens[i]))
        {
            try{
                values[i] = std::stof(tokens[i]);
            }
            catch (...)
            {
                error_msg = std::string("invalid variable:") + tokens[i] + " in " + eq;
                return false;
            }
        }
        else
        {
            if(tokens[i].find_first_of('(') == 0)
            {
                if(tokens[i].back() != ')')
                {
                    error_msg = std::string("invalid parentheses:") + tokens[i] + " in " + eq;
                    return false;
                }
                tokens[i] = tokens[i].substr(1,tokens[i].size()-2);
            }

            if(!equation<out>(operands[i] = x,tokens[i],error_msg))
                return false;
        }
    }
    while(!op.empty())
    {
        size_t first_op = 0;
        auto pos = std::find_if(op.begin(),op.end(),[](char c){return c == '*' || c == '/';});
        if(pos != op.end())
            first_op = pos - op.begin();

        if(operands[first_op].empty())
        {
            if(operands[first_op+1].empty())
            {
                error_msg = std::string("empty operands found in:") + eq;
                return false;
            }
            if constexpr(!std::is_void_v<out>) out() << "compute " << values[first_op] << op[first_op] << tokens[first_op+1];
            equation(values[first_op],operands[first_op+1],op[first_op]);
            operands[first_op].swap(operands[first_op+1]);
        }
        else
        {
            if(operands[first_op+1].empty())
            {
                if constexpr(!std::is_void_v<out>) out() << "compute " << tokens[first_op] << op[first_op] << values[first_op+1];
                equation(operands[first_op],values[first_op+1],op[first_op]);
            }
            else
            {
                if constexpr(!std::is_void_v<out>) out() << "compute " << tokens[first_op] << op[first_op] << tokens[first_op+1];
                equation(operands[first_op],operands[first_op+1],op[first_op]);
            }
        }
        if(op.size() == 1)
        {
            if constexpr(std::is_same_v<buf_image_type,image_type>)
                x.swap(operands[0]);
            else
                x = operands[0];
            return true;
        }
        op.erase(op.begin() + first_op);
        operands.erase(operands.begin() + first_op + 1);
        values.erase(values.begin() + first_op + 1);
    }
    return true;
}

}//namespace tipl

#endif//CMD_HPP
