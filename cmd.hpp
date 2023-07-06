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

namespace tipl{


template<typename image_loader,typename image_type>
bool command(image_type& data,tipl::vector<3>& vs,tipl::matrix<4,4>& T,bool& is_mni,
             std::string cmd,std::string param1,std::string& error_msg)
{
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

    if constexpr (std::is_floating_point<typename image_type::value_type>::value)
    {
        if(cmd.find("morphology") != std::string::npos)
        {
            tipl::image<image_type::dimension,char> mask(data.shape());
            tipl::par_for(mask.size(),[&](size_t pos)
            {
                mask[pos] = data[pos] > typename image_type::value_type(0) ? 1 : 0;
            });
            if(!command<image_loader>(mask,vs,T,is_mni,cmd,param1,error_msg))
                return false;
            tipl::par_for(mask.size(),[&](size_t pos)
            {
                data[pos] = typename image_type::value_type(mask[pos]);
            });
            return true;
        }
    }
    if(cmd == "morphology_defragment")
    {
        tipl::morphology::for_each_label(data,[](tipl::image<3,char>& mask){tipl::morphology::defragment(mask);});
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
        tipl::morphology::for_each_label(data,[](tipl::image<3,char>& mask){tipl::morphology::smoothing(mask);});
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
            tipl::normalize_mt(data,255.0f);
            tipl::upper_threshold_mt(data,255);
        }
        else
            tipl::normalize_mt(data);
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
        T[0] = -T[0];
        T[4] = -T[4];
        T[8] = -T[8];
        return true;
    }
    if(cmd == "header_flip_y")
    {
        T[7] += T[5]*float(data.height()-1);
        T[1] = -T[1];
        T[5] = -T[5];
        T[9] = -T[9];
        return true;
    }

    if(cmd == "header_flip_z")
    {
        T[11] += T[10]*float(data.depth()-1);
        T[2] = -T[2];
        T[6] = -T[6];
        T[10] = -T[10];
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
        return true;
    }
    if(cmd == "swap_xz")
    {
        tipl::swap_xz(data);
        return true;
    }
    if(cmd == "swap_yz")
    {
        tipl::swap_yz(data);
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
        if(!command<image_loader>(data,vs,T,is_mni,"translocate",std::to_string(-range_min[0]) + " " +
                                    std::to_string(-range_min[1]) + " " +
                                    std::to_string(-range_min[2]),error_msg))

            return false;

        if(!command<image_loader>(data,vs,T,is_mni,"resize",std::to_string(range_max[0]) + " " +
                                    std::to_string(range_max[1]) + " " +
                                    std::to_string(range_max[2]),error_msg))
            return false;
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
            tipl::resample_mt(data,new_data,tipl::transformation_matrix<float>(tipl::from_space(U).to(T)));
            new_data.swap(data);
            vs = new_vs;
            return true;
        }
        // flip in x y z
        if(T[0]*U[0] < 0)
        {
            command<image_loader>(data,vs,T,is_mni,"header_flip_x","",error_msg);
            command<image_loader>(data,vs,T,is_mni,"flip_x","",error_msg);
            return command<image_loader>(data,vs,T,is_mni,cmd,param1,error_msg);
        }
        if(T[5]*U[5] < 0)
        {
            command<image_loader>(data,vs,T,is_mni,"header_flip_y","",error_msg);
            command<image_loader>(data,vs,T,is_mni,"flip_y","",error_msg);
            return command<image_loader>(data,vs,T,is_mni,cmd,param1,error_msg);
        }
        if(T[10]*U[10] < 0)
        {
            command<image_loader>(data,vs,T,is_mni,"header_flip_z","",error_msg);
            command<image_loader>(data,vs,T,is_mni,"flip_z","",error_msg);
            return command<image_loader>(data,vs,T,is_mni,cmd,param1,error_msg);
        }
        // consider voxel size
        if(T[0] != U[0] || T[5] != U[5] || T[10] != U[10])
        {
            if(!command<image_loader>(data,vs,T,is_mni,"regrid",std::to_string(new_vs[0])+" "+std::to_string(new_vs[1])+" "+std::to_string(new_vs[2]),error_msg))
                return false;
            T[0] = U[0];
            T[5] = U[5];
            T[10] = U[10];
            vs = new_vs;
            return command<image_loader>(data,vs,T,is_mni,cmd,param1,error_msg);
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
            tipl::resample(data,new_data,m);
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
        image_type new_data(tipl::shape<3>(w,h,d));
        auto shift = tipl::vector<3,int>(new_data.shape()) - tipl::vector<3,int>(data.shape());
        shift /= 2;
        tipl::draw(data,new_data,shift);
        data.swap(new_data);
        T[3] -= T[0]*shift[0];
        T[7] -= T[5]*shift[1];
        T[11] -= T[10]*shift[2];
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
        if(is_label_image(data))
            tipl::resample_mt<tipl::interpolation::nearest>(data,J,T1);
        else
            tipl::resample_mt<tipl::interpolation::cubic>(data,J,T1);
        data.swap(J);
        vs = new_vs;
        T = T*nT;
        return true;
    }

    if(cmd == "add_value")
    {
        add_constant_mt(data,std::stof(param1));
        return true;
    }
    if(cmd == "multiply_value")
    {
        multiply_constant_mt(data,std::stof(param1));
        return true;
    }
    if(cmd == "lower_threshold")
    {
        lower_threshold_mt(data.begin(),data.end(),std::stof(param1));
        return true;
    }
    if(cmd == "upper_threshold")
    {
        upper_threshold_mt(data.begin(),data.end(),std::stof(param1));
        return true;
    }

    if(cmd == "threshold")
    {
        float value = std::stof(param1);
        tipl::par_for(data.size(),[&](size_t i)
        {
            data[i] = data[i] < value ? 0.0f : 1.0f;
        });
        return true;
    }

    if(cmd == "multiply_image" || cmd == "add_image" || cmd == "minus_image")
    {
        tipl::image<3> rhs(data.shape());
        if(!image_loader::load_to_space(param1.c_str(),rhs,T))
        {
            error_msg = "cannot open file:";
            error_msg += param1;
            return false;
        }
        if(cmd == "multiply_image")
            data *= rhs;
        if(cmd == "add_image")
            data += rhs;
        if(cmd == "minus_image")
            data -= rhs;
        return true;
    }
    if(cmd == "save")
    {
        image_loader nii;
        nii.set_image_transformation(T,is_mni);
        nii.set_voxel_size(vs);
        nii << data;
        return nii.save_to_file(param1.c_str());
    }
    if(cmd == "open")
    {
        image_loader nii;
        if(!nii.load_from_file(param1.c_str()))
            return false;
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

}//namespace tipl

#endif//CMD_HPP
