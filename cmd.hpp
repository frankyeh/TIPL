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
#include "morphology/morphology.hpp"

namespace tipl{


template<typename image_loader,typename image_type>
bool command(image_type& data,tipl::vector<3>& vs,tipl::matrix<4,4>& T,bool& is_mni,
             std::string cmd,std::string param1,std::string& error_msg)
{
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
    if(cmd == "flip_x")
    {
        T[3] += T[0]*float(data.width()-1);
        T[0] = -T[0];
        T[4] = -T[4];
        T[8] = -T[8];
        tipl::flip_x(data);
        return true;
    }


    if(cmd == "flip_y")
    {
        T[7] += T[5]*float(data.height()-1);
        T[1] = -T[1];
        T[5] = -T[5];
        T[9] = -T[9];
        tipl::flip_y(data);
        return true;
    }

    if(cmd == "flip_z")
    {
        T[11] += T[10]*float(data.depth()-1);
        T[2] = -T[2];
        T[6] = -T[6];
        T[10] = -T[10];
        tipl::flip_z(data);
        return true;
    }
    if(cmd == "swap_xy")
    {
        T = tipl::matrix<4,4>({0,1,0,0,
                               1,0,0,0,
                               0,0,1,0,
                               0,0,0,1})*T;
        tipl::swap_xy(data);
        std::swap(vs[0],vs[1]);
        return true;
    }
    if(cmd == "swap_xz")
    {
        T = tipl::matrix<4,4>({0,0,1,0,
                               0,1,0,0,
                               1,0,0,0,
                               0,0,0,1})*T;
        tipl::swap_xz(data);
        std::swap(vs[0],vs[2]);
        return true;
    }
    if(cmd == "swap_yz")
    {
        T = tipl::matrix<4,4>({1,0,0,0,
                               0,0,1,0,
                               0,1,0,0,
                               0,0,0,1})*T;
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
        if(!command<image_loader>(data,vs,T,is_mni,"translocation",std::to_string(-range_min[0]) + " " +
                                    std::to_string(-range_min[1]) + " " +
                                    std::to_string(-range_min[2]),error_msg))

            return false;

        if(!command<image_loader>(data,vs,T,is_mni,"resize",std::to_string(range_max[0]) + " " +
                                    std::to_string(range_max[1]) + " " +
                                    std::to_string(range_max[2]),error_msg))
            return false;
        return true;
    }
    if(cmd == "translocation")
    {
        std::istringstream in(param1);
        image_type new_data(data.shape());

        if(param1.find(".") != std::string::npos)
        {
            tipl::transformation_matrix<float> m;
            m.sr[0] = m.sr[4] = m.sr[8] = 1.0f;
            in >> m.shift[0] >> m.shift[1] >> m.shift[2];
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
            int dx,dy,dz;
            in >> dx >> dy >> dz;
            tipl::draw(data,new_data,tipl::vector<3,int>(dx,dy,dz));
            T[3] -= T[0]*float(dx);
            T[7] -= T[5]*float(dy);
            T[11] -= T[10]*float(dz);
        }
        data.swap(new_data);
        return true;
    }
    if(cmd == "regrid")
    {
        float nv = std::stof(param1);
        if(nv == 0.0f)
        {
            error_msg = "invalid resolution";
            return false;
        }
        tipl::vector<3> new_vs(nv,nv,nv);
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
        tipl::draw(data,new_data,tipl::vector<3,int>(0,0,0));
        data.swap(new_data);
        return true;
    }
    if(cmd == "multiply_image" || cmd == "add_image" || cmd == "minus_image")
    {
        image_loader nii;
        if(!nii.load_from_file(param1.c_str()))
        {
            error_msg = "cannot open file:";
            error_msg += param1;
            return false;
        }
        tipl::image<3> mask;
        nii.get_untouched_image(mask);
        if(mask.shape() != data.shape())
        {
            error_msg = "invalid mask file:";
            error_msg += param1;
            error_msg += " The dimension does not match:";
            std::ostringstream out;
            out << mask.shape() << " vs " << data.shape();
            error_msg += out.str();
            return false;
        }
        if(cmd == "multiply_image")
            data *= mask;
        if(cmd == "add_image")
            data += mask;
        if(cmd == "minus_image")
            data -= mask;
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
