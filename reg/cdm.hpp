#ifndef CDM_HPP
#define CDM_HPP
#include "tipl/numerical/basic_op.hpp"
#include "tipl/numerical/dif.hpp"
#include "tipl/filter/gaussian.hpp"
#include "tipl/filter/filter_model.hpp"
#include "tipl/utility/multi_thread.hpp"
#include "tipl/numerical/resampling.hpp"
#include "tipl/numerical/statistics.hpp"
#include "tipl/numerical/window.hpp"
#include <iostream>
#include <limits>
#include <vector>

namespace tipl
{
namespace reg
{

template<class pixel_type,size_t dimension>
void cdm_average_img(const std::vector<image<pixel_type,dimension> >& Ji, image<pixel_type,dimension>& J0)
{
    J0 = Ji[0];
    for(unsigned int index = 1;index < Ji.size();++index)
        add(J0.begin(),J0.end(),Ji[index].begin());
    divide_constant(J0.begin(),J0.end(),(float)Ji.size());
}

template<class pixel_type,size_t dimension>
double cdm_img_dif(const image<pixel_type,dimension>& I0,
                    const image<pixel_type,dimension>& I1)
{
    double value = 0;
    for (int index = 0; index < I0.size(); ++index)
    {
        pixel_type tmp = I0[index]-I1[index];
        value += tmp*tmp;
    }
    return value;
}

template<class pixel_type,size_t dimension>
double cdm_img_dif(const std::vector<image<pixel_type,dimension> >& Ji,const image<pixel_type,dimension>& J0)
{
    double next_dif = 0;
    for(unsigned int index = 0;index < Ji.size();++index)
        next_dif += cdm_img_dif(J0,Ji[index]);
    return next_dif;
}


template<class pixel_type,size_t dimension>
double cdm_contrast(const image<pixel_type,dimension>& J0,
                     const image<pixel_type,dimension>& Ji)
{
    double value1 = 0,value2 = 0;
    for (int index = 0; index < J0.size(); ++index)
    {
        double tmp = Ji[index];
        value1 += tmp*J0[index];
        value2 += tmp*tmp;
    }
    if(value2 == 0.0)
        return 1.0;
    return value1/value2;
}

template<class pixel_type,size_t dimension>
void cdm_update_contrast(const std::vector<image<pixel_type,dimension> >& Ji,
                          std::vector<double>& contrast)
{
    image<pixel_type,dimension> J0;
    cdm_average_img(Ji,J0);
    contrast.resize(Ji.size());
    for (unsigned int index = 0; index < Ji.size(); ++index)
        contrast[index] = cdm_contrast(J0,Ji[index]);
    double sum_contrast = std::accumulate(contrast.begin(),contrast.end(),0.0);
    sum_contrast /= Ji.size();
    divide_constant(contrast.begin(),contrast.end(),sum_contrast);
}


// trim the image size to uniform
template<class pixel_type,unsigned int dimension,class crop_type>
void cdm_trim_images(std::vector<image<pixel_type,dimension> >& I,
                      crop_type& crop_from,crop_type& crop_to)
{
    crop_from.resize(I.size());
    crop_to.resize(I.size());
    vector<dimension,int> min_from,max_to;
    for(int index = 0;index < I.size();++index)
    {
        crop(I[index],crop_from[index],crop_to[index]);
        if(index == 0)
        {
            min_from = crop_from[0];
            max_to = crop_to[0];
        }
        else
            for(int dim = 0;dim < dimension;++dim)
            {
                min_from[dim] = std::min(crop_from[index][dim],min_from[dim]);
                max_to[dim] = std::max(crop_to[index][dim],max_to[dim]);
            }
    }
    max_to -= min_from;
    int safe_margin = std::accumulate(max_to.begin(),max_to.end(),0.0)/(float)dimension/2.0;
    max_to += safe_margin;
    geometry<dimension> geo(max_to.begin());
    // align with respect to the min_from
    for(int index = 0;index < I.size();++index)
    {
        image<float,dimension> new_I(geo);
        vector<dimension,int> pos(crop_from[index]);
        pos -= min_from;
        pos += safe_margin/2;
        draw(I[index],new_I,pos);
        new_I.swap(I[index]);
        crop_from[index] -= pos;
        crop_to[index] = crop_from[index];
        crop_to[index] += geo.begin();
    }

}

template<class value_type,size_t dimension>
class poisson_equation_solver;

template<class value_type>
class poisson_equation_solver<value_type,2>
{
    typedef typename filter::pixel_manip<value_type>::type manip_type;
public:
    template<class image_type>
    void operator()(image_type& src)
    {
        image_type dest(src.geometry());
        int w = src.width();
        int shift[4];
        shift[0] = 1;
        shift[1] = -1;
        shift[2] = w;
        shift[3] = -w;
        for(int i = 0;i < 4;++i)
        if (shift[i] >= 0)
        {
            auto iter1 = dest.begin() + shift[i];
            auto iter2 = src.begin();
            auto end = dest.end();
            for (;iter1 < end;++iter1,++iter2)
                *iter1 += *iter2;
        }
        else
        {
            auto iter1 = dest.begin();
            auto iter2 = src.begin() + (-shift[i]);
            auto end = src.end();
            for (;iter2 < end;++iter1,++iter2)
                *iter1 += *iter2;
        }
        dest.swap(src);
    }
};
template<class value_type>
class poisson_equation_solver<value_type,3>
{
    typedef typename filter::pixel_manip<value_type>::type manip_type;
public:
    template<class image_type>
    void operator()(image_type& src)
    {
        image_type dest(src.geometry());
        int w = src.width();
        int wh = src.width()*src.height();
        int shift[6];
        shift[0] = 1;
        shift[1] = -1;
        shift[2] = w;
        shift[3] = -w;
        shift[4] = wh;
        shift[5] = -wh;
        for(int i = 0;i < 6;++i)
        {
            if (shift[i] >= 0)
            {
                int s = shift[i];
                par_for(dest.size()-s,[&dest,&src,s](int index){
                    dest[index+s] += src[index];
                });
            }
            else
            {
                int s = -shift[i];
                par_for(dest.size()-s,[&dest,&src,s](int index){
                    dest[index] += src[index+s];
                });
            }
        }
        dest.swap(src);
    }
};

template<class pixel_type,class vtor_type,unsigned int dimension,class terminate_type>
void cdm_group(const std::vector<image<pixel_type,dimension> >& I,// original images
          std::vector<image<vtor_type,dimension> >& d,// displacement field
          float theta,float reg,terminate_type& terminated)
{
    if (I.empty())
        return;
    unsigned int n = I.size();
    geometry<dimension> geo = I[0].geometry();

    d.resize(n);
    // preallocated memory so that there will be no memory re-allocation in upsampling
    for (int index = 0;index < n;++index)
        d[index].resize(geo);

    // multi resolution
    if (*std::min_element(geo.begin(),geo.end()) > 16)
    {
        //downsampling
        std::vector<image<pixel_type,dimension> > rI(n);
        for (int index = 0;index < n;++index)
            downsample_with_padding(I[index],rI[index]);
        cdm_group(rI,d,theta/2,reg,terminated);
        // upsampling deformation
        for (int index = 0;index < n;++index)
        {
            upsample_with_padding(d[index],d[index],geo);
            d[index] * 2.0f;
        }
    }
    std::cout << "dimension:" << geo[0] << "x" << geo[1] << "x" << geo[2] << std::endl;
    image<pixel_type,dimension> J0(geo);// the potential template
    std::vector<image<pixel_type,dimension> > Ji(n);// transformed I
    std::vector<image<vtor_type,dimension> > new_d(n);// new displacements
    std::vector<double> contrast(n);
    double current_dif = std::numeric_limits<double>::max();
    for (double dis = 0.5;dis > theta;)
    {
        // calculate Ji
        for (unsigned int index = 0;index < n;++index)
            compose_displacement(I[index],d[index],Ji[index]);


        // calculate contrast
        cdm_update_contrast(Ji,contrast);

        // apply contrast
        multiply(Ji.begin(),Ji.end(),contrast.begin());

        cdm_average_img(Ji,J0);

        double next_dif = cdm_img_dif(Ji,J0);
        if(next_dif >= current_dif)
        {
            dis *= 0.5;
            std::cout << "detail=(" << dis << ")" << std::flush;
            // roll back
            d.swap(new_d);
            current_dif = std::numeric_limits<double>::max();
            continue;
        }
        std::cout << next_dif;
        current_dif = next_dif;
        // initialize J0
        double max_d = 0;
        for (unsigned int index = 0;index < n && !terminated;++index)
        {
            std::cout << "." << std::flush;

            // gradient of Ji
            gradient_sobel(J0,new_d[index]);

            // dJi*sign(Ji-J0)
            for(unsigned int i = 0;i < Ji[index].size();++i)
                if(Ji[index][i] < J0[i])
                    new_d[index][i] = -new_d[index][i];

            {
                image<pixel_type,dimension> temp;
                jacobian_determinant_dis(d[index],temp);
                compose_displacement(temp,d[index],Ji[index]);
            }

            for(unsigned int i = 0;i < Ji[index].size();++i)
                new_d[index][i] *= Ji[index][i];

            //io::nifti header;
            //header << Ji[index];
            //header.save_to_file("c:/1.nii");

            // solving the poisson equation using Jacobi method
            {
                image<vtor_type,dimension> solve_d(new_d[index].geometry());
                for(int iter = 0;iter < 20;++iter)
                {
                    poisson_equation_solver<vtor_type,dimension>()(solve_d);
                    add_mt(solve_d,new_d[index]);
                    divide_constant_mt(solve_d,dimension*2);
                }
                new_d[index].swap(solve_d);
                minus_constant(new_d[index].begin(),new_d[index].end(),new_d[index][0]);
            }

            for (unsigned int i = 0; i < geo.size(); ++i)
                if (new_d[index][i]*new_d[index][i] > max_d)
                    max_d = std::sqrt(new_d[index][i]*new_d[index][i]);
        }
        // calculate the lambda
        double lambda = -dis/max_d;
        for (unsigned int index = 0;index < n && !terminated;++index)
        {
            new_d[index] *= lambda;
            add(new_d[index].begin(),new_d[index].end(),d[index].begin());
        }
        d.swap(new_d);
    }
    std::cout << std::endl;
}

/*
 *  The intensity between It and Is has to be matched
 *  std::pair<double,double> r = linear_regression(Is.begin(),Is.end(),It.begin());
        for(unsigned int index = 0;index < Is.size();++index)
            Is[index] = std::max<float>(0,Is[index]*r.first+r.second);
 */
template<class pixel_type,class vtor_type,unsigned int dimension,class terminate_type>
double cdm(const image<pixel_type,dimension>& It,
            const image<pixel_type,dimension>& Is,
            image<vtor_type,dimension>& d,// displacement field
            terminate_type& terminated,
            float resolution = 2.0,
            float cdm_smoothness = 0.3f,
            unsigned int steps = 30)
{
    if(It.geometry() != Is.geometry() || It.empty())
        throw "Invalid cdm input image";
    geometry<dimension> geo = It.geometry();
    d.resize(geo);
    // multi resolution
    if (*std::min_element(geo.begin(),geo.end()) > 32)
    {
        //downsampling
        image<pixel_type,dimension> rIs,rIt;
        downsample_with_padding(It,rIt);
        downsample_with_padding(Is,rIs);
        float r = cdm(rIt,rIs,d,terminated,resolution/2.0,cdm_smoothness,steps);
        upsample_with_padding(d,d,geo);
        d *= 2.0f;
        if(resolution > 1.0)
            return r;
    }
    image<pixel_type,dimension> Js;// transformed I
    image<vtor_type,dimension> new_d(d.geometry());// new displacements
    double max_t = (double)(*std::max_element(It.begin(),It.end()));
    double max_s = (double)(*std::max_element(Is.begin(),Is.end()));
    if(max_t == 0.0 || max_s == 0.0)
        return 0.0;
    double theta = 0.0;
    unsigned int window_size = 3;
    float inv_d2 = 0.5f/dimension;
    float cdm_smoothness2 = 1.0f-cdm_smoothness;
    int shift[dimension]={0};
    shift[0] = 1;
    for(int i = 1;i < dimension;++i)
        shift[i] = shift[i-1]*geo[i-1];
    float r,prev_r = 0.0;
    for (unsigned int index = 0;index < steps && !terminated;++index)
    {
        compose_displacement(Is,d,Js);
        r = tipl::correlation(Js.begin(),Js.end(),It.begin());
        if(r <= prev_r)
        {
            new_d.swap(d);
            break;
        }
        // dJ(cJ-I)
        gradient_sobel(Js,new_d);
        Js.for_each_mt([&](pixel_type&,pixel_index<dimension>& index){
            if(It[index.index()] == 0.0 || It.geometry().is_edge(index))
            {
                new_d[index.index()] = vtor_type();
                return;
            }
            std::vector<pixel_type> Itv,Jv;
            get_window(index,It,window_size,Itv);
            get_window(index,Js,window_size,Jv);
            double a,b,r2;
            linear_regression(Jv.begin(),Jv.end(),Itv.begin(),a,b,r2);
            if(a <= 0.0f)
                new_d[index.index()] = vtor_type();
            else
                new_d[index.index()] *= (Js[index.index()]*a+b-It[index.index()]);
        });
        // solving the poisson equation using Jacobi method
        image<vtor_type,dimension> solve_d(new_d);
        multiply_constant_mt(solve_d,-inv_d2);
        for(int iter = 0;iter < window_size*2 && !terminated;++iter)
        {
            image<vtor_type,dimension> new_solve_d(new_d.geometry());
            par_for(solve_d.size(),[&](int pos)
            {
                for(int d = 0;d < dimension;++d)
                {
                    int p1 = pos-shift[d];
                    int p2 = pos+shift[d];
                    if(p1 >= 0)
                       new_solve_d[pos] += solve_d[p1];
                    if(p2 < solve_d.size())
                       new_solve_d[pos] += solve_d[p2];
                }
                new_solve_d[pos] -= new_d[pos];
                new_solve_d[pos] *= inv_d2;
            });
            solve_d.swap(new_solve_d);
        }
        minus_constant_mt(solve_d,solve_d[0]);

        new_d = solve_d;
        if(theta == 0.0f)
        {
            par_for(new_d.size(),[&](int i)
            {
               float l = new_d[i].length();
               if(l > theta)
                   theta = l;
            });
        }
        multiply_constant_mt(new_d,0.5f/theta);
        add(new_d,d);

        image<vtor_type,dimension> new_ds(new_d);
        filter::gaussian2(new_ds);
        par_for(new_d.size(),[&](int i){
           new_ds[i] *= cdm_smoothness;
           new_d[i] *= cdm_smoothness2;
           new_d[i] += new_ds[i];
        });
        new_d.swap(d);
    }
    return r;
}


}// namespace reg
}// namespace image
#endif // DMDM_HPP
