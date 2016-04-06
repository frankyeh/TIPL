#ifndef DMDM_HPP
#define DMDM_HPP
#include "image/numerical/basic_op.hpp"
#include "image/numerical/dif.hpp"
#include "image/filter/gaussian.hpp"
#include "image/filter/filter_model.hpp"
#include <iostream>
#include <limits>
#include <vector>
//#include "image/io/nifti.hpp"


namespace image
{
namespace reg
{

template<class pixel_type,size_t dimension>
void dmdm_average_img(const std::vector<basic_image<pixel_type,dimension> >& Ji, image::basic_image<pixel_type,dimension>& J0)
{
    J0 = Ji[0];
    for(unsigned int index = 1;index < Ji.size();++index)
        image::add(J0.begin(),J0.end(),Ji[index].begin());
    image::divide_constant(J0.begin(),J0.end(),(float)Ji.size());
}

template<class pixel_type,size_t dimension>
double dmdm_img_dif(const basic_image<pixel_type,dimension>& I0,
                    const basic_image<pixel_type,dimension>& I1)
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
double dmdm_img_dif(const std::vector<basic_image<pixel_type,dimension> >& Ji,const image::basic_image<pixel_type,dimension>& J0)
{
    double next_dif = 0;
    for(unsigned int index = 0;index < Ji.size();++index)
        next_dif += dmdm_img_dif(J0,Ji[index]);
    return next_dif;
}


template<class pixel_type,size_t dimension>
double dmdm_contrast(const basic_image<pixel_type,dimension>& J0,
                     const basic_image<pixel_type,dimension>& Ji)
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
void dmdm_update_contrast(const std::vector<basic_image<pixel_type,dimension> >& Ji,
                          std::vector<double>& contrast)
{
    image::basic_image<pixel_type,dimension> J0;
    dmdm_average_img(Ji,J0);
    contrast.resize(Ji.size());
    for (unsigned int index = 0; index < Ji.size(); ++index)
        contrast[index] = dmdm_contrast(J0,Ji[index]);
    double sum_contrast = std::accumulate(contrast.begin(),contrast.end(),0.0);
    sum_contrast /= Ji.size();
    image::divide_constant(contrast.begin(),contrast.end(),sum_contrast);
}


// trim the image size to uniform
template<class pixel_type,unsigned int dimension,class crop_type>
void dmdm_trim_images(std::vector<image::basic_image<pixel_type,dimension> >& I,
                      crop_type& crop_from,crop_type& crop_to)
{
    crop_from.resize(I.size());
    crop_to.resize(I.size());
    image::vector<dimension,int> min_from,max_to;
    for(int index = 0;index < I.size();++index)
    {
        image::trim(I[index],crop_from[index],crop_to[index]);
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
    image::geometry<dimension> geo(max_to.begin());
    // align with respect to the min_from
    for(int index = 0;index < I.size();++index)
    {
        image::basic_image<float,dimension> new_I(geo);
        image::vector<dimension,int> pos(crop_from[index]);
        pos -= min_from;
        pos += safe_margin/2;
        image::draw(I[index],new_I,pos);
        new_I.swap(I[index]);
        crop_from[index] -= pos;
        crop_to[index] = crop_from[index];
        crop_to[index] += geo.begin();
    }

}

template<class image_type>
void dmdm_downsample(const image_type& I,image_type& rI)
{
    geometry<image_type::dimension> pad_geo(I.geometry());
    for(unsigned int dim = 0;dim < image_type::dimension;++dim)
        ++pad_geo[dim];
    basic_image<class image_type::value_type,image_type::dimension> pad_I(pad_geo);
    image::draw(I,pad_I,pixel_index<image_type::dimension>());
    downsampling(pad_I,rI);
}

template<class image_type,class geo_type>
void dmdm_upsample(const image_type& I,image_type& uI,const geo_type& geo)
{
    basic_image<class image_type::value_type,image_type::dimension> new_I;
    upsampling(I,new_I);
    new_I *= 2.0;
    uI.resize(geo);
    image::draw(new_I,uI,pixel_index<image_type::dimension>());
}

template<class value_type,size_t dimension>
class poisson_equation_solver;

template<class value_type>
class poisson_equation_solver<value_type,2>
{
    typedef typename image::filter::pixel_manip<value_type>::type manip_type;
public:
    template<class image_type>
    void operator()(image_type& src)
    {
        std::vector<manip_type> dest(src.size());
        int w = src.width();
        image::filter::add_weight<1>(dest,src,-1);
        image::filter::add_weight<1>(dest,src,1);
        image::filter::add_weight<1>(dest,src,-w);
        image::filter::add_weight<1>(dest,src,w);
        std::copy(dest.begin(),dest.end(),src.begin());
    }
};
template<class value_type>
class poisson_equation_solver<value_type,3>
{
    typedef typename image::filter::pixel_manip<value_type>::type manip_type;
public:
    template<class image_type>
    void operator()(image_type& src)
    {
        std::vector<manip_type> dest(src.size());
        int w = src.width();
        int wh = src.width()*src.height();
        image::filter::add_weight<1>(dest,src,1);
        image::filter::add_weight<1>(dest,src,-1);
        image::filter::add_weight<1>(dest,src,w);
        image::filter::add_weight<1>(dest,src,-w);
        image::filter::add_weight<1>(dest,src,wh);
        image::filter::add_weight<1>(dest,src,-wh);
        std::copy(dest.begin(),dest.end(),src.begin());
    }
};

template<class pixel_type,class vtor_type,unsigned int dimension,class terminate_type>
void dmdm(const std::vector<basic_image<pixel_type,dimension> >& I,// original images
          std::vector<basic_image<vtor_type,dimension> >& d,// displacement field
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
        std::vector<basic_image<pixel_type,dimension> > rI(n);
        for (int index = 0;index < n;++index)
            dmdm_downsample(I[index],rI[index]);
        dmdm(rI,d,theta/2,reg,terminated);
        // upsampling deformation
        for (int index = 0;index < n;++index)
            dmdm_upsample(d[index],d[index],geo);
    }
    std::cout << "dimension:" << geo[0] << "x" << geo[1] << "x" << geo[2] << std::endl;
    basic_image<pixel_type,dimension> J0(geo);// the potential template
    std::vector<basic_image<pixel_type,dimension> > Ji(n);// transformed I
    std::vector<basic_image<vtor_type,dimension> > new_d(n);// new displacements
    std::vector<double> contrast(n);
    double current_dif = std::numeric_limits<double>::max();
    for (double dis = 0.5;dis > theta;)
    {
        // calculate Ji
        for (unsigned int index = 0;index < n;++index)
            image::compose_displacement(I[index],d[index],Ji[index]);


        // calculate contrast
        dmdm_update_contrast(Ji,contrast);

        // apply contrast
        image::multiply(Ji.begin(),Ji.end(),contrast.begin());

        dmdm_average_img(Ji,J0);

        double next_dif = dmdm_img_dif(Ji,J0);
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
            image::gradient_sobel(J0,new_d[index]);

            // dJi*sign(Ji-J0)
            for(unsigned int i = 0;i < Ji[index].size();++i)
                if(Ji[index][i] < J0[i])
                    new_d[index][i] = -new_d[index][i];

            {
                basic_image<pixel_type,dimension> temp;
                image::jacobian_determinant_dis(d[index],temp);
                image::compose_displacement(temp,d[index],Ji[index]);
            }

            for(unsigned int i = 0;i < Ji[index].size();++i)
                new_d[index][i] *= Ji[index][i];

            //image::io::nifti header;
            //header << Ji[index];
            //header.save_to_file("c:/1.nii");

            // solving the poisson equation using Jacobi method
            {
                basic_image<vtor_type,dimension> solve_d(new_d[index].geometry());
                for(int iter = 0;iter < 20;++iter)
                {
                    poisson_equation_solver<vtor_type,dimension>()(solve_d);
                    image::add(solve_d.begin(),solve_d.end(),new_d[index].begin());
                    image::divide_constant(solve_d.begin(),solve_d.end(),dimension*2);
                }
                new_d[index] = solve_d;
                image::minus_constant(new_d[index].begin(),new_d[index].end(),new_d[index][0]);
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
            image::add(new_d[index].begin(),new_d[index].end(),d[index].begin());
        }
        d.swap(new_d);
    }
    std::cout << std::endl;
}


template<class pixel_type,class vtor_type,unsigned int dimension,class terminate_type>
void dmdm_pair(const basic_image<pixel_type,dimension>& It,
            const basic_image<pixel_type,dimension>& Is,
            basic_image<vtor_type,dimension>& d,// displacement field
            float theta,terminate_type& terminated,unsigned int steps = 20)
{
    geometry<dimension> geo = It.geometry();
    d.resize(geo);
    // multi resolution
    if (*std::min_element(geo.begin(),geo.end()) > 16)
    {
        //downsampling
        basic_image<pixel_type,dimension> rIs,rIt;
        dmdm_downsample(It,rIt);
        dmdm_downsample(Is,rIs);
        dmdm_pair(rIt,rIs,d,theta/2,terminated,steps*2);
        dmdm_upsample(d,d,geo);
    }
    basic_image<pixel_type,dimension> Js,jdet;// transformed I
    basic_image<vtor_type,dimension> new_d(geo),gIt;// new displacements
    double dis = theta;
    image::gradient_sobel(It,gIt);
    for (unsigned int index = 0;index < steps;++index)
    {
        // calculate Ji
        image::jacobian_determinant_dis(d,jdet);
        image::compose_displacement(Is,d,Js);

        // calculate contrast, apply contrast
        Js *= dmdm_contrast(It,Js);

        // dJi*sign(Ji-J0)
        for(unsigned int i = 0;i < new_d.size();++i)
            new_d[i] = (Js[i] < It[i]) ? -gIt[i]:gIt[i];

        image::compose_displacement(jdet,d,Js);
        image::multiply(new_d.begin(),new_d.end(),Js.begin());

        // solving the poisson equation using Jacobi method
        {
            basic_image<vtor_type,dimension> solve_d(new_d.geometry());
            for(int iter = 0;iter < 20;++iter)
            {
                poisson_equation_solver<vtor_type,dimension>()(solve_d);
                image::add(solve_d.begin(),solve_d.end(),new_d.begin());
                image::divide_constant(solve_d.begin(),solve_d.end(),dimension*2);
            }
            new_d = solve_d;
            image::minus_constant(new_d.begin(),new_d.end(),new_d[0]);
        }

        double max_d = 0;
        for (unsigned int i = 0; i < geo.size(); ++i)
            if (new_d[i]*new_d[i] > max_d)
                max_d = std::sqrt(new_d[i]*new_d[i]);
        new_d *= -dis/max_d;
        image::add(d.begin(),d.end(),new_d.begin());
    }
}


}// namespace reg
}// namespace image
#endif // DMDM_HPP
