#ifndef GRADIENT_MAGNITUDE_HPP_INCLUDED
#define GRADIENT_MAGNITUDE_HPP_INCLUDED

#include "filter_model.hpp"
//---------------------------------------------------------------------------
namespace image
{

namespace filter
{

template<class value_type,size_t dimension>
class gradient_magnitude_filter_imp;

template<class value_type>
class gradient_magnitude_filter_imp<value_type,1>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<class image_type>
    void operator()(image_type& src)
    {
        typedef image::basic_image<class image_type::value_type,image_type::dimension> image_buf_type;
        image_buf_type gx;
        gradient_2x(src,gx);
        absolute_value(gx.begin(),gx.end());
        gx.swap(src);
    }
};

template<class value_type>
class gradient_magnitude_filter_imp<value_type,2>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<class image_type>
    void operator()(image_type& src)
    {
        typedef image::basic_image<typename image_type::value_type,image_type::dimension> image_buf_type;
        image_buf_type gx;
        gradient_2x(src,gx);

        image_buf_type gy;
        gradient_2y(src,gy);

        for(size_t index = 0;index < src.size();++index)
        {
            float fx = gx[index];
            float fy = gy[index];
            src[index] = std::sqrt(fx*fx+fy*fy);
        }
    }
};


template<class value_type>
class gradient_magnitude_filter_imp<value_type,3>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<class image_type>
    void operator()(image_type& src)
    {
        typedef image::basic_image<typename image_type::value_type,image_type::dimension> image_buf_type;
        image_buf_type gx;
        gradient_2x(src,gx);

        image_buf_type gy;
        gradient_2y(src,gy);

        image_buf_type gz;
        gradient_2z(src,gz);

        for(size_t index = 0;index < src.size();++index)
        {
            float fx = gx[index];
            float fy = gy[index];
            float fz = gz[index];
            src[index] = std::sqrt(fx*fx+fy*fy+fz*fz);
        }
    }
};

template<class image_type>
void gradient_magnitude(image_type& src)
{
    gradient_magnitude_filter_imp<typename image_type::value_type,image_type::dimension>()(src);
}




}

}

#endif // GRADIENT_MAGNITUDE_HPP_INCLUDED
