#ifndef GRADIENT_MAGNITUDE_HPP_INCLUDED
#define GRADIENT_MAGNITUDE_HPP_INCLUDED

#include "filter_model.hpp"
//---------------------------------------------------------------------------
namespace tipl
{

namespace filter
{

template<typename value_type,size_t dimension>
class gradient_magnitude_filter_imp;

template<typename value_type>
class gradient_magnitude_filter_imp<value_type,1>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<typename image_type>
    void operator()(image_type& src)
    {
        typedef tipl::image<image_type::dimension,typename image_type::value_type> image_buf_type;
        image_buf_type gx;
        gradient_2x(src,gx);
        absolute_value(gx.begin(),gx.end());
        gx.swap(src);
    }
};

template<typename value_type>
class gradient_magnitude_filter_imp<value_type,2>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<typename image_type>
    void operator()(image_type& src)
    {
        typedef tipl::image<image_type::dimension,typename image_type::value_type> image_buf_type;
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


template<typename value_type>
class gradient_magnitude_filter_imp<value_type,3>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<typename image_type>
    void operator()(image_type& src)
    {
        typedef tipl::image<image_type::dimension,typename image_type::value_type> image_buf_type;
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

template<typename image_type>
void gradient_magnitude(image_type& src)
{
    gradient_magnitude_filter_imp<image_type::dimension,typename image_type::value_type>()(src);
}




}

}

#endif // GRADIENT_MAGNITUDE_HPP_INCLUDED
