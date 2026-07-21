//---------------------------------------------------------------------------
#ifndef GRADIENT_MAGNITUDE_HPP_INCLUDED
#define GRADIENT_MAGNITUDE_HPP_INCLUDED

#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>
#include "filter_model.hpp"
//---------------------------------------------------------------------------
namespace tipl
{
namespace filter
{
namespace detail
{

template<typename image_type>
void gradient_magnitude_impl(image_type& src)
{
    using out_type = typename image_type::value_type;
    using work_type = typename pixel_manip<out_type>::type;

    static_assert(std::is_arithmetic_v<out_type>,
                  "gradient_magnitude requires a scalar image");
    static_assert(image_type::dimension >= 1 && image_type::dimension <= 3);

    if(src.empty())
        return;

    std::vector<work_type> in(src.size());
    tipl::serial_or_parallel(src,[&](size_t i)
    {
        in[i] = pixel_manip<out_type>::to_work(src[i]);
    });

    const size_t w = src.width(),wh = src.plane_size();

    tipl::serial_or_parallel(src,[&](size_t i)
    {
        size_t x = i%w;
        work_type gx = in[x+1 < w ? i+1 : i]-in[x ? i-1 : i];
        work_type magnitude;

        if constexpr(image_type::dimension == 1)
            magnitude = gx < 0 ? -gx : gx;
        else
        {
            const size_t h = src.height();
            size_t y = (i/w)%h;
            work_type gy = in[y+1 < h ? i+w : i]-
                           in[y ? i-w : i];

            if constexpr(image_type::dimension == 2)
                magnitude = std::sqrt(gx*gx+gy*gy);
            else
            {
                const size_t d = src.depth();
                size_t z = i/wh;
                work_type gz = in[z+1 < d ? i+wh : i]-
                               in[z ? i-wh : i];
                magnitude = std::sqrt(gx*gx+gy*gy+gz*gz);
            }
        }

        src[i] = pixel_manip<out_type>::to_pixel(magnitude);
    });
}

}

template<typename image_type>
image_type& gradient_magnitude(image_type& src)
{
    detail::gradient_magnitude_impl(src);
    return src;
}

template<typename image_type,
         std::enable_if_t<!std::is_lvalue_reference_v<image_type>,int> = 0>
image_type&& gradient_magnitude(image_type&& src)
{
    detail::gradient_magnitude_impl(src);
    return std::move(src);
}

}
}

#endif // GRADIENT_MAGNITUDE_HPP_INCLUDED
