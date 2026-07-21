//---------------------------------------------------------------------------
#ifndef GAUSSIAN_HPP
#define GAUSSIAN_HPP
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
void gaussian_impl(image_type& src,unsigned int bandwidth)
{
    using out_type = typename image_type::value_type;
    using work_type = std::conditional_t<
        std::is_integral_v<out_type>,
        std::conditional_t<sizeof(out_type) == 1,float,double>,
        typename pixel_manip<out_type>::type>;

    static_assert(image_type::dimension >= 1 && image_type::dimension <= 3);

    if(src.empty() || !bandwidth)
        return;

    std::vector<work_type> a(src.begin(),src.end()),b(src.size());
    const size_t w = src.width(),wh = src.plane_size();

    // Repeated separable [1 2 1] filtering:
    // bandwidth 1 -> 3 samples; bandwidth 2 -> 5 samples.
    for(unsigned int k = 0;k < bandwidth;++k)
    {
        tipl::serial_or_parallel(src,[&](size_t i)
        {
            size_t x = i%w;
            auto center = a[i];
            center += center;
            b[i] = a[x ? i-1 : i]+center+a[x+1 < w ? i+1 : i];
        });
        a.swap(b);

        if constexpr(image_type::dimension >= 2)
        {
            const size_t h = src.height();
            tipl::serial_or_parallel(src,[&](size_t i)
            {
                size_t y = (i/w)%h;
                auto center = a[i];
                center += center;
                b[i] = a[y ? i-w : i]+center+a[y+1 < h ? i+w : i];
            });
            a.swap(b);
        }

        if constexpr(image_type::dimension >= 3)
        {
            const size_t d = src.depth();
            tipl::serial_or_parallel(src,[&](size_t i)
            {
                size_t z = i/wh;
                auto center = a[i];
                center += center;
                b[i] = a[z ? i-wh : i]+center+a[z+1 < d ? i+wh : i];
            });
            a.swap(b);
        }
    }

    const double scale = std::ldexp(
        1.0,-int(2*size_t(bandwidth)*image_type::dimension));

    tipl::serial_or_parallel(src,[&](size_t i)
    {
        if constexpr(std::is_integral_v<out_type>)
            src[i] = out_type(std::round(a[i]*scale));
        else
            src[i] = a[i]*scale;
    });
}

}

template<typename image_type>
image_type& gaussian(image_type& src,unsigned int bandwidth = 1)
{
    detail::gaussian_impl(src,bandwidth);
    return src;
}

template<typename image_type,
         std::enable_if_t<!std::is_lvalue_reference_v<image_type>,int> = 0>
image_type&& gaussian(image_type&& src,unsigned int bandwidth = 1)
{
    detail::gaussian_impl(src,bandwidth);
    return std::move(src);
}

}
}
#endif
