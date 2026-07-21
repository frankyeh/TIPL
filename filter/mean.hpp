//---------------------------------------------------------------------------
#ifndef MEAN_FILTER_HPP
#define MEAN_FILTER_HPP

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
void mean_impl(image_type& src)
{
    using out_type = typename image_type::value_type;
    using work_type = typename pixel_manip<out_type>::type;

    static_assert(image_type::dimension >= 1 && image_type::dimension <= 3);

    if(src.empty())
        return;

    std::vector<work_type> a(src.size()),b(src.size());
    tipl::serial_or_parallel(src,[&](size_t i)
    {
        a[i] = pixel_manip<out_type>::to_work(src[i]);
    });

    const size_t w = src.width(),wh = src.plane_size();

    tipl::serial_or_parallel(src,[&](size_t i)
    {
        size_t x = i%w;
        b[i] = a[x ? i-1 : i]+a[i]+a[x+1 < w ? i+1 : i];
    });
    a.swap(b);

    if constexpr(image_type::dimension >= 2)
    {
        const size_t h = src.height();
        tipl::serial_or_parallel(src,[&](size_t i)
        {
            size_t y = (i/w)%h;
            b[i] = a[y ? i-w : i]+a[i]+a[y+1 < h ? i+w : i];
        });
        a.swap(b);
    }

    if constexpr(image_type::dimension >= 3)
    {
        const size_t d = src.depth();
        tipl::serial_or_parallel(src,[&](size_t i)
        {
            size_t z = i/wh;
            b[i] = a[z ? i-wh : i]+a[i]+a[z+1 < d ? i+wh : i];
        });
        a.swap(b);
    }

    constexpr unsigned int divisor =
        image_type::dimension == 1 ? 3 :
            image_type::dimension == 2 ? 9 : 27;

    tipl::serial_or_parallel(src,[&](size_t i)
    {
        auto value = a[i];
        value /= divisor;
        src[i] = pixel_manip<out_type>::to_pixel(value);
    });
}

}

template<typename image_type>
image_type& mean(image_type& src)
{
    detail::mean_impl(src);
    return src;
}

template<typename image_type,
         std::enable_if_t<!std::is_lvalue_reference_v<image_type>,int> = 0>
image_type&& mean(image_type&& src)
{
    detail::mean_impl(src);
    return std::move(src);
}

}
}
#endif
