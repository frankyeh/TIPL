//---------------------------------------------------------------------------
#ifndef LAPLACIAN_FILTER_HPP
#define LAPLACIAN_FILTER_HPP

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
void laplacian_impl(image_type& src)
{
    using out_type = typename image_type::value_type;
    using work_type = typename pixel_manip<out_type>::type;

    static_assert(image_type::dimension >= 1 && image_type::dimension <= 3);

    if(src.empty())
        return;

    std::vector<work_type> in(src.size()),out(src.size());
    tipl::serial_or_parallel(src,[&](size_t i)
    {
        in[i] = pixel_manip<out_type>::to_work(src[i]);
    });

    const size_t w = src.width(),wh = src.plane_size();

    tipl::serial_or_parallel(src,[&](size_t i)
    {
        size_t x = i%w;
        auto center = in[i],v = in[x ? i-1 : i];
        center += center;
        v += in[x+1 < w ? i+1 : i];
        v -= center;

        if constexpr(image_type::dimension >= 2)
        {
            size_t y = (i/w)%src.height();
            v += in[y ? i-w : i];
            v += in[y+1 < src.height() ? i+w : i];
            v -= center;
        }

        if constexpr(image_type::dimension >= 3)
        {
            size_t z = i/wh;
            v += in[z ? i-wh : i];
            v += in[z+1 < src.depth() ? i+wh : i];
            v -= center;
        }
        out[i] = v;
    });

    tipl::serial_or_parallel(src,[&](size_t i)
    {
        auto v = out[i];

        if constexpr(std::is_same_v<out_type,tipl::rgb>)
            for(unsigned int j = 0;j < 3;++j)
                if(v[j] < 0)
                    v[j] = -v[j];
                else if constexpr(std::is_unsigned_v<out_type>)
                    if(v < 0)
                        v = -v;

        src[i] = pixel_manip<out_type>::to_pixel(v);
    });
}

}

template<typename image_type>
image_type& laplacian(image_type& src)
{
    detail::laplacian_impl(src);
    return src;
}

template<typename image_type,
         std::enable_if_t<!std::is_lvalue_reference_v<image_type>,int> = 0>
image_type&& laplacian(image_type&& src)
{
    detail::laplacian_impl(src);
    return std::move(src);
}

}
}
#endif
