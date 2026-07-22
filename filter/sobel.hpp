//---------------------------------------------------------------------------
#ifndef SOBEL_FILTER_HPP
#define SOBEL_FILTER_HPP

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
void sobel_impl(image_type& src)
{
    using out_type = typename image_type::value_type;
    using work_type = typename pixel_manip<out_type>::type;

    static_assert(image_type::dimension == 2 || image_type::dimension == 3);
    static_assert(std::is_arithmetic_v<out_type> ||
                  std::is_same_v<out_type,tipl::rgb>);

    if(src.empty())
        return;

    std::vector<work_type> input(src.size()),a(src.size()),
        b(src.size()),magnitude(src.size());

    tipl::serial_or_parallel(src.size(),[&](size_t i)
    {
        input[i] = pixel_manip<out_type>::to_work(src[i]);
    });

    const size_t w = src.width(),h = src.height(),wh = src.plane_size();

    // [1 2 1]
    auto smooth = [&](const auto& in,auto& out,size_t step,size_t length)
    {
        tipl::serial_or_parallel(src.size(),[&](size_t i)
        {
            size_t p = (i/step)%length;
            auto center = in[i];
            center += center;
            out[i] = in[p ? i-step : i]+center+
                     in[p+1 < length ? i+step : i];
        });
    };

    // [-1 0 1]
    auto derivative = [&](const auto& in,auto& out,size_t step,size_t length)
    {
        tipl::serial_or_parallel(src.size(),[&](size_t i)
        {
            size_t p = (i/step)%length;
            out[i] = in[p+1 < length ? i+step : i]-
                     in[p ? i-step : i];
        });
    };

    auto absolute = [](work_type v)
    {
        if constexpr(std::is_same_v<out_type,tipl::rgb>)
        {
            for(size_t j = 0;j < 3;++j)
                if(v[j] < 0)
                    v[j] = -v[j];
        }
        else if(v < 0)
            v = -v;
        return v;
    };

    auto set_abs = [&](const auto& g)
    {
        tipl::serial_or_parallel(src.size(),[&](size_t i)
        {
            magnitude[i] = absolute(g[i]);
        });
    };

    auto add_abs = [&](const auto& g)
    {
        tipl::serial_or_parallel(src.size(),[&](size_t i)
        {
            magnitude[i] += absolute(g[i]);
        });
    };

    // Gx = D(x)S(y)[S(z)]
    derivative(input,a,1,w);
    smooth(a,b,w,h);
    if constexpr(image_type::dimension == 3)
        smooth(b,magnitude,wh,src.depth());
    else
        magnitude.swap(b);
    set_abs(magnitude);

    // Gy = S(x)D(y)[S(z)]
    derivative(input,a,w,h);
    smooth(a,b,1,w);
    if constexpr(image_type::dimension == 3)
        smooth(b,a,wh,src.depth()),add_abs(a);
    else
        add_abs(b);

    // Gz = S(x)S(y)D(z)
    if constexpr(image_type::dimension == 3)
    {
        derivative(input,a,wh,src.depth());
        smooth(a,b,1,w);
        smooth(b,a,w,h);
        add_abs(a);
    }

    tipl::serial_or_parallel(src.size(),[&](size_t i)
    {
        src[i] = pixel_manip<out_type>::to_pixel(magnitude[i]);
    });
}

}

template<typename image_type>
image_type& sobel(image_type& src)
{
    detail::sobel_impl(src);
    return src;
}

template<typename image_type,
         std::enable_if_t<!std::is_lvalue_reference_v<image_type>,int> = 0>
image_type&& sobel(image_type&& src)
{
    detail::sobel_impl(src);
    return std::move(src);
}

}
}
#endif
