//---------------------------------------------------------------------------
#ifndef CANNY_EDGE_HPP_INCLUDED
#define CANNY_EDGE_HPP_INCLUDED

#include <cmath>
#include <cstddef>
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
void canny_edge_impl(image_type& src)
{
    using out_type = typename image_type::value_type;
    using work_type = typename pixel_manip<out_type>::type;

    static_assert(std::is_arithmetic_v<out_type>,
                  "canny_edge requires a scalar image");
    static_assert(image_type::dimension == 2 || image_type::dimension == 3);

    if(!src.size())
        return;

    const size_t w = src.width(),h = src.height(),wh = src.plane_size();
    std::vector<work_type> mag(src.size()),tmp(src.size()),
        gx(src.size()),gy(src.size()),gz;

    tipl::serial_or_parallel(src,[&](size_t i)
    {
        mag[i] = pixel_manip<out_type>::to_work(src[i]);
    });

    if constexpr(image_type::dimension == 3)
        gz.resize(src.size());

    // [1 2 1]
    auto smooth = [&](const auto& in,auto& out,size_t step,size_t length)
    {
        tipl::serial_or_parallel(src,[&](size_t i)
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
        tipl::serial_or_parallel(src,[&](size_t i)
        {
            size_t p = (i/step)%length;
            out[i] = in[p+1 < length ? i+step : i]-
                     in[p ? i-step : i];
        });
    };

    // Gx = D(x)S(y)[S(z)]
    derivative(mag,tmp,1,w);
    smooth(tmp,gx,w,h);
    if constexpr(image_type::dimension == 3)
        smooth(gx,tmp,wh,src.depth()),gx.swap(tmp);

    // Gy = S(x)D(y)[S(z)]
    derivative(mag,tmp,w,h);
    smooth(tmp,gy,1,w);
    if constexpr(image_type::dimension == 3)
        smooth(gy,tmp,wh,src.depth()),gy.swap(tmp);

    // Gz = S(x)S(y)D(z)
    if constexpr(image_type::dimension == 3)
    {
        derivative(mag,tmp,wh,src.depth());
        smooth(tmp,gz,1,w);
        smooth(gz,tmp,w,h);
        gz.swap(tmp);
    }

    // Reuse mag for gradient magnitude.
    tipl::serial_or_parallel(src,[&](size_t i)
    {
        if constexpr(image_type::dimension == 2)
            mag[i] = std::sqrt(gx[i]*gx[i]+gy[i]*gy[i]);
        else
            mag[i] = std::sqrt(gx[i]*gx[i]+gy[i]*gy[i]+gz[i]*gz[i]);
    });

    auto assign = [&](size_t i,work_type value)
    {
        src[i] = pixel_manip<out_type>::to_pixel(value);
    };

    if constexpr(image_type::dimension == 2)
    {
        constexpr work_type tan67 = work_type(2.414213562373095);

        tipl::serial_or_parallel(src,[&](size_t i)
        {
            work_type m = mag[i];
            if(m == 0)
                return assign(i,0);

            work_type fx = gx[i],fy = gy[i];
            work_type ax = fx < 0 ? -fx : fx;
            work_type ay = fy < 0 ? -fy : fy;
            int dx = 0,dy = 0;

            if(ax > ay*tan67)
                dx = 1;
            else if(ay > ax*tan67)
                dy = 1;
            else
                dx = 1,dy = (fx > 0) == (fy > 0) ? 1 : -1;

            size_t x = i%w,y = (i/w)%h;
            if((dx && (!x || x+1 == w)) ||
                (dy && (!y || y+1 == h)))
                return assign(i,m);

            std::ptrdiff_t shift =
                dx+std::ptrdiff_t(dy)*std::ptrdiff_t(w);
            if(mag[size_t(std::ptrdiff_t(i)-shift)] > m ||
                mag[size_t(std::ptrdiff_t(i)+shift)] > m)
                m = 0;
            assign(i,m);
        });
    }
    else
    {
        constexpr work_type inv_sqrt3 =
            work_type(0.5773502691896258);
        const size_t d = src.depth();

        tipl::serial_or_parallel(src,[&](size_t i)
        {
            work_type m = mag[i];
            if(m == 0)
                return assign(i,0);

            work_type threshold = m*inv_sqrt3;
            int dx = gx[i] >= threshold ? 1 :
                         gx[i] <= -threshold ? -1 : 0;
            int dy = gy[i] >= threshold ? 1 :
                         gy[i] <= -threshold ? -1 : 0;
            int dz = gz[i] >= threshold ? 1 :
                         gz[i] <= -threshold ? -1 : 0;

            size_t x = i%w,y = (i/w)%h,z = i/wh;
            if((dx && (!x || x+1 == w)) ||
                (dy && (!y || y+1 == h)) ||
                (dz && (!z || z+1 == d)))
                return assign(i,m);

            std::ptrdiff_t shift =
                dx+std::ptrdiff_t(dy)*std::ptrdiff_t(w)+
                std::ptrdiff_t(dz)*std::ptrdiff_t(wh);
            if(mag[size_t(std::ptrdiff_t(i)-shift)] > m ||
                mag[size_t(std::ptrdiff_t(i)+shift)] > m)
                m = 0;
            assign(i,m);
        });
    }
}

}

template<typename image_type>
image_type& canny_edge(image_type& src)
{
    detail::canny_edge_impl(src);
    return src;
}

template<typename image_type,
         std::enable_if_t<!std::is_lvalue_reference_v<image_type>,int> = 0>
image_type&& canny_edge(image_type&& src)
{
    detail::canny_edge_impl(src);
    return std::move(src);
}

}
}

#endif // CANNY_EDGE_HPP_INCLUDED
