#ifndef ANISOTROPIC_DIFFUSION_HPP_INCLUDED
#define ANISOTROPIC_DIFFUSION_HPP_INCLUDED

#include <array>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>
#include "filter_model.hpp"
#include "../mt.hpp"

namespace tipl
{
namespace filter
{
namespace detail
{

template<bool inverse,typename image_type>
void anisotropic_diffusion_impl(image_type& src,float conductance_parameter,unsigned int iteration)
{
    using out_type = typename image_type::value_type;
    using work_type = typename pixel_manip<out_type>::type;
    constexpr size_t dimension = image_type::dimension;

    if(src.empty() || !iteration || !conductance_parameter)
        return;

    const size_t size = src.size();
    const auto shape = src.shape();

    std::vector<work_type> image(size),gradient(size),update(size);
    tipl::serial_or_parallel(image.size(),[&](size_t i){image[i] = pixel_manip<out_type>::to_work(src[i]);});

    std::array<size_t,dimension> steps,lengths;
    std::array<std::vector<size_t>,dimension> lines;
    size_t step = 1;

    for(size_t dim = 0;dim < dimension;++dim)
    {
        const size_t length = shape[dim],block = step*length;
        steps[dim] = step;
        lengths[dim] = length;

        auto& line = lines[dim];
        line.reserve(size/length);
        for(size_t pos = 0;pos < size;pos += block)
            for(size_t p = pos,end = pos+step;p < end;++p)
                line.push_back(p);

        step = block;
    }

    double conductance =
        double(conductance_parameter)*conductance_parameter;
    if constexpr(!inverse)
        conductance *= 1.3862943611198906;

    for(unsigned int iter = 0;iter < iteration;++iter)
    {
        bool first = true;

        for(size_t dim = 0;dim < dimension;++dim)
        {
            const size_t step = steps[dim];
            const size_t length = lengths[dim];
            const auto& line = lines[dim];

            if(length < 2)
                continue;

            tipl::serial_or_parallel(size,line.size(),[&](size_t index)
            {
                size_t p = line[index];
                gradient[p] = work_type();

                for(size_t n = 1;n < length;++n)
                {
                    p += step;
                    gradient[p] = image[p]-image[p-step];
                }
            });

            const double K = double(tipl::square_sum(gradient))*conductance/double(size);

            if(K <= 0.0)
                continue;

            const double invK = 1.0/K;

            tipl::serial_or_parallel(size,line.size(),[&](size_t index)
            {
                size_t p = line[index];
                work_type previous = 0;

                for(size_t n = 1;n < length;++n)
                {
                    p += step;

                    double g = gradient[p];
                    double q = g*g*invK;
                    work_type current;

                    if constexpr(inverse)
                        current = work_type(g/(1.0+q));
                    else
                        current = work_type(g*std::exp2(-q));

                    if(first)
                        update[p-step] = current-previous;
                    else
                        update[p-step] += current-previous;

                    previous = current;
                }

                if(first)
                    update[p] = -previous;
                else
                    update[p] -= previous;
            });

            first = false;
        }

        if(first)
            break;

        tipl::serial_or_parallel(image.size(),[&](size_t i){image[i] += std::ldexp(update[i],-int(dimension));});
    }
    tipl::serial_or_parallel(image.size(),[&](size_t i){src[i] = pixel_manip<out_type>::to_pixel(image[i]);});
}

}

template<typename image_type>
image_type& anisotropic_diffusion(
    image_type& src,float conductance_parameter = 1.0f,
    unsigned int iteration = 5)
{
    detail::anisotropic_diffusion_impl<false>(
        src,conductance_parameter,iteration);
    return src;
}

template<typename image_type>
image_type& anisotropic_diffusion_inv(
    image_type& src,float conductance_parameter = 1.0f,
    unsigned int iteration = 5)
{
    detail::anisotropic_diffusion_impl<true>(
        src,conductance_parameter,iteration);
    return src;
}

template<typename image_type,
         std::enable_if_t<!std::is_lvalue_reference_v<image_type>,int> = 0>
image_type&& anisotropic_diffusion(
    image_type&& src,float conductance_parameter = 1.0f,
    unsigned int iteration = 5)
{
    anisotropic_diffusion(
        static_cast<image_type&>(src),conductance_parameter,iteration);
    return std::move(src);
}

template<typename image_type,
         std::enable_if_t<!std::is_lvalue_reference_v<image_type>,int> = 0>
image_type&& anisotropic_diffusion_inv(
    image_type&& src,float conductance_parameter = 1.0f,
    unsigned int iteration = 5)
{
    anisotropic_diffusion_inv(
        static_cast<image_type&>(src),conductance_parameter,iteration);
    return std::move(src);
}

}
}

#endif
