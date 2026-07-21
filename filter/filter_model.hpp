#ifndef FILTER_MODEL_HPP
#define FILTER_MODEL_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include "../utility/basic_image.hpp"
#include "../utility/rgb_image.hpp"
#include "../numerical/numerical.hpp"

namespace tipl
{
namespace filter
{

template<typename PixelType>
struct pixel_manip
{
    using type = std::conditional_t<
        std::is_integral_v<PixelType>,
        std::conditional_t<(sizeof(PixelType) <= 2),float,double>,
        PixelType>;

    static type to_work(PixelType value)
    {
        return type(value);
    }

    static PixelType to_pixel(type value)
    {
        if constexpr(std::is_integral_v<PixelType>)
        {
            value = std::round(value);
            return PixelType(std::clamp(
                value,
                type(std::numeric_limits<PixelType>::lowest()),
                type(std::numeric_limits<PixelType>::max())));
        }
        else
            return value;
    }
};

template<>
struct pixel_manip<tipl::rgb>
{
    using type = tipl::vector<3,float>;

    static type to_work(tipl::rgb value)
    {
        type result;
        result[0] = value.r;
        result[1] = value.g;
        result[2] = value.b;
        return result;
    }

    static tipl::rgb to_pixel(const type& value)
    {
        return tipl::rgb(
            std::clamp<int>(std::lround(value[0]),0,255),
            std::clamp<int>(std::lround(value[1]),0,255),
            std::clamp<int>(std::lround(value[2]),0,255));
    }
};

}
}

#endif
