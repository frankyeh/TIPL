#ifndef FILTER_MODEL_HPP
#define FILTER_MODEL_HPP
#include <algorithm> // for std::min
#include "../utility/basic_image.hpp"
#include "../utility/rgb_image.hpp"
#include "../numerical/numerical.hpp"

namespace tipl
{
namespace filter
{

struct rgb_manip
{
    int r = 0, g = 0, b = 0;

    rgb_manip() = default;
    rgb_manip(tipl::rgb color) : r(color.r), g(color.g), b(color.b) {}

    rgb_manip operator+(const rgb_manip& rhs) const { return rgb_manip(*this) += rhs; }
    rgb_manip operator-(const rgb_manip& rhs) const { return rgb_manip(*this) -= rhs; }

    rgb_manip& operator+=(const rgb_manip& rhs) { r += rhs.r; g += rhs.g; b += rhs.b; return *this; }
    rgb_manip& operator-=(const rgb_manip& rhs) { r -= rhs.r; g -= rhs.g; b -= rhs.b; return *this; }

    rgb_manip& operator+=(int value) { r += value; g += value; b += value; return *this; }
    rgb_manip& operator*=(int value) { r *= value; g *= value; b *= value; return *this; }
    rgb_manip& operator/=(int value) { r /= value; g /= value; b /= value; return *this; }

    rgb_manip& operator=(tipl::rgb color) { r = color.r; g = color.g; b = color.b; return *this; }
    rgb_manip& operator+=(tipl::rgb color) { r += color.r; g += color.g; b += color.b; return *this; }
    rgb_manip& operator-=(tipl::rgb color) { r -= color.r; g -= color.g; b -= color.b; return *this; }

    rgb_manip& operator>>=(unsigned char value) { r >>= value; g >>= value; b >>= value; return *this; }
    rgb_manip& operator<<=(unsigned char value) { r <<= value; g <<= value; b <<= value; return *this; }

    void abs()
    {
        if (r < 0) r = -r;
        if (g < 0) g = -g;
        if (b < 0) b = -b;
    }

    tipl::rgb to_rgb() const
    {
        return tipl::rgb(std::min(255, r), std::min(255, g), std::min(255, b));
    }
};

template<typename PixelType>    struct pixel_manip { typedef PixelType type; };
template<>  struct pixel_manip<unsigned char>   { typedef short type; };
template<>  struct pixel_manip<char>            { typedef short type; };
template<>  struct pixel_manip<short>           { typedef int type; };
template<>  struct pixel_manip<unsigned short>  { typedef int type; };
template<>  struct pixel_manip<tipl::rgb>       { typedef rgb_manip type; };

template<typename value_type, size_t w>
struct weight {
    constexpr value_type operator()(value_type value) const { return value * w; }
};

template<typename value_type>
struct weight<value_type, 1> {
    constexpr value_type operator()(value_type value) const { return value; }
};

template<size_t weight_value, typename dest_type, typename src_type>
void add_weight(dest_type& dest, const src_type& src, int64_t shift)
{
    weight<typename dest_type::value_type, weight_value> w;
    auto d_it = shift >= 0 ? dest.begin() + shift : dest.begin();
    auto s_it = shift >= 0 ? src.begin() : src.begin() - shift;
    auto d_end = dest.end();
    auto s_end = src.end();
    while (d_it < d_end && s_it < s_end)
    {
        *d_it += w(*s_it);
        ++d_it;
        ++s_it;
    }
}

template<size_t weight_value, typename dest_type, typename src_type>
void minus_weight(dest_type& dest, const src_type& src, int64_t shift)
{
    weight<typename dest_type::value_type, weight_value> w;
    auto d_it = shift >= 0 ? dest.begin() + shift : dest.begin();
    auto s_it = shift >= 0 ? src.begin() : src.begin() - shift;
    auto d_end = dest.end();
    auto s_end = src.end();

    while (d_it < d_end && s_it < s_end)
    {
        *d_it -= w(*s_it);
        ++d_it;
        ++s_it;
    }
}

} // namespace filter
} // namespace tipl

#endif // FILTER_MODEL_HPP
