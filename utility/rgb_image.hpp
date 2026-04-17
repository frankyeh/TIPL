#ifndef PIXEL_VALUE_HPP
#define PIXEL_VALUE_HPP

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include "basic_image.hpp"

namespace tipl
{

struct rgb
{
    union
    {
        uint32_t color;
        struct
        {
            uint8_t b;
            uint8_t g;
            uint8_t r;
            uint8_t a;
        };
        uint8_t data[4];
    };

    constexpr rgb() : color(0) {}
    constexpr rgb(uint32_t color_) : color(color_) {}
    constexpr rgb(int color_) : color(static_cast<uint32_t>(color_)) {}

    // Unified RGB(A) constructor
    template<typename T>
    constexpr rgb(T r_, T g_, T b_, T a_ = 0) :
        b(static_cast<uint8_t>(b_)), g(static_cast<uint8_t>(g_)),
        r(static_cast<uint8_t>(r_)), a(static_cast<uint8_t>(a_)) {}

    // Unified Grayscale constructor
    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
    constexpr rgb(T gray) :
        b(static_cast<uint8_t>(gray)), g(static_cast<uint8_t>(gray)),
        r(static_cast<uint8_t>(gray)), a(0) {}

    // Helper for grayscale casting
    template<typename T>
    constexpr T to_gray() const
    {
        return static_cast<T>((static_cast<uint16_t>(r) + g + b) / 3.0);
    }

    explicit operator uint8_t() const { return to_gray<uint8_t>(); }
    explicit operator short() const   { return to_gray<short>(); }
    explicit operator int() const     { return to_gray<int>(); }
    explicit operator float() const   { return to_gray<float>(); }
    operator uint32_t() const         { return color; }

    template<typename T, typename std::enable_if<std::is_fundamental<T>::value, bool>::type = true>
    rgb& operator=(const T* v)
    {
        r = static_cast<uint8_t>(std::clamp(v[0], T(0), T(255)));
        g = static_cast<uint8_t>(std::clamp(v[1], T(0), T(255)));
        b = static_cast<uint8_t>(std::clamp(v[2], T(0), T(255)));
        return *this;
    }

    template<typename T, typename std::enable_if<std::is_class<T>::value, bool>::type = true>
    rgb& operator=(const T& v)
    {
        using U = typename T::value_type;
        r = static_cast<uint8_t>(std::clamp(v[0], U(0), U(255)));
        g = static_cast<uint8_t>(std::clamp(v[1], U(0), U(255)));
        b = static_cast<uint8_t>(std::clamp(v[2], U(0), U(255)));
        return *this;
    }

    template<typename T, typename std::enable_if<std::is_fundamental<T>::value, bool>::type = true>
    rgb& operator=(T gray)
    {
        r = g = b = static_cast<uint8_t>(std::clamp(gray, T(0), T(255)));
        return *this;
    }

    rgb& operator=(uint32_t color_) { color = color_; return *this; }
    rgb& operator=(int color_)      { color = static_cast<uint32_t>(color_); return *this; }

    uint8_t& operator[](size_t index)             { return data[index]; }
    const uint8_t& operator[](size_t index) const { return data[index]; }

    // Color Math Constants
    static constexpr double PI = 3.14159265358979323846;
    static constexpr double TWO_PI = 2.0 * PI;
    static constexpr double PI_OVER_3 = PI / 3.0;

    double hue() const
    {
        double r_g = r - g;
        double r_b = r - b;
        double g_b = g - b;
        double t1 = (r_g + r_b) * 0.5;
        double t2 = std::sqrt(r_g * r_g + r_b * g_b);

        if (t2 == 0.0) return PI * 0.5;

        double theta = std::acos(t1 / t2);
        return (b > g) ? (TWO_PI - theta) : theta;
    }

    double saturation() const
    {
        uint8_t min_rgb = std::min({r, g, b});
        double sum = static_cast<double>(r) + g + b;

        if (sum == 0.0) return 0.0;
        return 1.0 - ((3.0 * min_rgb) / sum);
    }

    double intensity() const
    {
        return (static_cast<double>(r) + g + b) / (3.0 * 255.0);
    }

    void from_hsi(double h, double s, double i)
    {
        double r_, g_, b_;
        if (h < TWO_PI / 3.0)
        {
            b_ = i * (1.0 - s);
            r_ = i * (1.0 + s * std::cos(h) / std::cos(PI_OVER_3 - h));
            g_ = 3.0 * i - r_ - b_;
        }
        else if (h < TWO_PI * 2.0 / 3.0)
        {
            h -= TWO_PI / 3.0;
            r_ = i * (1.0 - s);
            g_ = i * (1.0 + s * std::cos(h) / std::cos(PI_OVER_3 - h));
            b_ = 3.0 * i - r_ - g_;
        }
        else
        {
            h -= TWO_PI * 2.0 / 3.0;
            g_ = i * (1.0 - s);
            b_ = i * (1.0 + s * std::cos(h) / std::cos(PI_OVER_3 - h));
            r_ = 3.0 * i - b_ - g_;
        }

        r = static_cast<uint8_t>(std::clamp(r_ * 255.0, 0.0, 255.0));
        g = static_cast<uint8_t>(std::clamp(g_ * 255.0, 0.0, 255.0));
        b = static_cast<uint8_t>(std::clamp(b_ * 255.0, 0.0, 255.0));
    }

    void from_hsl(double h, double s, double l)
    {
        h /= PI_OVER_3;
        double c = (1.0 - std::abs(2.0 * l - 1.0)) * s;
        double x = c * (1.0 - std::abs(std::fmod(h, 2.0) - 1.0));
        double r_ = 0, g_ = 0, b_ = 0;

        int h_i = static_cast<int>(h) % 6;
        switch(h_i) {
            case 0: r_ = c; g_ = x; break;
            case 1: r_ = x; g_ = c; break;
            case 2: g_ = c; b_ = x; break;
            case 3: g_ = x; b_ = c; break;
            case 4: r_ = x; b_ = c; break;
            case 5: r_ = c; b_ = x; break;
        }

        double m = l - c * 0.5;
        r = static_cast<uint8_t>(std::clamp((r_ + m) * 256.0, 0.0, 255.0));
        g = static_cast<uint8_t>(std::clamp((g_ + m) * 256.0, 0.0, 255.0));
        b = static_cast<uint8_t>(std::clamp((b_ + m) * 256.0, 0.0, 255.0));
    }

    static rgb generate(int color_gen)
    {
        static constexpr double GOLDEN_RATIO_CONJ = 0.618033988749895;
        static constexpr double var2[8] = {0.0, -0.2, 0.1, -0.15, 0.05, -0.2, 0.1, -0.15};

        rgb color;
        double h = std::fmod(color_gen * GOLDEN_RATIO_CONJ, 2.0) * PI;
        double s = 0.85 + (((color_gen / 13) % 2) ? -0.1 : 0.1);
        double l = 0.7 + var2[(color_gen / 13) % 8];
        color.from_hsl(h, s, l);
        return color;
    }

    static rgb generate_hue(int color_gen)
    {
        static constexpr double GOLDEN_RATIO_CONJ = 0.618033988749895;
        rgb color;
        color.from_hsl(std::fmod(color_gen * GOLDEN_RATIO_CONJ, 2.0) * PI, 0.85, 0.7);
        return color;
    }

    rgb& operator|=(const rgb& rhs)
    {
        r |= rhs.r; g |= rhs.g; b |= rhs.b;
        return *this;
    }

    rgb& operator&=(const rgb& rhs)
    {
        r &= rhs.r; g &= rhs.g; b &= rhs.b;
        return *this;
    }

    bool operator==(const rgb& rhs) const { return color == rhs.color; }
    bool operator!=(const rgb& rhs) const { return color != rhs.color; }
};

using color_image = image<2, rgb>;
using grayscale_image = image<2, uint8_t>;

} // namespace tipl

#endif // PIXEL_VALUE_HPP
