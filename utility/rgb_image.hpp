#ifndef PIXEL_VALUE_HPP
#define PIXEL_VALUE_HPP
#include <cmath>
#include <algorithm>
#include "basic_image.hpp"

namespace tipl
{


struct rgb
{
    union
    {
        unsigned int color;
        struct
        {
            unsigned char b;
            unsigned char g;
            unsigned char r;
            unsigned char a;
		};
        unsigned char data[4];
	};
    rgb(void): color(0) {}
    rgb(unsigned int color_): color(color_) {}
    rgb(int color_): color(color_) {}
    rgb(const rgb& rhs): color(rhs.color) {}
    template<typename value_type>
    rgb(value_type r_, value_type g_, value_type b_):
        b(uint8_t(b_)), g(uint8_t(g_)), r(uint8_t(r_)), a(0) {}
    template<typename value_type>
    rgb(value_type r_,value_type g_,value_type b_,value_type a_):
        b(uint8_t(b_)), g(uint8_t(g_)), r(uint8_t(r_)), a(uint8_t(a_)) {}
    rgb(unsigned char gray):
        b(gray), g(gray), r(gray), a(0) {}
    rgb(float gray):
        b(uint8_t(gray)), g(uint8_t(gray)), r(uint8_t(gray)), a(0) {}
    rgb(double gray):
        b(uint8_t(gray)), g(uint8_t(gray)), r(uint8_t(gray)), a(0) {}
    operator unsigned char() const
    {
        return (unsigned char)((((short)r) + ((short)g) + ((short)b)) / 3);
    }
    operator short() const
    {
        return (((short)r) + ((short)g) + ((short)b)) / 3;
    }
    operator int() const
    {
        return (((int)r) + ((int)g) + ((int)b)) / 3;
    }
    operator float() const
    {
        return (((float)r) + ((float)g) + ((float)b)) / 3.0f;
    }
    operator unsigned int() const
    {
        return color;
    }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    rgb& operator=(const T* v)
    {
        data[0] = std::max(T(0),std::min(T(255),v[0]));
        data[1] = std::max(T(0),std::min(T(255),v[1]));
        data[2] = std::max(T(0),std::min(T(255),v[2]));
        return *this;
    }
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    rgb& operator=(const T& v)
    {
        using U = typename T::value_type;
        data[0] = std::max(U(0),std::min(U(255),v[0]));
        data[1] = std::max(U(0),std::min(U(255),v[1]));
        data[2] = std::max(U(0),std::min(U(255),v[2]));
        return *this;
    }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    rgb& operator=(T gray)
    {
        r = g = b = uint8_t(std::max(T(0),std::min(T(255),gray)));
        return *this;
    }
    rgb& operator=(const rgb& rhs)
    {
        color = rhs.color;
        return *this;
    }
    rgb& operator=(unsigned int color_)
    {
        color = color_;
        return *this;
    }
    rgb& operator=(int color_)
    {
        color = color_;
        return *this;
    }
    unsigned char& operator[](unsigned char index)
    {
        return data[index];
    }
    const unsigned char& operator[](unsigned char index) const
    {
        return data[index];
    }

	double hue(void)
	{
		short r_g = r;
		short r_b = r;
		short g_b = g;
		r_g -= g;
		r_b -= b;
		g_b -= b;
        double t1 = r_g + r_b;
        double t2 = r_g * r_g + r_b * g_b;
		t1 /= 2.0;
		t2 = std::sqrt(t2);
		if(t2 == 0.0)
			return 1.57079632679489661923; //PI/2
		else
            return (b > g) ? 3.14159265358979323846 * 2.0 - std::acos(t1 / t2) : std::acos(t1 / t2);
	}
	double saturation(void)
	{
		unsigned char min_rgb = (r > b && g > b) ? b : ((r > g) ? g : r);
        double sum = ((double)r + (double)g + (double)b);
		double s = min_rgb;
		s *= 3.0;
		if(sum != 0.0)
			s /= sum;
		else
			return 0.0;
        return 1.0 - s;
	}
	double intensity(void)
	{
		double i = r;
		i += g;
		i += b;
        return i / 3.0 / 255.0;
	}
    void from_hsi(double h, double s, double i)
	{
        double r_, g_, b_;
        if(h < 3.14159265358979323846 * 2.0 / 3.0)
		{
            b_ = i * (1.0 - s) * 255.0;
            r_ = 255 * i * (1.0 + s * std::cos(h) / std::cos(3.14159265358979323846 / 3.0 - h));
            g_ = 255.0 * 3.0 * i - r_ - b_;
		}
        else if(h < 3.14159265358979323846 * 4.0 / 3.0)
		{
            h -= 3.14159265358979323846 * 2.0 / 3.0;
            r_ = i * (1.0 - s) * 255.0;
            g_ = 255 * i * (1.0 + s * std::cos(h) / std::cos(3.14159265358979323846 / 3.0 - h));
            b_ = 255.0 * 3.0 * i - r_ - g_;
		}
		else
		{
            h -= 3.14159265358979323846 * 4.0 / 3.0;
            g_ = i * (1.0 - s) * 255.0;
            b_ = 255 * i * (1.0 + s * std::cos(h) / std::cos(3.14159265358979323846 / 3.0 - h));
            r_ = 255.0 * 3.0 * i - b_ - g_;
		}
        r = (r_ >= 255.0) ? 255 : (unsigned char)r_;
        g = (g_ >= 255.0) ? 255 : (unsigned char)g_;
        b = (b_ >= 255.0) ? 255 : (unsigned char)b_;
	}
    void from_hsl(double h, double s, double l)
    {
        h /= 3.14159265358979323846/3.0;
        double c = (1.0-std::abs(l+l-1.0))*s;
        double x = c*(1.0-std::abs(h-std::floor(h/2)*2-1.0));
        double r_ = 0,g_ = 0,b_ = 0;
        if(h < 1)
        {
            r_ = c;
            g_ = x;
        }
        else
            if(h < 2)
            {
                r_ = x;
                g_ = c;
            }
            else
                if(h < 3)
                {
                    g_ = c;
                    b_ = x;
                }
                else
                    if(h < 4)
                    {
                        g_ = x;
                        b_ = c;
                    }
                    else
                        if(h < 5)
                        {
                            r_ = x;
                            b_ = c;
                        }
                        else
                            {
                                r_ = c;
                                b_ = x;
                            }
        double m = l-c*0.5;
        r_ += m;
        g_ += m;
        b_ += m;
        r_ *= 256.0;
        g_ *= 256.0;
        b_ *= 256.0;
        r = (r_ >= 255.0) ? 255 : (unsigned char)r_;
        g = (g_ >= 255.0) ? 255 : (unsigned char)g_;
        b = (b_ >= 255.0) ? 255 : (unsigned char)b_;
    }

    static rgb generate(int color_gen)
    {
        tipl::rgb color;
        double var2[8] = {0.0,-0.2, 0.1, -0.15, 0.05 ,-0.2,0.1,-0.15};
        color.from_hsl(std::fmod(color_gen*0.618033988749895, 2.0)*3.14159265358979323846,0.85+(((color_gen/13)%2)?-0.1:0.1),0.7+var2[(color_gen/13)%8]);
        return color;
    }
    static rgb generate_hue(int color_gen)
    {
        tipl::rgb color;
        color.from_hsl(std::fmod(color_gen*0.618033988749895, 2.0)*3.14159265358979323846,0.85,0.7);
        return color;
    }
    void operator|=(const rgb& rhs)
    {
        r = rhs.r | r;
        g = rhs.g | g;
        b = rhs.b | b;
    }
    void operator&=(const rgb& rhs)
    {
        r = rhs.r & r;
        g = rhs.g & g;
        b = rhs.b & b;
    }
    bool operator==(const rgb& rhs) const
    {
        return color == rhs.color;
    }
    bool operator!=(const rgb& rhs) const
    {
        return color != rhs.color;
    }

};


using color_image = image<2,rgb> ;
using grayscale_image = image<2,unsigned char>;

//---------------------------------------------------------------------------
}
#endif
