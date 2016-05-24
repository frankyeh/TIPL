#ifndef PIXEL_VALUE_HPP
#define PIXEL_VALUE_HPP
#include <cmath>
namespace image
{


struct rgb_color
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
    rgb_color(void): color(0) {}
    rgb_color(unsigned int color_): color(color_) {}
    rgb_color(int color_): color(color_) {}
    rgb_color(const rgb_color& rhs): color(rhs.color) {}
    rgb_color(unsigned char r_, unsigned char g_, unsigned char b_):
        b(b_), g(g_), r(r_), a(0) {}
    rgb_color(unsigned char r_, unsigned char g_, unsigned char b_, unsigned char a_):
        b(b_), g(g_), r(r_), a(a_) {}
    rgb_color(unsigned char gray):
        b(gray), g(gray), r(gray), a(0) {}
    rgb_color(float gray):
        b((unsigned char)gray), g((unsigned char)gray), r((unsigned char)gray), a(0) {}

    operator unsigned char() const
    {
        return (unsigned char)((((short)r) + ((short)g) + ((short)b)) / 3);
    }
    operator short() const
    {
        return (((short)r) + ((short)g) + ((short)b)) / 3;
    }
    operator float() const
    {
        return (((float)r) + ((float)g) + ((float)b)) / 3.0f;
    }
    operator unsigned int() const
    {
        return color;
    }
    const rgb_color& operator=(unsigned int color_)
    {
        color = color_;
        return *this;
    }
    const rgb_color& operator=(int color_)
    {
        color = color_;
        return *this;
    }
    const rgb_color& operator=(const rgb_color& rhs)
    {
        color = rhs.color;
        return *this;
    }
    const rgb_color& operator=(unsigned char gray)
    {
        r = g = b = gray;
        return *this;
    }
    const rgb_color& operator=(unsigned short gray)
    {
        r = g = b = (unsigned char)gray;
        return *this;
    }
    const rgb_color& operator=(short gray)
    {
        r = g = b = (unsigned char)gray;
        return *this;
    }
    const rgb_color& operator=(float gray)
    {
        r = g = b = (unsigned char)gray;
        return *this;
    }
    const rgb_color& operator=(double gray)
    {
        r = g = b = (unsigned char)gray;
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

    bool operator==(const rgb_color& rhs) const
    {
        return color == rhs.color;
    }
    bool operator!=(const rgb_color& rhs) const
    {
        return color != rhs.color;
    }

};
//---------------------------------------------------------------------------
}
#endif
