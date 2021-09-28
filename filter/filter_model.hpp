//---------------------------------------------------------------------------
#ifndef FILTER_MODEL_HPP
#define FILTER_MODEL_HPP
#include "../utility/basic_image.hpp"
#include "../numerical/numerical.hpp"
//---------------------------------------------------------------------------
namespace tipl
{

namespace filter
{

struct rgb_manip
{
    int r,g,b;

    rgb_manip(tipl::rgb color):r(color.r),g(color.g),b(color.b) {}
    rgb_manip(void):r(0),g(0),b(0) {}
public:
    rgb_manip operator+(const rgb_manip& rhs)
    {
        rgb_manip result(*this);
        result.r += rhs.r;
        result.g += rhs.g;
        result.b += rhs.b;
        return result;
    }
    rgb_manip operator-(const rgb_manip& rhs)
    {
        rgb_manip result(*this);
        result.r -= rhs.r;
        result.g -= rhs.g;
        result.b -= rhs.b;
        return result;
    }
    void operator+=(const rgb_manip& rhs)
    {
        r += rhs.r;
        g += rhs.g;
        b += rhs.b;
    }
    void operator-=(const rgb_manip& rhs)
    {
        r -= rhs.r;
        g -= rhs.g;
        b -= rhs.b;
    }
    void operator+=(int value)
    {
        r += value;
        g += value;
        b += value;
    }
    void operator=(rgb color)
    {
        r = color.r;
        g = color.g;
        b = color.b;
    }
    void operator+=(rgb color)
    {
        r += color.r;
        g += color.g;
        b += color.b;
    }
    void operator-=(rgb color)
    {
        r -= color.r;
        g -= color.g;
        b -= color.b;
    }
    void operator*=(int value)
    {
        r *= value;
        g *= value;
        b *= value;
    }
    void operator/=(int value)
    {
        r /= value;
        g /= value;
        b /= value;
    }
    void operator>>=(unsigned char value)
    {
        r >>= value;
        g >>= value;
        b >>= value;
    }
    void operator<<=(unsigned char value)
    {
        r <<= value;
        g <<= value;
        b <<= value;
    }
    void abs(void)
    {
        if (r < 0) r = -r;
        if (g < 0) g = -g;
        if (b < 0) b = -b;
    }
    tipl::rgb to_rgb(void) const
    {
        return tipl::rgb(std::min(255,r),std::min(255,g),std::min(255,b));
    }
};


template<typename PixelType>    struct pixel_manip {typedef PixelType type;};
template<>  struct pixel_manip<unsigned char>   {typedef short type;};
template<>  struct pixel_manip<char>            {typedef short type;};
template<>  struct pixel_manip<short>           {typedef int type;};
template<>  struct pixel_manip<unsigned short>  {typedef int type;};
template<>  struct pixel_manip<tipl::rgb>{typedef rgb_manip type;};


template<typename value_type,size_t w>  struct weight               {value_type operator()(value_type value)    {return value*w;}};
template<typename value_type>           struct weight<value_type,1> {value_type operator()(value_type value)    {return value;}};
template<typename value_type>           struct weight<value_type,2> {value_type operator()(value_type value)    {return value+value;}};

template<>  struct weight<unsigned int,3> {unsigned int operator()(unsigned int value){return (value << 1) + value;}};
template<>  struct weight<unsigned int,4> {unsigned int operator()(unsigned int value){return value << 2;}};
template<>  struct weight<unsigned int,5> {unsigned int operator()(unsigned int value){return (value << 2) + value;}};
template<>  struct weight<unsigned int,6> {unsigned int operator()(unsigned int value){return (value << 2) + (value << 1);}};

template<>  struct weight<int,3> {int operator()(int value){return (value << 1) + value;}};
template<>  struct weight<int,4> {int operator()(int value){return value << 2;}};
template<>  struct weight<int,5> {int operator()(int value){return (value << 2) + value;}};
template<>  struct weight<int,6> {int operator()(int value){return (value << 2) + (value << 1);}};

template<>  struct weight<unsigned short,3> {unsigned short operator()(unsigned short value){return (value << 1) + value;}};
template<>  struct weight<unsigned short,4> {unsigned short operator()(unsigned short value){return value << 2;}};
template<>  struct weight<unsigned short,5> {unsigned short operator()(unsigned short value){return (value << 2) + value;}};
template<>  struct weight<unsigned short,6> {unsigned short operator()(unsigned short value){return (value << 2) + (value << 1);}};

template<>  struct weight<short,3> {short operator()(short value){return (value << 1) + value;}};
template<>  struct weight<short,4> {short operator()(short value){return value << 2;}};
template<>  struct weight<short,5> {short operator()(short value){return (value << 2) + value;}};
template<>  struct weight<short,6> {short operator()(int value){return (value << 2) + (value << 1);}};

template<>  struct weight<unsigned char,3> {unsigned char operator()(unsigned char value){return (value << 1) + value;}};
template<>  struct weight<unsigned char,4> {unsigned char operator()(unsigned char value){return value << 2;}};
template<>  struct weight<unsigned char,5> {unsigned char operator()(unsigned char value){return (value << 2) + value;}};
template<>  struct weight<unsigned char,6> {unsigned char operator()(unsigned char value){return (value << 2) + (value << 1);}};

template<>  struct weight<char,3> {char operator()(char value){return (value << 1) + value;}};
template<>  struct weight<char,4> {char operator()(char value){return value << 2;}};
template<>  struct weight<char,5> {char operator()(char value){return (value << 2) + value;}};
template<>  struct weight<char,6> {char operator()(int value){return (value << 2) + (value << 1);}};

template<size_t weight_value,typename dest_type,typename src_type>
void add_weight(dest_type& dest,const src_type& src,int shift)
{
    weight<typename dest_type::value_type,weight_value> w;
    if (shift >= 0)
    {
        typename dest_type::iterator iter1 = dest.begin() + shift;
        typename src_type::const_iterator iter2 = src.begin();
        typename dest_type::iterator end = dest.end();
        for (;iter1 < end;++iter1,++iter2)
            *iter1 += w(*iter2);
    }
    else
    {
        typename dest_type::iterator iter1 = dest.begin();
        typename src_type::const_iterator iter2 = src.begin() + (-shift);
        typename src_type::const_iterator end = src.end();
        for (;iter2 < end;++iter1,++iter2)
            *iter1 += w(*iter2);
    }
}


template<size_t weight_value,typename dest_type,typename src_type>
void minus_weight(dest_type& dest,const src_type& src,int shift)
{
    weight<typename dest_type::value_type,weight_value> w;
    if (shift >= 0)
    {
        typename dest_type::iterator iter1 = dest.begin() + shift;
        typename src_type::const_iterator iter2 = src.begin();
        typename dest_type::iterator end = dest.end();
        for (;iter1 < end;++iter1,++iter2)
            *iter1 -= w(*iter2);
    }
    else
    {
        typename dest_type::iterator iter1 = dest.begin();
        typename src_type::const_iterator iter2 = src.begin() + (-shift);
        typename src_type::const_iterator end = src.end();
        for (;iter2 < end;++iter1,++iter2)
            *iter1 -= w(*iter2);
    }
}


}

}
#endif
