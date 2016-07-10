#ifndef COLOR_MAP_HPP
#define COLOR_MAP_HPP
#include <vector>
#include <fstream>
#include <iterator>
#include "image/utility/basic_image.hpp"
#include "image/numerical/basic_op.hpp"
namespace image{

inline unsigned char color_spectrum_value(unsigned char center, unsigned char value)
{
    unsigned char dif = center > value ? center-value:value-center;
    if(dif < 32)
        return 255;
    dif -= 32;
    if(dif >= 64)
        return 0;
    return 255-(dif << 2);
}

struct color_bar : public image::color_image{
public:
    color_bar(unsigned int width,unsigned int height)
    {
        resize(image::geometry<2>(width,height));
    }
    void two_color(image::rgb_color from_color,image::rgb_color to_color)
    {
        resize(image::geometry<2>(20,256));
        for(unsigned int index = 1;index < height();++index)
        {
            float findex = (float)index/(float)height();
            image::rgb_color color;
            for(unsigned char rgb_index = 0;rgb_index < 3;++rgb_index)
                color[rgb_index] = (float)from_color[rgb_index]*findex+(float)to_color[rgb_index]*(1.0-findex);
            std::fill(begin()+index*width()+1,begin()+(index+1)*width()-1,color);
        }
    }
    void spectrum(void)
    {
        for(unsigned int index = 1;index < height();++index)
        {
            float findex = (float)index*(float)255.0/height();
            image::rgb_color color;
            color.r = image::color_spectrum_value(64,findex);
            color.g = image::color_spectrum_value(128,findex);
            color.b = image::color_spectrum_value(128+64,findex);
            std::fill(begin()+index*width()+1,begin()+(index+1)*width()-1,color);
        }
    }
};

struct color_map{
    std::vector<image::vector<3,float> > color;
public:
    color_map(void):color(256){}
    size_t size(void)const{return color.size();}
    const image::vector<3,float>& operator[](unsigned int index) const{return color[255,index];}
    image::vector<3,float> min_color(void)const{return color.front();}
    image::vector<3,float> max_color(void)const{return color.back();}
    void two_color(image::rgb_color from_color,image::rgb_color to_color)
    {
        color.resize(256);
        for(unsigned int index = 0;index < 256;++index)
        {
            float findex = (float)index/255.0;
            for(unsigned char rgb_index = 0;rgb_index < 3;++rgb_index)
                color[index][rgb_index] = ((float)to_color[rgb_index]*findex+(float)from_color[rgb_index]*(1.0-findex))/255.0;
        }
    }
    void spectrum(void)
    {
        color.resize(256);
        for(unsigned int index = 0;index < 256;++index)
        {
            color[index][0] = (float)image::color_spectrum_value(128+64,index)/255.0;
            color[index][1] = (float)image::color_spectrum_value(128,index)/255.0;
            color[index][2] = (float)image::color_spectrum_value(64,index)/255.0;
        }
    }
};


struct color_map_rgb{
    std::vector<image::rgb_color> color;
public:
    color_map_rgb(void):color(256){}
    unsigned int size(void)const{return color.size();}
    const image::rgb_color& operator[](unsigned int index) const{return color[index];}
    image::rgb_color min_color(void)const{return color.front();}
    image::rgb_color max_color(void)const{return color.back();}
    void two_color(image::rgb_color from_color,image::rgb_color to_color)
    {
        for(unsigned int index = 0;index < 256;++index)
        {
            for(unsigned char rgb_index = 0;rgb_index < 3;++rgb_index)
                color[index][rgb_index] = std::min<short>(255,((float)to_color[rgb_index]*index+(float)from_color[rgb_index]*(255-index))/255.0);
        }
    }
    void spectrum(void)
    {
        for(unsigned int index = 0;index < 256;++index)
        {
            color[index][2] = image::color_spectrum_value(128+64,index);
            color[index][1] = image::color_spectrum_value(128,index);
            color[index][0] = image::color_spectrum_value(64,index);
        }
    }
    bool load_from_file(const char* file_name)
    {
        std::ifstream in(file_name);
        if(!in)
            return false;
        std::vector<float> values;
        std::copy(std::istream_iterator<float>(in),
                  std::istream_iterator<float>(),std::back_inserter(values));
        float max_value = *std::max_element(values.begin(),values.end());
        if(max_value < 2.0 && max_value != 0.0)
        {
            for(unsigned int i = 0;i < values.size();++i)
                values[i] = std::max<int>(0,std::min<int>(255,std::floor(values[i]*256.0/max_value)));
        }
        if(values.size() < 3)
            return false;
        color.clear();
        for(unsigned int i = 2;i < values.size();i += 3)
            color.push_back(image::rgb_color(values[i-2],values[i-1],values[i]));
        return true;
    }
};

template<class value_type>
struct value_to_color{
private:
    value_type min_value,max_value,r;
    image::color_map_rgb map;
public:
    value_to_color(void):min_value(0),r(1){}
    image::rgb_color min_color(void)const{return map.min_color();}
    image::rgb_color max_color(void)const{return map.max_color();}

    void set_range(value_type min_value_,value_type max_value_)
    {
        min_value = min_value_;
        max_value = max_value_;
        max_value_ -= min_value_;
        r = (max_value_ == 0.0) ? 1.0:(float)map.size()/max_value_;
    }
    void set_color_map(const image::color_map_rgb& rhs)
    {
        map = rhs;
        r = max_value-min_value;
        r = (r == 0.0) ? 1.0:(float)map.size()/r;
    }
    void two_color(image::rgb_color from_color,image::rgb_color to_color)
    {
        map.two_color(from_color,to_color);
    }

    const image::rgb_color& operator[](value_type value)const
    {
        value -= min_value;
        value *= r;
        int ivalue = std::floor(value);
        if(ivalue < 0)
            ivalue = 0;
        if(ivalue >= map.size())
            ivalue = map.size()-1;
        return map[ivalue];
    }
    template<class value_image_type,class color_image_type>
    void convert(const value_image_type& I1,color_image_type& I2) const
    {
        I2.resize(I1.geometry());
        for(unsigned int i = 0;i < I1.size();++i)
            I2[i] = (*this)[I1[i]];
    }
};

}



#endif//COLOR_MAP_HPP

 
