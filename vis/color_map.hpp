#ifndef COLOR_MAP_HPP
#define COLOR_MAP_HPP
#include <vector>
#include "image/utility/basic_image.hpp"

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
            float findex = (float)index*255.0/height();
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
    const image::vector<3,float>& operator[](unsigned int index){return color[std::min<unsigned int>(255,index)];}
    void two_color(image::rgb_color from_color,image::rgb_color to_color)
    {
        color.resize(256);
        for(unsigned int index = 0;index < 256;++index)
        {
            float findex = (float)index/255.0;
            for(unsigned char rgb_index = 0;rgb_index < 3;++rgb_index)
                color[index][2-rgb_index] =
                    ((float)to_color[rgb_index]*findex+(float)from_color[rgb_index]*(1.0-findex))/255.0;
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
    const image::rgb_color& operator[](unsigned int index){return color[std::min<unsigned int>(255,index)];}
    void two_color(image::rgb_color from_color,image::rgb_color to_color)
    {
        for(unsigned int index = 0;index < 256;++index)
        {
            for(unsigned char rgb_index = 0;rgb_index < 3;++rgb_index)
                color[index][2-rgb_index] =
                        std::min<short>(255,((float)to_color[rgb_index]*index+(float)from_color[rgb_index]*(255-index))/255.0);
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
};

}



#endif//COLOR_MAP_HPP

 

