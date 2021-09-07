//---------------------------------------------------------------------------
#ifndef WINDOW_HPP
#define WINDOW_HPP
#include <vector>
#include "../utility/pixel_index.hpp"

namespace tipl
{
//---------------------------------------------------------------------------
template<typename ImageType,typename PixelType>
void get_window(const pixel_index<2>& index,const ImageType& image,unsigned int width,std::vector<PixelType>& pixels)
{
    unsigned int fx = (index.x() > width) ? index.x() - width:0;
    unsigned int fy = (index.y() > width) ? index.y() - width:0;
    unsigned int tx = std::min<size_t>(index.x() + width,image.width()-1);
    unsigned int ty = std::min<size_t>(index.y() + width,image.height()-1);
    unsigned int y_index = fy*image.width()+fx;
    for (unsigned int y = fy;y <= ty;++y,y_index += image.width())
    {
        unsigned int x_index = y_index;
        for (unsigned int x = fx;x <= tx;++x,++x_index)
            pixels.push_back(image[x_index]);
    }
}
//---------------------------------------------------------------------------
template<typename ImageType,typename PixelType>
void get_window(const pixel_index<2>& index,const ImageType& image,std::vector<PixelType>& pixels)
{
    unsigned int width = image.width();
    unsigned int height = image.height();
    bool have_left = index.x() >= 1;
    bool have_right = index.x()+1 < width;
    if (index.y() >= 1)
    {
        unsigned int base_index = index.index()-width;
        if (have_left)
            pixels.push_back(image[base_index-1]);

        pixels.push_back(image[base_index]);

        if (have_right)
            pixels.push_back(image[base_index+1]);
    }

    {
        if (have_left)
            pixels.push_back(image[index.index()-1]);

        pixels.push_back(image[index.index()]);

        if (have_right)
            pixels.push_back(image[index.index()+1]);
    }

    if (index.y()+1 < height)
    {
        unsigned int base_index = index.index()+width;
        if (have_left)
            pixels.push_back(image[base_index-1]);

        pixels.push_back(image[base_index]);

        if (have_right)
            pixels.push_back(image[base_index+1]);
    }
}
//---------------------------------------------------------------------------
template<typename ImageType,typename PixelType>
void get_window(const pixel_index<3>& index,const ImageType& image,unsigned int width,std::vector<PixelType>& pixels)
{
    unsigned int wh = image.width()*image.height();
    unsigned int fx = (index.x() > width) ? index.x() - width:0;
    unsigned int fy = (index.y() > width) ? index.y() - width:0;
    unsigned int fz = (index.z() > width) ? index.z() - width:0;
    unsigned int tx = std::min<size_t>(index.x() + width,image.width()-1);
    unsigned int ty = std::min<size_t>(index.y() + width,image.height()-1);
    unsigned int tz = std::min<size_t>(index.z() + width,image.depth()-1);
    unsigned int z_index = (fz*image.height()+fy)*image.width()+fx;
    for (unsigned int z = fz;z <= tz;++z,z_index += wh)
    {
        unsigned int y_index = z_index;
        for (unsigned int y = fy;y <= ty;++y,y_index += image.width())
        {
            unsigned int x_index = y_index;
            for (unsigned int x = fx;x <= tx;++x,++x_index)
                pixels.push_back(image[x_index]);
        }
    }
}
//---------------------------------------------------------------------------
template<typename ImageType,typename PixelType>
void get_window(const pixel_index<3>& index,const ImageType& image,std::vector<PixelType>& pixels)
{
    pixels.clear();
    pixels.reserve(27);
    unsigned int z_offset = image.geometry().plane_size();
    unsigned int y_offset = image.width();
    bool have_left = index.x() >= 1;
    bool have_right = index.x()+1 < image.width();
    bool has_top = index.y() >= 1;
    bool has_bottom = index.y()+1 < image.height();
    if (index.z() >= 1)
    {
        if (has_top)
        {
            if (have_left)
                pixels.push_back(image[index.index()-1-y_offset-z_offset]);

            pixels.push_back(image[index.index()  -y_offset-z_offset]);

            if (have_right)
                pixels.push_back(image[index.index()+1-y_offset-z_offset]);
        }
        {
            if (have_left)
                pixels.push_back(image[index.index()-1-z_offset]);

            pixels.push_back(image[index.index()  -z_offset]);

            if (have_right)
                pixels.push_back(image[index.index()+1-z_offset]);
        }
        if (has_bottom)
        {
            if (have_left)
                pixels.push_back(image[index.index()-1+y_offset-z_offset]);

            pixels.push_back(image[index.index()  +y_offset-z_offset]);

            if (have_right)
                pixels.push_back(image[index.index()+1+y_offset-z_offset]);
        }
    }

    {
        if (has_top)
        {
            if (have_left)
                pixels.push_back(image[index.index()-1-y_offset]);

            pixels.push_back(image[index.index()  -y_offset]);

            if (have_right)
                pixels.push_back(image[index.index()+1-y_offset]);
        }
        {
            if (have_left)
                pixels.push_back(image[index.index()-1]);

            pixels.push_back(image[index.index()]);

            if (have_right)
                pixels.push_back(image[index.index()+1]);
        }
        if (has_bottom)
        {
            if (have_left)
                pixels.push_back(image[index.index()-1+y_offset]);

            pixels.push_back(image[index.index()  +y_offset]);

            if (have_right)
                pixels.push_back(image[index.index()+1+y_offset]);
        }

    }
    if (index.z()+1 < image.depth())
    {
        if (has_top)
        {
            if (have_left)
                pixels.push_back(image[index.index()-1-y_offset+z_offset]);

            pixels.push_back(image[index.index()  -y_offset+z_offset]);

            if (have_right)
                pixels.push_back(image[index.index()+1-y_offset+z_offset]);
        }
        {
            if (have_left)
                pixels.push_back(image[index.index()-1+z_offset]);

            pixels.push_back(image[index.index()  +z_offset]);

            if (have_right)
                pixels.push_back(image[index.index()+1+z_offset]);
        }
        if (has_bottom)
        {
            if (have_left)
                pixels.push_back(image[index.index()-1+y_offset+z_offset]);

            pixels.push_back(image[index.index()  +y_offset+z_offset]);

            if (have_right)
                pixels.push_back(image[index.index()+1+y_offset+z_offset]);
        }
    }

}
}
#endif
