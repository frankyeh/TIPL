//---------------------------------------------------------------------------
#ifndef WINDOW_HPP
#define WINDOW_HPP
#include <vector>
#include "../def.hpp"
#include "../utility/pixel_index.hpp"

namespace tipl
{

template<int width,int dim>
struct get_window_size;

template<int width>
struct get_window_size<width,2>{
    static constexpr size_t value = (width+width+1)*(width+width+1);
};

template<int width>
struct get_window_size<width,3>{
    static constexpr size_t value = (width+width+1)*(width+width+1)*(width+width+1);
};

template<typename ImageType,typename IteratorType>
__INLINE__ void iterate_x(const ImageType& I,int64_t index,int x_upper,int x_lower,IteratorType& iter)
{
    auto x_index = index;
    for (int dx = 0;dx <= x_upper;++dx)
    {
        *iter = I[x_index];
        ++iter;
        ++x_index;
    }
    x_index = index-1;
    for (int dx = 1;dx <= x_lower;++dx)
    {
        *iter = I[x_index];
        ++iter;
        --x_index;
    }
}

template<typename ImageType,typename IteratorType>
__INLINE__ void iterate_xy(const ImageType& I,int64_t index,int x_upper,int x_lower,int y_upper,int y_lower,IteratorType& iter)
{
    auto y_index = index;
    for (int dy = 0;dy <= y_upper;++dy)
    {
        iterate_x(I,y_index,x_upper,x_lower,iter);
        y_index += I.width();
    }
    y_index = index-I.width();
    for (int dy = 1;dy <= y_lower;++dy)
    {
        iterate_x(I,y_index,x_upper,x_lower,iter);
        y_index -= I.width();
    }
}

template<typename ImageType,typename IteratorType>
__INLINE__ void iterate_xyz(const ImageType& I,int64_t index,
                     int x_upper,int x_lower,int y_upper,int y_lower,int z_upper,int z_lower,IteratorType& iter)
{
    auto z_index = index;
    for (int dz = 0;dz <= z_upper;++dz)
    {
        iterate_xy(I,z_index,x_upper,x_lower,y_upper,y_lower,iter);
        z_index += I.plane_size();
    }

    z_index = index-I.plane_size();
    for (int dz = 1;dz <= z_lower;++dz)
    {
        iterate_xy(I,z_index,x_upper,x_lower,y_upper,y_lower,iter);
        z_index -= I.plane_size();
    }
}


template<int width,typename ImageType,typename IteratorType>
__DEVICE_HOST__ IteratorType get_window_at_width(const pixel_index<2>& index,const ImageType& I,IteratorType iter)
{
    int x_upper = std::min<int>(I.width()-index.x()-1,width);
    int y_upper = std::min<int>(I.height()-index.y()-1,width);
    int x_lower = std::min<int>(index.x(),width);
    int y_lower = std::min<int>(index.y(),width);
    iterate_xy(I,index.index(),x_upper,x_lower,y_upper,y_lower,iter);
    return iter;
}

//---------------------------------------------------------------------------
template<int width,typename ImageType,typename IteratorType>
__DEVICE_HOST__ IteratorType get_window_at_width(const pixel_index<3>& index,const ImageType& I,IteratorType iter)
{
    unsigned int x_upper = std::min<size_t>(I.width()-index.x()-1,width);
    unsigned int y_upper = std::min<size_t>(I.height()-index.y()-1,width);
    unsigned int z_upper = std::min<size_t>(I.depth()-index.z()-1,width);
    unsigned int x_lower = std::min<size_t>(index.x(),width);
    unsigned int y_lower = std::min<size_t>(index.y(),width);
    unsigned int z_lower = std::min<size_t>(index.z(),width);
    iterate_xyz(I,index.index(),x_upper,x_lower,y_upper,y_lower,z_upper,z_lower,iter);
    return iter;
}


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
    unsigned int z_offset = image.shape().plane_size();
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
