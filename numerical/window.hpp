//---------------------------------------------------------------------------
#ifndef WINDOW_HPP
#define WINDOW_HPP
#include <vector>
#include <algorithm>
#include "../def.hpp"
#include "../utility/pixel_index.hpp"

namespace tipl
{

template<size_t width,int dim>
struct get_window_size;

template<size_t width>
struct get_window_size<width,2>{
    static constexpr size_t value = (width+width+1)*(width+width+1);
};

template<size_t width>
struct get_window_size<width,3>{
    static constexpr size_t value = (width+width+1)*(width+width+1)*(width+width+1);
};

template<typename ImageType,typename IteratorType>
__INLINE__ void iterate_x(const ImageType& I,int64_t index,int x_upper,int x_lower,IteratorType value,int& size)
{
    auto x_index = index;
    for (int dx = 0; dx <= x_upper; ++dx, ++size, ++x_index)
        value[size] = I[x_index];
    x_index = index-1;
    for (int dx = 1; dx <= x_lower; ++dx, ++size, --x_index)
        value[size] = I[x_index];
}

template<typename ImageType,typename IteratorType>
__INLINE__ void iterate_xy(const ImageType& I,int64_t index,int x_upper,int x_lower,int y_upper,int y_lower,IteratorType value,int& size)
{
    const int64_t w = I.width();
    auto y_index = index;
    for (int dy = 0; dy <= y_upper; ++dy)
    {
        iterate_x(I, y_index, x_upper, x_lower, value, size);
        y_index += w;
    }
    y_index = index - w;
    for (int dy = 1; dy <= y_lower; ++dy)
    {
        iterate_x(I, y_index, x_upper, x_lower, value, size);
        y_index -= w;
    }
}

template<typename ImageType,typename IteratorType>
__INLINE__ void iterate_xyz(const ImageType& I,int64_t index,
                     int x_upper,int x_lower,int y_upper,int y_lower,int z_upper,int z_lower,IteratorType value,int& size)
{
    const int64_t wh = I.plane_size();
    auto z_index = index;
    for (int dz = 0; dz <= z_upper; ++dz)
    {
        iterate_xy(I, z_index, x_upper, x_lower, y_upper, y_lower, value, size);
        z_index += wh;
    }

    z_index = index - wh;
    for (int dz = 1; dz <= z_lower; ++dz)
    {
        iterate_xy(I, z_index, x_upper, x_lower, y_upper, y_lower, value, size);
        z_index -= wh;
    }
}


template<int width,typename ImageType,typename IteratorType>
__INLINE__ int get_window_at_width(const pixel_index<2>& index,const ImageType& I,IteratorType iter)
{
    int x_upper = I.width()-index.x()-1;
    int y_upper = I.height()-index.y()-1;
    int x_lower = index.x();
    int y_lower = index.y();
    if(x_upper > width)        x_upper = width;
    if(y_upper > width)        y_upper = width;
    if(x_lower > width)        x_lower = width;
    if(y_lower > width)        y_lower = width;
    int size = 0;
    iterate_xy(I, index.index(), x_upper, x_lower, y_upper, y_lower, iter, size);
    return size;
}

//---------------------------------------------------------------------------
template<int width,typename ImageType,typename IteratorType>
__INLINE__ int get_window_at_width(const pixel_index<3>& index,const ImageType& I,IteratorType iter)
{
    int x_upper = I.width()-index.x()-1;
    int y_upper = I.height()-index.y()-1;
    int z_upper = I.depth()-index.z()-1;
    int x_lower = index.x();
    int y_lower = index.y();
    int z_lower = index.z();
    if(x_upper > width)        x_upper = width;
    if(y_upper > width)        y_upper = width;
    if(z_upper > width)        z_upper = width;
    if(x_lower > width)        x_lower = width;
    if(y_lower > width)        y_lower = width;
    if(z_lower > width)        z_lower = width;
    int size = 0;
    iterate_xyz(I, index.index(), x_upper, x_lower, y_upper, y_lower, z_upper, z_lower, iter, size);
    return size;
}

//---------------------------------------------------------------------------
template<typename T>
auto get_window(const pixel_index<2>& index,const T& image,unsigned int width)
{
    std::vector<typename T::value_type> pixels;
    pixels.reserve((width * 2 + 1) * (width * 2 + 1));

    const size_t w = image.width();
    const size_t h = image.height();
    const int cx = index.x();
    const int cy = index.y();
    const int i_width = static_cast<int>(width);

    const size_t fx = static_cast<size_t>(std::max(0, cx - i_width));
    const size_t fy = static_cast<size_t>(std::max(0, cy - i_width));
    const size_t tx = static_cast<size_t>(std::min<int>(w - 1, cx + i_width));
    const size_t ty = static_cast<size_t>(std::min<int>(h - 1, cy + i_width));

    size_t y_index = fy * w + fx;
    for (size_t y = fy; y <= ty; ++y, y_index += w)
    {
        size_t x_index = y_index;
        for (size_t x = fx; x <= tx; ++x, ++x_index)
            pixels.push_back(image[x_index]);
    }
    return pixels;
}

//---------------------------------------------------------------------------
template<typename T>
auto get_window(const pixel_index<2>& index,const T& image)
{
    std::vector<typename T::value_type> pixels;
    pixels.reserve(9);

    const size_t w = image.width();
    const size_t h = image.height();
    const size_t idx = index.index();
    const int cx = index.x();
    const int cy = index.y();

    const bool have_left = cx >= 1;
    const bool have_right = cx + 1 < w;

    if (cy >= 1)
    {
        const size_t base_index = idx - w;
        if (have_left) pixels.push_back(image[base_index - 1]);
        pixels.push_back(image[base_index]);
        if (have_right) pixels.push_back(image[base_index + 1]);
    }
    {
        if (have_left) pixels.push_back(image[idx - 1]);
        pixels.push_back(image[idx]);
        if (have_right) pixels.push_back(image[idx + 1]);
    }
    if (cy + 1 < h)
    {
        const size_t base_index = idx + w;
        if (have_left) pixels.push_back(image[base_index - 1]);
        pixels.push_back(image[base_index]);
        if (have_right) pixels.push_back(image[base_index + 1]);
    }
    return pixels;
}

//---------------------------------------------------------------------------
template<typename T>
auto get_window(const pixel_index<3>& index,const T& image,unsigned int width)
{
    std::vector<typename T::value_type> pixels;
    pixels.reserve((width * 2 + 1) * (width * 2 + 1) * (width * 2 + 1));

    const size_t w = image.width();
    const size_t h = image.height();
    const size_t d = image.depth();
    const size_t wh = image.plane_size();

    const int cx = index.x();
    const int cy = index.y();
    const int cz = index.z();
    const int i_width = static_cast<int>(width);

    const size_t fx = static_cast<size_t>(std::max(0, cx - i_width));
    const size_t fy = static_cast<size_t>(std::max(0, cy - i_width));
    const size_t fz = static_cast<size_t>(std::max(0, cz - i_width));
    const size_t tx = static_cast<size_t>(std::min<int>(w - 1, cx + i_width));
    const size_t ty = static_cast<size_t>(std::min<int>(h - 1, cy + i_width));
    const size_t tz = static_cast<size_t>(std::min<int>(d - 1, cz + i_width));

    size_t z_index = fz * wh + fy * w + fx;
    for (size_t z = fz; z <= tz; ++z, z_index += wh)
    {
        size_t y_index = z_index;
        for (size_t y = fy; y <= ty; ++y, y_index += w)
        {
            size_t x_index = y_index;
            for (size_t x = fx; x <= tx; ++x, ++x_index)
                pixels.push_back(image[x_index]);
        }
    }
    return pixels;
}

//---------------------------------------------------------------------------
template<typename T>
auto get_window(const pixel_index<3>& index,const T& image)
{
    std::vector<typename T::value_type> pixels;
    pixels.reserve(27);

    const size_t w = image.width();
    const size_t h = image.height();
    const size_t d = image.depth();
    const size_t wh = image.plane_size();
    const size_t idx = index.index();

    const int cx = index.x();
    const int cy = index.y();
    const int cz = index.z();

    const bool have_left = cx >= 1;
    const bool have_right = cx + 1 < w;
    const bool has_top = cy >= 1;
    const bool has_bottom = cy + 1 < h;

    auto add_plane = [&](size_t plane_base) {
        if (has_top) {
            const size_t base = plane_base - w;
            if (have_left) pixels.push_back(image[base - 1]);
            pixels.push_back(image[base]);
            if (have_right) pixels.push_back(image[base + 1]);
        }
        {
            if (have_left) pixels.push_back(image[plane_base - 1]);
            pixels.push_back(image[plane_base]);
            if (have_right) pixels.push_back(image[plane_base + 1]);
        }
        if (has_bottom) {
            const size_t base = plane_base + w;
            if (have_left) pixels.push_back(image[base - 1]);
            pixels.push_back(image[base]);
            if (have_right) pixels.push_back(image[base + 1]);
        }
    };

    if (cz >= 1)
        add_plane(idx - wh);

    add_plane(idx);

    if (cz + 1 < d)
        add_plane(idx + wh);

    return pixels;
}

}
#endif
