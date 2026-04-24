//---------------------------------------------------------------------------
#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <algorithm>
#include "../def.hpp"
#include "../utility/pixel_index.hpp"
#include "../utility/basic_image.hpp"
#include "../utility/rgb_image.hpp"
namespace tipl
{

template<typename value_type>   struct add_type{using type = value_type;};
template<> struct add_type<char>            {using type = int16_t;};
template<> struct add_type<unsigned char>   {using type = uint16_t;};
template<> struct add_type<short>           {using type = int32_t;};
template<> struct add_type<unsigned short>  {using type = uint32_t;};
template<> struct add_type<int>             {using type = int64_t;};
template<> struct add_type<unsigned int>    {using type = uint64_t;};
template<> struct add_type<rgb>             {using type = vector<3,uint16_t>;};


template<typename T>
struct interp_type{
    using type = std::conditional_t<std::is_integral_v<T>, float, T>;
    __INLINE__ static void assign(T& result,type v)
    {
        if constexpr(std::is_signed<T>::value)
            result = T(v);
        else
            result = (v <= 0.0f ? T(0):T(v));
    }
};

template<>
struct interp_type<rgb>{
    using type = tipl::vector<3,float>;
    __INLINE__ static void assign(rgb& result,const type& v)
    {
        result.r = (v[0] <= 0.0f ? uint8_t(0) : uint8_t(v[0]));
        result.g = (v[1] <= 0.0f ? uint8_t(0) : uint8_t(v[1]));
        result.b = (v[2] <= 0.0f ? uint8_t(0) : uint8_t(v[2]));
    }
};


template<int dim,typename vtype>
struct interp_type<tipl::vector<dim,vtype> >{
    using type = tipl::vector<dim, typename interp_type<vtype>::type>;
    __INLINE__ static void assign(tipl::vector<dim,vtype>& result,const type& v)
    {
        result = v;
    }
};

//---------------------------------------------------------------------------
namespace interpolator{

template<typename image_type,typename data_iterator_type,typename weighting_iterator,typename output_type>
__INLINE__ void weighted_sum(const image_type& I,data_iterator_type from,data_iterator_type to,weighting_iterator w,output_type& result_)
{
    using value_type = typename interp_type<output_type>::type;
    value_type result(I[*from]*(*w));
    for (++from,++w;from != to;++from,++w)
    {
        value_type v(I[*from]);
        v *= (*w);
        result += v;
    }
    interp_type<output_type>::assign(result_,result);
}

template<typename image_type, typename data_iterator_type, typename weighting_iterator, typename output_type>
__INLINE__ void weight_majority(const image_type& I, data_iterator_type from, data_iterator_type to, weighting_iterator w, output_type& result_)
{
    using val_type = typename interp_type<output_type>::type;
    using weight_type = typename std::remove_reference<decltype(*w)>::type;

    // Stack array eliminates massive heap allocations inside the interpolation loop.
    // Max reference count is 8 (for 3D linear).
    val_type unique_values[8];
    weight_type weights[8];
    int count = 0;

    for (; from != to; ++from, ++w)
    {
        auto val = I[*from];
        int i = 0;
        for (; i < count; ++i) {
            if (unique_values[i] == val) {
                weights[i] += *w;
                break;
            }
        }
        if (i == count && count < 8) {
            unique_values[count] = val;
            weights[count] = *w;
            count++;
        }
    }

    int best_idx = 0;
    for (int i = 1; i < count; ++i) {
        if (weights[i] > weights[best_idx])
            best_idx = i;
    }

    interp_type<output_type>::assign(result_, unique_values[best_idx]);
}


template<unsigned int dimension>
struct nearest{};

template<>
struct nearest<1>
{
    int64_t x;
    template<typename VTorType>
    __INLINE__ bool get_location(const shape<1>& geo,const VTorType& location)
    {
        x = std::round(location);
        if (x < 0 || x >= geo[0])
            return false;
        return true;
    }

    //---------------------------------------------------------------------------
    template<typename ImageType,typename VTorType,typename PixelType>
    __INLINE__ bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (get_location(source.shape(),location))
        {
            pixel = source[x];
            return true;
        }
        return false;
    }

    //---------------------------------------------------------------------------
    template<typename ImageType,typename PixelType>
    __INLINE__ void estimate(const ImageType& source,PixelType& pixel)
    {
        pixel = source[x];
    }
};


template<>
struct nearest<2>
{
    int64_t x,y;
    size_t index = 0;
    template<typename VTorType>
    __INLINE__ bool get_location(const shape<2>& geo,const VTorType& location)
    {
        x = std::round(location[0]);
        y = std::round(location[1]);
        if (x < 0 || y < 0 || x >= geo[0] || y >= geo[1])
            return false;
        index = x+y*geo[0];
        return true;
    }

    //---------------------------------------------------------------------------
    template<typename ImageType,typename VTorType,typename PixelType>
    __INLINE__ bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (get_location(source.shape(),location))
        {
            pixel = source[index];
            return true;
        }
        return false;
    }

    //---------------------------------------------------------------------------
    template<typename ImageType,typename PixelType>
    __INLINE__ void estimate(const ImageType& source,PixelType& pixel)
    {
        pixel = source[index];
    }
};


template<>
struct nearest<3>
{
    int64_t x,y,z;
    size_t index = 0;
    template<typename VTorType>
    __INLINE__ bool get_location(const shape<3>& geo,const VTorType& location)
    {
        x = std::round(location[0]);
        y = std::round(location[1]);
        z = std::round(location[2]);
        if (x < 0 || y < 0 || z < 0 || x >= geo[0] || y >= geo[1] || z >= geo[2])
            return false;
        index = x+(z*geo[1]+y)*geo[0];
        return true;
    }

    //---------------------------------------------------------------------------
    template<typename ImageType,typename VTorType,typename PixelType>
    __INLINE__ bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (get_location(source.shape(),location))
        {
            pixel = source[index];
            return true;
        }
        return false;
    }

    //---------------------------------------------------------------------------
    template<typename ImageType,typename PixelType>
    __INLINE__ void estimate(const ImageType& source,PixelType& pixel)
    {
        pixel = source[index];
    }
};

template<int dimension>
struct linear{};

template<>
struct linear<1>
{
    static const unsigned int ref_count = 2;
    float ratio[ref_count];
    size_t dindex[ref_count];

    template<typename VTorType>
    __INLINE__ bool get_location(const shape<1>& geo,const VTorType& location)
    {
        float x = location;
        if (x < 0)
            return false;
        size_t ix = size_t(x);
        if (ix + 1 >= geo[0])
            return false;
        float p = x - float(ix);
        dindex[0] = ix;
        dindex[1] = dindex[0] + 1;

        ratio[0] = 1.0f - p;
        ratio[1] = p;
        return true;
    }

    //---------------------------------------------------------------------------
    template<typename ImageType,typename VTorType,typename PixelType>
    __INLINE__ bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (get_location(source.shape(),location))
        {
            weighted_sum(source,dindex,dindex+ref_count,ratio,pixel);
            return true;
        }
        return false;
    }

    //---------------------------------------------------------------------------
    template<typename ImageType,typename PixelType>
    __INLINE__ void estimate(const ImageType& source,PixelType& pixel)
    {
        weighted_sum(source,dindex,dindex+ref_count,ratio,pixel);
    }
};

template<>
struct linear<2>
{
    static const unsigned int ref_count = 4;
    float ratio[ref_count];
    size_t dindex[ref_count];

    template<typename VTorType>
    __INLINE__ bool get_location(const shape<2>& geo,const VTorType& location)
    {
        float x = location[0];
        float y = location[1];
        if (x < 0 || y < 0)
            return false;
        size_t ix = size_t(x);
        size_t iy = size_t(y);
        size_t w = geo[0];
        if (ix + 1 >= w || iy + 1 >= geo[1])
            return false;

        dindex[0] = iy * w + ix;
        dindex[1] = dindex[0] + 1;
        dindex[2] = dindex[0] + w;
        dindex[3] = dindex[2] + 1;

        float p0 = x - float(ix);
        float p1 = y - float(iy);
        float n0 = 1.0f - p0;
        float n1 = 1.0f - p1;
        ratio[0] = n0*n1;
        ratio[1] = p0*n1;
        ratio[2] = n0*p1;
        ratio[3] = p0*p1;
        return true;
    }

    //---------------------------------------------------------------------------
    template<typename ImageType,typename VTorType,typename PixelType>
    __INLINE__ bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (get_location(source.shape(),location))
        {
            weighted_sum(source,dindex,dindex+ref_count,ratio,pixel);
            return true;
        }
        return false;
    }

    //---------------------------------------------------------------------------
    template<typename ImageType,typename PixelType>
    __INLINE__ void estimate(const ImageType& source,PixelType& pixel)
    {
        weighted_sum(source,dindex,dindex+ref_count,ratio,pixel);

    }
};

template<>
struct linear<3>
{
    static const unsigned int ref_count = 8;
    float ratio[ref_count];
    size_t dindex[ref_count];

    template<typename VTorType>
    __INLINE__ bool get_location(const shape<3>& geo,const VTorType& location)
    {
        float x = location[0];
        float y = location[1];
        float z = location[2];
        if (x < 0 || y < 0 || z < 0)
            return false;
        size_t ix = size_t(x);
        size_t iy = size_t(y);
        size_t iz = size_t(z);
        size_t w = geo[0];
        if (ix + 1 >= w || iy + 1 >= geo[1] || iz + 1 >= geo[2])
            return false;

        float p0 = x - float(ix);
        float p1 = y - float(iy);
        float p2 = z - float(iz);
        float n0 = 1.0f - p0;
        float n1 = 1.0f - p1;
        float n2 = 1.0f - p2;

        size_t wh = geo.plane_size();
        dindex[0] = iz * wh + iy * w + ix;
        dindex[1] = dindex[0] + 1;
        dindex[2] = dindex[0] + w;
        dindex[3] = dindex[2] + 1;
        dindex[4] = dindex[0] + wh;
        dindex[5] = dindex[1] + wh;
        dindex[6] = dindex[2] + wh;
        dindex[7] = dindex[3] + wh;

        float n0n1 = n0 * n1;
        float p0n1 = p0 * n1;
        float n0p1 = n0 * p1;
        float p0p1 = p0 * p1;

        ratio[0] = n0n1 * n2;
        ratio[1] = p0n1 * n2;
        ratio[2] = n0p1 * n2;
        ratio[3] = p0p1 * n2;
        ratio[4] = n0n1 * p2;
        ratio[5] = p0n1 * p2;
        ratio[6] = n0p1 * p2;
        ratio[7] = p0p1 * p2;
        return true;
    }

    //---------------------------------------------------------------------------
    template<typename ImageType,typename VTorType,typename PixelType>
    __INLINE__ bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (get_location(source.shape(),location))
        {
            weighted_sum(source,dindex,dindex+ref_count,ratio,pixel);
            return true;
        }
        return false;
    }
    //---------------------------------------------------------------------------
    template<typename ImageType,typename PixelType>
    __INLINE__ void estimate(const ImageType& source,PixelType& pixel)
    {
        weighted_sum(source,dindex,dindex+ref_count,ratio,pixel);
    }
};


template<int dimension>
struct majority : public linear<dimension>
{
    //---------------------------------------------------------------------------
    template<typename ImageType,typename VTorType,typename PixelType>
    __INLINE__ bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (linear<dimension>::get_location(source.shape(),location))
        {
            weight_majority(source,linear<dimension>::dindex,linear<dimension>::dindex+linear<dimension>::ref_count,linear<dimension>::ratio,pixel);
            return true;
        }
        return false;
    }

    //---------------------------------------------------------------------------
    template<typename ImageType,typename PixelType>
    __INLINE__ void estimate(const ImageType& source,PixelType& pixel)
    {
        weight_majority(source,linear<dimension>::dindex,linear<dimension>::dindex+linear<dimension>::ref_count,linear<dimension>::ratio,pixel);
    }
};


/** Interpolation on the unit interval without exact derivatives
 * p[0] sampled at floor(x)-1
 * p[1] sampled at floor(x)
 * p[2] sampled at floor(x)+1
 * p[3] sampled at floor(x)+2
 * value to estimate is at 0<x<1
 * -1  2 -1  0    x^3
 * 3 -5  0  2    x^2
 * CINT(p,x) = [ p[0] p[1] p[2] [3] ] [ -3  4  1  0 ] [x  ]    * 0.5
 * 1 -1  0  0    1
 *
 */

// this scaled by 0.5
template<typename iterator_type>
__INLINE__ typename std::iterator_traits<iterator_type>::value_type
    cubic_imp(iterator_type p, float x,float x2,float x3)
{
    typedef typename std::iterator_traits<iterator_type>::value_type value_type;
    value_type p1_p2 = (p[1] - p[2]);
    value_type p1_p2_2 = p1_p2 + p1_p2;
    value_type p1_p2_3 = p1_p2_2 + p1_p2;
    value_type p1_p2_4 = p1_p2_2 + p1_p2_2;
    value_type _p0_p3 = p[3] - p[0];
    return (p[1]+p[1]) +((_p0_p3  + p1_p2_3)*x3+(p[0] - p[1]- p1_p2_4 - _p0_p3)*x2+(-p[0]+p[2])*x);
}

// this scaled by 0.25
template<typename iterator_type>
__INLINE__ typename std::iterator_traits<iterator_type>::value_type
    cubic_imp(iterator_type p, float x, float x2,float x3,float y, float y2,float y3)
{
    typedef typename std::iterator_traits<iterator_type>::value_type value_type;
    value_type arr[4];
    arr[0] = cubic_imp(p, y, y2, y3);
    arr[1] = cubic_imp(p+4, y, y2, y3);
    arr[2] = cubic_imp(p+8, y, y2, y3);
    arr[3] = cubic_imp(p+12, y, y2, y3);
    return cubic_imp(arr, x, x2, x3);
}

// this scaled by 0.125
template<typename iterator_type>
__INLINE__ typename std::iterator_traits<iterator_type>::value_type
    cubic_imp(iterator_type p,float x, float x2,float x3,
                                             float y, float y2,float y3,
                                             float z, float z2,float z3)
{
    typedef typename std::iterator_traits<iterator_type>::value_type value_type;
    value_type arr[4];
    arr[0] = cubic_imp(p, y, y2, y3, z, z2, z3);
    arr[1] = cubic_imp(p+16, y, y2, y3, z, z2, z3);
    arr[2] = cubic_imp(p+32, y, y2, y3, z, z2, z3);
    arr[3] = cubic_imp(p+48, y, y2, y3, z, z2, z3);
    return cubic_imp(arr, x, x2, x3);
}

template<unsigned int dimension>
struct cubic{};

template<>
struct cubic<1>{

    float dx,dx2,dx3;
    int64_t dindex[4];

    template<typename VTorType>
    __INLINE__ bool get_location(const shape<1>& geo,const VTorType& location)
    {
        float x = location;
        if (x < 0 || x > geo[0])
            return false;
        int64_t ix = x;
        dx = x - float(ix);
        dx2 = dx*dx;
        dx3 = dx2*dx;
        int64_t x_shift[4];
        int64_t max_x = geo[0] - 1;
        x_shift[1] = std::min<int64_t>(ix,max_x);
        x_shift[0] = std::max<int64_t>(0,ix-1);
        x_shift[2] = std::min<int64_t>(ix+1,max_x);
        x_shift[3] = std::min<int64_t>(ix+2,max_x);
        for(int i = 0; i <= 3; ++i)
            dindex[i] = x_shift[i];
        return true;
    }

    template<typename ImageType,typename VTorType,typename PixelType>
    __INLINE__ bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (get_location(source.shape(),location))
        {
            estimate(source,pixel);
            return true;
        }
        return false;
    }
    template<typename ImageType,typename PixelType>
    __INLINE__ void estimate(const ImageType& source,PixelType& pixel)
    {
        typename interp_type<PixelType>::type p[4];
        for(unsigned int index = 0;index < 4;++index)
            p[index] = source[dindex[index]];
        interp_type<PixelType>::assign(pixel,cubic_imp(p,dx,dx2,dx3)*0.5);
    }
    template<typename ImageType>
    __INLINE__ void estimate(const ImageType& source,tipl::rgb& pixel)
    {
        for(char i = 0;i < 3;++i)
        {
            float p[4];
            for(unsigned int index = 0;index < 4;++index)
                p[index] = source[dindex[index]][i];
            pixel[i] = cubic_imp(p,dx,dx2,dx3)*0.5;
        }
    }
};


template<>
struct cubic<2>{

    float dx,dx2,dx3,dy,dy2,dy3;
    int64_t dindex[16];

    template<typename VTorType>
    __INLINE__ bool get_location(const shape<2>& geo,const VTorType& location)
    {
        float x = location[0];
        float y = location[1];
        if (x < 0 || y < 0 || x > geo[0] || y > geo[1])
            return false;
        int64_t ix = x;
        int64_t iy = y;
        dx = x - float(ix);
        dy = y - float(iy);
        dx2 = dx*dx;
        dy2 = dy*dy;
        dx3 = dx2*dx;
        dy3 = dy2*dy;

        int64_t w = geo[0];
        int64_t wh = geo.plane_size();
        int64_t max_x = w - 1;
        int64_t max_y = wh - w;

        int64_t x_shift[4],y_shift[4];
        x_shift[1] = std::min<int64_t>(ix,max_x);
        x_shift[0] = std::max<int64_t>(0,ix-1);
        x_shift[2] = std::min<int64_t>(ix+1,max_x);
        x_shift[3] = std::min<int64_t>(ix+2,max_x);

        int64_t y_base = iy * w;
        y_shift[1] = std::min<int64_t>(y_base,max_y);
        y_shift[0] = std::max<int64_t>(0,y_shift[1]-w);
        y_shift[2] = std::min<int64_t>(y_shift[1]+w,max_y);
        y_shift[3] = std::min<int64_t>(y_shift[1]+(w << 1),max_y);

        for(int i = 0, index = 0; i <= 3; ++i)
            for(int j = 0; j <= 3; ++j, ++index)
                    dindex[index] = x_shift[i] + y_shift[j];
        return true;
    }

    template<typename ImageType,typename VTorType,typename PixelType>
    __INLINE__ bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (get_location(source.shape(),location))
        {
            estimate(source,pixel);
            return true;
        }
        return false;
    }
    template<typename ImageType,typename PixelType>
    __INLINE__ void estimate(const ImageType& source,PixelType& pixel)
    {
        typename interp_type<PixelType>::type p[16];
        for(unsigned int index = 0;index < 16;++index)
            p[index] = source[dindex[index]];
        interp_type<PixelType>::assign(pixel,cubic_imp(p,dx,dx2,dx3,dy,dy2,dy3)*0.25);
    }
    template<typename ImageType>
    __INLINE__ void estimate(const ImageType& source,tipl::rgb& pixel)
    {
        for(char i = 0;i < 3;++i)
        {
            float p[16];
            for(unsigned int index = 0;index < 16;++index)
                p[index] = source[dindex[index]][i];
            pixel[i] = cubic_imp(p,dx,dx2,dx3,dy,dy2,dy3)*0.25;
        }
    }
};


template<>
struct cubic<3>{

    float dx,dx2,dx3,dy,dy2,dy3,dz,dz2,dz3;
    int64_t dindex[64];

    template<typename VTorType>
    __INLINE__ bool get_location(const shape<3>& geo,const VTorType& location)
    {
        float x = location[0];
        float y = location[1];
        float z = location[2];
        if (x < 0 || y < 0 || z < 0 || x > geo[0] || y > geo[1] || z > geo[2])
            return false;

        int64_t ix = x;
        int64_t iy = y;
        int64_t iz = z;
        dx = x - float(ix);
        dy = y - float(iy);
        dz = z - float(iz);
        dx2 = dx*dx;
        dy2 = dy*dy;
        dz2 = dz*dz;
        dx3 = dx2*dx;
        dy3 = dy2*dy;
        dz3 = dz2*dz;

        int64_t w = geo[0];
        int64_t wh = geo.plane_size();
        int64_t sz = geo.size();

        int64_t max_x = w - 1;
        int64_t max_y = wh - w;
        int64_t max_z = sz - wh;

        int64_t x_shift[4],y_shift[4],z_shift[4];
        x_shift[1] = std::min<int64_t>(ix,max_x);
        x_shift[0] = std::max<int64_t>(0,ix-1);
        x_shift[2] = std::min<int64_t>(ix+1,max_x);
        x_shift[3] = std::min<int64_t>(ix+2,max_x);

        int64_t y_base = iy * w;
        y_shift[1] = std::min<int64_t>(y_base,max_y);
        y_shift[0] = std::max<int64_t>(0,y_shift[1]-w);
        y_shift[2] = std::min<int64_t>(y_shift[1]+w,max_y);
        y_shift[3] = std::min<int64_t>(y_shift[1]+(w << 1),max_y);

        int64_t z_base = iz * wh;
        z_shift[1] = std::min<int64_t>(z_base,max_z);
        z_shift[0] = std::max<int64_t>(0,z_shift[1]-wh);
        z_shift[2] = std::min<int64_t>(z_shift[1]+wh,max_z);
        z_shift[3] = std::min<int64_t>(z_shift[1]+(wh << 1),max_z);

        for(int i = 0, index = 0; i <= 3; ++i)
            for(int j = 0; j <= 3; ++j)
                for(int k = 0; k <= 3; ++k, ++index)
                    dindex[index] = x_shift[i] + y_shift[j] + z_shift[k];
        return true;
    }

    template<typename ImageType,typename VTorType,typename PixelType>
    __INLINE__ bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (get_location(source.shape(),location))
        {
            estimate(source,pixel);
            return true;
        }
        return false;
    }
    template<typename ImageType,typename PixelType>
    __INLINE__ void estimate(const ImageType& source,PixelType& pixel)
    {
        typename interp_type<PixelType>::type pos[64];
        for(unsigned int index = 0;index < 64;++index)
            pos[index] = source[dindex[index]];
        interp_type<PixelType>::assign(pixel,cubic_imp(pos,dx,dx2,dx3,dy,dy2,dy3,dz,dz2,dz3)*0.125);
    }
};



}//interpolation

enum interpolation{nearest,majority,linear,cubic,check};

template<interpolation type = linear,typename ImageType,typename VTorType,typename PixelType>
__INLINE__ bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
{
    if constexpr(type == nearest)
        return tipl::interpolator::nearest<ImageType::dimension>().estimate(source,location,pixel);
    if constexpr(type == majority)
        return tipl::interpolator::majority<ImageType::dimension>().estimate(source,location,pixel);
    if constexpr(type == linear)
        return tipl::interpolator::linear<ImageType::dimension>().estimate(source,location,pixel);
    if constexpr(type == cubic)
        return tipl::interpolator::cubic<ImageType::dimension>().estimate(source,location,pixel);
    return false;
}

template<interpolation type = linear,typename ImageType,typename VTorType>
__INLINE__ auto estimate(const ImageType& source,const VTorType& location)
{
    auto result = typename ImageType::value_type();
    if constexpr(type == nearest)
        tipl::interpolator::nearest<ImageType::dimension>().estimate(source,location,result);
    if constexpr(type == majority)
        tipl::interpolator::majority<ImageType::dimension>().estimate(source,location,result);
    if constexpr(type == linear)
        tipl::interpolator::linear<ImageType::dimension>().estimate(source,location,result);
    if constexpr(type == cubic)
        tipl::interpolator::cubic<ImageType::dimension>().estimate(source,location,result);
    return result;
}

template <int dim,typename value_type,template <typename...> typename stype>
template <typename pos_type, typename>
__INLINE__ value_type image<dim,value_type,stype>::operator[](const pos_type& pos) const
{
    return tipl::estimate(*this, pos);
}


}//tipl
#endif
