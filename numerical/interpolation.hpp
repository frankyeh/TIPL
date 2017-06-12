//---------------------------------------------------------------------------
#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP
#include "image/utility/basic_image.hpp"
#include "index_algorithm.hpp"

namespace image
{
//---------------------------------------------------------------------------

template<typename value_type>
struct interpolator{
    typedef value_type type;
    static value_type assign(value_type v)
    {
        return v;
    }
};

template<>
struct interpolator<char>{
    typedef float type;
    static char assign(float v)
    {
        return v;
    }
};

template<>
struct interpolator<unsigned char>{
    typedef float type;
    static unsigned char assign(float v)
    {
        return std::max<float>(0.0f,v);
    }
};

template<>
struct interpolator<short>{
    typedef float type;
    static short assign(float v)
    {
        return v;
    }
};

template<>
struct interpolator<unsigned short>{
    typedef float type;
    static unsigned short assign(float v)
    {
        return std::max<float>(0.0f,v);
    }
};

template<>
struct interpolator<int>{
    typedef float type;
    static int assign(float v)
    {
        return v;
    }
};


template<>
struct interpolator<unsigned int>{
    typedef float type;
    static unsigned int assign(float v)
    {
        return std::max<float>(0.0f,v);
    }
};


template<int dim,typename vtype>
struct interpolator<image::vector<dim,vtype> >{
    typedef image::vector<dim,typename interpolator<vtype>::type> type;
    static image::vector<dim,vtype> assign(image::vector<dim,vtype> v)
    {
        return v;
    }
};


template<class storage_type,class iterator_type>
class const_reference_iterator
{
private:
    const storage_type* storage;
    iterator_type iter;
public:
    typedef typename storage_type::value_type value_type;
    const_reference_iterator(const storage_type& storage_,iterator_type iter_):storage(&storage_),iter(iter_) {}
public:
    value_type operator*(void) const
    {
        return (*storage)[*iter];
    }
    value_type operator[](unsigned int index) const
    {
        return (*storage)[iter[index]];
    }
public:
    bool operator==(const const_reference_iterator& rhs)
    {
        return iter == rhs.iter;
    }
    bool operator!=(const const_reference_iterator& rhs)
    {
        return iter != rhs.iter;
    }
    bool operator<(const const_reference_iterator& rhs)
    {
        return iter < rhs.iter;
    }
    void operator++(void)
    {
        ++iter;
    }
    void operator--(void)
    {
        --iter;
    }
};


template<class value_type>
struct weighting_sum
{
    template<class data_iterator_type,class weighting_iterator,class output_type>
    void operator()(data_iterator_type from,data_iterator_type to,weighting_iterator w,output_type& result_)
    {

        interpolator<value_type>::type result = (*from)*(*w);
        for (++from,++w;from != to;++from,++w)
            result += (*from)*(*w);
        result_ = result;
    }
};

template<>
struct weighting_sum<rgb_color>
{
    template<class data_iterator_type,class weighting_iterator,class output_type>
    void operator()(data_iterator_type from,data_iterator_type to,weighting_iterator w,output_type& result_)
    {
        float sum_r = 0.0;
        float sum_g = 0.0;
        float sum_b = 0.0;
        for (;from != to;++from,++w)
        {
            rgb_color color = *from;
            float weighting = *w;
            sum_r += ((float)color.r)*weighting;
            sum_g += ((float)color.g)*weighting;
            sum_b += ((float)color.b)*weighting;
        }
        result_ = image::rgb_color((unsigned char)sum_r,(unsigned char)sum_g,(unsigned char)sum_b);
    }

};

struct linear_weighting
{
    template<class value_type>
    void operator()(value_type&) {}
};

struct gaussian_radial_basis_weighting
{
    double sd;
	gaussian_radial_basis_weighting(void):sd(1.0){}
    template<class value_type>
    void operator()(value_type& value)
    {
        value_type dx = (1.0-value)/sd;
        dx *= dx;
        dx *= 0.5;
        value = std::exp(-dx);
    }
};

template<class weighting_function,unsigned int dimension>
struct interpolation{};

template<class weighting_function>
struct interpolation<weighting_function,2>
{
    static const unsigned int ref_count = 4;
    float ratio[ref_count];
    int dindex[ref_count];
	weighting_function weighting;

    template<class VTorType>
    bool get_location(const geometry<2>& geo,const VTorType& location)
    {
        float p[2],n[2];
        float x = location[0];
        float y = location[1];
        if (x < 0 || y < 0)
            return false;
        float fx = std::floor(x);
        float fy = std::floor(y);
        unsigned int ix = fx;
        unsigned int iy = fy;
        if (ix + 1 >= geo[0] || iy + 1>= geo[1])
            return false;
        p[1] = y-fy;
        p[0] = x-fx;
        dindex[ 0] = iy*geo[0] + ix;
        dindex[1] = dindex[0] + 1;
        dindex[2] = dindex[0] + geo[0];
        dindex[3] = dindex[2] + 1;

        n[0] = 1.0 - p[0];
        n[1] = 1.0 - p[1];

        weighting(p[0]);
        weighting(p[1]);
        weighting(n[0]);
        weighting(n[1]);

        ratio[0] = n[0]*n[1];
        ratio[1] = p[0]*n[1];
        ratio[2] = n[0]*p[1];
        ratio[3] = p[0]*p[1];
        return true;
    }

    //---------------------------------------------------------------------------
    template<class ImageType,class VTorType,class PixelType>
    bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (get_location(source.geometry(),location))
        {
            weighting_sum<typename interpolator<PixelType>::type>()(const_reference_iterator<ImageType,int*>(source,dindex),
                const_reference_iterator<ImageType,int*>(source,dindex+ref_count),ratio,pixel);
            return true;
        }
        return false;
    }

    //---------------------------------------------------------------------------
    template<class ImageType,class PixelType>
    void estimate(const ImageType& source,PixelType& pixel)
    {
        weighting_sum<typename interpolator<PixelType>::type>()(const_reference_iterator<ImageType,int*>(source,dindex),
                const_reference_iterator<ImageType,int*>(source,dindex+ref_count),ratio,pixel);
    }
};

template<class weighting_function>
struct interpolation<weighting_function,3>
{
    static const unsigned int ref_count = 8;
    float ratio[ref_count];
    int dindex[ref_count];
        weighting_function weighting;

    template<class VTorType>
    bool get_location(const geometry<3>& geo,const VTorType& location)
    {
        float p[3],n[3];
        float x = location[0];
        float y = location[1];
        float z = location[2];
        if (x < 0 || y < 0 || z < 0)
            return false;
        float fx = std::floor(x);
        float fy = std::floor(y);
        float fz = std::floor(z);
        int ix = fx;
        int iy = fy;
        int iz = fz;
        if (ix + 1 >= geo[0] || iy + 1 >= geo[1] || iz + 1 >= geo[2])
            return false;
        p[1] = y-fy;
        p[0] = x-fx;
        p[2] = z-fz;
        unsigned int wh = geo.plane_size();
        dindex[ 0] = iz*wh + iy*geo[0] + ix;
        dindex[1] = dindex[0] + 1;
        dindex[2] = dindex[0] + geo[0];
        dindex[3] = dindex[2] + 1;
        dindex[4] = dindex[0] + wh;
        dindex[5] = dindex[1] + wh;
        dindex[6] = dindex[2] + wh;
        dindex[7] = dindex[3] + wh;

        n[0] = (float)1.0-p[0];
        n[1] = (float)1.0-p[1];
        n[2] = (float)1.0-p[2];

        weighting(p[0]);
        weighting(p[1]);
        weighting(p[2]);
        weighting(n[0]);
        weighting(n[1]);
        weighting(n[2]);

        ratio[0] = n[0]*n[1];
        ratio[1] = p[0]*n[1];
        ratio[2] = n[0]*p[1];
        ratio[3] = p[0]*p[1];
        ratio[4] = ratio[0];
        ratio[5] = ratio[1];
        ratio[6] = ratio[2];
        ratio[7] = ratio[3];
        ratio[0] *= n[2];
        ratio[1] *= n[2];
        ratio[2] *= n[2];
        ratio[3] *= n[2];
        ratio[4] *= p[2];
        ratio[5] *= p[2];
        ratio[6] *= p[2];
        ratio[7] *= p[2];
        return true;
    }

    //---------------------------------------------------------------------------
    template<class ImageType,class VTorType,class PixelType>
    bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (get_location(source.geometry(),location))
        {
            weighting_sum<typename interpolator<PixelType>::type>()(const_reference_iterator<ImageType,int*>(source,dindex),
                const_reference_iterator<ImageType,int*>(source,dindex+ref_count),ratio,pixel);
            return true;
        }
        return false;
    }

    //---------------------------------------------------------------------------
    template<class ImageType,class PixelType>
    void estimate(const ImageType& source,PixelType& pixel)
    {
        weighting_sum<typename interpolator<PixelType>::type>()(const_reference_iterator<ImageType,int*>(source,dindex),
                const_reference_iterator<ImageType,int*>(source,dindex+ref_count),ratio,pixel);
    }
};


/** Interpolation on the unit interval without exact derivatives
 * p[0] sampled at floor(x)-1
 * p[1] sampled at floor(x)
 * p[2] sampled at floor(x)+1
 * p[3] sampled at floor(x)+2
 * value to estimate is at 0<x<1
 *                                      -1  2 -1  0    x^3
 *                                       3 -5  0  2    x^2
 * CINT(p,x) = [ p[0] p[1] p[2] [3] ] [ -3  4  1  0 ] [x  ]    *  0.5
 *                                       1 -1  0  0    1
 *
 */

// this scaled by 0.5
template<class iterator_type>
inline typename std::iterator_traits<iterator_type>::value_type
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
template<class iterator_type>
inline typename std::iterator_traits<iterator_type>::value_type
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
template<class iterator_type>
inline typename std::iterator_traits<iterator_type>::value_type
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
struct cubic_interpolation{};


template<>
struct cubic_interpolation<2>{

    float dx,dx2,dx3,dy,dy2,dy3;
    int dindex[16];

    template<class VTorType>
    bool get_location(const geometry<2>& geo,const VTorType& location)
    {
        float x = location[0];
        float y = location[1];
        if (x < 0 || y < 0 || x > geo[0] || y > geo[1])
            return false;
        float fx = std::floor(x);
        float fy = std::floor(y);
        int ix = x;
        int iy = y;
        dx = x-fx;
        dy = y-fy;
        dx2 = dx*dx;
        dy2 = dy*dy;
        dx3 = dx2*dx;
        dy3 = dy2*dy;
        int x_shift[4],y_shift[4];
        int max_x = geo.width()-1;
        x_shift[1] = std::min<int>(ix,max_x);
        x_shift[0] = std::max<int>(0,ix-1);
        x_shift[2] = std::min<int>(ix+1,max_x);
        x_shift[3] = std::min<int>(ix+2,max_x);

        int max_y = geo.plane_size()-geo.width();
        y_shift[1] = std::min<int>(iy*geo.width(),max_y);
        y_shift[0] = std::max<int>(0,y_shift[1]-geo.width());
        y_shift[2] = std::min<int>(y_shift[1]+geo.width(),max_y);
        y_shift[3] = std::min<int>(y_shift[1]+geo.width()+geo.width(),max_y);

        for(int x = 0,index = 0;x <= 3;++x)
            for(int y = 0;y <= 3;++y,++index)
                    dindex[index] = x_shift[x] + y_shift[y];
        return true;
    }

    template<class ImageType,class VTorType,class PixelType>
    bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (get_location(source.geometry(),location))
        {
            estimate(source,pixel);
            return true;
        }
        return false;
    }
    template<class ImageType,class PixelType>
    void estimate(const ImageType& source,PixelType& pixel)
    {
        typename interpolator<PixelType>::type p[16];
        for(unsigned int index = 0;index < 16;++index)
            p[index] = source[dindex[index]];
        pixel = interpolator<PixelType>::assign(cubic_imp(p,dx,dx2,dx3,dy,dy2,dy3)*0.25);
    }
};


template<>
struct cubic_interpolation<3>{

    float dx,dx2,dx3,dy,dy2,dy3,dz,dz2,dz3;
    int dindex[64];

    template<class VTorType>
    bool get_location(const geometry<3>& geo,const VTorType& location)
    {
        float x = location[0];
        float y = location[1];
        float z = location[2];
        if (x < 0 || y < 0 || z < 0 || x > geo[0] || y > geo[1] || z > geo[2])
            return false;
        float fx = std::floor(x);
        float fy = std::floor(y);
        float fz = std::floor(z);
        int ix = x;
        int iy = y;
        int iz = z;
        dx = x-fx;
        dy = y-fy;
        dz = z-fz;
        dx2 = dx*dx;
        dy2 = dy*dy;
        dz2 = dz*dz;
        dx3 = dx2*dx;
        dy3 = dy2*dy;
        dz3 = dz2*dz;
        int x_shift[4],y_shift[4],z_shift[4];
        int max_x = geo.width()-1;
        x_shift[1] = std::min<int>(ix,max_x);
        x_shift[0] = std::max<int>(0,ix-1);
        x_shift[2] = std::min<int>(ix+1,max_x);
        x_shift[3] = std::min<int>(ix+2,max_x);

        int max_y = geo.plane_size()-geo.width();
        y_shift[1] = std::min<int>(iy*geo.width(),max_y);
        y_shift[0] = std::max<int>(0,y_shift[1]-geo.width());
        y_shift[2] = std::min<int>(y_shift[1]+geo.width(),max_y);
        y_shift[3] = std::min<int>(y_shift[1]+geo.width()+geo.width(),max_y);

        int max_z = geo.size()-geo.plane_size();
        z_shift[1] = std::min<int>(iz*geo.plane_size(),max_z);
        z_shift[0] = std::max<int>(0,z_shift[1]-geo.plane_size());
        z_shift[2] = std::min<int>(z_shift[1]+geo.plane_size(),max_z);
        z_shift[3] = std::min<int>(z_shift[1]+geo.plane_size()+geo.plane_size(),max_z);

        for(int x = 0,index = 0;x <= 3;++x)
            for(int y = 0;y <= 3;++y)
                for(int z = 0;z <= 3;++z,++index)
                    dindex[index] = x_shift[x] + y_shift[y] + z_shift[z];
        return true;
    }

    template<class ImageType,class VTorType,class PixelType>
    bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (get_location(source.geometry(),location))
        {
            estimate(source,pixel);
            return true;
        }
        return false;
    }
    template<class ImageType,class PixelType>
    void estimate(const ImageType& source,PixelType& pixel)
    {
        typename interpolator<PixelType>::type pos[64];
        for(unsigned int index = 0;index < 64;++index)
            pos[index] = source[dindex[index]];
        pixel = interpolator<PixelType>::assign(cubic_imp(pos,dx,dx2,dx3,dy,dy2,dy3,dz,dz2,dz3)*0.125);
    }
};

enum interpolation_type {linear, cubic};


template<class ImageType,class VTorType,class PixelType>
bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel,interpolation_type type = linear)
{
    if(type == linear)
        return interpolation<linear_weighting,ImageType::dimension>().estimate(source,location,pixel);
    if(type == cubic)
        return cubic_interpolation<ImageType::dimension>().estimate(source,location,pixel);
    return false;
}

template<class ImageType,class VTorType>
typename ImageType::value_type estimate(const ImageType& source,const VTorType& location,interpolation_type type = linear)
{
    typename ImageType::value_type result(0);
    if(type == linear)
    {
        interpolation<linear_weighting,ImageType::dimension>().estimate(source,location,result);
        return result;
    }
    if(type == cubic)
    {
        cubic_interpolation<ImageType::dimension>().estimate(source,location,result);
        return result;
    }
    return result;
}



}
#endif
