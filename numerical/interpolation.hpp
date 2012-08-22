//---------------------------------------------------------------------------
#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP
#include "image/utility/basic_image.hpp"
#include "index_algorithm.hpp"

namespace image
{
//---------------------------------------------------------------------------

template<typename storage_type,typename iterator_type>
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


template<typename value_type>
struct weighting_sum
{
    template<typename data_iterator_type,typename weighting_iterator,typename output_type>
    void operator()(data_iterator_type from,data_iterator_type to,weighting_iterator w,output_type& result_)
    {
        output_type result = (*from)*(*w);
        for (++from,++w;from != to;++from,++w)
            result += (*from)*(*w);
        result_ = result;
    }
};

template<>
struct weighting_sum<unsigned char>
{
    template<typename data_iterator_type,typename weighting_iterator,typename output_type>
    void operator()(data_iterator_type from,data_iterator_type to,weighting_iterator w,output_type& result_)
    {
        float result = (*from)*(*w);
        for (++from,++w;from != to;++from,++w)
            result += (*from)*(*w);
        result_ = result;
    }
};

template<>struct weighting_sum<unsigned short>
{
    template<typename data_iterator_type,typename weighting_iterator,typename output_type>
    void operator()(data_iterator_type from,data_iterator_type to,weighting_iterator w,output_type& result_)
{
    weighting_sum<unsigned char>()(from,to,w,result_);
}
};

template<>struct weighting_sum<short>
{
    template<typename data_iterator_type,typename weighting_iterator,typename output_type>
    void operator()(data_iterator_type from,data_iterator_type to,weighting_iterator w,output_type& result_)
{
    weighting_sum<unsigned char>()(from,to,w,result_);
}
};

template<>
struct weighting_sum<rgb_color>
{
    template<typename data_iterator_type,typename weighting_iterator,typename output_type>
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
    template<typename value_type>
    void operator()(value_type& value) {}
};

struct gaussian_radial_basis_weighting
{
    double sd;
	gaussian_radial_basis_weighting(void):sd(1.0){}
    template<typename value_type>
    void operator()(value_type& value)
    {
        value_type dx = (1.0-value)/sd;
        dx *= dx;
        dx /= 2.0;
        value = std::exp(-dx);
    }
};

template<typename weighting_function,unsigned int dimension>
struct interpolation{};

template<typename weighting_function>
struct interpolation<weighting_function,2>
{
    static const unsigned int ref_count = 4;
    float ratio[ref_count];
    int dindex[ref_count];
	weighting_function weighting;

    template<typename VTorType>
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
        dindex[0] = iy*geo[0] + ix;
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
    template<typename ImageType,typename VTorType,typename PixelType>
    bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (get_location(source.geometry(),location))
        {
			weighting_sum<PixelType>()(const_reference_iterator<ImageType,int*>(source,dindex),
                const_reference_iterator<ImageType,int*>(source,dindex+ref_count),ratio,pixel);
            return true;
        }
        return false;
    }

    //---------------------------------------------------------------------------
    template<typename ImageType,typename PixelType>
    void estimate(const ImageType& source,PixelType& pixel)
    {
        weighting_sum<PixelType>()(const_reference_iterator<ImageType,int*>(source,dindex),
                const_reference_iterator<ImageType,int*>(source,dindex+ref_count),ratio,pixel);
    }
};

template<typename weighting_function>
struct interpolation<weighting_function,3>
{
    static const unsigned int ref_count = 8;
    float ratio[ref_count];
    int dindex[ref_count];
	weighting_function weighting;

    template<typename VTorType>
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
        unsigned int ix = fx;
        unsigned int iy = fy;
        unsigned int iz = fz;
        if (ix + 1 >= geo[0] || iy + 1 >= geo[1] || iz + 1 >= geo[2])
            return false;
        p[1] = y-fy;
        p[0] = x-fx;
        p[2] = z-fz;
        unsigned int wh = geo.plane_size();
        dindex[0] = iz*wh + iy*geo[0] + ix;
        dindex[1] = dindex[0] + 1;
        dindex[2] = dindex[0] + geo[0];
        dindex[3] = dindex[2] + 1;
        dindex[4] = dindex[0] + wh;
        dindex[5] = dindex[1] + wh;
        dindex[6] = dindex[2] + wh;
        dindex[7] = dindex[3] + wh;

        n[0] = 1.0-p[0];
        n[1] = 1.0-p[1];
        n[2] = 1.0-p[2];

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
    template<typename ImageType,typename VTorType,typename PixelType>
    bool estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
    {
        if (get_location(source.geometry(),location))
        {
			weighting_sum<PixelType>()(const_reference_iterator<ImageType,int*>(source,dindex),
                const_reference_iterator<ImageType,int*>(source,dindex+ref_count),ratio,pixel);
            return true;
        }
        return false;
    }

    //---------------------------------------------------------------------------
    template<typename ImageType,typename PixelType>
    void estimate(const ImageType& source,PixelType& pixel)
    {
        weighting_sum<PixelType>()(const_reference_iterator<ImageType,int*>(source,dindex),
                const_reference_iterator<ImageType,int*>(source,dindex+ref_count),ratio,pixel);
    }
};

template<typename ImageType,typename VTorType,typename PixelType>
bool linear_estimate(const ImageType& source,const VTorType& location,PixelType& pixel)
{
	return interpolation<linear_weighting,ImageType::dimension>().estimate(source,location,pixel);
}

template<typename ImageType,typename VTorType>
typename ImageType::value_type linear_estimate(const ImageType& source,const VTorType& location)
{
	typename ImageType::value_type result(0);
	interpolation<linear_weighting,ImageType::dimension>().estimate(source,location,result);
	return result;
}


}
#endif
