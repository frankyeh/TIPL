#ifndef FFT_HPP_INCLUDED
#define FFT_HPP_INCLUDED
#include "image/utility/basic_image.hpp"
#include "image/utility/geometry.hpp"
#include <cmath>
#include <vector>
#include <stdexcept>

namespace image
{
template<class image_type>
void fft_shift_x(image_type& I)
{
    int half_w = I.width() >> 1;
    int half_w_1 = half_w-1;
    int w_1 = I.width()-1;
    int quater_w = half_w >> 1;
    typename image_type::iterator iter1 = I.begin();
    typename image_type::iterator end = I.end();
    for(;iter1 != end;iter1 += I.width())
    {
        typename image_type::iterator iter2 = iter1+half_w;
        for(int x = 0,rx = half_w_1;x < quater_w;++x,--rx)
        {
            std::swap(iter1[x],iter1[rx]);
            std::swap(iter2[x],iter2[rx]);
        }
    }
}

template<class image_type>
void fft_shift_y(image_type& I)
{
    int w = I.width();
    int half_wh = I.plane_size() >> 1;
    int half_wh_1 = half_wh-w;
    int quater_wh = half_wh >> 1;
    typename image_type::iterator iter1 = I.begin();
    typename image_type::iterator end = I.end();
    for(;iter1 != end;iter1 += I.plane_size())
    {
        typename image_type::iterator iter1_x = iter1;
        typename image_type::iterator iter1_x_end = iter1+w;
        typename image_type::iterator iter2_x = iter1_x+half_wh;
        for(;iter1_x != iter1_x_end;++iter1_x,++iter2_x)
        for(int y = 0,ry = half_wh_1;y < quater_wh;y+=w,ry-=w)
        {
            std::swap(iter1_x[y],iter1_x[ry]);
            std::swap(iter2_x[y],iter2_x[ry]);
        }
    }
}

template<class image_type>
void fft_shift_z(image_type& I)
{
    int wh = I.plane_size();
    int half_size = I.size() >> 1;
    int half_size_1 = half_size-wh;
    int quater_size = half_size >> 1;
    typename image_type::iterator iter1 = I.begin();
    typename image_type::iterator iter2 = iter1+half_size;
    typename image_type::iterator end = iter1+wh;
    for(;iter1 != end;++iter1,++iter2)
    {
        for(int z = 0,rz = half_size_1;z < quater_size;z+=wh,rz-=wh)
        {
            std::swap(iter1[z],iter1[rz]);
            std::swap(iter2[z],iter2[rz]);
        }
    }
}

template<class value_type>
void fft_shift(basic_image<value_type,2>& I)
{
    fft_shift_x(I);
    fft_shift_y(I);
}

template<class value_type>
void fft_shift(basic_image<value_type,3>& I)
{
    fft_shift_x(I);
    fft_shift_y(I);
    fft_shift_z(I);
}

template<class value_type>
value_type fft_round_up_size(value_type num_)
{
    unsigned int num = num_;
    unsigned int result = 1;
    bool need_padding = false;
    while(num > 1)
    {
        need_padding |= (num & 1);
        num >>= 1;
        ++result;
    }
    return need_padding ? 1 << result : num_;
}
template<class geo_type>
geo_type fft_round_up_geometry(const geo_type& geo)
{
    geo_type geo2;
    for(int dim = 0;dim < geo_type::dimension;++dim)
        geo2[dim] = fft_round_up_size(geo[dim]);
    return geo2;
}
template<class image_type,class pos_type>
void fft_round_up(image_type& I,pos_type& from,pos_type& to)
{
    image_type newI(fft_round_up_geometry(I.geometry()));
    for(int dim = 0;dim < image_type::dimension;++dim)
    {
        from[dim] = (newI.geometry()[dim]-I.geometry()[dim]) >> 1;
        to[dim] = from[dim] + I.geometry()[dim];
    }
    image::draw(I,newI,from);
    I.swap(newI);
}
template<class image_type,class pos_type>
void fft_round_down(image_type& I,const pos_type& from,const pos_type& to)
{
    image::crop(I,from,to);
}

template<unsigned int dimension,class float_type = float>
class fftn
{
protected:
    geometry<dimension> geo;
    std::vector<std::vector<size_t> > swap_pair1,swap_pair2;
    std::vector<std::vector<float_type> > wr,wi;
    std::vector<std::vector<float_type> > iwr,iwi;
protected:
    template<class ImageType>
    void fft(ImageType& real,ImageType& img,bool invert) const
    {
        unsigned int nprev = 1;
        for(int dim = 0;dim < dimension;++dim)
        {
            typename std::vector<size_t>::const_iterator iter = swap_pair1[dim].begin();
            typename std::vector<size_t>::const_iterator end = swap_pair1[dim].end();
            typename std::vector<size_t>::const_iterator iter2 = swap_pair2[dim].begin();
            for(; iter != end; ++iter,++iter2)
            {
                std::swap(real[*iter],real[*iter2]);
                std::swap(img[*iter],img[*iter2]);
            }

            typename std::vector<float_type>::const_iterator r_iter = invert ? iwr[dim].begin() : wr[dim].begin();
            typename std::vector<float_type>::const_iterator i_iter = invert ? iwi[dim].begin() : wi[dim].begin();
            unsigned int ip2 = nprev*geo[dim];
            unsigned int ifp2;
            unsigned int ip3 = geo.size();
            typename ImageType::value_type rvalue,ivalue,real_t,img_t;
            for(unsigned int ifp1 = nprev; ifp1 < ip2; ifp1 = ifp2)
            {
                ifp2 = ifp1 << 1;
                for(unsigned int i3 = 0; i3 < ifp1; i3 += nprev)
                {
                    float_type wr = *(r_iter++);
                    float_type wi = *(i_iter++);
                    for(unsigned int i1 = i3; i1 < i3 + nprev;++i1)
                        for(unsigned int k1 = i1; k1 < ip3; k1 += ifp2)
                        {
                            unsigned int k2 = k1+ifp1;
                            rvalue = real_t = real[k2];
                            ivalue = img_t = img[k2];
                            rvalue *= wr;
                            ivalue *= wr;
                            rvalue -= img_t*wi;
                            ivalue += real_t*wi;
                            real[k2] = real[k1] - rvalue;
                            img[k2] = img[k1] - ivalue;
                            real[k1] += rvalue;
                            img[k1] += ivalue;
                        }
                }
            }
            nprev *= geo[dim];
        }
    }
public:
    // the geometry has to power 2
    fftn(const geometry<dimension>& geo_):geo(geo_),
        swap_pair1(dimension),swap_pair2(dimension),wr(dimension),wi(dimension),iwr(dimension),iwi(dimension)
    {
        unsigned int ntot = geo.size(),nprev = 1,nrem,ip1,ip2,i2rev,i3rev,ibit,ip3 = ntot << 1;
        for(int dim = 0;dim < dimension;++dim)
        {
            unsigned int dim_bandwith = geo[dim];
            nrem = ntot/(dim_bandwith* nprev);
            ip1 = nprev << 1;
            ip2 = ip1*dim_bandwith;
            i2rev = 0;
            for(unsigned int i2 = 0; i2 < ip2; i2 += ip1)
            {
                if(i2 < i2rev)
                {
                    for(unsigned int i1 = i2; i1<i2+ip1-1; i1 += 2)
                        for(unsigned int i3=i1; i3 < ip3; i3 += ip2)
                        {
                            i3rev = i2rev + i3 - i2;
                            swap_pair1[dim].push_back(i3 >> 1);
                            swap_pair2[dim].push_back(i3rev >> 1);
                        }
                }
                ibit = ip2 >> 1;
                while(ibit >= ip1 && i2rev+1 > ibit)
                {
                    i2rev -= ibit;
                    ibit >>= 1;
                }
                i2rev += ibit;
            }
            nprev *= dim_bandwith;
            for(unsigned int index = 1; index < dim_bandwith; index <<= 1)
            {
                float_type theta = 3.141592653589793238462643/((float)index);
                float_type temp = std::sin(0.5*theta);
                float_type itemp = -temp;
                float_type r=1.0,i=0.0,ir=1.0,ii=0.0;
                float_type pr = (-2.0*temp*temp),pi = std::sin(theta);
                for(unsigned int j = 0; j < index; ++j)
                {
                    wr[dim].push_back(r);
                    wi[dim].push_back(i);
                    temp = r;
                    r+=temp*pr-i*pi;
                    i+=i*pr+temp*pi;
                    iwr[dim].push_back(ir);
                    iwi[dim].push_back(ii);
                    itemp = ir;
                    ir+=itemp*pr+ii*pi;
                    ii+=ii*pr-itemp*pi;
                }
            }
        }
    }
    template<class ImageType>
    void apply(ImageType& real,ImageType& img) const
    {
        if(real.size() != geo.size())
            throw std::runtime_error("Inconsistent image size");
        img.clear();
        img.resize(geo);
        fft(real,img,false);
    }
    template<class ImageType>
    void apply_inverse(ImageType& real,ImageType& img) const
    {
        if(real.size() != geo.size() || img.size() != geo.size())
            throw std::runtime_error("Inconsistent image size");
        fft(real,img,true);
    }
    template<class ImageType,class KernelType>
    void convolve(ImageType& real,const KernelType& k)
    {
        ImageType img(geo);
        apply(real,img);
        real *= k;
        img *= k;
        apply_inverse(real,img);
    }
};

/**

    fx = 0, (1, ..., n-1),-n,(-n+1,...,-1)
    fy = .....

    1. first symmetry
    real[(1, ..., n-1)][fy][fz] = real[(-n+1,...,-1)][fy][fz];
    img[(1, ..., n-1)][fy][fz] = -img[(-n+1,...,-1)][fy][fz];

    2. second symmetry
    real[fx = 0][fy][fz] = real[fx = 0][-fy][-fz]
    img[fx = 0][fy][fz] = -img[fx = 0][-fy][-fz]

*/

template<class value_type,class float_type>
void realfftn_rotate_real_pair(value_type& real_from,value_type& real_to,
                 value_type& img_from,value_type& img_to,
                 float_type wr,float_type wi,float_type c2)
{
    value_type h1r(real_from),h1i(img_from),h2i(real_from),h2r(img_from);
    h1r += real_to;h1r *= 0.5;
    h2i -= real_to;h2i *= c2;
    h2r += img_to;h2r *= -c2;
    h1i -= img_to;h1i *= 0.5;
    value_type wiri(h2r);wiri *= wr;
    value_type wrri(h2i);wrri *= wr;
    h2i *= wi;
    h2r *= wi;
    wiri -= h2i;
    wrri += h2r;
    real_from = h1r;
    real_from += wiri;
    real_to   =  h1r;
    real_to   -= wiri;
    img_from  =  h1i;
    img_from  += wrri;
    img_to    = -h1i;
    img_to    += wrri;
}

template<class ImageType>
void realfftn_rotate_real(ImageType& real,ImageType& img,geometry<2>& geo,
                 bool invert_fft)
{
    float c2= -0.5*(invert_fft ? -1.0 : 1.0);
    float theta=(invert_fft ? -1.0 : 1.0)*(3.141592653589793238462643/geo[1]);
    float wtemp=std::sin(0.5*theta);
    float wpr = -2.0*wtemp*wtemp;
    float wpi = std::sin(theta);
    float wr = 1.0,wi = 0.0;
    int y_index_end = geo.size() >> 1;
    for(int y_index = 0;y_index <= y_index_end;y_index += geo[0])
    {
        int ry_index = geo.size()-y_index;
        {
            for(int x = 0;x < geo[0];++x)
            {
                int rx = (x ? geo[0]-x: 0);
                int from = x + y_index;
                int to = rx + ry_index;
                realfftn_rotate_real_pair(real[from],real[to],img[from],img[to],wr,wi,c2);
            }
        }
        wtemp=wr;
        wr+=wtemp*wpr-wi*wpi;
        wi+=wi*wpr+wtemp*wpi;
    }
}

template<class ImageType>
void realfftn_rotate_real(ImageType& real,ImageType& img,geometry<3>& geo,
                 bool invert_fft)
{
    float c2= -0.5*(invert_fft ? -1.0 : 1.0);
    float theta=(invert_fft ? -1.0 : 1.0)*(3.141592653589793238462643/geo[2]);
    float wtemp=std::sin(0.5*theta);
    float wpr = -2.0*wtemp*wtemp;
    float wpi = std::sin(theta);
    float wr = 1.0,wi = 0.0;
    int z_index_end = geo.size() >> 1;
    for(int z_index = 0;z_index <= z_index_end;z_index += geo.plane_size())
    {
        int rz_index = geo.size()-z_index;
        for(int y_index = 0;y_index < geo.plane_size();y_index += geo[0])
        {
            int ry_index = (y_index ? geo.plane_size()-y_index: 0);
            int yz_index = z_index + y_index;
            int ryz_index = rz_index + ry_index;
            for(int x = 0;x < geo[0];++x)
            {
                int rx = (x ? geo[0]-x: 0);
                int from = x + yz_index;
                int to = rx + ryz_index;
                realfftn_rotate_real_pair(real[from],real[to],img[from],img[to],wr,wi,c2);
            }
        }
        wtemp=wr;
        wr+=wtemp*wpr-wi*wpi;
        wi+=wi*wpr+wtemp*wpi;
    }
}

template<unsigned int dimension,class float_type = float>
class realfftn : public fftn<dimension,float_type> {

public:
    geometry<dimension> ext_geo; // the frequency geometry + fy = -n
    geometry<dimension> image_geo;
    geometry<dimension> half_size(geometry<dimension> geo_)
    {
        geo_[dimension-1] >>= 1;
        return geo_;
    }

public:
    realfftn(const geometry<dimension>& geo_):fftn<dimension,float_type>(half_size(geo_)),ext_geo(half_size(geo_)),image_geo(geo_)
    {
        ++ext_geo[dimension-1];
    }
    template<class ImageType>
    void apply(ImageType& real,ImageType& img)
    {
        if(real.geometry() != image_geo)
            throw std::runtime_error("Inconsistent image size");
        image::geometry<dimension> geo(fftn<dimension,float_type>::geo);
        img.resize(geo);
        int block_size = image_geo.size()/image_geo[dimension-1];
        // dispatch data to real and img
        typename ImageType::iterator iter = real.begin();
        typename ImageType::iterator end = real.end();
        typename ImageType::iterator real_iter = real.begin();
        typename ImageType::iterator img_iter = img.begin();
        for(;iter != end;iter += (block_size << 1),real_iter+=block_size,img_iter+=block_size)
        {
            std::copy(iter,iter+block_size,real_iter);
            std::copy(iter+block_size,iter + (block_size << 1),img_iter);
        }
        real.resize(geo);
        fft(real,img,false);

        // prepare the fy = -n data
        real.resize(ext_geo);
        img.resize(ext_geo);
        int size = ext_geo.size()-geo.size();
        std::copy(real.begin(),real.begin()+size,real.end()-size);
        std::copy(img.begin(),img.begin()+size,img.end()-size);
        realfftn_rotate_real(real,img,fftn<dimension,float_type>::geo,false);
    }
    template<class ImageType>
    void apply_inverse(ImageType& real,ImageType& img)
    {
        image::geometry<dimension> geo(fftn<dimension,float_type>::geo);
        if(real.geometry() != ext_geo || img.geometry() != ext_geo)
            throw std::runtime_error("Inconsistent image size");

        realfftn_rotate_real(real,img,fftn<dimension,float_type>::geo,true);
        fft(real,img,true);
        ImageType new_real(image_geo);

        int block_size = image_geo.size()/image_geo[dimension-1];
        typename ImageType::iterator iter = new_real.begin();
        typename ImageType::iterator end = new_real.end();
        typename ImageType::iterator real_iter = real.begin();
        typename ImageType::iterator img_iter = img.begin();
        for(;iter != end;iter += (block_size << 1),real_iter+=block_size,img_iter+=block_size)
        {
            std::copy(real_iter,real_iter+block_size,iter);
            std::copy(img_iter,img_iter+block_size,iter+block_size);
        }

        real.swap(new_real);
    }
    template<class ImageType,class KernelType>
    void convolve(ImageType& real,const KernelType& k)
    {
        ImageType img;
        apply(real,img);
        real *= k;
        img *= k;
        apply_inverse(real,img);
    }
};


}
#endif // FFT_HPP_INCLUDED
