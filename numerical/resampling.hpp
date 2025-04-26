//---------------------------------------------------------------------------
#ifndef RESAMPLING_HPP
#define RESAMPLING_HPP
#include "../utility/basic_image.hpp"
#include "transformation.hpp"
#include "numerical.hpp"
#include "basic_op.hpp"
#include "statistics.hpp"
#include "interpolation.hpp"

#ifdef __CUDACC__
#include "../cu.hpp"
#endif

namespace tipl
{

template<typename IntegerType>
class PixelAdapter
{
private:
    unsigned int r_16;
public:
    PixelAdapter(void):r_16(0) {}
    void add(IntegerType gray,unsigned int w_16)
    {
        r_16 += ((unsigned int)gray)*w_16;
    }
    void add(IntegerType gray,const unsigned int* w)
    {
        r_16 += w[gray];
    }
    IntegerType get(void) const
    {
        return r_16 >> 16;
    }
};

template<>
class PixelAdapter<rgb>
{
private:
    unsigned int r_16,g_16,b_16;
public:
    PixelAdapter<rgb>(void):r_16(0),g_16(0),b_16(0) {}
    void add(rgb color,unsigned int w_16)
    {
        r_16 += ((unsigned int)color.r)*w_16;
        g_16 += ((unsigned int)color.g)*w_16;
        b_16 += ((unsigned int)color.b)*w_16;
    }
    void add(rgb color,const unsigned int* w_16)
    {
        r_16 += w_16[color.r];
        g_16 += w_16[color.g];
        b_16 += w_16[color.b];
    }
    rgb get(void) const
    {
        return rgb(r_16 >> 16,g_16 >> 16,b_16 >> 16);
    }
};
//---------------------------------------------------------------------------
template<typename PixelType>
void thumb(const image<2,PixelType>& from,image<2,PixelType>& to)
{
    std::vector<PixelAdapter<PixelType> > to_buffer(to.width());
    enum MoveType {MoveFrom,MoveTo};

    const unsigned int value_16 = 1 << 16;
    unsigned int height_r_16 = ((double)to.height())/((double)from.height())*((double)((unsigned int)1 << 16));
    unsigned int width_r_16 = ((double)to.width())/((double)from.width())*((double)((unsigned int)1 << 16));
    unsigned int height_r_8 = height_r_16 >> 8;
    unsigned int width_r_8 = width_r_16 >> 8;
    unsigned int rr_16 = ((double)to.size())/((double)from.size())*((double)((unsigned int)1 << 16));

    unsigned int tmp_rr_16;
    bool rr_buffer;

    //std::vector<unsigned int> rbuffer_16(PixelAdapter<PixelType>::max);
    //for (unsigned int index = 0;index < PixelAdapter<PixelType>::max;++index)
    //    rbuffer_16[index] = rr_16*index;

    unsigned int next_from_y_16 = height_r_16;
    unsigned int next_to_y_16 = value_16;
    unsigned int from_y = 0;
    unsigned int to_y = 0;
    unsigned int from_y_index = 0;
    unsigned int to_y_index = 0;
    unsigned int y_16 = 0;
    MoveType y_move = MoveFrom;
    while (1)
    {
        unsigned int y_interval_8;
        if (next_from_y_16 < next_to_y_16)
        {
            y_interval_8 = (y_move == MoveFrom) ? height_r_8 : (next_from_y_16-y_16) >> 8;
            y_move = MoveFrom;
        }
        else
        {
            y_interval_8 = (y_move == MoveTo) ? (1 << 8) : (next_to_y_16-y_16) >> 8;
            y_move = MoveTo;
        }

        if (!(rr_buffer = (y_interval_8 == height_r_8)))
            tmp_rr_16 = width_r_8 * y_interval_8;
        {
            unsigned int next_from_x_16 = width_r_16;
            unsigned int next_to_x_16 = value_16;
            unsigned int from_x = 0;
            unsigned int to_x = 0;
            unsigned int x_16 = 0;
            MoveType x_move = MoveFrom;
            while (1)
            {
                unsigned int w_16;
                if (next_from_x_16 < next_to_x_16)
                {
                    if (x_move == MoveFrom)
                    {
                        if (rr_buffer)
                            to_buffer[to_x].add(from[from_y_index + from_x],rr_16);
                        else
                            to_buffer[to_x].add(from[from_y_index + from_x],tmp_rr_16);
                    }
                    else
                        to_buffer[to_x].add(from[from_y_index + from_x],
                                            ((next_from_x_16-x_16) >> 8)* (y_interval_8));
                    x_move = MoveFrom;
                    x_16 = next_from_x_16;
                    ++from_x;
                    next_from_x_16 += width_r_16;
                    if (from_x >= from.width())
                        break;
                }
                else
                {
                    w_16 = (x_move == MoveTo) ? (y_interval_8 << 8) :
                           ((next_to_x_16-x_16) >> 8) * y_interval_8;
                    x_move = MoveTo;

                    to_buffer[to_x].add(from[from_y_index + from_x],w_16);

                    x_16 = next_to_x_16;
                    ++to_x;
                    next_to_x_16 += value_16;
                    if (to_x >= to.width())
                        break;
                }
            }
        }

        if (y_move == MoveFrom)
        {
            y_16 = next_from_y_16;
            ++from_y;
            from_y_index += from.width();
            next_from_y_16 += height_r_16;
            if (from_y >= from.height())
                break;
        }
        else
        {
            y_16 = next_to_y_16;
            ++to_y;
            next_to_y_16 += value_16;
            if (to_y >= to.height())
                break;
            for (unsigned int index = 0;index < to.width();++index)
                to[to_y_index + index] = to_buffer[index].get();
            to_buffer.resize(0);
            to_buffer.resize(to.width());
            to_y_index += to.width();
        }
    }
    for (unsigned int index = 0;index < to.width();++index)
        to[to_y_index + index] = to_buffer[index].get();

}

template<typename value_type>
struct pixel_average{
    value_type operator()(value_type l,value_type r) const
    {
        return (l+r)/2;
    }
};

template<>
struct pixel_average<short>{
    short operator()(short l,short r) const
    {
        int avg = l;
        avg += r;
        return avg >> 1;
    }
};

template<>
struct pixel_average<int>{
    int operator()(int l,int r) const
    {
        return (l+r) >> 1;
    }
};

template<>
struct pixel_average<float>{
    float operator()(float l,float r) const
    {
        return (l+r)*0.5f;
    }
};
template<>
struct pixel_average<tipl::rgb>{
    tipl::rgb operator()(tipl::rgb c1,tipl::rgb c2) const
    {
        short r = c1.r;
        short g = c1.g;
        short b = c1.b;
        r += c2.r;
        g += c2.g;
        b += c2.b;
        return tipl::rgb(r >> 1,g >> 1,b >> 1);
    }
};

template<typename IteratorType,typename OutputIterator>
OutputIterator upsampling_x(IteratorType from,IteratorType to,OutputIterator out,int width)
{
    using value_type = typename std::iterator_traits<IteratorType>::value_type;
    IteratorType line_iter = to;
    out += (to-from) << 1;
    OutputIterator result = out;
    do
    {
        to -= width;
        --line_iter;
        *(--out) = *line_iter;
        *(--out) = *line_iter;
        if(line_iter != to)
        do{
            --line_iter;
            --out;
            *out = pixel_average<value_type>()(*line_iter,*(line_iter+1));
            *(--out) = *line_iter;
        }
        while(line_iter != to);

        if(to == from)
            break;
        line_iter = to;
    }
    while(1);
    return result;
}

template<typename IteratorType,typename OutputIterator>
OutputIterator upsampling_x_nearest(IteratorType from,IteratorType to,OutputIterator out,int width)
{
    IteratorType line_iter = to;
    out += (to-from) << 1;
    OutputIterator result = out;
    do
    {
        to -= width;
        --line_iter;
        *(--out) = *line_iter;
        *(--out) = *line_iter;
        if(line_iter != to)
        do{
            --line_iter;
            *(--out) = *line_iter;
            *(--out) = *line_iter;
        }
        while(line_iter != to);

        if(to == from)
            break;
        line_iter = to;
    }
    while(1);
    return result;
}


template<typename IteratorType,typename OutputIterator>
OutputIterator upsampling_y(IteratorType from,IteratorType to,OutputIterator out,int width,int height)
{
    size_t plane_size = size_t(width)*size_t(height);
    IteratorType plane_iter = to;
    IteratorType plane_end;
    out += (to-from) << 1;
    OutputIterator result = out;
    do
    {
        to -= plane_size;

        plane_end = plane_iter;
        plane_iter -= width;

        out -= width;
        std::copy(plane_iter,plane_end,out);
        out -= width;
        std::copy(plane_iter,plane_end,out);

        if(plane_iter != to)
        do{
            out -= width;
            std::copy(plane_iter,plane_end,out);

            plane_end = plane_iter;
            plane_iter -= width;

            for(int i = 0; i < width;++i)
                out[i] = pixel_average<typename std::iterator_traits<IteratorType>::value_type>()(out[i],plane_iter[i]);

            out -= width;
            std::copy(plane_iter,plane_end,out);

        }
        while(plane_iter != to);

        if(to == from)
            break;
        plane_iter = to;
    }
    while(1);
    return result;
}

template<typename IteratorType,typename OutputIterator>
OutputIterator upsampling_y_nearest(IteratorType from,IteratorType to,OutputIterator out,int width,int height)
{
    size_t plane_size = size_t(width)*size_t(height);
    IteratorType plane_iter = to;
    IteratorType plane_end;
    out += (to-from) << 1;
    OutputIterator result = out;
    do
    {
        to -= plane_size;

        plane_end = plane_iter;
        plane_iter -= width;

        out -= width;
        std::copy(plane_iter,plane_end,out);
        out -= width;
        std::copy(plane_iter,plane_end,out);

        if(plane_iter != to)
        do{
            out -= width;
            std::copy(plane_iter,plane_end,out);
            plane_end = plane_iter;
            plane_iter -= width;
            std::copy(plane_iter,plane_end,out);
            out -= width;
            std::copy(plane_iter,plane_end,out);

        }
        while(plane_iter != to);

        if(to == from)
            break;
        plane_iter = to;
    }
    while(1);
    return result;
}

template<typename IteratorType,typename OutputIterator>
IteratorType upsampling_z(IteratorType from,IteratorType to,OutputIterator out,int width,int height,int depth)
{
    return upsampling_y(from,to,out,width*height,depth);
}

template<typename IteratorType,typename OutputIterator>
IteratorType upsampling_z_nearest(IteratorType from,IteratorType to,OutputIterator out,int width,int height,int depth)
{
    return upsampling_y_nearest(from,to,out,width*height,depth);
}

template<typename ImageType1,typename ImageType2>
void upsampling(const ImageType1& in,ImageType2& out)
{
    shape<ImageType1::dimension> geo(in.shape());
    shape<ImageType1::dimension> new_geo(in.shape());
    for(int dim = 0;dim < ImageType1::dimension;++dim)
        new_geo[dim] <<= 1;
    out.resize(new_geo);
    typename ImageType2::iterator end_iter =
            upsampling_x(in.begin(),in.begin()+geo.size(),out.begin(),geo.width());
    
    size_t plane_size = new_geo[0];
    for(int dim = 1;dim < ImageType1::dimension;++dim)
    {
        end_iter = upsampling_y(out.begin(),end_iter,out.begin(),plane_size,geo[dim]);
        plane_size *= new_geo[dim];
    }

}

template<typename ImageType1,typename ImageType2>
void upsampling_nearest(const ImageType1& in,ImageType2& out)
{
    shape<ImageType1::dimension> geo(in.shape());
    shape<ImageType1::dimension> new_geo(in.shape());
    for(int dim = 0;dim < ImageType1::dimension;++dim)
        new_geo[dim] <<= 1;
    out.resize(new_geo);
    typename ImageType2::iterator end_iter =
            upsampling_x_nearest(in.begin(),in.begin()+geo.size(),out.begin(),geo.width());

    size_t plane_size = new_geo[0];
    for(int dim = 1;dim < ImageType1::dimension;++dim)
    {
        end_iter = upsampling_y_nearest(out.begin(),end_iter,out.begin(),plane_size,geo[dim]);
        plane_size *= new_geo[dim];
    }

}

template<typename ImageType>
void upsampling(ImageType& in)
{
    upsampling(in,in);
}


template<typename ImageType>
void upsampling_nearest(ImageType& in)
{
    upsampling_nearest(in,in);
}

template<typename value_type>
struct downsampling_facade{
    value_type operator()(value_type v1,value_type v2)
    {
        v1 += v2;
        v1 /= 2;
        return v1;
    }
};
template<>
struct downsampling_facade<int>{
    int operator()(int v1,int v2)
    {
        v1 += v2;
        return v1 >= 0 ? v1>> 1: v1/2;
    }
};
template<>
struct downsampling_facade<short>{
    short operator()(short v1,short v2)
    {
        int v = v1;
        v += v2;
        return v >= 0 ? (v >> 1): v/2;
    }
};
template<>
struct downsampling_facade<char>{
    char operator()(char v1,char v2)
    {
        short v = v1;
        v += v2;
        return v >= 0 ? (v >> 1): v/2;
    }
};
template<>
struct downsampling_facade<unsigned int>{
    unsigned int operator()(unsigned int v1,unsigned int v2)
    {
         return (v1+v2) >> 1;
    }
};
template<>
struct downsampling_facade<unsigned short>{
    unsigned short operator()(unsigned short v1,unsigned short v2)
    {
         return ((unsigned int)v1+(unsigned int)v2) >> 1;
    }
};
template<>
struct downsampling_facade<unsigned char>{
    unsigned char operator()(unsigned char v1,unsigned char v2)
    {
         return ((unsigned short)v1+(unsigned short)v2) >> 1;
    }
};

template<>
struct downsampling_facade<tipl::rgb>{
    tipl::rgb operator()(tipl::rgb v1,tipl::rgb v2)
    {
        unsigned short b = v1[0];
        b += v2[0];
        unsigned short g = v1[1];
        g += v2[1];
        unsigned short r = v1[2];
        r += v2[2];
        b >>= 1;
        g >>= 1;
        r >>= 1;
        return tipl::rgb(r,g,b);
    }
};

template<typename IteratorType,typename OutputIterator>
OutputIterator downsampling_x(IteratorType from,IteratorType to,OutputIterator out,int width)
{
    typedef typename std::iterator_traits<IteratorType>::value_type value_type;
    downsampling_facade<value_type> average;
    int half_width = width >> 1;
    for(;from != to;from += width)
    {
        IteratorType read_end = from + (half_width << 1);
        for(IteratorType read = from;read != read_end;++out,read += 2)
            *out = average((*read),*(read+1));
    }
    return out;
}

template<typename IteratorType,typename OutputIterator>
OutputIterator downsampling_y(IteratorType from,IteratorType to,OutputIterator out,int width,int height)
{
    typedef typename std::iterator_traits<IteratorType>::value_type value_type;
    downsampling_facade<value_type> average;
    size_t half_height = height >> 1;
    size_t width2 = width << 1;
    size_t plane_size = size_t(width)*size_t(height);
    size_t plane_size2 = size_t(half_height)*size_t(width2);
    for(;from != to;from += plane_size)
    {
        IteratorType line_end = from+plane_size2;
        for(IteratorType line = from;line != line_end;line += width)
        {
            for(IteratorType end_line = line + width;line != end_line;++line,++out)
                *out = average(*line,line[width]);
        }
    }
    return out;
}
template<typename IteratorType,typename OutputIterator>
IteratorType downsampling_z(IteratorType from,IteratorType to,OutputIterator out,int width,int height,int depth)
{
    return downsampling_y(from,to,out,width*height,depth);
}

template<typename IteratorType,typename OutputIterator>
IteratorType downsampling_z_with_padding(IteratorType from,IteratorType to,OutputIterator out,int width,int height,int depth)
{
    return downsampling_y_with_padding(from,to,out,width*height,depth);
}


template<typename ImageType1,typename ImageType2>
void downsampling(const ImageType1& in,ImageType2& out)
{
    out.resize(in.shape());
    shape<ImageType1::dimension> new_geo(in.shape());
    typename ImageType2::iterator end_iter = downsampling_x(in.begin(),in.end(),out.begin(),in.width());
    new_geo[0] >>= 1;
    size_t plane_size = new_geo[0];
    for(int dim = 1;dim < ImageType1::dimension;++dim)
    {
        end_iter = downsampling_y(out.begin(),end_iter,out.begin(),plane_size,in.shape()[dim]);
        new_geo[dim] = (in.shape()[dim] >> 1);
        plane_size *= new_geo[dim];
    }
    out.resize(new_geo);
}

template<typename T,typename U,typename V>
__INLINE__ void downsample_with_padding_imp(const pixel_index<T::dimension>& pos1,
                                            T& in,U& out,V& shift)
{
    constexpr int buf_count = (T::dimension == 3 ? 8 : 4);
    using value_type = typename T::value_type;
    pixel_index<T::dimension> pos2((v(pos1)*2).begin(),in.shape());
    char has = 0;
    if(pos2[0]+1 < in.width())
        has += 1;
    if(pos2[1]+1 < in.height())
        has += 2;
    if constexpr(T::dimension == 3)
    {
        if(pos2[2]+1 < in.depth())
            has += 4;
    }
    value_type buf[buf_count];
    typename add_type<value_type>::type out_value = buf[0] = in[pos2.index()];
    for(int i = 1 ;i < buf_count;++i)
    {
        auto h = has & i;
        out_value += (buf[i] = ((h == i) ? in[pos2.index()+shift[i]] : buf[h]));
    }
    if constexpr(std::is_integral<decltype(out_value)>::value)
        out[pos1.index()] = out_value >> (T::dimension);
    else
        out[pos1.index()] = out_value/buf_count;
}

#ifdef __CUDACC__


template<typename T,typename U,typename V>
__global__ void downsample_with_padding_cuda_kernel(T in,U out,V shift)
{
    using value_type = typename T::value_type;
    TIPL_FOR(index,out.size())
    {
        downsample_with_padding_imp(pixel_index<T::dimension>(index,out.shape()),in,out,shift);
    }
}

#endif

template<typename T,typename U>
void downsample_with_padding(const T& in,U& out)
{
    shape<T::dimension> out_shape(((v(in.shape())+1)/2).begin());
    if(out.size() < out_shape.size())
        out.resize(out_shape);

    std::vector<size_t> shift(T::dimension == 3 ? 8 : 4);
    shift[0] = 0;
    shift[1] = 1;
    shift[2] = in.width();
    shift[3] = 1+in.width();
    if constexpr(T::dimension == 3)
    {
        shift[4] = in.plane_size();
        shift[5] = in.plane_size()+1;
        shift[6] = in.plane_size()+in.width();
        shift[7] = in.plane_size()+1+in.width();
    }

    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        device_vector<size_t> shift_(shift.begin(),shift.end());
        TIPL_RUN(downsample_with_padding_cuda_kernel,out.size())
                (tipl::make_shared(in),tipl::make_shared(out),tipl::make_shared(shift_));
        #endif
    }
    else
    {
        par_for(tipl::begin_index(out_shape),tipl::end_index(out_shape),[&]
                (const auto& pos1)
        {
            downsample_with_padding_imp(pos1,in,out,shift);
        });
    }
    out.resize(out_shape);

}

template<typename ImageType>
void downsampling(ImageType& in)
{
    downsampling(in,in);
}

template<typename ImageType>
void downsample_with_padding(ImageType& in)
{
    ImageType out;
    downsample_with_padding(in,out);
    in.swap(out);
}



template<typename IteratorType,typename OutputIterator>
OutputIterator downsampling_no_average_x(IteratorType from,IteratorType to,OutputIterator out,int width)
{
    int half_width = width >> 1;
    for(;from != to;from += width)
    {
        IteratorType read_end = from + (half_width << 1);
        for(IteratorType read = from;read != read_end;++out,read += 2)
            *out = *read;
        if(width & 1) // the padding x dimension by 1
        {
            *out = *read_end;
            ++out;
        }
    }
    return out;
}

template<typename IteratorType,typename OutputIterator>
OutputIterator downsampling_no_average_y(IteratorType from,IteratorType to,OutputIterator out,int width,int height)
{
    int half_height = height >> 1;
    size_t plane_size = size_t(width)*size_t(height);
    size_t plane_size2 = (size_t(half_height) << 1 )*size_t(width);
    for(;from != to;from += plane_size)
    {
        IteratorType line_end = from+plane_size2;
        for(IteratorType line = from;line != line_end;line += width)
        {
            for(IteratorType end_line = line + width;line != end_line;++line,++out)
                *out = *line;
        }
        if(height & 1) // padding y dimension by 1
        {
            for(IteratorType plane_end = line_end + width;line_end != plane_end;++line_end,++out)
                *out = *line_end;
        }
    }
    return out;
}

template<typename ImageType1,typename ImageType2>
void downsample_no_average(const ImageType1& in,ImageType2& out)
{
    out.resize(in.shape());
    tipl::shape<ImageType1::dimension> new_geo(in.shape());
    typename ImageType2::iterator end_iter = downsampling_no_average_x(in.begin(),in.end(),out.begin(),in.width());
    new_geo[0] = ((new_geo[0]+1) >> 1);
    size_t plane_size = new_geo[0];
    for(int dim = 1;dim < ImageType1::dimension;++dim)
    {
        end_iter = downsampling_no_average_y(out.begin(),end_iter,out.begin(),plane_size,in.shape()[dim]);
        new_geo[dim] = ((in.shape()[dim]+1) >> 1);
        plane_size *= new_geo[dim];
    }
    out.resize(new_geo);
}
#ifdef __CUDACC__
template<tipl::interpolation itype,typename T1,typename T2>
__global__ void upsample_with_padding_cuda_kernel(T1 from,T2 to)
{
    TIPL_FOR(index,to.size())
    {
        tipl::estimate<itype>(from,
            tipl::vector<T1::dimension>(tipl::pixel_index<T1::dimension>(index,to.shape()))*=0.5,
            to[index]);
    }
}
#endif
template<interpolation type = linear,typename T>
void upsample_with_padding(const T& in,T& out)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(upsample_with_padding_cuda_kernel<linear>,out.size())
                (tipl::make_shared(in),tipl::make_shared(out));
        #endif
    }
    else
    {
        par_for(begin_index(out.shape()),end_index(out.shape()),[&]
                (const pixel_index<T::dimension>& pos)
        {
            vector<T::dimension> v(pos);
            v *= 0.5f;
            estimate<type>(in,v,out[pos.index()]);
        });
    }

}

template<interpolation type = linear,typename T>
void upsample_with_padding(T& in,const shape<T::dimension>& geo)
{
    T new_d(geo);
    upsample_with_padding<type>(in,new_d);
    new_d.swap(in);
}


template<typename PixelType>
void shrink(const tipl::image<3,PixelType>& image,
            tipl::image<3,PixelType>& buffer,
            unsigned int scale)
{
    size_t slice_size = image.plane_size();
    const PixelType* slice = image.begin();
    buffer.resize(tipl::shape<3>(
                      image.width()/scale,
                      image.height()/scale,
                      image.depth()/scale));
    std::fill(buffer.begin(),buffer.end(),0);
    for (unsigned int index = 0;index < buffer.depth();++index)
    {
        PixelType* buffer_iter = buffer.begin() + size_t(index)*buffer.plane_size();
        tipl::image<2,PixelType> buffer_slice(
            tipl::shape<2>(image.width() /scale,
                               image.height() /scale));
        for (unsigned int j = 0;j < scale;++j,slice += slice_size)
        {
            std::fill(buffer_slice.begin(),buffer_slice.end(),0);

            const PixelType* line = slice;
            for (unsigned int k = 0;k < buffer_slice.height();++k)
            {
                PixelType* buffer_slice_iter = buffer_slice.begin() +
                                               k*buffer_slice.width();

                for (unsigned int l = 0;l < scale;++l,line += image.width())
                {

                    const PixelType* iter = line;
                    for (unsigned int m = 0;m < buffer_slice.width();++m)
                    {
                        PixelType sum = 0;
                        for (unsigned int n = 0;n < scale;++n,++iter)
                            sum += *iter > 0 ? *iter : 0;
                        buffer_slice_iter[m] += sum / scale;
                    }
                }
            }
            for (unsigned int k = 0;k < buffer_slice.size();++k)
                buffer_iter[k] += buffer_slice[k] / scale;
        }
    }
    for (unsigned int index = 0;index < buffer.size();++index)
        buffer[index] /= scale;
}

template<typename PixelType>
void fast_resample(const tipl::image<3,PixelType>& source_image,
                   tipl::image<3,PixelType>& des_image)
{
    double dx = (double)(source_image.width()-1)/(double)(des_image.width()-1);
    double dy = (double)(source_image.height()-1)/(double)(des_image.height()-1);
    double dz = (double)(source_image.depth()-1)/(double)(des_image.depth()-1);
    double maxx = source_image.width()-1;
    double maxy = source_image.height()-1;
    double maxz = source_image.depth()-1;
    double coord[3];
    coord[2] = 0.5;
    auto wh = source_image.plane_size();
    auto w = source_image.width();
    for (unsigned int z = 0,index = 0;z < des_image.depth();++z,coord[2] += dz)
    {
        if (coord[2] > maxz)
            coord[2] = maxz;
        size_t index_z = size_t(coord[2])*wh;
        coord[1] = 0.5;
        for (unsigned int y = 0;y < des_image.height();++y,coord[1] += dy)
        {
            if (coord[1] > maxy)
                coord[1] = maxy;
            size_t index_y = index_z + size_t(coord[1])*w;
            coord[0] = 0.5;
            for (unsigned int x = 0;x < des_image.width();++x,++index,coord[0] += dx)
            {
                if (coord[0] > maxx)
                    coord[0] = maxx;
                size_t index_x = index_y + size_t(coord[0]);
                des_image[index] = source_image[index_x];
            }
        }
    }
}

template<typename PixelType>
void fast_resample(const tipl::image<2,PixelType>& source_image,
                   tipl::image<2,PixelType>& des_image)
{
    double dx = (double)(source_image.width()-1)/(double)(des_image.width()-1);
    double dy = (double)(source_image.height()-1)/(double)(des_image.height()-1);
    double maxx = source_image.width()-1;
    double maxy = source_image.height()-1;
    double coord[2];
    coord[1] = 0.5;
    unsigned int w = source_image.width();
    for (unsigned int y = 0,index = 0;y < des_image.height();++y,coord[1] += dy)
    {
        if (coord[1] > maxy)
            coord[1] = maxy;
        size_t index_y = size_t(coord[1])*w;
        coord[0] = 0.5;
        for (unsigned int x = 0;x < des_image.width();++x,++index,coord[0] += dx)
        {
            if (coord[0] > maxx)
                coord[0] = maxx;
            size_t index_x = index_y + size_t(coord[0]);
            des_image[index] = source_image[index_x];
        }
    }
}


template<typename pixel_type>
void homogenize(tipl::image<3,pixel_type>& I,tipl::image<3,pixel_type>& J,int block_size = 20)
{
    if(I.shape() != J.shape())
        return;
    double r = tipl::correlation(I.begin(),I.end(),J.begin());
    if(r < 0.80)
    {
        tipl::normalize(I,max_value(J.begin(),J.end()));
        return;
    }
    float distance_scale = 1.0/(float)block_size/(float)block_size;
    tipl::image<3,float> v_map(I.shape()),w_map(I.shape());
    for(int z = block_size;z < J.depth()-block_size;z += block_size)
        for(int y = block_size;y < J.height()-block_size;y += block_size)
            for(int x = block_size;x < J.width()-block_size;x += block_size)
            {
                std::vector<float> Iv,Jv,dis2;
                std::vector<size_t> locations;
                for_each_neighbors(tipl::pixel_index<3>(x,y,z,I.shape()),I.shape(),block_size,[&](const auto& pos)
                {
                    int dx = pos[0]-x;
                    int dy = pos[1]-y;
                    int dz = pos[2]-z;
                    dis2.push_back((dx*dx+dy*dy+dz*dz)*distance_scale);
                    Iv.push_back(I[pos.index()]);
                    Jv.push_back(J[pos.index()]);
                    locations.push_back(pos.index());
                });
                double a,b,r2;
                tipl::linear_regression(Iv.begin(),Iv.end(),Jv.begin(),a,b,r2);
                for(int i = 0; i < locations.size();++i)
                {
                    float v = Iv[i]*a+b;
                    float w = std::exp(-dis2[i]*0.5)*r2;
                    if(w == 0.0)
                        continue;
                    auto index = locations[i];
                    v_map[index] = (v_map[index]*w_map[index] + v*w)/(w_map[index]+w);
                    w_map[index] += w;
                }
            }
    tipl::upper_lower_threshold(v_map,0.0f,*std::max_element(J.begin(),J.end()));
    I = v_map;
}

template<typename T>
void match_signal(const T& VG,T& VFF)
{
    std::vector<float> x,y;
    x.reserve(VG.size());
    y.reserve(VG.size());
    for(size_t index = 0;index < VG.size();++index)
        if(VG[index] > 0 && VFF[index] > 0)
        {
            x.push_back(VFF[index]);
            y.push_back(VG[index]);
        }
    std::pair<double,double> r = tipl::linear_regression(x.begin(),x.end(),y.begin());
    for(size_t index = 0;index < VG.size();++index)
        if(VG[index] > 0 && VFF[index] > 0)
            VFF[index] = std::max<float>(0,VFF[index]*r.first+r.second);
        else
            VFF[index] = 0;
}

template<typename T>
void match_signal_kernel(const T& VG,T& VFF)
{
    typedef typename T::value_type value_type;
    value_type max_value = *std::max_element(VFF.begin(),VFF.end());

    std::vector<unsigned int> count(256);
    std::vector<float> sum(256);

    for(size_t index = 0;index < VG.size();++index)
        if(VG[index] > 0 && VFF[index] > 0)
        {
            int v = std::min<int>(255,std::round((float)VFF[index]/(float)max_value*255.499f));
            sum[v] += VG[index];
            ++count[v];
        }

    for(unsigned int index = 0;index < sum.size();++index)
    {
        if(count[index] != 0)
        {
            sum[index] /= count[index];
            continue;
        }
        if(index != 0)
            sum[index] = sum[index-1];
    }
    // smoothing
    tipl::image<1,float> value(tipl::shape<1>(256));
    value[0] = sum[0];
    value[255] = sum[255];
    for(unsigned int index = 1;index+1 < sum.size();++index)
        value[index] = (sum[index]+sum[index]+sum[index-1]+sum[index+1])*0.25;

    tipl::par_for(VG.size(),[&](size_t index)
    {
        if(VG[index] > 0 && VFF[index] > 0)
        {
            float v = (float)VFF[index]/(float)max_value*255.499f;
            float fv = std::floor(v);
            int floor_index = std::max<int>(0,fv);
            int ceil_index = std::min<int>(255,floor_index+1);
            float ceil_w = v-fv;
            float floor_w = 1.0f-ceil_w;
            VFF[index] = value[floor_index]*floor_w+value[ceil_index]*ceil_w;
        }
    else
        VFF[index] = 0;
    });
}

template<tipl::interpolation Type = linear,typename ImageType1,typename ImageType2,typename T>
void scale(const ImageType1& from,ImageType2&& to,const tipl::vector<ImageType1::dimension>& s)
{
    par_for(begin_index(to.shape()),end_index(to.shape()),
                [&](const auto& index)
    {
        vector<ImageType1::dimension> pos(index);
        multiply(pos,s);
        estimate<Type>(from,pos,to[index.index()]);
    });
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<tipl::interpolation itype,typename T1,typename T2,typename U>
__global__ void resample_cuda_kernel(T1 from,T2 to,U trans)
{
    TIPL_FOR(index,to.size())
    {
        tipl::pixel_index<3> pos(index,to.shape());
        tipl::vector<3> v;
        trans(pos,v);
        tipl::estimate<itype>(from,v,to[index]);
    }
}
#endif

template<tipl::interpolation itype = linear,
         typename T,typename U,typename V,
         typename std::enable_if<!std::is_same_v<std::decay_t<U>,shape<T::dimension> >,bool>::type = true>
void resample(const T& from,U&& to,const tipl::transformation_matrix<V,T::dimension>& trans)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN_STREAM(resample_cuda_kernel<itype>,to.size(),nullptr)
                (tipl::make_shared(from),tipl::make_shared(to),trans);
        if(cudaPeekAtLastError() != cudaSuccess)
            throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
        #endif
    }
    else
    {
        tipl::par_for(tipl::begin_index(to.shape()),tipl::end_index(to.shape()),
                    [&](const auto& index)
        {
            tipl::vector<T::dimension> pos;
            trans(index,pos);
            estimate<itype>(from,pos,to[index.index()]);
        });
    }
}

template<tipl::interpolation itype = linear,typename T,typename U,typename V,
         typename std::enable_if<!std::is_same_v<std::decay_t<U>,shape<T::dimension> >,bool>::type = true>
inline void resample(const T& from,U&& to,const V& trans)
{
    resample<itype>(from,to,tipl::transformation_matrix<typename V::value_type,T::dimension>(trans));
}

template<tipl::interpolation itype = linear,typename T,typename U>
inline auto resample(const T& from,const shape<T::dimension>& shape,const U& trans)
{
    typename T::buffer_type to(shape);
    resample<itype>(from,to,trans);
    return to;
}

template<tipl::interpolation Type = linear,typename ImageType1,typename ImageType2,typename transform_type>
void resample_dis(const ImageType1& from,ImageType2&& to,const transform_type& transform,const tipl::image<3,tipl::vector<3> >& dis)
{
    tipl::shape<ImageType1::dimension> geo(to.shape());
    for (tipl::pixel_index<ImageType1::dimension> index(geo);index < geo.size();++index)
    {
        tipl::vector<ImageType1::dimension,double> pos;
        transform(index,pos);
        pos += dis[index.index()];
        estimate<Type>(from,pos,to[index.index()]);
    }
}

template<tipl::interpolation Type = linear,typename ImageType,typename value_type>
void resample(ImageType& from,const tipl::transformation_matrix<value_type,ImageType::dimension>& transform)
{
    tipl::image<ImageType::dimension,typename ImageType::value_type> I(from.shape());
    for (tipl::pixel_index<ImageType::dimension> index(from.shape());index < from.size();++index)
    {
        tipl::vector<ImageType::dimension,value_type> pos;
        transform(index,pos);
        estimate<Type>(from,pos,I[index.index()]);
    }
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<tipl::interpolation itype,typename T1,typename T2>
__global__ void scale_cuda_kernel(T1 from,T2 to,double dx,double dy,double dz)
{
    TIPL_FOR(index,to.size())
    {
        tipl::pixel_index<3> pos(index,to.shape());
        tipl::vector<3> v(pos);
        v[0] *= dx;
        v[1] *= dy;
        v[2] *= dz;
        tipl::estimate<itype>(from,v,to[index]);
    }
}
#endif
template<tipl::interpolation itype = linear,typename T1,typename T2,typename std::enable_if<T1::dimension==3,bool>::type = true>
void scale(const T1& source_image,T2& des_image)
{
    double dx = double(source_image.width()-1)/double(des_image.width()-1);
    double dy = double(source_image.height()-1)/double(des_image.height()-1);
    double dz = double(source_image.depth()-1)/double(des_image.depth()-1);
    if constexpr(memory_location<T1>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(scale_cuda_kernel<itype>,des_image.size())
                (tipl::make_shared(source_image),tipl::make_shared(des_image),dx,dy,dz);
        #endif
    }
    else
    {
        double maxx = source_image.width()-1;
        double maxy = source_image.height()-1;
        double maxz = source_image.depth()-1;
        double coord[3]={0.0,0.0,0.0};
        for (unsigned int z = 0,index = 0;z < des_image.depth();++z,coord[2] += dz)
        {
            if (coord[2] > maxz)
                coord[2] = maxz;
            coord[1] = 0.0;
            for (unsigned int y = 0;y < des_image.height();++y,coord[1] += dy)
            {
                if (coord[1] > maxy)
                    coord[1] = maxy;
                coord[0] = 0.0;
                for (unsigned int x = 0;x < des_image.width();++x,++index,coord[0] += dx)
                {
                    if (coord[0] > maxx)
                        coord[0] = maxx;
                    tipl::estimate<itype>(source_image,coord,des_image[index]);
                }
            }
        }
    }

}

template<tipl::interpolation Type = linear,typename T1,typename T2,typename std::enable_if<T1::dimension==2,bool>::type = true>
void scale(const T1& source_image,T2& des_image)
{
    double dx = double(source_image.width()-1)/double(des_image.width()-1);
    double dy = double(source_image.height()-1)/double(des_image.height()-1);
    double maxx = source_image.width()-1;
    double maxy = source_image.height()-1;
    double coord[2] ={0.0,0.0};
    for (unsigned int y = 0,index = 0;y < des_image.height();++y,coord[1] += dy)
    {
        if (coord[1] > maxy)
            coord[1] = maxy;
        coord[0] = 0.0;
        for (unsigned int x = 0;x < des_image.width();++x,++index,coord[0] += dx)
        {
            if (coord[0] > maxx)
                coord[0] = maxx;
            tipl::estimate<Type>(source_image,coord,des_image[index]);
        }
    }
}

template<tipl::interpolation Type = linear,typename T1,typename T2,typename T3,typename std::enable_if<T1::dimension==3,bool>::type = true>
void scale(const T1& source_image,T2& des_image,const T3& ratio)
{
    des_image.resize(tipl::shape<3>(uint32_t(std::ceil(float(source_image.width())*ratio[0])),
                                    uint32_t(std::ceil(float(source_image.height())*ratio[1])),
                                    uint32_t(std::ceil(float(source_image.depth())*ratio[2]))));
    if(des_image.empty())
        return;
    tipl::transformation_matrix<double,T1::dimension> T;
    T.sr[0] = 1.0/double(ratio[0]);
    T.sr[4] = 1.0/double(ratio[1]);
    T.sr[8] = 1.0/double(ratio[2]);
    resample<Type>(source_image,des_image,T);
}


}
#endif
