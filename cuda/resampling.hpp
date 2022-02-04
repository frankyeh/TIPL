#ifndef CUDA_RESAMPLE_HPP
#define CUDA_RESAMPLE_HPP


#ifdef __CUDACC__
#include "../utility/basic_image.hpp"
#include "../numerical/interpolation.hpp"
namespace tipl{


template<typename T,typename U,typename V>
__global__ void downsample_with_padding_cuda_kernel(T in,U out,V shift)
{
    using value_type = typename T::value_type;
    size_t stride = blockDim.x*gridDim.x;
    for(size_t index = threadIdx.x + blockIdx.x*blockDim.x;
        index < out.size();index += stride)
    {
        pixel_index<3> pos1(index,out.shape());
        pixel_index<3> pos2(pos1[0]<<1,pos1[1]<<1,pos1[2]<<1,in.shape());
        char has = 0;
        if(pos2[0]+1 < in.width())
            has += 1;
        if(pos2[1]+1 < in.height())
            has += 2;
        if(pos2[2]+1 < in.depth())
            has += 4;
        value_type buf[8];
        typename sum_type<value_type>::type out_value = buf[0] = in[pos2.index()];
        for(int i = 1 ;i < 8;++i)
        {
            auto h = has & i;
            out_value += (buf[i] = ((h == i) ? in[pos2.index()+shift[i]] : buf[h]));
        }
        if constexpr(std::is_integral<value_type>::value)
            out[pos1.index()] = out_value >> 3;
        else
            out[pos1.index()] = out_value/8;
    }
}

template<typename T,typename U>
void downsample_with_padding_cuda(const T& in,U& out,bool sync = true)
{
    if constexpr(T::dimension==3)
    {
        std::vector<size_t> shift(8);
        shift[0] = 0;
        shift[1] = 1;
        shift[2] = in.width();
        shift[3] = 1+in.width();
        shift[4] = in.plane_size();
        shift[5] = in.plane_size()+1;
        shift[6] = in.plane_size()+in.width();
        shift[7] = in.plane_size()+1+in.width();

        device_vector<size_t> shift_(shift.begin(),shift.end());
        out.resize(shape<3>((in.width()+1)/2,(in.height()+1)/2,(in.depth()+1)/2));
        downsample_with_padding_cuda_kernel
                <<<std::min<int>((out.size()+255)/256,256),256>>>(
                        tipl::make_shared(in),tipl::make_shared(out),
                        tipl::make_shared(shift_));

    }
    if(sync)
        cudaDeviceSynchronize();
}


template<tipl::interpolation itype,typename T1,typename T2>
__global__ void upsample_with_padding_cuda_kernel(T1 from,T2 to)
{
    size_t stride = blockDim.x*gridDim.x;
    for(size_t index = threadIdx.x + blockIdx.x*blockDim.x;
        index < to.size();index += stride)
    {
        tipl::estimate<itype>(from,
            tipl::vector<3>(tipl::pixel_index<3>(index,to.shape()))*=0.5,
            to[index]);
    }
}


template<typename image_type>
void upsample_with_padding_cuda(const image_type& in,image_type& out,bool sync = true)
{
    upsample_with_padding_cuda_kernel<linear>
            <<<std::min<int>((out.size()+255)/256,256),256>>>(
                tipl::make_shared(in),tipl::make_shared(out));
    if(sync)
        cudaDeviceSynchronize();
}

template<typename image_type>
void upsample_with_padding_cuda(image_type& in,const shape<image_type::dimension>& geo,bool sync = true)
{
    image_type new_d(geo);
    upsample_with_padding_cuda(in,new_d,sync);
    new_d.swap(in);
}



template<tipl::interpolation itype,typename T1,typename T2,typename U>
__global__ void resample_cuda_kernel(T1 from,T2 to,U trans)
{
    size_t stride = blockDim.x*gridDim.x;
    for(size_t index = threadIdx.x + blockIdx.x*blockDim.x;
        index < to.size();index += stride)
    {
        tipl::pixel_index<3> pos(index,to.shape());
        tipl::vector<3> v;
        trans(pos,v);
        tipl::estimate<itype>(from,v,to[index]);
    }
}


template<tipl::interpolation itype = linear,typename T,typename T2,typename U>
inline void resample_cuda(const T& from,T2& to,const U& trans,bool sync = true,cudaStream_t stream = nullptr)
{
    if(stream)
        resample_cuda_kernel<itype>
                <<<std::min<int>((to.size()+255)/256,256),256,0,stream>>>(
                    tipl::make_shared(from),
                    tipl::make_shared(to),trans);
    else
    resample_cuda_kernel<itype>
            <<<std::min<int>((to.size()+255)/256,256),256>>>(
                tipl::make_shared(from),
                tipl::make_shared(to),trans);
    if(cudaPeekAtLastError() != cudaSuccess)
        throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
    if(sync)
        cudaDeviceSynchronize();
}

}

#endif//__CUDACC__

#endif//CUDA_RESAMPLE_HPP
