#ifndef CUDA_KERNEL_HPP
#define CUDA_KERNEL_HPP
#ifdef __CUDACC__
#include "../def.hpp"

namespace tipl {


template<typename T,typename Fun>
__global__ void cuda_for_kernel(size_t size,T from,Fun f)
{
    size_t stride = blockDim.x*gridDim.x;
    for(size_t index = threadIdx.x + blockIdx.x*blockDim.x;
        index < size;index += stride)
            f(from+index);
}

template <typename T,typename Func,typename std::enable_if<
              std::is_integral<T>::value ||
              std::is_class<T>::value,bool>::type = true>
inline void cuda_for(T from,T to,Func&& f, bool sync = true,unsigned int thread_count = 256)
{
    if(to == from)
        return;
    size_t size = to-from;
    size_t grid_size = (size+thread_count-1)/thread_count;
    cuda_for_kernel<<<(grid_size > thread_count ? thread_count:grid_size),thread_count>>>(size,from,f);
    if(cudaPeekAtLastError() != cudaSuccess)
        throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
    if(sync)
        cudaDeviceSynchronize();
}

template <typename T,typename Func,typename std::enable_if<std::is_integral<T>::value,bool>::type = true>
inline void cuda_for(T size, Func&& f, bool sync = true, unsigned int thread_count = 256)
{
    if(!size)
	return;
    cuda_for(T(0),size,std::move(f),sync,thread_count);
}

template <typename T,typename Func,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
inline void cuda_for(T& c, Func&& f, bool sync = true, unsigned int thread_count = 256)
{
    cuda_for(c.begin(),c.end(),std::move(f),sync,thread_count);
}



}//namespace tipl
#endif//__CUDACC__
#endif//CUDA_KERNEL_HPP
