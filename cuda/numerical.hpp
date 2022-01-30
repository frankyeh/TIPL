#ifndef CUDA_NUMERICAL_HPP
#define CUDA_NUMERICAL_HPP

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include "mem.hpp"

namespace tipl{

template<typename T>
std::pair<typename T::value_type,typename T::value_type>
minmax_value_cuda(const T& data)
{
    if(data.empty())
        return std::make_pair(0,0);
    auto result = thrust::minmax_element(data.begin_thrust(),data.end_thrust());
    return std::make_pair(*result.first,*result.second);
}

template<typename T,typename U>
void normalize_upper_lower_cuda(const T& in,U& out,float upper_limit = 255.0f)
{
    using value_type = typename U::value_type;
    std::pair<value_type,value_type> min_max(minmax_value_cuda(in));
    value_type range = min_max.second-min_max.first;
    if(range == 0)
        return;
    using namespace thrust::placeholders;
    upper_limit /= range;
    thrust::transform(in.begin_thrust(),in.end_thrust(),out.begin_thrust(), (_1 -min_max.first)*upper_limit);
}

template<typename T>
__INLINE__ typename T::value_type sum_cuda(const T& data,typename T::value_type init = 0)
{
    return thrust::reduce(data.begin_thrust(),data.end_thrust(),init);
}


template<typename T,typename U>
__global__ void add_constant_cuda_kernel(T I,U v)
{
    size_t stride = blockDim.x*gridDim.x;
    for(size_t index = threadIdx.x + blockIdx.x*blockDim.x;index < I.size();index += stride)
        I[index] += v;
}

template<typename T,typename U>
void add_constant_cuda(T& I,U v)
{
    add_constant_cuda_kernel<<<std::min<int>((I.size()+255)/256,256),256>>>
        (tipl::make_shared(I),v);
}

template<typename T1,typename T2,typename U>
__global__ void add_constant_cuda_kernel(T1 I,T2 out,U v)
{
    size_t stride = blockDim.x*gridDim.x;
    for(size_t index = threadIdx.x + blockIdx.x*blockDim.x;
        index < I.size();index += stride)
        out[index] = I[index] + v;
}


template<typename T1,typename T2,typename U>
void add_constant_cuda(const T1& I,T2& out,U v,bool sync = true)
{
    add_constant_cuda_kernel<<<std::min<int>((I.size()+255)/256,256),256>>>
        (tipl::make_shared(I),
         tipl::make_shared(out),v);
    if(sync)
        cudaDeviceSynchronize();
}

template<typename T1,typename T2,typename U>
__global__ void multiply_constant_cuda_kernel(T1 I,T2 out,U v)
{
    size_t stride = blockDim.x*gridDim.x;
    for(size_t index = threadIdx.x + blockIdx.x*blockDim.x;
        index < I.size();index += stride)
        out[index] = I[index] * v;
}


template<typename T1,typename T2,typename U>
void multiply_constant_cuda(const T1& I,T2& out,U v,bool sync = true)
{
    multiply_constant_cuda_kernel<<<std::min<int>((I.size()+255)/256,256),256>>>
        (tipl::make_shared(I),
         tipl::make_shared(out),v);
    if(sync)
        cudaDeviceSynchronize();
}


}

#endif//__CUDACC__


#endif//CUDA_NUMERICAL_HPP
