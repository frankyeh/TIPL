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
    auto result = thrust::minmax_element(thrust::device,
                                         data.get(),data.get()+data.size());
    return std::make_pair(device_eval(result.first),device_eval(result.second));
}

template<typename T1, typename T2,typename value_type>
__global__  void normalize_upper_lower_cuda_kernel(T1 in,T2 out,value_type min,value_type coef)
{
    TIPL_FOR(index,in.size())
        out[index] = value_type(in[index]-min)*coef;
}

template<typename T,typename U>
inline void normalize_upper_lower_cuda(const T& in,U& out,float upper_limit = 255.0f)
{
    auto min_max = minmax_value_cuda(in);
    auto range = min_max.second-min_max.first;
    float coef = range == 0 ? 0.0f:float(upper_limit/range);
    TIPL_RUN(normalize_upper_lower_cuda_kernel,in.size())
        (tipl::make_shared(in),tipl::make_shared(out),float(min_max.first),coef);
}

template<typename T,
         typename std::enable_if<std::is_fundamental<typename T::value_type>::value,bool>::type = true>
inline typename T::value_type sum_cuda(const T& data,typename T::value_type init = 0,cudaStream_t stream = nullptr)
{
    if(stream)
        return thrust::reduce(thrust::cuda::par.on(stream),data.get(),data.get()+data.size(),init);
    return thrust::reduce(thrust::device,data.get(),data.get()+data.size(),init);
}

template<typename T,
         typename std::enable_if<std::is_class<typename T::value_type>::value,bool>::type = true>
inline typename T::value_type sum_cuda(const T& data,typename T::value_type init = typename T::value_type(),cudaStream_t stream = nullptr)
{
    if(stream)
        return thrust::reduce(thrust::cuda::par.on(stream),data.get(),data.get()+data.size(),init);
    return thrust::reduce(thrust::device,data.get(),data.get()+data.size(),init);
}

template<typename T,typename U>
__global__ void add_cuda_kernel(T I,U I2)
{
    TIPL_FOR(index,I.size())
        I[index] += I2[index];
}

template<typename T,typename U>
inline void add_cuda(T& I,const U& I2)
{
    TIPL_RUN(add_cuda_kernel,I.size())
        (tipl::make_shared(I),tipl::make_shared(I2));
}


template<typename T,typename U>
__global__ void add_constant_cuda_kernel(T I,U v)
{
    TIPL_FOR(index,I.size())
        I[index] += v;
}

template<typename T,typename U>
inline void add_constant_cuda(T& I,U v)
{
    TIPL_RUN(add_constant_cuda_kernel,I.size())
        (tipl::make_shared(I),v);
}

template<typename T1,typename T2,typename U>
__global__ void add_constant_cuda_kernel(T1 I,T2 out,U v)
{
    TIPL_FOR(index,I.size())
        out[index] = I[index] + v;
}


template<typename T1,typename T2,typename U>
inline void add_constant_cuda(const T1& I,T2& out,U v)
{
    TIPL_RUN(add_constant_cuda_kernel,I.size())
        (tipl::make_shared(I),
         tipl::make_shared(out),v);
}

template<typename T1,typename T2,typename U>
__global__ void multiply_constant_cuda_kernel(T1 I,T2 out,U v)
{
    TIPL_FOR(index,I.size())
        out[index] = I[index] * v;
}


template<typename T1,typename T2,typename U>
inline void multiply_constant_cuda(const T1& I,T2& out,U v)
{
    TIPL_RUN(multiply_constant_cuda_kernel,I.size())
        (tipl::make_shared(I),
         tipl::make_shared(out),v);
}

template<typename T1,typename U>
__global__ void multiply_constant_cuda_kernel(T1 I,U v)
{
    TIPL_FOR(index,I.size())
        I[index] *= v;
}


template<typename T1,typename U>
inline void multiply_constant_cuda(T1& I,U v)
{
    TIPL_RUN(multiply_constant_cuda_kernel,I.size())
        (tipl::make_shared(I),v);
}


}

#endif//__CUDACC__


#endif//CUDA_NUMERICAL_HPP
