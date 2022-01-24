#ifndef CUDA_NUMERICAL_HPP
#define CUDA_NUMERICAL_HPP

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


namespace tipl{

template<typename iterator_type>
std::pair<typename std::iterator_traits<iterator_type>::value_type,typename std::iterator_traits<iterator_type>::value_type>
minmax_value_cuda(iterator_type iter,iterator_type end)
{
    if(iter == end)
        return std::make_pair(0,0);
    auto dp = thrust::device_pointer_cast(iter);
    auto result = thrust::minmax_element(dp,dp+(end-iter));
    return std::make_pair(*result.first,*result.second);
}



template<typename InputIter,typename OutputIter>
void normalize_upper_lower_cuda(InputIter from,InputIter to,OutputIter out,float upper_limit = 255.0)
{
    typedef typename std::iterator_traits<InputIter>::value_type value_type;
    std::pair<value_type,value_type> min_max(minmax_value_cuda(from,to));
    value_type range = min_max.second-min_max.first;
    if(range == 0)
        return;
    auto dp_from = thrust::device_pointer_cast(from);
    auto dp_out = thrust::device_pointer_cast(out);
    using namespace thrust::placeholders;
    upper_limit /= range;
    thrust::transform(dp_from,dp_from + (to-from),dp_out, (_1 -min_max.first)*upper_limit);
}

template<typename ImageType>
__INLINE__ ImageType::value_type accumulate(const ImageType& I,ImageType::value_type init = 0)
{
    auto beg = thrust::device_pointer_cast(I.get());
    auto end = beg + I.size();
    return thrust::reduce(beg,end,init);
}


}

#endif//__CUDACC__


#endif//CUDA_NUMERICAL_HPP
