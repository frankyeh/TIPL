#ifndef CUDA_RESAMPLE_HPP
#define CUDA_RESAMPLE_HPP


#ifdef __CUDACC__

#include "../numerical/interpolation.hpp"

namespace tipl{

template<tipl::interpolation itype,typename T,typename U>
__global__ void resample_cuda_kernel(const T* from,T* to,const U* trans_,tipl::shape<3> from_shape)
{
    const tipl::transformation_matrix<U>& trans = *reinterpret_cast<const tipl::transformation_matrix<U>* >(trans_);
    tipl::pixel_index<3> index(blockIdx.x,blockIdx.y,threadIdx.x,tipl::shape<3>(gridDim.x,gridDim.y,blockDim.x));
    tipl::vector<3> v;
    trans(index,v);
    tipl::estimate<itype>(tipl::make_image(from,from_shape),v,to[index.index()]);
}

template<tipl::interpolation itype = linear,typename T,typename T2,typename U>
void resample_cuda(const T& dfrom,T2& dto,const U& trans,bool sync = true)
{
    resample_cuda_kernel<itype><<<dim3(dto.width(),dto.height()),dto.depth()>>>
        (dfrom.get(),dto.get(),tipl::device_memory<typename U::value_type>(trans).get(),dfrom.shape());
    if(sync)
        cudaDeviceSynchronize();
}

}

#endif//__CUDACC__

#endif//CUDA_RESAMPLE_HPP
