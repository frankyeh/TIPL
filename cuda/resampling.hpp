#ifndef CUDA_RESAMPLE_HPP
#define CUDA_RESAMPLE_HPP


#ifdef __CUDACC__
#include "../utility/basic_image.hpp"
#include "../numerical/interpolation.hpp"
namespace tipl{

template<tipl::interpolation itype,typename T1,typename T2,typename U>
__global__ void resample_cuda_kernel(const_pointer_image<3,T1> from,
                                     pointer_image<3,T2> to,
                                      U trans)
{
    tipl::pixel_index<3> index(blockIdx.x,blockIdx.y,threadIdx.x,tipl::shape<3>(gridDim.x,gridDim.y,blockDim.x));
    tipl::vector<3> v;
    trans(index,v);
    tipl::estimate<itype>(from,v,to[index.index()]);
}


template<tipl::interpolation itype = linear,typename T,typename T2,typename U>
inline void resample_cuda(const T& from,T2& to,const U& trans,bool sync = true)
{
    resample_cuda_kernel<itype,typename T::value_type,typename T2::value_type>
            <<<dim3(to.width(),to.height()),to.depth()>>>(from,to,trans);
    if(sync)
        cudaDeviceSynchronize();
}

}

#endif//__CUDACC__

#endif//CUDA_RESAMPLE_HPP
