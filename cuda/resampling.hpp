#ifndef CUDA_RESAMPLE_HPP
#define CUDA_RESAMPLE_HPP


#ifdef __CUDACC__
#include "../utility/basic_image.hpp"
#include "../numerical/interpolation.hpp"
namespace tipl{

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
inline void resample_cuda(const T& from,T2& to,const U& trans,bool sync = true)
{
    resample_cuda_kernel<itype>
            <<<std::min<int>((to.size()+255)/256,256),256>>>(
                tipl::make_shared(from),
                tipl::make_shared(to),trans);
    if(sync)
        cudaDeviceSynchronize();
}

}

#endif//__CUDACC__

#endif//CUDA_RESAMPLE_HPP
