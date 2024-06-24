#ifndef TIPL_DEF
#define TIPL_DEF

namespace tipl{


#if defined(TIPL_USE_CUDA)
constexpr bool use_cuda = true;
#else
constexpr bool use_cuda = false;
#endif

enum memory_location_type{
    CPU,
    CUDA,
    HIP
};

template<typename container>
struct memory_location{
    static constexpr memory_location_type at = CPU;
};

}

#ifdef __CUDACC__
#define __DEVICE_HOST__ __device__ __host__
#define __INLINE__ __forceinline__ __device__ __host__
#define __DEVICE__ __device__
#define __HOST__ __host__
#define TIPL_RUN_STREAM(kernel_function,total_size,stream) kernel_function<<<std::min<int>((total_size+255)/256,256),256,0,stream>>>
#define TIPL_RUN(kernel_function,total_size) kernel_function<<<std::min<int>((total_size+255)/256,256),256>>>
#define TIPL_FOR(index,total_size) \
    size_t tipl_for_stride = blockDim.x*gridDim.x; \
    for(size_t index = threadIdx.x + blockIdx.x*blockDim.x; \
        index < total_size;index += tipl_for_stride)
#else
#define __DEVICE_HOST__
#define __INLINE__ inline
#define __DEVICE__
#define __HOST__
#endif

#endif//TIPL_DEF
