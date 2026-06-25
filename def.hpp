#ifndef TIPL_DEF
#define TIPL_DEF

namespace tipl{


#if defined(TIPL_USE_CUDA)
constexpr bool use_cuda = true;
inline bool has_gpu = true;
#else
constexpr bool use_cuda = false;
inline bool has_gpu = false;
#endif


enum memory_location_type{
    CPU,
    CUDA,
    HIP
};

template<typename container>
struct memory_location;

template<typename container>
struct memory_location<const container> {
    static constexpr memory_location_type at = memory_location<container>::at;
};

template<typename container>
struct memory_location<container&> {
    static constexpr memory_location_type at = memory_location<container>::at;
};

template<typename container>
struct memory_location<container&&> {
    static constexpr memory_location_type at = memory_location<container>::at;
};

template<typename container>
struct memory_location<const container&> {
    static constexpr memory_location_type at = memory_location<container>::at;
};

}

#ifdef __CUDACC__
#define __DEVICE_HOST__ __device__ __host__
#define __INLINE__ __forceinline__ __device__ __host__
#define __DEVICE__ __device__
#define __HOST__ __host__

#define TIPL_RUN_STREAM(kernel_function, total_size, stream) \
    kernel_function<<<std::min<size_t>(((total_size) + 255) / 256, 32768), 256, 0, stream>>>

#define TIPL_RUN(kernel_function, total_size) \
    kernel_function<<<std::min<size_t>(((total_size) + 255) / 256, 32768), 256>>>

#define TIPL_FOR(index, total_size) \
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; \
        index < (total_size); \
        index += blockDim.x * gridDim.x)
#else
#define __DEVICE_HOST__
#define __INLINE__ inline
#define __DEVICE__
#define __HOST__
#endif

#endif//TIPL_DEF
