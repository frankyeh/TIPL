#ifndef TIPL_DEF
#define TIPL_DEF

namespace tipl{


template <typename T>
struct function_traits
    : public function_traits<decltype(&T::operator())>
{};
template <typename ClassType, typename ReturnType, typename Arg1,typename... Args>
struct function_traits<ReturnType(ClassType::*)(Arg1,Args...) const>
{
    using class_type = ClassType;
    using return_type = ReturnType;
    using arg_type = Arg1;
    static constexpr int arg_num = sizeof...(Args)+1;
};



#if defined(TIPL_USE_CUDA)
constexpr bool use_cuda = true;
#else
constexpr bool use_cuda = false;
#endif

#ifdef INCLUDE_NLOHMANN_JSON_HPP_
constexpr bool use_xeus_cling = true;
#else
constexpr bool use_xeus_cling = false;
#endif

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
