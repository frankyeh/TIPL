#ifndef TIPL_DEF
#define TIPL_DEF

namespace tipl{


template <typename T>
struct function_traits
    : public function_traits<decltype(&T::operator())>
{};
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const>
{
    static constexpr int arg_num = sizeof...(Args);
};



#ifdef TIPL_USE_CUDA
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
#else
#define __DEVICE_HOST__
#define __INLINE__ inline
#endif

#endif//TIPL_DEF
