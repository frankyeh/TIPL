#ifndef TIPL_DEF
#define TIPL_DEF

namespace tipl{
enum backend{seq,mt,cuda};

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
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#else
#define __DEVICE_HOST__
#define __INLINE__ inline
#endif

#endif//TIPL_DEF
