#ifdef INCLUDE_NLOHMANN_JSON_HPP_
#define USING_XEUS_CLING
#endif

#ifdef __CUDACC__
#define __DEVICE_HOST__ __device__ __host__
#else
#define __DEVICE_HOST__
#endif

