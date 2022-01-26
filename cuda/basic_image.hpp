#ifndef CUDA_BASIC_IMAGE_HPP
#define CUDA_BASIC_IMAGE_HPP


#ifdef __CUDACC__

#include "../utility/basic_image.hpp"
#include "mem.hpp"

namespace tipl{

template<int dim,typename vtype = float>
using device_image = tipl::image<dim,vtype,device_memory>;
template<int dim,typename vtype = float>
using host_image = tipl::image<dim,vtype,host_memory>;

}

#endif//__CUDACC__

#endif//CUDA_BASIC_IMAGE_HPP
