#ifndef CUDA_BASIC_IMAGE_HPP
#define CUDA_BASIC_IMAGE_HPP


#include "../utility/basic_image.hpp"
#include "mem.hpp"

namespace tipl{

template<int dim,typename vtype = float>
using device_image = tipl::image<dim,vtype,device_vector>;
template<int dim,typename vtype = float>
using host_image = tipl::image<dim,vtype,host_vector>;

}


#endif//CUDA_BASIC_IMAGE_HPP
