#ifndef CUDA_CDM_HPP
#define CUDA_CDM_HPP


#ifdef __CUDACC__
#include "../utility/basic_image.hpp"
#include "../numerical/interpolation.hpp"
#include "numerical.hpp"

namespace tipl{

namespace reg{
// calculate dJ(cJ-I)

template<typename T1,typename T2,typename T3>
__global__ void cdm_get_gradient_cuda_kernel(T1 Js,T1 It,T2 new_d,T3 r2_256)
{
    size_t stride = blockDim.x*gridDim.x;
    for(size_t index = threadIdx.x + blockIdx.x*blockDim.x;
        index < Js.size();index += stride)
    {
        tipl::pixel_index<3> pos(index,Js.shape());
        if(It[index] == 0.0f || Js[index] == 0.0f || It.shape().is_edge(pos))
        {
            new_d[index] = typename T2::value_type();
            continue;
        }
        // calculate gradient
        new_d[index][0] = Js[index+1]-Js[index-1];
        new_d[index][1] = Js[index+Js.width()]-Js[index-Js.width()];
        new_d[index][2] = Js[index+Js.plane_size()]-Js[index-Js.plane_size()];


        typename T1::value_type Itv[get_window_size<2,3>::value];
        typename T1::value_type Jsv[get_window_size<2,3>::value];
        get_window_at_width<2>(pos,It,Itv);
        auto size = get_window_at_width<2>(pos,Js,Jsv);

        float a,b,r2;
        linear_regression(Jsv,Jsv+size,Itv,a,b,r2);
        if(a <= 0.0f)
            new_d[index] = typename T2::value_type();
        else
        {
            new_d[index] *= r2*(Js[index]*a+b-It[index]);
            atomicAdd(r2_256.begin()+threadIdx.x,r2);
        }
    }
}

template<typename image_type,typename dis_type>
inline float cdm_get_gradient_cuda(const image_type& Js,const image_type& It,dis_type& new_d)
{
    device_vector<float> r2_256(256);
    cdm_get_gradient_cuda_kernel
            <<<std::min<int>((Js.size()+255)/256,256),256>>>(
                tipl::make_shared(Js),
                tipl::make_shared(It),
                tipl::make_shared(new_d),
                tipl::make_shared(r2_256));
    cudaDeviceSynchronize();
    return thrust::reduce(r2_256.begin_thrust(),r2_256.end_thrust(),0.0)/float(Js.size());
}


template<tipl::interpolation itype,typename T1,typename T2,typename T3>
__global__ void compose_displacement_cuda_kernel(T1 from,T2 dis,T3 to)
{
    size_t stride = blockDim.x*gridDim.x;
    for(size_t index = threadIdx.x + blockIdx.x*blockDim.x;
        index < to.size();index += stride)
    {
        if(dis[index] == typename T2::value_type())
            to[index] = from[index];
        else
            tipl::estimate<itype>(from,
               typename T2::value_type(tipl::pixel_index<3>(index,to.shape())) += dis[index],to[index]);
    }
}


template<tipl::interpolation itype = tipl::interpolation::linear,
         typename T1,typename T2,typename T3>
inline void compose_displacement_cuda(const T1& from,const T2& dis,T3& to,bool sync = true)
{
    to.resize(from.shape());
    compose_displacement_cuda_kernel<itype,typename T1::value_type,typename T2::value_type,typename T3::value_type>
            <<<std::min<int>((to.size()+255)/256,256),256>>>(
                tipl::make_shared(from),
                tipl::make_shared(dis),
                tipl::make_shared(to));
    if(sync)
        cudaDeviceSynchronize();
}

template<typename T>
__global__ void cdm_solve_poisson_cuda_kernel2(T new_d,T solve_d,T new_solve_d)
{
    const float inv_d2 = 0.5f/3.0f;
    int w = new_d.width();
    int wh = new_d.plane_size();
    size_t stride = blockDim.x*gridDim.x;
    for(int64_t index = threadIdx.x + blockIdx.x*blockDim.x;
        index < new_d.size();index += stride)
    {
        typename T::value_type v;
        {
            int64_t p1 = index-1;
            int64_t p2 = index+1;
            if(p1 >= 0)
               v += solve_d[p1];
            if(p2 < solve_d.size())
               v += solve_d[p2];
        }
        {
            int64_t p1 = index-w;
            int64_t p2 = index+w;
            if(p1 >= 0)
               v += solve_d[p1];
            if(p2 < solve_d.size())
               v += solve_d[p2];
        }
        {
            int64_t p1 = index-wh;
            int64_t p2 = index+wh;
            if(p1 >= 0)
               v += solve_d[p1];
            if(p2 < solve_d.size())
               v += solve_d[p2];
        }
        v -= new_d[index];
        v *= inv_d2;
        new_solve_d[index] = v;
    }
}

template<typename T,typename terminated_type>
inline void cdm_solve_poisson_cuda(T& new_d,terminated_type& terminated)
{
    T solve_d(new_d.shape()),new_solve_d(new_d.shape());
    multiply_constant_cuda(new_d,solve_d,-0.5f/3.0f);

    size_t grid_dim = std::min<int>((new_d.size()+255)/256,256);
    for(int iter = 0;iter < 12 && !terminated;++iter)
    {
        cdm_solve_poisson_cuda_kernel2<<<grid_dim,256>>>(
                    tipl::make_shared(new_d),
                    tipl::make_shared(solve_d),
                    tipl::make_shared(new_solve_d));
        cudaDeviceSynchronize();
        solve_d.swap(new_solve_d);
    }
    new_d.swap(solve_d);
}


}//reg
}//tipl

#endif//__CUDACC__

#endif//CUDA_CDM_HPP
