#ifndef CUDA_CDM_HPP
#define CUDA_CDM_HPP


#ifdef __CUDACC__
#include <functional>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include "../utility/basic_image.hpp"
#include "../numerical/interpolation.hpp"
#include "numerical.hpp"


namespace tipl{

namespace reg{



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
        {
            typename T2::value_type v(tipl::pixel_index<3>(index,to.shape()));
            v += dis[index];
            tipl::estimate<itype>(from,v,to[index]);
        }
    }
}


template<tipl::interpolation itype = tipl::interpolation::linear,
         typename T1,typename T2,typename T3>
inline void compose_displacement_cuda(const T1& from,const T2& dis,T3& to)
{
    to.clear();
    to.resize(from.shape());
    compose_displacement_cuda_kernel<itype>
            <<<std::min<int>((to.size()+255)/256,256),256>>>(
                tipl::make_shared(from),
                tipl::make_shared(dis),
                tipl::make_shared(to));
}


//---------------------------------------------------------------------------
template<typename T>
inline void accumulate_displacement_cuda(T& v0,const T& vv)
{
    {
        T nv;
        compose_displacement_cuda(v0,vv,nv);
        v0.swap(nv);
    }
    add_cuda(v0,vv);
}


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
    return thrust::reduce(thrust::device,r2_256.get(),r2_256.get()+r2_256.size(),0.0)/float(Js.size());
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
    multiply_constant_cuda(new_d,solve_d,float(-0.5f/3.0f));

    size_t grid_dim = std::min<int>((new_d.size()+255)/256,256);
    for(int iter = 0;iter < 12 && !terminated;++iter)
    {
        cdm_solve_poisson_cuda_kernel2<<<grid_dim,256>>>(
                    tipl::make_shared(new_d),
                    tipl::make_shared(solve_d),
                    tipl::make_shared(new_solve_d));
        solve_d.swap(new_solve_d);
    }
    new_d.swap(solve_d);
}


struct cdm_accumulate_dis_vector_length
{
    template<typename T>
    __INLINE__ float operator()(const T &v) const
    {
        return v.length();
    }
};

template<typename dist_type,typename value_type>
void cdm_accumulate_dis_cuda(dist_type& d,dist_type& new_d,value_type& theta,float speed)
{
    if(theta == 0.0f)
    {
        theta = thrust::transform_reduce(thrust::device,
                    new_d.get(),
                    new_d.get()+d.size(),
                    cdm_accumulate_dis_vector_length(),
                        0.0f,thrust::maximum<float>());
    }
    if(theta == 0.0)
        return;
    multiply_constant_cuda(new_d,speed/theta);
    accumulate_displacement_cuda(d,new_d);
}


template<typename T>
__global__ void cdm_constraint_cuda_kernel(T d,T dd,float constraint_length)
{
    size_t shift[T::dimension];
    shift[0] = 1;
    shift[1] = d.width();
    shift[2] = d.plane_size();
    size_t stride = blockDim.x*gridDim.x;
    for(size_t index = threadIdx.x + blockIdx.x*blockDim.x;
        index < d.size();index += stride)
    {
        for(unsigned int dim = 0;dim < 3;++dim)
        {
            size_t index_with_shift = index + shift[dim];
            if(index_with_shift >= d.size())
                break;
            float dis = d[index_with_shift][dim] - d[index][dim];
            if(dis < 0)
                dis *= 0.25f;
            else
            {
                if(dis > constraint_length)
                    dis = 0.25f*(dis-constraint_length);
                else
                    continue;
            }
            atomicAdd(&dd[index][dim],dis);
            atomicAdd(&dd[index_with_shift][dim],-dis);
        }
    }
}

template<typename dist_type>
void cdm_constraint_cuda(dist_type& d,float constraint_length)
{
    dist_type dd(d.shape());
    cdm_constraint_cuda_kernel<<<std::min<int>((d.size()+255)/256,256),256>>>(
                    tipl::make_shared(d),
                    tipl::make_shared(dd),
                    constraint_length);
    add_cuda(d,dd);
}

template<typename image_type,typename dist_type,typename terminate_type>
float cdm_cuda(const image_type& It,
               const image_type& Is,
               dist_type& d,// displacement field
               terminate_type& terminated,
               cdm_param param = cdm_param())
{
    if(It.shape() != Is.shape())
        throw "Inconsistent image dimension";
    auto geo = It.shape();
    d.resize(It.shape());

    // multi resolution
    if (*std::min_element(geo.begin(),geo.end()) > param.min_dimension)
    {
        //downsampling
        image_type rIs,rIt;
        downsample_with_padding_cuda(It,rIt);
        downsample_with_padding_cuda(Is,rIs);
        cdm_param param2 = param;
        param2.resolution /= 2.0f;
        float r = cdm_cuda(rIt,rIs,d,terminated,param2);
        multiply_constant_cuda(d,2.0f);
        upsample_with_padding_cuda(d,geo);
        if(param.resolution > 1.0f)
            return r;
    }

    image_type Js;// transformed I
    dist_type new_d(d.shape());
    float theta = 0.0;

    std::deque<float> r,iter;
    for (unsigned int index = 0;index < param.iterations && !terminated;++index)
    {
        compose_displacement_cuda(Is,d,Js);
        // dJ(cJ-I)
        r.push_back(cdm_get_gradient_cuda(Js,It,new_d));
        iter.push_back(index);
        if(!cdm_improved(r,iter))
            break;
        // solving the poisson equation using Jacobi method
        cdm_solve_poisson_cuda(new_d,terminated);
        cdm_accumulate_dis_cuda(d,new_d,theta,param.speed);
        cdm_constraint_cuda(d,param.constraint);
    }
    return r.front();
}

template<typename image_type,typename dist_type,typename terminate_type>
float cdm2_cuda(const image_type& It,const image_type& It2,
           const image_type& Is,const image_type& Is2,
           dist_type& d,// displacement field
           terminate_type& terminated,
           cdm_param param = cdm_param())
{
    if(It.shape() != It2.shape() ||
       It.shape() != Is.shape() ||
       It.shape() != Is2.shape())
        throw "Inconsistent image dimension";
    auto geo = It.shape();
    d.resize(It.shape());

    // multi resolution
    if (*std::min_element(geo.begin(),geo.end()) > param.min_dimension)
    {
        //downsampling
        image_type rIs,rIt,rIs2,rIt2;
        downsample_with_padding_cuda(It,rIt);
        downsample_with_padding_cuda(Is,rIs);
        downsample_with_padding_cuda(It2,rIt2);
        downsample_with_padding_cuda(Is2,rIs2);
        cdm_param param2 = param;
        param2.resolution /= 2.0f;
        float r = cdm2_cuda(rIt,rIt2,rIs,rIs2,d,terminated,param2);
        multiply_constant_cuda(d,2.0f);
        upsample_with_padding_cuda(d,geo);
        if(param.resolution > 1.0f)
            return r;
    }

    image_type Js,Js2;// transformed I
    dist_type new_d(d.shape()),new_d2(d.shape());// new displacements
    float theta = 0.0;

    std::deque<float> r,iter;
    for (unsigned int index = 0;index < param.iterations && !terminated;++index)
    {
        compose_displacement_cuda(Is,d,Js);
        compose_displacement_cuda(Is2,d,Js2);
        // dJ(cJ-I)
        r.push_back((cdm_get_gradient_cuda(Js,It,new_d)+cdm_get_gradient_cuda(Js2,It2,new_d2))*0.5f);
        iter.push_back(index);
        if(!cdm_improved(r,iter))
            break;
        add_cuda(new_d,new_d2);
        // solving the poisson equation using Jacobi method
        cdm_solve_poisson_cuda(new_d,terminated);
        cdm_accumulate_dis_cuda(d,new_d,theta,param.speed);
        cdm_constraint_cuda(d,param.constraint);
    }
    return r.front();
}

}//reg
}//tipl

#endif//__CUDACC__

#endif//CUDA_CDM_HPP
