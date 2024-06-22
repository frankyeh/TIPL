#ifndef CDM_HPP
#define CDM_HPP
#include "../def.hpp"
#include "../numerical/basic_op.hpp"
#include "../numerical/dif.hpp"
#include "../filter/gaussian.hpp"
#include "../filter/filter_model.hpp"
#include "../mt.hpp"
#include "../numerical/resampling.hpp"
#include "../numerical/statistics.hpp"
#include "../numerical/window.hpp"
#include "../numerical/index_algorithm.hpp"
#include <iostream>
#include <limits>
#include <vector>

namespace tipl
{
namespace reg
{
template<typename image_type>
void cdm_pre(image_type& I)
{
    if(I.empty())
        return;
    float mean = float(tipl::mean(I));
    if(mean != 0.0f)
        I *= 1.0f/mean;
}
template<typename image_type>
void cdm_pre(image_type& It,image_type& It2,
             image_type& Is,image_type& Is2)
{
    std::thread t1([&](){cdm_pre(It);});
    std::thread t2([&](){cdm_pre(It2);});
    std::thread t3([&](){cdm_pre(Is);});
    std::thread t4([&](){cdm_pre(Is2);});
    t1.join();
    t2.join();
    t3.join();
    t4.join();
}

template<typename cost_type>
bool cdm_improved(cost_type& cost,cost_type& iter)
{
    if(cost.size() > 5)
    {
        float a,b,r2;
        linear_regression(iter.begin(),iter.end(),cost.begin(),a,b,r2);
        if(a > 0.0f)
            return false;
        if(cost.size() > 7)
        {
            cost.pop_front();
            iter.pop_front();
        }
    }
    return true;
}

struct cdm_param{
    float resolution = 2.0f;
    float speed = 0.3f;
    float smoothing = 0.1f;
    unsigned int iterations = 200;
    unsigned int min_dimension = 8;
};

template<typename T,typename U,typename V,typename W>
__INLINE__ void cdm_get_gradient_imp(const pixel_index<T::dimension>& index,
                                     const T& Js,const U& It,V& new_d,W& cost_map)
{
    typename T::value_type Itv[get_window_size<2,T::dimension>::value];
    typename T::value_type Jsv[get_window_size<2,T::dimension>::value];
    get_window_at_width<2>(index,It,Itv);
    auto size = get_window_at_width<2>(index,Js,Jsv);

    float a,b,r2;
    linear_regression(Jsv,Jsv+size,Itv,a,b,r2);
    if(a > 0.0f)
    {
        auto g = gradient_at(Js,index);
        auto pos = index.index();
        g *= r2*(Js[pos]*a+b-It[pos]);
        new_d[pos] += g;
        cost_map[pos] = -r2;
    }
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T1,typename T2,typename T3>
__global__ void cdm_get_gradient_cuda_kernel(T1 Js,T1 It,T2 new_d,T3 cost_map)
{
    TIPL_FOR(index,Js.size())
    {
        cdm_get_gradient_imp(tipl::pixel_index<T1::dimension>(index,Js.shape()),Js,It,new_d,cost_map);
    }
}
#endif

// calculate dJ(cJ-I)
template<typename image_type1,typename image_type2,typename dis_type>
inline float cdm_get_gradient(const image_type1& Js,const image_type2& It,dis_type& new_d)
{
    typename image_type1::buffer_type cost_map(Js.shape());
    if constexpr(memory_location<image_type1>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(cdm_get_gradient_cuda_kernel,Js.size())
                (tipl::make_shared(Js),
                 tipl::make_shared(It),
                 tipl::make_shared(new_d),
                 tipl::make_shared(cost_map));
        if (cudaGetLastError() != cudaSuccess)
            throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
        #endif
    }
    else
    {
        tipl::par_for(Js.size(),[&](size_t index)
        {
            cdm_get_gradient_imp(tipl::pixel_index<image_type1::dimension>(index,Js.shape()),Js,It,new_d,cost_map);
        });
    }
    return float(tipl::mean(cost_map));
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__

template<typename T>
__global__ void cdm_solve_poisson_cuda_kernel(T new_d,T solve_d,T new_solve_d)
{
    const int w = new_d.width();
    const int wh = new_d.plane_size();
    const int size = new_d.size();
    const int size_1 = new_d.size()-1;
    const int size_w = new_d.size()-w;
    const int size_wh = new_d.size()-wh;
    constexpr float inv_d2 = 0.5f / float(T::dimension);
    TIPL_FOR(pos,new_d.size())
    {
        auto v = new_solve_d[pos];
        if(pos >= 1)
           v += solve_d[pos-1];
        if(pos < size_1)
           v += solve_d[pos+1];
        if(pos >= w)
           v += solve_d[pos-w];
        if(pos < size_w)
           v += solve_d[pos+w];
        if constexpr(T::dimension == 3)
        {
            if(pos >= wh)
               v += solve_d[pos-wh];
            if(pos < size_wh)
               v += solve_d[pos+wh];
        }
        v -= new_d[pos];
        v *= inv_d2;
        new_solve_d[pos] = v;
    }
}
#endif

template<typename T,typename terminated_type>
void cdm_solve_poisson(T& new_d,terminated_type& terminated)
{
    T solve_d(new_d),new_solve_d(new_d.shape());
    multiply_constant(solve_d,float(-0.5f/float(T::dimension)));

    for(int iter = 0;iter < 12 && !terminated;++iter)
    {
        if constexpr(memory_location<T>::at == CUDA)
        {
            #ifdef __CUDACC__
            cdm_solve_poisson_cuda_kernel<<<std::min<int>((new_d.size()+255)/256,256),256>>>(
                        tipl::make_shared(new_d),
                        tipl::make_shared(solve_d),
                        tipl::make_shared(new_solve_d));
            if (cudaGetLastError() != cudaSuccess)
                throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
            #endif
        }
        else
        {
            const int w = new_d.width();
            const int wh = new_d.plane_size();
            const int size = new_d.size();
            const int size_1 = new_d.size()-1;
            const int size_w = new_d.size()-w;
            const int size_wh = new_d.size()-wh;
            constexpr float inv_d2 = 0.5f / float(T::dimension);
            tipl::par_for(solve_d.size(),[&](int pos)
            {
                auto v = new_solve_d[pos];
                if(pos >= 1)
                   v += solve_d[pos-1];
                if(pos < size_1)
                   v += solve_d[pos+1];
                if(pos >= w)
                   v += solve_d[pos-w];
                if(pos < size_w)
                   v += solve_d[pos+w];
                if constexpr(T::dimension == 3)
                {
                    if(pos >= wh)
                        v += solve_d[pos-wh];
                    if(pos < size_wh)
                        v += solve_d[pos+wh];
                }
                v -= new_d[pos];
                v *= inv_d2;
                new_solve_d[pos] = v;
            });
        }
        solve_d.swap(new_solve_d);
    }
    new_d.swap(solve_d);
}

struct cdm_dis_vector_length
{
    template<typename T>
    __INLINE__ float operator()(const T &v) const
    {
        return v.length();
    }
};

template<typename dist_type>
inline float cdm_max_displacement_length(dist_type& new_d)
{
    if constexpr(memory_location<dist_type>::at == CUDA)
    {
        #ifdef __CUDACC__
        return thrust::transform_reduce(thrust::device,
                        new_d.data(),new_d.data()+new_d.size(),
                        cdm_dis_vector_length(),
                            0.0f,thrust::maximum<float>());
        #endif
    }
    else
    {
        float theta = 0.0f;
        par_for(new_d.size(),[&](int i)
        {
            float l = new_d[i].length();
            if(l > theta)
               theta = l;
        });
        return theta;
    }
}
//---------------------------------------------------------------------------
template<typename T,typename U>
__INLINE__ void cdm_smooth_imp(T& d,U& dd,size_t cur_index,float w_6,float w_1)
{
    size_t cur_index_with_shift = cur_index + 1;
    tipl::vector<T::dimension> v;
    if(cur_index_with_shift < d.size())
        v += d[cur_index_with_shift];
    if(cur_index >= 1)
        v += d[cur_index-1];
    cur_index_with_shift = cur_index + d.width();
    if(cur_index_with_shift < d.size())
        v += d[cur_index_with_shift];
    if(cur_index >= d.width())
        v += d[cur_index-d.width()];
    if constexpr(T::dimension == 3)
    {
        cur_index_with_shift = cur_index + d.plane_size();
        if(cur_index_with_shift < d.size())
            v += d[cur_index_with_shift];
        if(cur_index >= d.plane_size())
            v += d[cur_index-d.plane_size()];
    }
    v *= w_6;
    dd[cur_index] = d[cur_index]*w_1+v;
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T>
__global__ void cdm_smooth_cuda_kernel(T d,T dd,float w_6,float w_1)
{
    TIPL_FOR(cur_index,d.size())
    {
        cdm_smooth_imp(d,dd,cur_index,w_6,w_1);
    }
}

#endif
template<typename dist_type>
void cdm_smooth(dist_type& d,float smoothing)
{
    if(smoothing == 0.0f)
        return;
    dist_type dd(d.shape());
    if constexpr(memory_location<dist_type>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(cdm_smooth_cuda_kernel,d.size())
                (tipl::make_shared(d),tipl::make_shared(dd),smoothing/6.0f,(1.0f-smoothing));
        if (cudaGetLastError() != cudaSuccess)
            throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
        #endif
    }
    else
    tipl::par_for(d.size(),[&](size_t cur_index)
    {
        cdm_smooth_imp(d,dd,cur_index,smoothing/float(2.0f*dist_type::dimension),(1.0f-smoothing));
    });
    dd.swap(d);
}



template<typename out_type = void,typename pointer_image_type,typename dist_type,typename terminate_type>
void cdm(std::vector<pointer_image_type> It,
          std::vector<pointer_image_type> Is,
           dist_type& d,// displacement field
           dist_type& inv_d,// displacement field
           terminate_type& terminated,
           cdm_param param = cdm_param())
{
    using image_type = typename pointer_image_type::buffer_type;
    using value_type = typename pointer_image_type::value_type;
    if(It.size() < Is.size())
        Is.resize(It.size());
    if(Is.size() < It.size())
        It.resize(Is.size());

    auto geo = It[0].shape();
    d.resize(geo);
    inv_d.resize(geo);
    // multi resolution
    if (min_value(geo) > param.min_dimension)
    {
        std::vector<image_type> rIt_buffer(It.size()),rIs_buffer(It.size());
        std::vector<pointer_image_type> rIt(It.size()),rIs(It.size());
        tipl::par_for(It.size(),[&](size_t i)
        {
            downsample_with_padding(It[i],rIt_buffer[i]);
            downsample_with_padding(Is[i],rIs_buffer[i]);
            rIt[i] = rIt_buffer[i];
            rIs[i] = rIs_buffer[i];
        },It.size());
        cdm_param param2 = param;
        param2.resolution /= 2.0f;
        cdm<out_type>(rIt,rIs,d,inv_d,terminated,param2);
        multiply_constant(d,2.0f);
        multiply_constant(inv_d,2.0f);
        upsample_with_padding(d,geo);
        upsample_with_padding(inv_d,geo);
        if(param.resolution > 1.0f)
            return;
    }


    float theta = 0.0;

    if constexpr(!std::is_void<out_type>::value)
        out_type() << "size:" << It[0].shape();


    std::deque<float> cost,iter;
    for (unsigned int index = 0;index < param.iterations && !terminated;++index)
    {
        std::vector<float> sub_cost(It.size());
        std::vector<dist_type> sub_new_d(It.size());
        tipl::par_for(It.size(),[&](size_t i)
        {
            image_type Js;
            compose_displacement(Is[i],d,Js);
            sub_new_d[i].resize(It[i].shape());
            sub_cost[i] = cdm_get_gradient(Js,It[i],sub_new_d[i]);
        },It.size());

        auto new_d = sub_new_d[0];
        for(size_t i = 1;i < sub_new_d.size();++i)
            add(new_d,sub_new_d[i]);

        cost.push_back(mean(sub_cost));
        iter.push_back(index);

        if constexpr(!std::is_void<out_type>::value)
            out_type() << "cost:" << cost.back();

        if(!cdm_improved(cost,iter))
            break;
        // solving the poisson equation using Jacobi method
        cdm_solve_poisson(new_d,terminated);

        if(theta == 0.0f)
            theta = cdm_max_displacement_length(new_d);
        if(theta == 0.0f)
            break;
        multiply_constant(new_d,param.speed/theta);
        //cdm_constraint(new_d);
        accumulate_displacement(d,new_d);
        d.swap(new_d);
        cdm_smooth(d,param.smoothing);
        invert_displacement(d,inv_d,2);
    }
    invert_displacement(d,inv_d);
}


}// namespace reg
}// namespace image
#endif // DMDM_HPP
