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
        // calculate gradient
        float data[6];
        connected_neighbors(index,Js,data);
        tipl::vector<3> g(data[1] - data[0],data[3] - data[2],data[5] - data[4]);
        if(data[0] == 0.0f || data[1] == 0.0f)
            g[0] = 0.0f;
        if(data[2] == 0.0f || data[3] == 0.0f)
            g[1] = 0.0f;
        if(data[4] == 0.0f || data[5] == 0.0f)
            g[2] = 0.0f;
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
        cdm_get_gradient_imp(tipl::pixel_index<3>(index,Js.shape()),Js,It,new_d,cost_map);
    }
}
#endif

// calculate dJ(cJ-I)
template<typename image_type,typename dis_type>
inline float cdm_get_gradient(const image_type& Js,const image_type& It,dis_type& new_d)
{
    image_type cost_map(Js.shape());
    if constexpr(memory_location<image_type>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(cdm_get_gradient_cuda_kernel,Js.size())
                (tipl::make_shared(Js),
                 tipl::make_shared(It),
                 tipl::make_shared(new_d),
                 tipl::make_shared(cost_map));
        #endif
    }
    else
    {
        tipl::par_for(Js.size(),[&](size_t index)
        {
            cdm_get_gradient_imp(tipl::pixel_index<3>(index,Js.shape()),Js,It,new_d,cost_map);
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
    constexpr float inv_d2 = 0.5f / 3.0f;
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
        if(pos >= wh)
           v += solve_d[pos-wh];
        if(pos < size_wh)
           v += solve_d[pos+wh];
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
    multiply_constant(solve_d,float(-0.5f/3.0f));

    for(int iter = 0;iter < 12 && !terminated;++iter)
    {
        if constexpr(memory_location<T>::at == CUDA)
        {
            #ifdef __CUDACC__
            cdm_solve_poisson_cuda_kernel<<<std::min<int>((new_d.size()+255)/256,256),256>>>(
                        tipl::make_shared(new_d),
                        tipl::make_shared(solve_d),
                        tipl::make_shared(new_solve_d));
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
            constexpr float inv_d2 = 0.5f / 3.0f;
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
                if(pos >= wh)
                   v += solve_d[pos-wh];
                if(pos < size_wh)
                   v += solve_d[pos+wh];
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
    tipl::vector<3> v;
    if(cur_index_with_shift < d.size())
        v += d[cur_index_with_shift];
    if(cur_index >= 1)
        v += d[cur_index-1];
    cur_index_with_shift = cur_index + d.width();
    if(cur_index_with_shift < d.size())
        v += d[cur_index_with_shift];
    if(cur_index >= d.width())
        v += d[cur_index-d.width()];
    cur_index_with_shift = cur_index + d.plane_size();
    if(cur_index_with_shift < d.size())
        v += d[cur_index_with_shift];
    if(cur_index >= d.plane_size())
        v += d[cur_index-d.plane_size()];
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

        #endif
    }
    else
    tipl::par_for(d.size(),[&](size_t cur_index)
    {
        cdm_smooth_imp(d,dd,cur_index,smoothing/6.0f,(1.0f-smoothing));
    });
    dd.swap(d);
}



template<typename out_type = void,typename image_type,typename dist_type,typename terminate_type>
float cdm2(const image_type& It,const image_type& It2,
           const image_type& Is,const image_type& Is2,
           dist_type& d,// displacement field
           dist_type& inv_d,// displacement field
           terminate_type& terminated,
           cdm_param param = cdm_param())
{
    bool has_dual = (It2.size() && Is2.size());
    auto geo = It.shape();
    d.resize(It.shape());
    inv_d.resize(It.shape());
    // multi resolution
    if (*std::min_element(geo.begin(),geo.end()) > param.min_dimension)
    {
        //downsampling
        image_type rIs,rIt,rIs2,rIt2;
        downsample_with_padding(It,rIt);
        downsample_with_padding(Is,rIs);
        if(has_dual)
        {
            downsample_with_padding(It2,rIt2);
            downsample_with_padding(Is2,rIs2);
        }
        cdm_param param2 = param;
        param2.resolution /= 2.0f;
        float r = cdm2<out_type>(rIt,rIt2,rIs,rIs2,d,inv_d,terminated,param2);
        multiply_constant(d,2.0f);
        multiply_constant(inv_d,2.0f);
        upsample_with_padding(d,geo);
        upsample_with_padding(inv_d,geo);
        if(param.resolution > 1.0f)
            return r;
    }


    float theta = 0.0;

    if constexpr(!std::is_void<out_type>::value)
        out_type() << "resolution:" << It.shape();


    std::deque<float> cost,iter;
    for (unsigned int index = 0;index < param.iterations && !terminated;++index)
    {
        image_type Js;
        compose_displacement(Is,d,Js);
        // dJ(cJ-I)
        dist_type new_d(It.shape());
        cost.push_back(cdm_get_gradient(Js,It,new_d));
        if(has_dual)
        {
            image_type Js2;
            compose_displacement(Is2,d,Js2);
            cost.back() += cdm_get_gradient(Js2,It2,new_d);
            cost.back() *= 0.5f;
        }

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
    return cost.front();
}


template<typename image_type,typename dist_type,typename terminate_type>
float cdm(const image_type& It,
            const image_type& Is,
            dist_type& d,// displacement field
            dist_type& inv_d,// displacement field
            terminate_type& terminated,
            cdm_param param = cdm_param())
{
    return cdm2(It,image_type(),Is,image_type(),d,inv_d,terminated,param);
}




}// namespace reg
}// namespace image
#endif // DMDM_HPP
