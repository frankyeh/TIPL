#ifndef CDM_HPP
#define CDM_HPP
#include "../numerical/basic_op.hpp"
#include "../numerical/dif.hpp"
#include "../filter/gaussian.hpp"
#include "../filter/filter_model.hpp"
#include "../mt.hpp"
#include "../numerical/resampling.hpp"
#include "../numerical/statistics.hpp"
#include "../numerical/window.hpp"
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


template<typename value_type,size_t dimension>
class poisson_equation_solver;

template<typename value_type>
class poisson_equation_solver<value_type,2>
{
    typedef typename filter::pixel_manip<value_type>::type manip_type;
public:
    template<typename image_type>
    void operator()(image_type& src)
    {
        image_type dest(src.shape());
        int w = src.width();
        int shift[4];
        shift[0] = 1;
        shift[1] = -1;
        shift[2] = w;
        shift[3] = -w;
        for(int i = 0;i < 4;++i)
        if (shift[i] >= 0)
        {
            auto iter1 = dest.begin() + shift[i];
            auto iter2 = src.begin();
            auto end = dest.end();
            for (;iter1 < end;++iter1,++iter2)
                *iter1 += *iter2;
        }
        else
        {
            auto iter1 = dest.begin();
            auto iter2 = src.begin() + (-shift[i]);
            auto end = src.end();
            for (;iter2 < end;++iter1,++iter2)
                *iter1 += *iter2;
        }
        dest.swap(src);
    }
};
template<typename value_type>
class poisson_equation_solver<value_type,3>
{
    typedef typename filter::pixel_manip<value_type>::type manip_type;
public:
    template<typename image_type>
    void operator()(image_type& src)
    {
        image_type dest(src.shape());
        int64_t w = src.width();
        int64_t wh = src.width()*src.height();
        int64_t shift[6];
        shift[0] = 1;
        shift[1] = -1;
        shift[2] = w;
        shift[3] = -w;
        shift[4] = wh;
        shift[5] = -wh;
        for(int i = 0;i < 6;++i)
        {
            if (shift[i] >= 0)
            {
                int64_t s = shift[i];
                par_for(dest.size()-s,[&dest,&src,s](size_t index){
                    dest[index+s] += src[index];
                });
            }
            else
            {
                int64_t s = -shift[i];
                par_for(dest.size()-s,[&dest,&src,s](size_t index){
                    dest[index] += src[index+s];
                });
            }
        }
        dest.swap(src);
    }
};

template<typename r_type>
bool cdm_improved(r_type& r,r_type& iter)
{
    if(r.size() > 5)
    {
        float a,b,r2;
        linear_regression(iter.begin(),iter.end(),r.begin(),a,b,r2);
        if(a < 0.0f)
            return false;
        if(r.size() > 7)
        {
            r.pop_front();
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
                                     const T& Js,const U& It,V& new_d,W& r2_map)
{
    if(It[index.index()] == 0.0 || Js[index.index()] == 0.0 ||
       It.shape().is_edge(index))
    {
        new_d[index.index()] = typename V::value_type();
        return;
    }
    // calculate gradient
    new_d[index.index()][0] = Js[index.index()+1]-Js[index.index()-1];
    new_d[index.index()][1] = Js[index.index()+Js.width()]-Js[index.index()-Js.width()];
    new_d[index.index()][2] = Js[index.index()+Js.plane_size()]-Js[index.index()-Js.plane_size()];

    typename T::value_type Itv[get_window_size<2,T::dimension>::value];
    typename T::value_type Jsv[get_window_size<2,T::dimension>::value];
    get_window_at_width<2>(index,It,Itv);
    auto size = get_window_at_width<2>(index,Js,Jsv);
    float a,b,r2;
    linear_regression(Jsv,Jsv+size,Itv,a,b,r2);
    if(a <= 0.0f)
        new_d[index.index()] = typename V::value_type();
    else
    {
        new_d[index.index()] *= r2*(Js[index.index()]*a+b-It[index.index()]);
        r2_map[index.index()] = r2;
    }
}
// calculate dJ(cJ-I)
template<typename image_type,typename dis_type>
float cdm_get_gradient(const image_type& Js,const image_type& It,dis_type& new_d)
{
    std::vector<float> r2_map(Js.size());
    tipl::par_for(tipl::begin_index(Js.shape()),tipl::end_index(Js.shape()),
                        [&](const pixel_index<image_type::dimension>& index)
    {
        cdm_get_gradient_imp(index,Js,It,new_d,r2_map);
    });
    return float(tipl::mean(r2_map));
}


template<typename T,typename terminated_type>
void cdm_solve_poisson(T& new_d,terminated_type& terminated)
{
    float inv_d2 = 0.5f/3.0f;
    T solve_d(new_d);
    multiply_constant_mt(solve_d,-inv_d2);

    int w = new_d.width();
    int wh = new_d.plane_size();
    for(int iter = 0;iter < 12 && !terminated;++iter)
    {
        T new_solve_d(new_d.shape());
        tipl::par_for(solve_d.size(),[&](int pos)
        {
            // boundary checking (p > 0 && p < width) is critical for
            // getting correct results from low resolution
            auto v = new_solve_d[pos];
            {
                int p1 = pos-1;
                int p2 = pos+1;
                if(p1 >= 0)
                   v += solve_d[p1];
                if(p2 < solve_d.size())
                   v += solve_d[p2];
            }
            {
                int p1 = pos-w;
                int p2 = pos+w;
                if(p1 >= 0)
                   v += solve_d[p1];
                if(p2 < solve_d.size())
                   v += solve_d[p2];
            }
            {
                int p1 = pos-wh;
                int p2 = pos+wh;
                if(p1 >= 0)
                   v += solve_d[p1];
                if(p2 < solve_d.size())
                   v += solve_d[p2];
            }
            v -= new_d[pos];
            v *= inv_d2;
            new_solve_d[pos] = v;
        });
        solve_d.swap(new_solve_d);
    }
    new_d.swap(solve_d);
}

template<typename dist_type>
float cdm_max_displacement_length(dist_type& new_d)
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

template<typename dist_type>
void cdm_constraint(dist_type& d)
{
    tipl::par_for(d.size(),[&](size_t cur_index)
    {
        auto v = d[cur_index];
        if(v[0] > 0.125f)
            v[0] = 0.125;
        if(v[0] < -0.125f)
            v[0] = -0.125;
        if(v[1] > 0.125f)
            v[1] = 0.125;
        if(v[1] < -0.125f)
            v[1] = -0.125;
        if(v[2] > 0.125f)
            v[2] = 0.125;
        if(v[2] < -0.125f)
            v[2] = -0.125;
        d[cur_index] = v;
    });

}


template<typename T,typename U>
__INLINE__ void cdm_smooth_imp(T& d,U& dd,size_t cur_index)
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
    dd[cur_index] = v;
}
template<typename dist_type>
void cdm_smooth(dist_type& d,float smoothing)
{
    if(smoothing == 0.0f)
        return;
    dist_type dd(d.shape());
    tipl::par_for(d.size(),[&](size_t cur_index)
    {
        cdm_smooth_imp(d,dd,cur_index);
    });
    multiply_constant(dd,smoothing/6.0f);
    multiply_constant(d,(1.0f-smoothing));
    add(d,dd);
}


template<typename image_type,typename dist_type,typename terminate_type>
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
        float r = cdm2(rIt,rIt2,rIs,rIs2,d,inv_d,terminated,param2);
        d *= 2.0f;
        inv_d *= 2.0f;
        upsample_with_padding(d,geo);
        upsample_with_padding(inv_d,geo);
        if(param.resolution > 1.0f)
            return r;
    }
    image_type Js,Js2;// transformed I
    dist_type new_d(It.shape()),new_d2(It2.shape());// new displacements
    float theta = 0.0;


    std::deque<float> r,iter;
    for (unsigned int index = 0;index < param.iterations && !terminated;++index)
    {
        compose_displacement(Is,d,Js);
        // dJ(cJ-I)
        r.push_back(cdm_get_gradient(Js,It,new_d));
        if(has_dual)
        {
            compose_displacement(Is2,d,Js2);
            r.back() += cdm_get_gradient(Js2,It2,new_d2);
            r.back() *= 0.5f;
        }

        iter.push_back(index);
        if(!cdm_improved(r,iter))
            break;
        if(has_dual)
            add(new_d,new_d2);
        // solving the poisson equation using Jacobi method
        cdm_solve_poisson(new_d,terminated);

        if(theta == 0.0f)
            theta = cdm_max_displacement_length(new_d);
        if(theta == 0.0f)
            break;
        multiply_constant(new_d,param.speed/theta);
        //cdm_constraint(new_d);
        accumulate_displacement(d,new_d);
        cdm_smooth(d,param.smoothing);
        invert_displacement_imp(d,inv_d,2);
    }
    invert_displacement_imp(d,inv_d);
    return r.front();
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
