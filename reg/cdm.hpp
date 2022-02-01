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
template<typename pixel_type,size_t dimension>
void cdm_average_img(const std::vector<image<dimension,pixel_type> >& Ji, image<dimension,pixel_type>& J0)
{
    J0 = Ji[0];
    for(unsigned int index = 1;index < Ji.size();++index)
        add(J0.begin(),J0.end(),Ji[index].begin());
    divide_constant(J0.begin(),J0.end(),float(Ji.size()));
}

template<typename pixel_type,size_t dimension>
float cdm_img_dif(const image<dimension,pixel_type>& I0,
                    const image<dimension,pixel_type>& I1)
{
    float value = 0;
    for (int index = 0; index < I0.size(); ++index)
    {
        pixel_type tmp = I0[index]-I1[index];
        value += tmp*tmp;
    }
    return value;
}

template<typename pixel_type,size_t dimension>
float cdm_img_dif(const std::vector<image<dimension,pixel_type> >& Ji,const image<dimension,pixel_type>& J0)
{
    float next_dif = 0;
    for(unsigned int index = 0;index < Ji.size();++index)
        next_dif += cdm_img_dif(J0,Ji[index]);
    return next_dif;
}


template<typename pixel_type,size_t dimension>
float cdm_contrast(const image<dimension,pixel_type>& J0,
                     const image<dimension,pixel_type>& Ji)
{
    float value1 = 0,value2 = 0;
    for (int index = 0; index < J0.size(); ++index)
    {
        float tmp = Ji[index];
        value1 += tmp*J0[index];
        value2 += tmp*tmp;
    }
    if(value2 == 0.0f)
        return 1.0f;
    return value1/value2;
}

template<typename pixel_type,size_t dimension>
void cdm_update_contrast(const std::vector<image<dimension,pixel_type> >& Ji,
                          std::vector<float>& contrast)
{
    image<dimension,pixel_type> J0;
    cdm_average_img(Ji,J0);
    contrast.resize(Ji.size());
    for (unsigned int index = 0; index < Ji.size(); ++index)
        contrast[index] = cdm_contrast(J0,Ji[index]);
    float sum_contrast = std::accumulate(contrast.begin(),contrast.end(),0.0f);
    sum_contrast /= Ji.size();
    divide_constant(contrast.begin(),contrast.end(),sum_contrast);
}


// trim the image size to uniform
template<typename pixel_type,unsigned int dimension,typename crop_type>
void cdm_trim_images(std::vector<image<dimension,pixel_type> >& I,
                      crop_type& crop_from,crop_type& crop_to)
{
    crop_from.resize(I.size());
    crop_to.resize(I.size());
    vector<dimension,int> min_from,max_to;
    for(int index = 0;index < I.size();++index)
    {
        crop(I[index],crop_from[index],crop_to[index]);
        if(index == 0)
        {
            min_from = crop_from[0];
            max_to = crop_to[0];
        }
        else
            for(int dim = 0;dim < dimension;++dim)
            {
                min_from[dim] = std::min(crop_from[index][dim],min_from[dim]);
                max_to[dim] = std::max(crop_to[index][dim],max_to[dim]);
            }
    }
    max_to -= min_from;
    int safe_margin = std::accumulate(max_to.begin(),max_to.end(),0.0f)/float(dimension)/2.0f;
    max_to += safe_margin;
    shape<dimension> geo(max_to.begin());
    // align with respect to the min_from
    for(int index = 0;index < I.size();++index)
    {
        image<dimension,float> new_I(geo);
        vector<dimension,int> pos(crop_from[index]);
        pos -= min_from;
        pos += safe_margin/2;
        draw(I[index],new_I,pos);
        new_I.swap(I[index]);
        crop_from[index] -= pos;
        crop_to[index] = crop_from[index];
        crop_to[index] += geo.begin();
    }

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
        int w = src.width();
        int wh = src.width()*src.height();
        int shift[6];
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
                int s = shift[i];
                par_for(dest.size()-s,[&dest,&src,s](int index){
                    dest[index+s] += src[index];
                });
            }
            else
            {
                int s = -shift[i];
                par_for(dest.size()-s,[&dest,&src,s](int index){
                    dest[index] += src[index+s];
                });
            }
        }
        dest.swap(src);
    }
};

template<typename pixel_type,typename vtor_type,unsigned int dimension,typename terminate_type>
void cdm_group(const std::vector<image<dimension,pixel_type> >& I,// original images
          std::vector<image<dimension,vtor_type> >& d,// displacement field
          float theta,float reg,terminate_type& terminated)
{
    if (I.empty())
        return;
    size_t n = I.size();
    shape<dimension> geo = I[0].shape();

    d.resize(n);
    // preallocated memory so that there will be no memory re-allocation in upsampling
    for (size_t index = 0;index < n;++index)
        d[index].resize(geo);

    // multi resolution
    if (*std::min_element(geo.begin(),geo.end()) > 16)
    {
        //downsampling
        std::vector<image<dimension,pixel_type> > rI(n);
        for (size_t index = 0;index < n;++index)
            downsample_with_padding(I[index],rI[index]);
        cdm_group(rI,d,theta/2,reg,terminated);
        // upsampling deformation
        for (size_t index = 0;index < n;++index)
        {
            upsample_with_padding(d[index],d[index],geo);
            d[index] * 2.0f;
        }
    }
    std::cout << "dimension:" << geo[0] << "x" << geo[1] << "x" << geo[2] << std::endl;
    image<dimension,pixel_type> J0(geo);// the potential template
    std::vector<image<dimension,pixel_type> > Ji(n);// transformed I
    std::vector<image<dimension,vtor_type> > new_d(n);// new displacements
    std::vector<float> contrast(n);
    float current_dif = std::numeric_limits<float>::max();
    for (float dis = 0.5;dis > theta;)
    {
        // calculate Ji
        for (unsigned int index = 0;index < n;++index)
            compose_displacement(I[index],d[index],Ji[index]);


        // calculate contrast
        cdm_update_contrast(Ji,contrast);

        // apply contrast
        multiply(Ji.begin(),Ji.end(),contrast.begin());

        cdm_average_img(Ji,J0);

        float next_dif = cdm_img_dif(Ji,J0);
        if(next_dif >= current_dif)
        {
            dis *= 0.5f;
            std::cout << "detail=(" << dis << ")" << std::flush;
            // roll back
            d.swap(new_d);
            current_dif = std::numeric_limits<float>::max();
            continue;
        }
        std::cout << next_dif;
        current_dif = next_dif;
        // initialize J0
        float max_d = 0;
        for (unsigned int index = 0;index < n && !terminated;++index)
        {
            std::cout << "." << std::flush;

            // gradient of Ji
            gradient_sobel(J0,new_d[index]);

            // dJi*sign(Ji-J0)
            for(unsigned int i = 0;i < Ji[index].size();++i)
                if(Ji[index][i] < J0[i])
                    new_d[index][i] = -new_d[index][i];

            {
                image<dimension,pixel_type> temp;
                jacobian_determinant_dis(d[index],temp);
                compose_displacement(temp,d[index],Ji[index]);
            }

            for(unsigned int i = 0;i < Ji[index].size();++i)
                new_d[index][i] *= Ji[index][i];

            //io::nifti header;
            //header << Ji[index];
            //header.save_to_file("c:/1.nii");

            // solving the poisson equation using Jacobi method
            {
                image<dimension,vtor_type> solve_d(new_d[index].shape());
                for(int iter = 0;iter < 20;++iter)
                {
                    poisson_equation_solver<vtor_type,dimension>()(solve_d);
                    add_mt(solve_d,new_d[index]);
                    divide_constant_mt(solve_d,dimension*2);
                }
                new_d[index].swap(solve_d);
                minus_constant(new_d[index].begin(),new_d[index].end(),new_d[index][0]);
            }

            for (unsigned int i = 0; i < geo.size(); ++i)
                if (new_d[index][i]*new_d[index][i] > max_d)
                    max_d = std::sqrt(new_d[index][i]*new_d[index][i]);
        }
        // calculate the lambda
        float lambda = -dis/max_d;
        for (unsigned int index = 0;index < n && !terminated;++index)
        {
            new_d[index] *= lambda;
            add(new_d[index].begin(),new_d[index].end(),d[index].begin());
        }
        d.swap(new_d);
    }
    std::cout << std::endl;
}



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
    float speed = 0.5f;
    float constraint = 2.0f;
    unsigned int iterations = 200;
    unsigned int min_dimension = 16;
    bool multi_resolution = true;
};


// calculate dJ(cJ-I)
template<typename image_type,typename dis_type>
float cdm_get_gradient(const image_type& Js,const image_type& It,dis_type& new_d)
{
    std::vector<double> accumulated_r2(std::thread::hardware_concurrency());
    tipl::par_for(tipl::begin_index(Js.shape()),tipl::end_index(Js.shape()),
                        [&](const pixel_index<image_type::dimension>& index,int id)
    {
        if(It[index.index()] == 0.0 || Js[index.index()] == 0.0 ||
           It.shape().is_edge(index))
        {
            new_d[index.index()] = typename dis_type::value_type();
            return;
        }
        // calculate gradient
        new_d[index.index()][0] = Js[index.index()+1]-Js[index.index()-1];
        new_d[index.index()][1] = Js[index.index()+Js.width()]-Js[index.index()-Js.width()];
        new_d[index.index()][2] = Js[index.index()+Js.plane_size()]-Js[index.index()-Js.plane_size()];

        typename image_type::value_type Itv[get_window_size<2,image_type::dimension>::value];
        typename image_type::value_type Jsv[get_window_size<2,image_type::dimension>::value];
        get_window_at_width<2>(index,It,Itv);
        auto size = get_window_at_width<2>(index,Js,Jsv);
        float a,b,r2;
        linear_regression(Jsv,Jsv+size,Itv,a,b,r2);
        if(a <= 0.0f)
            new_d[index.index()] = typename dis_type::value_type();
        else
        {
            new_d[index.index()] *= r2*(Js[index.index()]*a+b-It[index.index()]);
            accumulated_r2[id] += r2;
        }
    });
    return std::accumulate(accumulated_r2.begin(),accumulated_r2.end(),0.0)/float(Js.size());
}


// calculate dJ(cJ-I)
template<typename image_type,typename dis_type>
float cdm_get_gradient_abs_dif(const image_type& Js,const image_type& It,dis_type& new_d)
{
    float accumulated_r2 = 0.0f;
    unsigned int r_num = 0;
    gradient_sobel(Js,new_d);
    tipl::par_for(tipl::begin_index(Js.shape()),tipl::end_index(Js.shape()),
    [&](const pixel_index<image_type::dimension>& index)
    {
        if(It[index.index()] == 0.0 ||
           Js[index.index()] == 0.0 ||
           It.shape().is_edge(index))
        {
            new_d[index.index()] = typename dis_type::value_type();
            return;
        }
        auto dif = Js[index.index()]-It[index.index()];
        new_d[index.index()] *= dif;
        accumulated_r2 += dif*dif;
        ++r_num;
    });
    return accumulated_r2/float(r_num);
}


/*
template<typename dis_type,typename terminated_type>
void cdm_solve_poisson(dis_type& new_d,terminated_type& terminated)
{
    float inv_d2 = 0.5f/dis_type::dimension;
    const unsigned int window_size = 3;
    dis_type solve_d(new_d);
    multiply_constant_mt(solve_d,-inv_d2);

    int shift[dis_type::dimension]={0};
    shift[0] = 1;
    for(int i = 1;i < dis_type::dimension;++i)
        shift[i] = shift[i-1]*new_d.shape()[i-1];

    for(int iter = 0;iter < window_size*2 && !terminated;++iter)
    {
        dis_type new_solve_d(new_d.shape());
        par_for(solve_d.size(),[&](int pos)
        {
            for(int d = 0;d < dis_type::dimension;++d)
            {
                int p1 = pos-shift[d];
                int p2 = pos+shift[d];
                if(p1 >= 0)
                   new_solve_d[pos] += solve_d[p1];
                if(p2 < solve_d.size())
                   new_solve_d[pos] += solve_d[p2];
            }
            new_solve_d[pos] -= new_d[pos];
            new_solve_d[pos] *= inv_d2;
        });
        solve_d.swap(new_solve_d);
    }
    minus_constant_mt(solve_d,solve_d[0]);
    new_d.swap(solve_d);
}
*/

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

template<typename dist_type,typename value_type>
void cdm_accumulate_dis(dist_type& d,dist_type& new_d,value_type& theta,float speed)
{
    if(theta == 0.0f)
        par_for(new_d.size(),[&](int i)
        {
           value_type l = new_d[i].length();
           if(l > theta)
               theta = l;
        });
    if(theta == 0.0)
        return;
    new_d *= speed/theta;
    tipl::accumulate_displacement(d,new_d);
}

template<typename dist_type>
void cdm_constraint(dist_type& d,float constraint_length)
{
    size_t shift[dist_type::dimension];
    shift[0] = 1;
    shift[1] = d.width();
    shift[2] = d.plane_size();
    dist_type dd(d.shape());
    tipl::par_for(d.size(),[&](size_t cur_index)
    {
        for(unsigned char dim = 0;dim < 3;++dim)
        {
            size_t cur_index_with_shift = cur_index + shift[dim];
            if(cur_index_with_shift >= d.size())
                break;
            float dis = d[cur_index_with_shift][dim] - d[cur_index][dim];
            if(dis < 0)
                dis *= 0.25f;
            else
            {
                if(dis > constraint_length)
                    dis = 0.25f*(dis-constraint_length);
                else
                    continue;
            }
            dd[cur_index][dim] += dis;
            dd[cur_index_with_shift][dim] -= dis;
        }
    });
    d += dd;
}

template<typename image_type,typename dist_type,typename terminate_type>
float cdm(const image_type& It,
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
    if (*std::min_element(geo.begin(),geo.end()) > param.min_dimension && param.multi_resolution)
    {
        //downsampling
        image_type rIs,rIt;
        downsample_with_padding(It,rIt);
        downsample_with_padding(Is,rIs);
        cdm_param param2 = param;
        param2.resolution /= 2.0f;
        float r = cdm(rIt,rIs,d,terminated,param2);
        d *= 2.0f;
        upsample_with_padding(d,geo);
        if(param.resolution > 1.0f)
            return r;
    }
    image_type Js;// transformed I
    dist_type new_d(d.shape());
    float theta = 0.0;

    std::deque<float> r,iter;
    for (unsigned int index = 0;index < param.iterations && !terminated;++index)
    {
        compose_displacement(Is,d,Js);
        // dJ(cJ-I)
        r.push_back(cdm_get_gradient(Js,It,new_d));
        iter.push_back(index);
        if(!cdm_improved(r,iter))
            break;
        // solving the poisson equation using Jacobi method
        cdm_solve_poisson(new_d,terminated);
        cdm_accumulate_dis(d,new_d,theta,param.speed);
        cdm_constraint(d,param.constraint);
    }
    return r.front();
}


template<typename image_type,typename dist_type,typename terminate_type>
float cdm2(const image_type& It,const image_type& It2,
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
        downsample_with_padding(It,rIt);
        downsample_with_padding(Is,rIs);
        downsample_with_padding(It2,rIt2);
        downsample_with_padding(Is2,rIs2);
        cdm_param param2 = param;
        param2.resolution /= 2.0f;
        float r = cdm2(rIt,rIt2,rIs,rIs2,d,terminated,param2);
        d *= 2.0f;
        upsample_with_padding(d,geo);
        if(param.resolution > 1.0f)
            return r;
    }
    image_type Js,Js2;// transformed I
    dist_type new_d(d.shape()),new_d2(d.shape());// new displacements
    float theta = 0.0;


    std::deque<float> r,iter;
    for (unsigned int index = 0;index < param.iterations && !terminated;++index)
    {
        compose_displacement(Is,d,Js);
        compose_displacement(Is2,d,Js2);
        // dJ(cJ-I)
        r.push_back((cdm_get_gradient(Js,It,new_d)+cdm_get_gradient(Js2,It2,new_d2))*0.5f);
        iter.push_back(index);
        if(!cdm_improved(r,iter))
            break;
        new_d += new_d2;
        // solving the poisson equation using Jacobi method
        cdm_solve_poisson(new_d,terminated);
        cdm_accumulate_dis(d,new_d,theta,param.speed);
        cdm_constraint(d,param.constraint);
    }
    return r.front();
}




}// namespace reg
}// namespace image
#endif // DMDM_HPP
