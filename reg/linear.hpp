#ifndef IMAGE_REG_HPP
#define IMAGE_REG_HPP
#include <limits>
#include <future>
#include <list>
#include <memory>
#include <cstdlib>     /* srand, rand */
#include <ctime>
#include "../numerical/interpolation.hpp"
#include "../numerical/numerical.hpp"
#include "../numerical/basic_op.hpp"
#include "../numerical/transformation.hpp"
#include "../numerical/optimization.hpp"
#include "../numerical/statistics.hpp"
#include "../numerical/resampling.hpp"
#ifdef __CUDACC__
#include "../cu.hpp"
#endif

namespace tipl
{

namespace reg
{

struct correlation
{
    typedef double value_type;
    template<typename T,typename U,typename V>
    __INLINE__ double operator()(const T& Ifrom,const U& Ito,const V& transform)
    {
        if(Ifrom.size() > Ito.size())
        {
            auto trans(transform);
            trans.inverse();
            return (*this)(Ito,Ifrom,trans);
        }
        typename T::buffer_type y(Ifrom.shape());
        tipl::resample(Ito,y,transform);
        float c = tipl::correlation(Ifrom.begin(),Ifrom.end(),y.begin());
        return -c*c;
    }
};

constexpr unsigned int mi_band_width = 6;
constexpr unsigned int mi_his_bandwidth = (1 << mi_band_width);





#ifdef __CUDACC__

template<typename T,typename V,typename U>
__global__ void mutual_information_cuda_kernel(T from,T to,V trans,U mutual_hist)
{
    TIPL_FOR(index,from.size())
    {
        tipl::pixel_index<T::dimension> pos(index,from.shape());
        tipl::vector<T::dimension> v;
        trans(pos,v);
        unsigned char to_index = 0;
        tipl::estimate<tipl::interpolation::linear>(to,v,to_index);
        atomicAdd(mutual_hist.begin() +
                  (uint32_t(from[index]) << mi_band_width) +to_index,1);
    }
}

template<typename T,typename U>
__global__ void mutual_information_cuda_kernel2(T from8_hist,T mutual_hist,U mu_log_mu)
{
    int32_t to8=0;
    for(int i=0; i < mi_his_bandwidth; ++i)
        to8 += mutual_hist[threadIdx.x + i*blockDim.x];

    size_t index = threadIdx.x + blockDim.x*blockIdx.x;

    // if mutual_hist is not zero,
    // the corresponding from8_hist and to8_hist won't be zero
    mu_log_mu[index] = mutual_hist[index] ?
        mutual_hist[index]*std::log(mutual_hist[index]/double(from8_hist[blockIdx.x])/double(to8))
            : 0.0;
}



struct mutual_information_cuda
{
    typedef double value_type;
    device_vector<int32_t> from8_hist;
    device_vector<unsigned char> from8,to8;
    std::mutex init_mutex;
public:
    template<typename ImageType,typename TransformType>
    double operator()(const ImageType& from_raw,const ImageType& to_raw,const TransformType& trans)
    {
        using DeviceImageType = device_image<ImageType::dimension,typename ImageType::value_type>;
        {
            std::scoped_lock<std::mutex> lock(init_mutex);
            if (from8_hist.empty() || to_raw.size() != to8.size() || from_raw.size() != from8.size())
            {
                to8.resize(to_raw.size());
                normalize_upper_lower2(DeviceImageType(to_raw),to8,mi_his_bandwidth-1);

                host_vector<unsigned char> host_from8;
                host_vector<int32_t> host_from8_hist;

                host_from8.resize(from_raw.size());
                normalize_upper_lower2(from_raw,host_from8,mi_his_bandwidth-1);
                histogram(host_from8,host_from8_hist,0,mi_his_bandwidth-1,mi_his_bandwidth);

                from8_hist = host_from8_hist;
                from8 = host_from8;

            }
        }

        device_vector<int32_t> mutual_hist(mi_his_bandwidth*mi_his_bandwidth);
        TIPL_RUN(mutual_information_cuda_kernel,from_raw.size())
                                (tipl::make_image(from8.data(),from_raw.shape()),
                                 tipl::make_image(to8.data(),to_raw.shape()),
                                 trans,
                                 tipl::make_shared(mutual_hist));
        if(cudaPeekAtLastError() != cudaSuccess)
            throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
        device_vector<double> mu_log_mu(mi_his_bandwidth*mi_his_bandwidth);
        mutual_information_cuda_kernel2<<<mi_his_bandwidth,mi_his_bandwidth>>>(
                        tipl::make_shared(from8_hist),
                        tipl::make_shared(mutual_hist),
                        tipl::make_shared(mu_log_mu));
        return -sum(mu_log_mu);

    }
};
#endif

template<typename T,typename U,typename V,typename W>
inline void get_mutual_info(T& mutual_hist_all,const U& to,const V& from,const W& trans)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__

        #endif
    }
    else
    {
        std::vector<T> mutual_hist(max_thread_count);
        for(auto& each : mutual_hist)
            each.resize(mutual_hist_all.shape());
        tipl::par_for<sequential_with_id>(tipl::begin_index(from.shape()),tipl::end_index(from.shape()),
                                       [&](const auto& index,int id)
        {
            tipl::vector<U::dimension> pos;
            trans(index,pos);
            unsigned char to_index = 0;
            tipl::estimate<tipl::interpolation::linear>(to,pos,to_index);
            mutual_hist[id][(uint32_t(from[index.index()]) << mi_band_width) + uint32_t(to_index)]++;
        });
        for(int i = 1;i < mutual_hist.size();++i)
            tipl::add(mutual_hist[0],mutual_hist[i]);
        mutual_hist_all.swap(mutual_hist[0]);
    }
}
template<typename T,typename U>
double get_mutual_info_sum(const T& mutual_hist,const U& from_hist)
{
    double sum = 0.0;
    std::vector<uint32_t> to_hist(mi_his_bandwidth);
    for (tipl::pixel_index<2> index(mutual_hist.shape());index < mutual_hist.size();++index)
        to_hist[index.x()] += mutual_hist[index.index()];
    for (tipl::pixel_index<2> index(mutual_hist.shape());index < mutual_hist.size();++index)
    {
        double mu = mutual_hist[index.index()];
        if (mu == 0.0f)
            continue;
        sum += mu*std::log(mu/double(from_hist[index.y()])/double(to_hist[index.x()]));
    }
    return sum;
}


struct mutual_information
{
    typedef double value_type;
    std::vector<unsigned int> from_hist;
    std::vector<unsigned char> from,to;
    std::mutex init_mutex;
public:
    template<typename ImageType1,typename ImageType2,typename TransformType>
    double operator()(const ImageType1& from_,const ImageType2& to_,const TransformType& transform)
    {
        {
            std::scoped_lock<std::mutex> lock(init_mutex);
            if (from_hist.empty() || to_.size() != to.size() || from_.size() != from.size())
            {
                to.resize(to_.size());
                from.resize(from_.size());
                normalize_upper_lower2(to_,to,mi_his_bandwidth-1);
                normalize_upper_lower2(from_,from,mi_his_bandwidth-1);
                histogram(from,from_hist,0,mi_his_bandwidth-1,mi_his_bandwidth);
            }
        }

        // obtain the histogram
        tipl::image<2,uint32_t> mutual_hist(tipl::shape<2>(mi_his_bandwidth,mi_his_bandwidth));
        get_mutual_info(mutual_hist,
                        make_image(to.data(),to_.shape()),
                        make_image(from.data(),from_.shape()),transform);

        return -get_mutual_info_sum(mutual_hist,from_hist);
    }
};


enum reg_type {none = 0,translocation = 1,rotation = 2,rigid_body = 3,scaling = 4,translocation_scaling = 5,rigid_scaling = 7,tilt = 8,affine = 15};
enum cost_type{corr,mutual_info};

const float narrow_bound[8] = {0.2f,-0.2f,      0.1f,-0.1f,    1.2f,0.8f,  0.05f,-0.05f};
const float reg_bound[8] =    {0.75f,-0.75f,    0.3f,-0.3f,    1.5f,0.7f,  0.15f,-0.15f};
const float large_bound[8] =  {1.0f,-1.0f,      1.2f,-1.2f,    2.0f,0.5f,  0.5f,-0.5f};

template<int dim,typename value_type,typename prog_type = void,typename out_type = void>
class linear_reg_param{
    static const int dimension = dim;
    using transform_type = affine_transform<float,dimension>;
    using vs_type = tipl::vector<dim>;
    using image_type = image<dim,value_type>;
    using pointer_image_type = const_pointer_image<dimension,value_type>;
public:
    std::vector<pointer_image_type> from,to;
    tipl::vector<dimension> from_vs,to_vs;
    transform_type arg_upper,arg_lower;
    transform_type& arg_min;
public:
    unsigned int count = 0,prog = 0,max_prog = 0;
    double precision = 0.001;
    bool line_search = true;
    size_t max_iterations = 10;
    std::vector<reg_type> reg_list = {translocation,translocation_scaling,rigid_scaling,affine};
private:
    std::vector<image_type> buffer;
    void down_sampling(void)
    {
        buffer.resize(from.size()+to.size());
        std::vector<pointer_image_type> from_to_list(buffer.size());
        tipl::par_for(buffer.size(),[&](size_t id)
        {
            downsample_with_padding(id < from.size() ? from[id]:to[id-from.size()],buffer[id]);
            from_to_list[id] = pointer_image_type(&buffer[id][0],buffer[id].shape());
        },buffer.size());
        from = std::vector<pointer_image_type>(from_to_list.begin(),from_to_list.begin()+from.size());
        to = std::vector<pointer_image_type>(from_to_list.begin()+from.size(),from_to_list.end());
        from_vs *= 2.0f;
        to_vs *= 2.0f;
    }
public:
    linear_reg_param(const std::vector<pointer_image_type>& from_,
                     const std::vector<pointer_image_type>& to_,transform_type& arg_min_):from(from_),to(to_),arg_min(arg_min_){}
    linear_reg_param(const linear_reg_param& rhs):
        from(rhs.from),to(rhs.to),from_vs(rhs.from_vs),to_vs(rhs.to_vs),
        arg_upper(rhs.arg_upper),arg_lower(rhs.arg_lower),arg_min(rhs.arg_min),
        count(rhs.count),max_prog(rhs.max_prog),
        precision(rhs.precision),line_search(rhs.line_search),
        max_iterations(rhs.max_iterations),reg_list(rhs.reg_list)
    {
    }
    void update_bound(transform_type& upper,transform_type& lower,reg_type type)
    {
        for (unsigned int index = 0; index < dimension; ++index)
        {
            if (!(type & translocation))
            {
                arg_upper.translocation[index] = arg_min.translocation[index];
                arg_lower.translocation[index] = arg_min.translocation[index];
            }
            if (!(type & scaling))
            {
                arg_upper.scaling[index] = arg_min.scaling[index];
                arg_lower.scaling[index] = arg_min.scaling[index];
            }
        }
        for (unsigned int index = 0; index < (dim == 3 ? 3 : 1); ++index)
        {
            if (!(type & rotation))
            {
                arg_upper.rotation[index] = arg_min.rotation[index];
                arg_lower.rotation[index] = arg_min.rotation[index];
            }
            if (!(type & tilt))
            {
                arg_upper.affine[index] = arg_min.affine[index];
                arg_lower.affine[index] = arg_min.affine[index];
            }
        }
    }
    void set_bound(reg_type type,const float* bound = reg_bound,bool absolute = true)
    {
        if(type == translocation)
            reg_list = {translocation};
        if(type == rotation)
            reg_list = {rotation};
        if(type == rigid_body)
            reg_list = {translocation,rigid_body};
        if(type == scaling)
            reg_list = {scaling};
        if(type == rigid_scaling)
            reg_list = {translocation,translocation_scaling,rigid_scaling};
        if(type == affine)
            reg_list = {translocation,translocation_scaling,rigid_scaling,affine};

        if(bound == narrow_bound)
            line_search = false;
        if(absolute)
        {
            arg_upper.clear();
            arg_lower.clear();
        }
        else
            arg_upper = arg_lower = arg_min;

        for (unsigned int index = 0; index < dimension; ++index)
        {
            if (type & translocation)
            {
                float range = std::max<float>(from[0].shape()[index]*from_vs[index],
                                              to[0].shape()[index]*to_vs[index])*0.5f;
                arg_upper.translocation[index] += range*bound[0];
                arg_lower.translocation[index] += range*bound[1];
            }
            if (type & scaling)
            {
                arg_upper.scaling[index] *= bound[4];
                arg_lower.scaling[index] *= bound[5];
            }
        }
        for (unsigned int index = 0; index < (dim == 3 ? 3 : 1); ++index)
        {
            if (type & rotation)
            {
                arg_upper.rotation[index] += 3.14159265358979323846f*bound[2]*(index == 0 ? 2.0f:1.0f);
                arg_lower.rotation[index] += 3.14159265358979323846f*bound[3]*(index == 0 ? 2.0f:1.0f);
            }
            if (type & tilt)
            {
                arg_upper.affine[index] += bound[6];
                arg_lower.affine[index] += bound[7];
            }
        }
    }
    template<typename cost_type,typename terminated_type>
    float optimize(std::vector<std::shared_ptr<cost_type> > cost_fun,terminated_type&& is_terminated)
    {

        double optimal_value;
        if(reg_list.back() == affine)
            max_iterations += 20;


        auto fun = [&](const transform_type& new_param)
        {
            ++count;
            float cost = 0.0f;
            for(size_t i = 0;i < cost_fun.size();++i)
                cost += (*cost_fun[i].get())(from[i],to[i],tipl::transformation_matrix<float,dim>(new_param,from[i].shape(),from_vs,to[i].shape(),to_vs));
            if constexpr(!std::is_void<out_type>::value)
            {
                out_type() << new_param;
                out_type() << "cost:" << optimal_value;
            }
            return cost;
        };
        optimal_value = fun(arg_min);
        for(size_t iter = 0;iter < reg_list.size();++iter)
        {
            auto cur_type = reg_list[iter];

            if constexpr(!std::is_void<prog_type>::value)
                prog_type()(prog++,max_prog);
            if constexpr(!std::is_void<out_type>::value)
                out_type() << "optimal cost:" << optimal_value;

            if(is_terminated())
                break;
            transform_type upper(arg_upper),lower(arg_lower);
            update_bound(upper,lower,cur_type);
            if(line_search)
            {
                if constexpr(!std::is_void<out_type>::value)
                    out_type() << "line search";
                tipl::optimization::line_search(arg_min.begin(),arg_min.end(),
                                                 upper.begin(),lower.begin(),fun,optimal_value,is_terminated);
                if constexpr(!std::is_void<out_type>::value)
                    out_type() << "quasi_newtons_minimize";
                tipl::optimization::quasi_newtons_minimize(arg_min.begin(),arg_min.end(),
                                                       upper.begin(),lower.begin(),fun,optimal_value,is_terminated,
                                                       precision);
            }
            else
            {
                if constexpr(!std::is_void<out_type>::value)
                    out_type() << "gradient_descent";
                tipl::optimization::gradient_descent(arg_min.begin(),arg_min.end(),
                                                     upper.begin(),lower.begin(),fun,optimal_value,is_terminated,
                                                     precision,max_iterations);
            }
        }
        if(!line_search)
            tipl::optimization::gradient_descent(arg_min.begin(),arg_min.end(),
                                                 arg_upper.begin(),arg_lower.begin(),fun,optimal_value,is_terminated,
                                                 precision,max_iterations);
        if constexpr(!std::is_void<out_type>::value)
            out_type() << "end cost:" << optimal_value;
        return optimal_value;
    }
    template<typename cost_type,typename terminated_type>
    __INLINE__ float optimize(terminated_type&& is_terminated)
    {
        std::vector<std::shared_ptr<cost_type> > cost_fun;
        auto size = (from.size() > to.size() ? to.size() : from.size());
        for(size_t i = 0;i < size;++i)
            cost_fun.push_back(std::make_shared<cost_type>());
        return optimize<cost_type>(cost_fun,std::forward<terminated_type>(is_terminated));
    }
    template<typename cost_type>
    __INLINE__ float optimize(bool& is_terminated)
    {
        max_prog += reg_list.size();
        return optimize<cost_type>([&](void){return is_terminated;});
    }

    template<typename cost_type,typename terminated_type>
    float optimize_mr(terminated_type&& terminated)
    {
        if constexpr(!std::is_void<out_type>::value)
        {
            out_type() << "resolution:" << from_vs;
            out_type() << "size:" << from[0].shape();
        }
        if(from[0].size() > (dim == 3 ? 64*64*64 : 64*64))
        {
            if constexpr(!std::is_void<out_type>::value)
                out_type() << "try lower resolution first";

            max_prog += reg_list.size();
            linear_reg_param low_reso_reg(*this);
            low_reso_reg.down_sampling();
            low_reso_reg.optimize_mr<cost_type>(std::forward<terminated_type>(terminated));
            max_prog = low_reso_reg.max_prog;
            prog = low_reso_reg.prog;
            if(line_search)
                line_search = false;
        }
        return optimize<cost_type>(std::forward<terminated_type>(terminated));
    }
    template<typename cost_type>
    __INLINE__ float optimize_mr(bool& is_terminated)
    {
        return optimize_mr<cost_type>([&](void){return is_terminated;});
    }
};


template<typename prog_type = void,typename out_type = void,typename T,typename U>
inline auto linear_reg(const std::vector<T>& template_image,tipl::vector<T::dimension> template_vs,
                       const std::vector<U>& subject_image,tipl::vector<T::dimension> subject_vs,
                       affine_transform<float,T::dimension>& arg_min)
{
    auto reg = std::make_shared<linear_reg_param<T::dimension,typename U::value_type,prog_type,out_type> >(template_image,subject_image,arg_min);
    reg->from_vs = template_vs;
    reg->to_vs = subject_vs;
    return reg;
}

}
}


#endif//IMAGE_REG_HPP
