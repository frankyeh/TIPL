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
#include "../segmentation/otsu.hpp"
#include "../morphology/morphology.hpp"
#include "../filter/sobel.hpp"
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
    template<typename ImageType1,typename ImageType2,typename TransformType>
    double operator()(const ImageType1& Ifrom,const ImageType2& Ito,const TransformType& transform,int)
    {
        if(Ifrom.size() > Ito.size())
        {
            auto trans(transform);
            trans.inverse();
            return (*this)(Ito,Ifrom,trans,0);
        }
        tipl::image<ImageType1::dimension,typename ImageType1::value_type> y(Ifrom.shape());
        tipl::resample(Ito,y,transform);
        float c = tipl::correlation(Ifrom.begin(),Ifrom.end(),y.begin());
        return -c*c;
    }
};


#ifdef __CUDACC__

template<typename T,typename V,typename U>
__global__ void mutual_information_cuda_kernel(T from,T to,V trans,U mutual_hist)
{
    constexpr int bandwidth = 6;
    constexpr int his_bandwidth = 64;
    TIPL_FOR(index,from.size())
    {
        tipl::pixel_index<3> pos(index,from.shape());
        tipl::vector<3> v;
        trans(pos,v);
        unsigned char to_index = 0;
        tipl::estimate<tipl::interpolation::linear>(to,v,to_index);
        atomicAdd(mutual_hist.begin() +
                  (uint32_t(from[index]) << bandwidth) +to_index,1);
    }
}

template<typename T,typename U>
__global__ void mutual_information_cuda_kernel2(T from8_hist,T mutual_hist,U mu_log_mu)
{
    constexpr int bandwidth = 6;
    constexpr int his_bandwidth = 64;
    int32_t to8=0;
    for(int i=0; i < his_bandwidth; ++i)
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
    device_image<3,unsigned char> from8;
    device_image<3,unsigned char> to8;
    std::mutex init_mutex;
    int device = 0;
    static constexpr int bandwidth = 6;
    static constexpr int his_bandwidth = 64;
public:
    template<typename ImageType,typename TransformType>
    double operator()(const ImageType& from_raw,const ImageType& to_raw,const TransformType& trans,int thread_id = 0)
    {
        using DeviceImageType = device_image<3,typename ImageType::value_type>;
        {
            std::scoped_lock<std::mutex> lock(init_mutex);
            if (from8_hist.empty() || to_raw.size() != to8.size() || from_raw.size() != from8.size())
            {
                cudaGetDevice(&device);
                to8.resize(to_raw.shape());
                normalize_upper_lower2(DeviceImageType(to_raw),to8,his_bandwidth-1);

                host_image<3,unsigned char> host_from8,host_to8;
                host_vector<int32_t> host_from8_hist;

                host_from8.resize(from_raw.shape());
                normalize_upper_lower2(from_raw,host_from8,his_bandwidth-1);
                histogram(host_from8,host_from8_hist,0,his_bandwidth-1,his_bandwidth);

                from8_hist = host_from8_hist;
                from8 = host_from8;

            }
            else
                cudaSetDevice(device);
        }

        device_vector<int32_t> mutual_hist(his_bandwidth*his_bandwidth);
        TIPL_RUN(mutual_information_cuda_kernel,from_raw.size())
                                (tipl::make_shared(from8),
                                 tipl::make_shared(to8),
                                 trans,
                                 tipl::make_shared(mutual_hist));
        if(cudaPeekAtLastError() != cudaSuccess)
            throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
        device_vector<double> mu_log_mu(his_bandwidth*his_bandwidth);
        mutual_information_cuda_kernel2<<<his_bandwidth,his_bandwidth>>>(
                        tipl::make_shared(from8_hist),
                        tipl::make_shared(mutual_hist),
                        tipl::make_shared(mu_log_mu));
        return -sum(mu_log_mu);

    }
};

#endif//__CUDACC__


struct mutual_information
{
    typedef double value_type;
    unsigned int band_width;
    unsigned int his_bandwidth;
    std::vector<unsigned int> from_hist;
    std::vector<unsigned char> from,to;
    std::mutex init_mutex;
public:
    mutual_information(unsigned int band_width_ = 6):band_width(band_width_),his_bandwidth(1 << band_width_) {}
public:
    template<typename ImageType1,typename ImageType2,typename TransformType>
    double operator()(const ImageType1& from_,const ImageType2& to_,const TransformType& transform,int)
    {
        {
            std::scoped_lock<std::mutex> lock(init_mutex);
            if (from_hist.empty() || to_.size() != to.size() || from_.size() != from.size())
            {
                to.resize(to_.size());
                from.resize(from_.size());
                normalize_upper_lower2(to_,to,his_bandwidth-1);
                normalize_upper_lower2(from_,from,his_bandwidth-1);
                histogram(from,from_hist,0,his_bandwidth-1,his_bandwidth);
            }
        }

        // obtain the histogram
        unsigned int thread_count = tipl::available_thread_count();

        tipl::shape<2> geo(his_bandwidth,his_bandwidth);
        std::vector<tipl::image<2,uint32_t> > mutual_hist(thread_count);
        for(int i = 0;i < mutual_hist.size();++i)
            mutual_hist[i].resize(geo);

        auto pto = tipl::make_image(to.data(),to_.shape());

        tipl::par_for(tipl::begin_index(from_.shape()),tipl::end_index(from_.shape()),
                       [&](const pixel_index<ImageType1::dimension>& index,int id)
        {
            if(id >= thread_count)
                id = 0;
            tipl::vector<3> pos;
            transform(index,pos);
            unsigned char to_index = 0;
            tipl::estimate<tipl::interpolation::linear>(pto,pos,to_index);
            mutual_hist[id][(uint32_t(from[index.index()]) << band_width) + uint32_t(to_index)]++;
        });

        for(int i = 1;i < mutual_hist.size();++i)
            tipl::add(mutual_hist[0],mutual_hist[i]);

        // calculate the cost
        {
            double sum = 0.0;
            std::vector<uint32_t> to_hist(his_bandwidth);
            for (tipl::pixel_index<2> index(geo);index < geo.size();++index)
                to_hist[index.x()] += mutual_hist[0][index.index()];
            for (tipl::pixel_index<2> index(geo);index < geo.size();++index)
            {
                double mu = mutual_hist[0][index.index()];
                if (mu == 0.0f)
                    continue;
                sum += mu*std::log(mu/double(from_hist[index.y()])/double(to_hist[index.x()]));
            }
            return -sum;
        }
    }
};


enum reg_type {none = 0,translocation = 1,rotation = 2,rigid_body = 3,scaling = 4,translocation_scaling = 5,rigid_scaling = 7,tilt = 8,affine = 15};
enum cost_type{corr,mutual_info};

const float narrow_bound[8] = {0.2f,-0.2f,      0.1f,-0.1f,    1.2f,0.8f,  0.05f,-0.05f};
const float reg_bound[8] =    {0.75f,-0.75f,    0.3f,-0.3f,    1.5f,0.7f,  0.15f,-0.15f};
const float large_bound[8] =  {1.0f,-1.0f,      1.2f,-1.2f,    2.0f,0.5f,  0.5f,-0.5f};

template<int dim,typename value_type>
class linear_reg_param{
    static const int dimension = dim;
    using transform_type = affine_transform<float>;
    using vs_type = tipl::vector<dim>;
    using image_type = image<dim,value_type>;
    using pointer_image_type = const_pointer_image<dim,value_type>;
public:
    std::vector<pointer_image_type> from,to;
    tipl::vector<3> from_vs,to_vs;
    transform_type arg_upper,arg_lower;
    transform_type& arg_min;
    reg_type type = affine;
public:
    unsigned int count = 0;
    double precision = 0.001;
    bool line_search = true;
    size_t max_iterations = 10;
private:
    void update_bound(transform_type& upper,transform_type& lower,reg_type type)
    {
        const int check_reg[transform_type::total_size] =
                                  {translocation,translocation,translocation,
                                   rotation,rotation,rotation,
                                   scaling,scaling,scaling,
                                   tilt,tilt,tilt};
        for (unsigned int index = 0; index < transform_type::total_size; ++index)
            if(!(type & check_reg[index]))
                upper[index] = lower[index] = arg_min[index];
    }
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
            from(rhs.from),to(rhs.to),arg_min(rhs.arg_min),
            from_vs(rhs.from_vs),to_vs(rhs.to_vs)
    {
        from_vs = rhs.from_vs;
        to_vs = rhs.to_vs;
        arg_upper = rhs.arg_upper;
        arg_lower = rhs.arg_lower;
        type = rhs.type;
        count = rhs.count;
        precision = rhs.precision;
        line_search = rhs.line_search;
        max_iterations = rhs.max_iterations;
    }
    void set_bound(const float* bound = reg_bound,bool absolute = true)
    {
        if(bound == narrow_bound)
            line_search = false;
        if(absolute)
        {
            arg_upper.clear();
            arg_lower.clear();
        }
        else
            arg_upper = arg_lower = arg_min;

        if (type & translocation)
            for (unsigned int index = 0; index < dimension; ++index)
            {
                float range = std::max<float>(from[0].shape()[index]*from_vs[index],
                                              to[0].shape()[index]*to_vs[index])*0.5f;
                arg_upper[index] += range*bound[0];
                arg_lower[index] += range*bound[1];
            }
        if (type & rotation)
            for (unsigned int index = dimension; index < dimension + dimension; ++index)
            {
                arg_upper[index] += 3.14159265358979323846f*bound[2]*(index == 0 ? 2.0f:1.0f);
                arg_lower[index] += 3.14159265358979323846f*bound[3]*(index == 0 ? 2.0f:1.0f);
            }

        if (type & scaling)
            for (unsigned int index = dimension + dimension; index < dimension+dimension+dimension; ++index)
            {
                arg_upper[index] *= bound[4];
                arg_lower[index] *= bound[5];
            }

        if (type & tilt)
            for (unsigned int index = dimension + dimension + dimension; index < transform_type::total_size; ++index)
            {
                arg_upper[index] += bound[6];
                arg_lower[index] += bound[7];
            }

    }
    template<typename cost_type,typename terminated_type>
    float optimize(std::vector<std::shared_ptr<cost_type> > cost_fun,terminated_type&& is_terminated)
    {
        std::vector<reg_type> reg_list;
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
        double optimal_value;
        if(type == affine)
            max_iterations += 20;

        auto fun = [&](const transform_type& new_param,int thread_id = 0)
        {
            ++count;
            std::vector<float> costs(cost_fun.size());
            tipl::par_for(cost_fun.size(),[&](size_t i)
            {
                costs[i] = (*cost_fun[i].get())(from[i],to[i],
                    tipl::transformation_matrix<double>(new_param,from[i].shape(),from_vs,to[i].shape(),to_vs),thread_id);
            },cost_fun.size());
            return tipl::sum(costs);
        };
        optimal_value = fun(arg_min);
        for(auto cur_type : reg_list)
        {
            if(is_terminated())
                break;
            transform_type upper(arg_upper),lower(arg_lower);
            update_bound(upper,lower,cur_type);
            if(line_search)
            {
                tipl::optimization::line_search(arg_min.begin(),arg_min.end(),
                                                 upper.begin(),lower.begin(),fun,optimal_value,is_terminated);
                tipl::optimization::quasi_newtons_minimize(arg_min.begin(),arg_min.end(),
                                                       upper.begin(),lower.begin(),fun,optimal_value,is_terminated,
                                                       precision);
            }
            else
                tipl::optimization::gradient_descent(arg_min.begin(),arg_min.end(),
                                                     upper.begin(),lower.begin(),fun,optimal_value,is_terminated,
                                                     precision,max_iterations);
        }
        if(!line_search)
            tipl::optimization::gradient_descent(arg_min.begin(),arg_min.end(),
                                                 arg_upper.begin(),arg_lower.begin(),fun,optimal_value,is_terminated,
                                                 precision,max_iterations);

        return optimal_value;
    }
    template<typename cost_type,typename terminated_type>
    __INLINE__ float optimize(terminated_type&& is_terminated)
    {
        std::vector<std::shared_ptr<cost_type> > cost_fun;
        auto size = (from.size() > to.size() ? to.size() : from.size());
        for(size_t i = 0;i < size;++i)
            cost_fun.push_back(std::make_shared<cost_type>());
        return optimize(cost_fun,is_terminated);
    }
    template<typename cost_type>
    __INLINE__ float optimize(bool& is_terminated)
    {
        return optimize<cost_type>([&](void){return is_terminated;});
    }
    template<typename cost_type,typename terminated_type>
    float optimize_mr(terminated_type&& terminated)
    {
        if(from[0].size() > 64*64*64)
        {
            linear_reg_param low_reso_reg(*this);
            low_reso_reg.down_sampling();
            low_reso_reg.optimize_mr<cost_type>(terminated);
            if(line_search)
                line_search = false;
        }
        return optimize<cost_type>(terminated);
    }
    template<typename cost_type>
    __INLINE__ float optimize_mr(bool& is_terminated)
    {
        return optimize_mr<cost_type>([&](void){return is_terminated;});
    }
};


template<typename T,typename U>
inline auto linear_reg(const std::vector<T>& template_image,tipl::vector<3> template_vs,
                       const std::vector<U>& subject_image,tipl::vector<3> subject_vs,
                       affine_transform<float>& arg_min)
{
    auto reg = std::make_shared<linear_reg_param<T::dimension,typename U::value_type> >(template_image,subject_image,arg_min);
    reg->from_vs = template_vs;
    reg->to_vs = subject_vs;
    return reg;
}

}
}


#endif//IMAGE_REG_HPP
