#ifndef IMAGE_REG_HPP
#define IMAGE_REG_HPP
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
    template<typename T,typename U,typename V>
    double operator()(const T& from,const U& to,const V& transform)
    {
        if(from.size() > to.size())
        {
            auto trans(transform);
            trans.inverse();
            return (*this)(to,from,trans);
        }
        float c = tipl::correlation(from,tipl::resample(to,from.shape(),transform));
        return 1.0f-c*c;
    }
};
#ifdef __CUDACC__

template <int dim,template <typename...> typename stype = std::vector>
struct correlation_cuda
{
    image<dim,unsigned char,stype> from,to;
    std::mutex init_mutex;
    template<typename T,typename U,typename V>
    double operator()(const T& from_,const U& to_,const V& transform)
    {
        if(from_.size() > to_.size())
        {
            auto trans(transform);
            trans.inverse();
            return (*this)(to_,from_,trans);
        }
        {
            std::scoped_lock<std::mutex> lock(init_mutex);
            if (to_.size() != to.size() || from_.size() != from.size())
            {
                if constexpr(std::is_same_v<unsigned char,typename T::value_type>)
                {
                    from = from_;
                    to = to_;
                }
                else
                {
                    using device_image_type = device_image<T::dimension,typename T::value_type>;
                    to.resize(to_.shape());
                    normalize_upper_lower2(device_image_type(to_),to,255.99f);

                    from.resize(from_.shape());
                    normalize_upper_lower2(device_image_type(from_),from_,255.99f);

                }
            }
        }
        float c = tipl::correlation(from,tipl::resample(to,from.shape(),transform));
        return 1.0f-c*c;
    }
};




#endif
constexpr unsigned int mi_band_width = 8;
constexpr unsigned int mi_his_bandwidth = (1 << mi_band_width);

#ifdef __CUDACC__
template<typename T,typename U,typename V>
__global__ void mutual_information_cuda_kernel2(T mutual_hist,U from8_hist,V mu_log_mu)
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
#endif
template<typename T,typename U>
inline double get_mutual_info_mean(const T& mutual_hist,const U& from_hist)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        device_vector<double> mu_log_mu(mi_his_bandwidth*mi_his_bandwidth);
        mutual_information_cuda_kernel2<<<mi_his_bandwidth,mi_his_bandwidth>>>(
                                tipl::make_shared(mutual_hist),
                                tipl::make_shared(from_hist),
                                tipl::make_shared(mu_log_mu));
        return mean(mu_log_mu);
        #endif
    }
    else
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
        return sum/float(mutual_hist.size());
    }
}

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
        atomicAdd(mutual_hist.begin() + (uint32_t(from[index]) << mi_band_width) +to_index,1);
    }
}

#endif

template<typename T,typename U,typename V,typename W>
inline void get_mutual_info(T& mutual_hist_all,const U& to,const V& from,const W& trans)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(mutual_information_cuda_kernel,from.size())
                                (from,to,trans,
                                 tipl::make_shared(mutual_hist_all));
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


template <int dim,template <typename...> typename stype = std::vector>
struct mutual_information
{
    typedef double value_type;
    stype<unsigned int> from_hist;
    image<dim,unsigned char,stype> from,to;
    std::mutex init_mutex;
public:
    template<typename ImageType1,typename ImageType2,typename TransformType>
    double operator()(const ImageType1& from_,const ImageType2& to_,const TransformType& transform)
    {
        {
            std::scoped_lock<std::mutex> lock(init_mutex);
            if (from_hist.empty() || to_.size() != to.size() || from_.size() != from.size())
            {
                if constexpr(memory_location<stype<unsigned int> >::at == CUDA)
                {
                    #ifdef __CUDACC__
                    if constexpr(std::is_same_v<unsigned char,typename ImageType1::value_type>)
                    {
                        host_vector<int32_t> host_from_hist;
                        histogram(from_,host_from_hist,0,mi_his_bandwidth-1,mi_his_bandwidth);
                        from_hist = host_from_hist;
                        from = from_;
                        to = to_;
                    }
                    else
                    {
                        using device_image_type = device_image<ImageType1::dimension,typename ImageType1::value_type>;
                        to.resize(to_.shape());
                        normalize_upper_lower2(device_image_type(to_),to,float(mi_his_bandwidth)-0.01f);

                        host_image<dim,unsigned char> host_from;
                        host_vector<int32_t> host_from_hist;

                        host_from.resize(from_.shape());
                        normalize_upper_lower2(from_,host_from,float(mi_his_bandwidth)-0.01f);
                        histogram(host_from,host_from_hist,0,mi_his_bandwidth-1,mi_his_bandwidth);

                        from_hist = host_from_hist;
                        from = host_from;
                    }
                    #endif
                }
                else
                {
                    if constexpr(std::is_same_v<unsigned char,typename ImageType1::value_type>)
                    {
                        to = to_;
                        from = from_;
                    }
                    else
                    {
                        normalize_upper_lower2(to_,to,float(mi_his_bandwidth)-0.01f);
                        normalize_upper_lower2(from_,from,float(mi_his_bandwidth)-0.01f);
                    }
                    histogram(from,from_hist,0,mi_his_bandwidth-1,mi_his_bandwidth);
                }
            }
        }
        image<2,int32_t,stype> mutual_hist(shape<2>(mi_his_bandwidth,mi_his_bandwidth));
        get_mutual_info(mutual_hist,make_shared(to),make_shared(from),transform);
        return -get_mutual_info_mean(mutual_hist,from_hist);
    }
};


enum reg_type {none = 0,translocation = 1,rotation = 2,rigid_body = 3,scaling = 4,translocation_scaling = 5,rigid_scaling = 7,tilt = 8,affine = 15};
enum cost_type{corr = 0,mutual_info = 1};

const float narrow_bound[3][8] = {{0.2f,-0.2f,      0.2f,-0.2f,    1.2f,0.8f,  0.05f,-0.05f},
                                  {0.2f,-0.2f,      0.1f,-0.1f,    1.2f,0.8f,  0.05f,-0.05f},
                                  {0.2f,-0.2f,      0.1f,-0.1f,    1.2f,0.8f,  0.05f,-0.05f}};
const float reg_bound[3][8] =    {{0.75f,-0.75f,    0.4f,-0.4f,    1.5f,0.7f,  0.1f,-0.1f},
                                  {0.75f,-0.75f,    0.25f,-0.25f,  1.5f,0.7f,  0.1f,-0.1f},
                                  {0.75f,-0.75f,    0.25f,-0.25f,  1.5f,0.7f,  0.1f,-0.1f}};
const float large_bound[3][8] =  {{1.0f,-1.0f,      0.5f,-0.5f,    2.0f,0.5f,  0.25f,-0.25f},
                                  {1.0f,-1.0f,      0.4f,-0.4f,    2.0f,0.5f,  0.25f,-0.25f},
                                  {1.0f,-1.0f,      0.4f,-0.4f,    2.0f,0.5f,  0.25f,-0.25f}};

template<int dim,typename value_type,typename out_type = void>
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
    size_t max_iterations = 128;
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
        precision(rhs.precision),
        max_iterations(rhs.max_iterations),reg_list(rhs.reg_list)
    {
    }
    std::pair<transform_type,transform_type> get_current_bound(reg_type type)
    {
        transform_type upper(arg_upper),lower(arg_lower);
        for (unsigned int index = 0; index < dimension; ++index)
        {
            if (!(type & translocation))
            {
                upper.translocation[index] = arg_min.translocation[index];
                lower.translocation[index] = arg_min.translocation[index];
            }
            if (!(type & scaling))
            {
                upper.scaling[index] = arg_min.scaling[index];
                lower.scaling[index] = arg_min.scaling[index];
            }
        }
        for (unsigned int index = 0; index < (dim == 3 ? 3 : 1); ++index)
        {
            if (!(type & rotation))
            {
                upper.rotation[index] = arg_min.rotation[index];
                lower.rotation[index] = arg_min.rotation[index];
            }
            if (!(type & tilt))
            {
                upper.affine[index] = arg_min.affine[index];
                lower.affine[index] = arg_min.affine[index];
            }
        }
        return std::make_pair(std::move(upper),std::move(lower));
    }
    void set_bound(reg_type type,const float bound[3][8] = reg_bound,bool absolute = true)
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
                arg_upper.translocation[index] += range*bound[index][0];
                arg_lower.translocation[index] += range*bound[index][1];
            }
            if (type & scaling)
            {
                arg_upper.scaling[index] *= bound[index][4];
                arg_lower.scaling[index] *= bound[index][5];
            }
        }
        for (unsigned int index = 0; index < (dim == 3 ? 3 : 1); ++index)
        {
            if (type & rotation)
            {
                arg_upper.rotation[index] += 3.14159265358979323846f*bound[index][2];
                arg_lower.rotation[index] += 3.14159265358979323846f*bound[index][3];
            }
            if (type & tilt)
            {
                arg_upper.affine[index] += bound[index][6];
                arg_lower.affine[index] += bound[index][7];
            }
        }
    }
    template<bool line_search,typename cost_type,typename terminated_type>
    float run_optimize(std::vector<std::shared_ptr<cost_type> > cost_fun,
                   terminated_type&& is_terminated)
    {
        double optimal_value;
        auto fun = [&](const transform_type& new_param)
        {
            ++count;
            float cost = 0.0f;
            for(size_t i = 0;i < cost_fun.size();++i)
            {
                tipl::transformation_matrix<float,dim> trans(new_param,from[i].shape(),from_vs,to[i].shape(),to_vs);
                cost += (*cost_fun[i].get())(from[i],to[i],trans);
            }
            cost /= cost_fun.size()*2;
            return cost;
        };
        optimal_value = fun(arg_min);

        for(size_t iter = 0;iter < reg_list.size() && !is_terminated();++iter)
        {
            auto cur_bound = get_current_bound(reg_list[iter]);
            if constexpr(line_search)
            {
                bool gradient_descent_ended = false;
                auto arg_min2 = arg_min;
                auto optimal_value2 = optimal_value;
                std::mutex m;
                size_t search_updated = 0;
                auto check_terminated = [&](void)
                {
                    {
                        std::lock_guard<std::mutex> lock(m);
                        if(optimal_value2 < optimal_value)
                        {
                            optimal_value = optimal_value2;
                            arg_min = arg_min2;
                            ++search_updated;
                        }
                        if(optimal_value < optimal_value2)
                        {
                            optimal_value2 = optimal_value;
                            arg_min2 = arg_min;
                        }
                    }
                    return is_terminated();
                };

                std::thread thread([&](void)
                    {
                        tipl::optimization::line_search(
                        arg_min2.begin(),arg_min2.end(),
                        cur_bound.first.begin(),cur_bound.second.begin(),fun,optimal_value2,
                                [&](){return gradient_descent_ended || check_terminated();});
                    });

                tipl::optimization::gradient_descent(
                    arg_min.begin(),arg_min.end(),
                    cur_bound.first.begin(),cur_bound.second.begin(),fun,optimal_value,check_terminated,
                    precision,max_iterations);
                gradient_descent_ended = true;
                thread.join();

                if constexpr(!std::is_void<out_type>::value)
                    out_type() << "reg:" << int(reg_list[iter])
                               << " search:" << search_updated
                               << " cost:" << optimal_value << " " << arg_min;
            }
            else
            {
                tipl::optimization::gradient_descent(
                    arg_min.begin(),arg_min.end(),
                    cur_bound.first.begin(),cur_bound.second.begin(),fun,optimal_value,is_terminated,
                    precision,max_iterations);
                if constexpr(!std::is_void<out_type>::value)
                    out_type() << "reg:" << int(reg_list[iter])
                               << " cost:" << optimal_value << " " << arg_min;
            }
        }
        return optimal_value;
    }
    template<typename cost_type,bool line_search = false,typename terminated_type>
    float optimize(terminated_type&& is_terminated)
    {
        std::vector<std::shared_ptr<cost_type> > cost_fun;
        auto size = std::min(from.size(),to.size());
        for(size_t i = 0;i < size;++i)
            cost_fun.push_back(std::make_shared<cost_type>());
        return run_optimize<line_search>(cost_fun,std::forward<terminated_type>(is_terminated));
    }
    template<typename cost_type>
    float optimize(bool& is_terminated)
    {
        max_prog += reg_list.size();
        return optimize<cost_type>([&](void){return is_terminated;});
    }

    template<typename cost_type,typename terminated_type>
    float optimize_mr(terminated_type&& terminated)
    {
        if(from[0].size() > (dim == 3 ? 128*128*128 : 128*128))
        {
            max_prog += reg_list.size();
            linear_reg_param low_reso_reg(*this);
            low_reso_reg.down_sampling();
            low_reso_reg.optimize_mr<cost_type>(std::forward<terminated_type>(terminated));
            max_prog = low_reso_reg.max_prog;
            prog = low_reso_reg.prog;
        }
        if constexpr(!std::is_void<out_type>::value)
            out_type() << "optimize at:" << from_vs << " size:" << from[0].shape();
        return optimize<cost_type,true>(std::forward<terminated_type>(terminated));
    }
    template<typename cost_type>
    float optimize_mr(bool& is_terminated)
    {
        return optimize_mr<cost_type>([&](void){return is_terminated;});
    }
};


template<typename out_type = void,typename T,typename U>
inline auto linear_reg(const std::vector<T>& template_image,tipl::vector<T::dimension> template_vs,
                       const std::vector<U>& subject_image,tipl::vector<T::dimension> subject_vs,
                       affine_transform<float,T::dimension>& arg_min)
{
    auto reg = std::make_shared<linear_reg_param<T::dimension,typename U::value_type,out_type> >(template_image,subject_image,arg_min);
    reg->from_vs = template_vs;
    reg->to_vs = subject_vs;
    return reg;
}


template<typename out_type = void,typename image_type,int dim>
tipl::vector<dim> adjust_to_vs(const image_type& from,
               const tipl::vector<dim>& from_vs,
               const image_type& to,
               const tipl::vector<dim>& to_vs)
{
    tipl::vector<dim> from_min,from_max,to_min,to_max;
    tipl::bounding_box(from,from_min,from_max,0);
    tipl::bounding_box(to,to_min,to_max,0);
    from_max -= from_min;
    to_max -= to_min;
    tipl::vector<dim> new_vs(to_vs);
    float r = (to_max[0] > 0.0f) ? from_max[0]*from_vs[0]/(to_max[0]*to_vs[0]) : 1.0f;
    if(r > 1.5f || r < 1.0f/1.5f)
    {
        new_vs *= r;
        if constexpr(!std::is_void<out_type>::value)
            out_type() << "large differences in fov found. adjust voxel size to perform linear registration\n"
                       << "old vs: " << to_vs << " new vs:" << new_vs;
    }
    return new_vs;
}

template<typename image_type>
auto make_list(const image_type& I,const image_type& I2)
{
    auto pI = tipl::make_shared(I);
    if(I2.empty())
        return std::vector<decltype(pI)>({pI});
    auto pI2 = tipl::make_shared(I2);
    return std::vector<decltype(pI)>({pI,pI2});
}
template<typename image_type>
auto make_list(const image_type& I)
{
    auto pI = tipl::make_shared(I);
    return std::vector<decltype(pI)>({pI});
}

#ifdef __CUDACC__
#include "../cu.hpp"
template<bool mr,typename out_type,typename value_type,int dim>
float optimize_mi_cuda(std::shared_ptr<tipl::reg::linear_reg_param<dim,value_type,out_type> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated)
{
    if constexpr(mr)
        return cost_type == tipl::reg::mutual_info ?
                reg->template optimize_mr<tipl::reg::mutual_information<dim,tipl::device_vector> >(terminated) :
                reg->template optimize_mr<tipl::reg::correlation_cuda<dim,tipl::device_vector> >(terminated);
    else
        return cost_type == tipl::reg::mutual_info ?
                reg->template optimize<tipl::reg::mutual_information<dim,tipl::device_vector> >(terminated) :
                reg->template optimize<tipl::reg::correlation_cuda<dim,tipl::device_vector> >(terminated);
}
#else
template<bool mr,typename out_type,typename value_type,int dim>
float optimize_mi_cuda(std::shared_ptr<tipl::reg::linear_reg_param<dim,value_type,out_type> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated);

#endif//__CUDACC__



template<typename out_type,int dim>
inline float linear_refine(std::vector<tipl::const_pointer_image<dim,unsigned char> > from,
                            tipl::vector<dim> from_vs,
                            std::vector<tipl::const_pointer_image<dim,unsigned char> > to,
                            tipl::vector<dim> to_vs,
                            tipl::affine_transform<float,dim>& arg,
                            tipl::reg::reg_type reg_type,
                            bool& terminated,
                            tipl::reg::cost_type cost_type = tipl::reg::mutual_info,
                            bool cuda = true)
{
    auto reg = tipl::reg::linear_reg<out_type>(from,from_vs,to,to_vs,arg);
    reg->set_bound(reg_type,tipl::reg::narrow_bound,false);
    float result = 0;
    if constexpr (tipl::use_cuda && dim == 3)
    {
        if(cuda)
            result = optimize_mi_cuda<false/*no mr*/,out_type>(reg,cost_type,terminated);
    }
    if(!result)
        result = (cost_type == tipl::reg::mutual_info ? reg->template optimize<tipl::reg::mutual_information<dim> >(terminated):

                                                        reg->template optimize<tipl::reg::correlation>(terminated));
    if constexpr(!std::is_void<out_type>::value)
        out_type() << "result: " << arg;
    return result;
}

template<typename out_type,int dim>
float linear(std::vector<tipl::const_pointer_image<dim,unsigned char> > from,
                             tipl::vector<dim> from_vs,
                             std::vector<tipl::const_pointer_image<dim,unsigned char> > to,
                             tipl::vector<dim> to_vs,
                              tipl::affine_transform<float,dim>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float bound[3][8] = tipl::reg::reg_bound,
                              tipl::reg::cost_type cost_type = tipl::reg::mutual_info,
                              bool cuda = true)
{
    if constexpr(!std::is_void<out_type>::value)
        out_type()
                << (reg_type == tipl::reg::affine? "affine" : "rigid body")
                << " registration using "
                << (cost_type == tipl::reg::mutual_info? "mutual info" : "correlation")
                << " on "
                << (cuda ? "gpu":"cpu");
    auto new_to_vs = to_vs;
    if constexpr(dim == 3)
    {
        if(reg_type == tipl::reg::affine)
            new_to_vs = adjust_to_vs<out_type>(from[0],from_vs,to[0],to_vs);
    }
    bool end = false;
    tipl::affine_transform<float,dim> surrogate_arg;
    std::shared_ptr<std::thread> update_arg;
    if(new_to_vs != to_vs)
    {
        update_arg = std::make_shared<std::thread>([&](void)
        {
            while(!end)
            {
                std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                arg = tipl::transformation_matrix<float,dim>(surrogate_arg,from[0],from_vs,to[0],new_to_vs).to_affine_transform(from[0],from_vs,to[0],to_vs);
            }
            arg = tipl::transformation_matrix<float,dim>(surrogate_arg,from[0],from_vs,to[0],new_to_vs).to_affine_transform(from[0],from_vs,to[0],to_vs);
        });
    }
    float result = std::numeric_limits<float>::max();

    auto reg = tipl::reg::linear_reg<out_type>(from,from_vs,to,new_to_vs,(new_to_vs == to_vs) ? arg : surrogate_arg);
    reg->set_bound(reg_type,bound);

    if constexpr (tipl::use_cuda && dim == 3)
    {
        if(cuda)
        {
            do{
                auto cost = (result == std::numeric_limits<float>::max()) ?
                        optimize_mi_cuda<true/*has mr*/,out_type>(reg,cost_type,terminated):
                        optimize_mi_cuda<false/*no mr*/,out_type>(reg,cost_type,terminated);
                if constexpr(!std::is_void<out_type>::value)
                    out_type() << "result cost: " << cost;
                if(cost >= result)
                    break;
                result = cost;
            }while(1);
        }
    }
    if(result == std::numeric_limits<float>::max())
    {
        do{
            auto cost =
                    (result == std::numeric_limits<float>::max()) ?
                    (cost_type == tipl::reg::mutual_info ? reg->template optimize_mr<tipl::reg::mutual_information<dim> >(terminated):
                                                           reg->template optimize_mr<tipl::reg::correlation>(terminated))
                    :
                    (cost_type == tipl::reg::mutual_info ? reg->template optimize<tipl::reg::mutual_information<dim> >(terminated):
                                                           reg->template optimize<tipl::reg::correlation>(terminated));

            if constexpr(!std::is_void<out_type>::value)
                out_type() << "result cost: " << cost;
            if(cost >= result)
                break;
            result = cost;
        }while(1);
    }

    result = linear_refine<out_type>(from,from_vs,to,new_to_vs,(new_to_vs == to_vs) ? arg : surrogate_arg,reg_type,terminated,cost_type,use_cuda);
    end = true;
    if(update_arg.get())
        update_arg->join();
    return result;
}
template<typename out_type = void,int dim>
auto linear(std::vector<tipl::const_pointer_image<dim,unsigned char> > from,
                                          tipl::vector<dim> from_vs,
                                          std::vector<tipl::const_pointer_image<dim,unsigned char> > to,
                                          tipl::vector<dim> to_vs,
                                          tipl::reg::reg_type reg_type,
                                          bool& terminated,
                                          const float bound[3][8] = tipl::reg::reg_bound,
                                          tipl::reg::cost_type cost_type = tipl::reg::mutual_info,
                                          bool use_cuda = true)
{
    tipl::affine_transform<float,dim> arg;
    linear<out_type>(from,from_vs,to,to_vs,arg,reg_type,terminated,bound,cost_type,use_cuda);
    return tipl::transformation_matrix<float,dim>(arg,from[0],from_vs,to[0],to_vs);
}


}
}


#endif//IMAGE_REG_HPP
