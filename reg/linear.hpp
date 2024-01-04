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
        tipl::resample_mt(Ito,y,transform);
        float c = tipl::correlation(Ifrom.begin(),Ifrom.end(),y.begin());
        return -c*c;
    }
};

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
                tipl::normalize_upper_lower(to_.begin(),to_.end(),to.begin(),his_bandwidth-1);
                tipl::normalize_upper_lower(from_.begin(),from_.end(),from.begin(),his_bandwidth-1);
                tipl::histogram(from,from_hist,0,his_bandwidth-1,his_bandwidth);
            }
        }

        // obtain the histogram
        unsigned int thread_count = std::max<int>(1,tipl::available_thread_count);

        tipl::shape<2> geo(his_bandwidth,his_bandwidth);
        std::vector<tipl::image<2,uint32_t> > mutual_hist(thread_count);
        for(int i = 0;i < mutual_hist.size();++i)
            mutual_hist[i].resize(geo);

        auto pto = tipl::make_image(to.data(),to_.shape());

        tipl::par_for(tipl::begin_index(from_.shape()),tipl::end_index(from_.shape()),
                       [&](const pixel_index<ImageType1::dimension>& index,int id)
        {
            tipl::vector<3> pos;
            transform(index,pos);
            unsigned char to_index = 0;
            tipl::estimate<tipl::interpolation::linear>(pto,pos,to_index);
            mutual_hist[id][(uint32_t(from[index.index()]) << band_width) + uint32_t(to_index)]++;
        },thread_count);

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

template<typename image_type1,typename image_type2>
class linear_reg_param{
    static const int dimension = 3;
    using transform_type = affine_transform<float>;
    using vs_type = tipl::vector<3>;
private:
    image_type1 from_buffer;
    image_type2 to_buffer;
public:
    const image_type1& from;
    const image_type2& to;
    transform_type& arg_min;
    tipl::vector<3> from_vs;
    tipl::vector<3> to_vs;
    transform_type arg_upper,arg_lower;
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

    template<typename T,typename U,typename V>
    void linear_mr_get_downsampled_images(std::list<T>& from_buffer,std::vector<U>& from_series,std::vector<V>& vs_series)
    {
        while(from_series.back().size() > 64*64*64)
        {
            from_buffer.push_back(U());
            downsample_with_padding(from_series.back(),from_buffer.back());
            // add one more layer
            from_series.push_back(tipl::make_image(&from_buffer.back()[0],from_buffer.back().shape()));
            vs_series.push_back(vs_series.back()*2.0f);
        }
        // remove those too large
        while(from_series.size() > 1 && from_series.front().width() > 512 && from_series.front().height() > 512)
            from_series.erase(from_series.begin());
    }

public:
    linear_reg_param(const image_type1& from_,const image_type2& to_,
                     transform_type& arg_min_):from(from_),to(to_),arg_min(arg_min_){}
    template<typename rhs_image_type1,typename rhs_image_type2,
             typename std::enable_if<!std::is_same<rhs_image_type1,image_type1>::value ||
                                     !std::is_same<rhs_image_type2,image_type2>::value,bool>::type = true>
    linear_reg_param(const linear_reg_param<rhs_image_type1,rhs_image_type2>& rhs):
            from_buffer(rhs.from),to_buffer(rhs.to),
            from(from_buffer),to(to_buffer),arg_min(rhs.arg_min),
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
        if(reg_bound == narrow_bound)
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
                float range = std::max<float>(from.shape()[index]*from_vs[index],
                                              to.shape()[index]*to_vs[index])*0.5f;
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

    template<typename cost_type>
    __INLINE__ float optimize(std::shared_ptr<cost_type> cost_fun,bool& is_terminated)
    {
        return optimize(cost_fun,[&](void){return is_terminated;});
    }
    template<typename cost_type,typename terminated_type>
    float optimize(std::shared_ptr<cost_type> cost_fun,terminated_type&& is_terminated)
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
            return (*cost_fun.get())(from,to,
                tipl::transformation_matrix<double>(new_param,from.shape(),from_vs,to.shape(),to_vs),thread_id);
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
                tipl::optimization::line_search_mt(arg_min.begin(),arg_min.end(),
                                                 upper.begin(),lower.begin(),fun,optimal_value,is_terminated);
                tipl::optimization::quasi_newtons_minimize_mt(arg_min.begin(),arg_min.end(),
                                                       upper.begin(),lower.begin(),fun,optimal_value,is_terminated,
                                                       precision);
            }
            else
                tipl::optimization::gradient_descent_mt(arg_min.begin(),arg_min.end(),
                                                     upper.begin(),lower.begin(),fun,optimal_value,is_terminated,
                                                     precision,max_iterations);
        }
        if(!line_search)
            tipl::optimization::gradient_descent_mt(arg_min.begin(),arg_min.end(),
                                                 arg_upper.begin(),arg_lower.begin(),fun,optimal_value,is_terminated,
                                                 precision,max_iterations);

        return optimal_value;
    }

    template<typename cost_type>
    __INLINE__ float optimize_mr(std::shared_ptr<cost_type> cost_fun,bool& is_terminated)
    {
        return optimize_mr(cost_fun,[&](void){return is_terminated;});
    }
    template<typename cost_type,typename terminated_type>
    float optimize_mr(std::shared_ptr<cost_type> cost_fun,terminated_type&& is_terminated)
    {
        std::list<image<image_type1::dimension,typename image_type1::value_type> > from_buffer;
        std::list<image<image_type1::dimension,typename image_type1::value_type> > to_buffer;

        std::vector<const_pointer_image<image_type1::dimension,typename image_type1::value_type> > from_series;
        std::vector<const_pointer_image<image_type1::dimension,typename image_type1::value_type> > to_series;
        std::vector<vs_type> from_vs_series;
        std::vector<vs_type> to_vs_series;

        // add original resolution as the first layer
        from_series.push_back(tipl::make_image(&from[0],from.shape()));
        to_series.push_back(tipl::make_image(&to[0],to.shape()));
        from_vs_series.push_back(from_vs);
        to_vs_series.push_back(to_vs);
        // create multiple resolution layers of the original image
        linear_mr_get_downsampled_images(from_buffer,from_series,from_vs_series);
        linear_mr_get_downsampled_images(to_buffer,to_series,to_vs_series);

        int from_index = int(from_series.size())-1;
        int to_index = int(to_series.size())-1;
        float result = 0.0f;
        bool previous_line_search = line_search;
        while(!is_terminated())
        {
            result = optimize(cost_fun,is_terminated);
            if(from_index == 0) // cost evaluated at "from" space
                break;
            from_index = std::max<int>(0,from_index-1);
            to_index = std::max<int>(0,to_index-1);
            line_search = false;
        }
        line_search = previous_line_search;
        return result;
    }
};


template<typename T,typename U>
auto linear_reg(const T& template_image,tipl::vector<3> template_vs,
                const U& subject_image,tipl::vector<3> subject_vs,
                affine_transform<float>& arg_min)
{
    auto reg = std::make_shared<linear_reg_param<T,U> >(template_image,subject_image,arg_min);
    reg->from_vs = template_vs;
    reg->to_vs = subject_vs;
    return reg;
}

}
}


#endif//IMAGE_REG_HPP
