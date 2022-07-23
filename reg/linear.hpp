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
    double operator()(const ImageType1& Ifrom,const ImageType2& Ito,const TransformType& transform,int thread = 0)
    {
        if(Ifrom.size() > Ito.size())
        {
            auto trans(transform);
            trans.inverse();
            return (*this)(Ito,Ifrom,trans);
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
    std::vector<unsigned char> from;
    std::vector<unsigned char> to;
    std::mutex init_mutex;
public:
    mutual_information(unsigned int band_width_ = 6):band_width(band_width_),his_bandwidth(1 << band_width_) {}
public:
    template<typename ImageType1,typename ImageType2,typename TransformType>
    double operator()(const ImageType1& from_,const ImageType2& to_,const TransformType& transform,int thread_id = 0)
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
        auto geo = from_.shape();
        unsigned int thread_count = std::thread::hardware_concurrency();


        std::vector<tipl::image<2,float> > mutual_hist(thread_count);
        std::vector<std::vector<float> > to_hist(thread_count);
        for(int i = 0;i < thread_count;++i)
        {
            mutual_hist[i].resize(tipl::shape<2>(his_bandwidth,his_bandwidth));
            to_hist[i].resize(his_bandwidth);
        }

        tipl::par_for(tipl::begin_index(geo),tipl::end_index(geo),
                       [&](const pixel_index<ImageType1::dimension>& index,int id)
        {
            tipl::interpolator::linear<ImageType1::dimension> interp;
            unsigned int from_index = ((unsigned int)from[index.index()]) << band_width;
            tipl::vector<ImageType1::dimension,float> pos;
            transform(index,pos);
            if (!interp.get_location(to_.shape(),pos))
            {
                to_hist[id][0] += 1.0;
                mutual_hist[id][from_index] += 1.0;
            }
            else
                for (unsigned int i = 0; i < tipl::interpolator::linear<ImageType1::dimension>::ref_count; ++i)
                {
                    auto weighting = interp.ratio[i];
                    unsigned int to_index = to[interp.dindex[i]];
                    to_hist[id][to_index] += weighting;
                    mutual_hist[id][from_index+ to_index] += weighting;
                }
        });

        for(int i = 1;i < thread_count;++i)
        {
            tipl::add(mutual_hist[0],mutual_hist[i]);
            tipl::add(to_hist[0],to_hist[i]);
        }

        // calculate the cost
        {
            double sum = 0.0;
            tipl::shape<2> geo(his_bandwidth,his_bandwidth);
            for (tipl::pixel_index<2> index(geo);index < geo.size();++index)
            {
                double mu = mutual_hist[0][index.index()];
                if (mu == 0.0f)
                    continue;
                sum += mu*std::log(mu/double(from_hist[index.y()])/double(to_hist[0][index.x()]));
            }
            return -sum;
        }
    }
};


enum reg_type {none = 0,translocation = 1,rotation = 2,rigid_body = 3,scaling = 4,rigid_scaling = 7,tilt = 8,affine = 15};
enum cost_type{corr,mutual_info};

const float narrow_bound[8] = {0.2f,-0.2f,0.1f, -0.1f, 1.5f,0.9f,0.1f,-0.1f};
const float reg_bound[8] =    {1.0f,-1.0f,0.25f,-0.25f,2.0f,0.5f,0.2f,-0.2f};
const float large_bound[8] =  {1.0f,-1.0f,1.2f, -1.2f, 4.0f,0.2f,0.5f,-0.5f};
template<typename image_type1,typename image_type2,typename vstype1,typename vstype2,typename transform_type>
void get_bound(const image_type1& from,const image_type2& to,
               vstype1 from_vs,vstype2 to_vs,
               const transform_type& trans,
               transform_type& upper_trans,
               transform_type& lower_trans,               
               reg_type type,const float* bound = reg_bound)
{
    const unsigned int dimension = image_type1::dimension;
    upper_trans = trans;
    lower_trans = trans;
    if (type & translocation)
    {
        for (unsigned int index = 0; index < dimension; ++index)
        {
            float range = std::max<float>(std::max<float>(from.shape()[index]*from_vs[index],to.shape()[index]*to_vs[index])*0.5f,
                                          std::fabs((float)from.shape()[index]*from_vs[index]-(float)to.shape()[index]*to_vs[index]));
            upper_trans[index] = range*bound[0];
            lower_trans[index] = range*bound[1];
        }
    }

    if (type & rotation)
    {
        for (unsigned int index = dimension; index < dimension + dimension; ++index)
        {
            upper_trans[index] = 3.14159265358979323846f*bound[2];
            lower_trans[index] = 3.14159265358979323846f*bound[3];
        }
    }

    if (type & scaling)
    {
        for (unsigned int index = dimension + dimension; index < dimension+dimension+dimension; ++index)
        {
            upper_trans[index] = bound[4];
            lower_trans[index] = bound[5];
        }
    }

    if (type & tilt)
    {
        for (unsigned int index = dimension + dimension + dimension; index < transform_type::total_size; ++index)
        {
            upper_trans[index] = bound[6];
            lower_trans[index] = bound[7];
        }
    }
}

template<typename image_type1,typename vs_type1,
         typename image_type2,typename vs_type2,
         typename cost_type>
class fun_adoptor{
public:
    const image_type1& from;
    const image_type2& to;
    const vs_type1& from_vs;
    const vs_type2& to_vs;
    std::shared_ptr<cost_type> fun;
    unsigned int count = 0;
    using value_type = double;
public:
    fun_adoptor(const image_type1& from_,const vs_type1& from_vs_,
                const image_type2& to_,const vs_type2& to_vs_):
                from(from_),to(to_),from_vs(from_vs_),to_vs(to_vs_),fun(new cost_type){}
    template<typename param_type>
    value_type operator()(const param_type& new_param,int thread_id = 0)
    {
        ++count;
        return (*fun.get())(from,to,
        tipl::transformation_matrix<typename param_type::value_type>(new_param,from.shape(),from_vs,to.shape(),to_vs),thread_id);
    }
};

template<typename cost_type,
         typename image_type1,typename vs_type1,
         typename image_type2,typename vs_type2>
__INLINE__ fun_adoptor<image_type1,vs_type1,image_type2,vs_type2,cost_type>
make_functor(const image_type1& from,const vs_type1& from_vs,const image_type2& to,const vs_type2& to_vs)
{
    return fun_adoptor<image_type1,vs_type1,image_type2,vs_type2,cost_type>(from,from_vs,to,to_vs);
}

template<typename CostFunctionType,typename image_type1,typename vs_type1,
         typename image_type2,typename vs_type2,
         typename transform_type,typename function>
float linear(const image_type1& from,const vs_type1& from_vs,
             const image_type2& to  ,const vs_type2& to_vs,
             transform_type& arg_min,
             reg_type rtype,
             function&& is_terminated,
             double precision = 0.01,
             bool line_search = true,
             const float* bound = reg_bound,
             size_t iterations = 3)
{
    reg_type reg_list[4] = {translocation,rigid_body,rigid_scaling,affine};
    auto fun = make_functor<CostFunctionType>(from,from_vs,to,to_vs);
    double optimal_value;
    if(rtype == affine)
        iterations += 2;
    optimal_value = fun(arg_min);
    transform_type upper,lower;
    if(line_search)
    for(int type = 0;type < 4 && reg_list[type] <= rtype && !is_terminated();++type)
    {
        tipl::reg::get_bound(from,to,from_vs,to_vs,arg_min,upper,lower,reg_list[type],bound);
        tipl::optimization::line_search_mt(arg_min.begin(),arg_min.end(),
                                             upper.begin(),lower.begin(),fun,optimal_value,is_terminated);
        tipl::optimization::quasi_newtons_minimize_mt(arg_min.begin(),arg_min.end(),
                                                   upper.begin(),lower.begin(),fun,optimal_value,is_terminated,
                                                   precision);
    }

    tipl::reg::get_bound(from,to,from_vs,to_vs,arg_min,upper,lower,rtype,bound);
    for(size_t i = 0;i < iterations;++i,precision *= 0.5f)
        tipl::optimization::quasi_newtons_minimize_mt(arg_min.begin(),arg_min.end(),
                                                   upper.begin(),lower.begin(),fun,optimal_value,is_terminated,
                                                   precision);
    return optimal_value;
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

template<typename CostFunctionType,typename image_type1,typename vs_type1,
         typename image_type2,typename vs_type2,
         typename transform_type,typename function>
float linear_mr(const image_type1& from,const vs_type1& from_vs,
                const image_type2& to  ,const vs_type2& to_vs,
                transform_type& arg_min,
                reg_type base_type,
                function&& is_terminated,
                double precision = 0.01,
                bool line_search = true,
                const float* bound = reg_bound,
                size_t iterations = 5)
{
    std::list<image<image_type1::dimension,typename image_type1::value_type> > from_buffer;
    std::list<image<image_type1::dimension,typename image_type1::value_type> > to_buffer;

    std::vector<const_pointer_image<image_type1::dimension,typename image_type1::value_type> > from_series;
    std::vector<const_pointer_image<image_type1::dimension,typename image_type1::value_type> > to_series;
    std::vector<vs_type1> from_vs_series;
    std::vector<vs_type2> to_vs_series;

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
    while(!is_terminated())
    {
        result = linear<CostFunctionType>(from_series[from_index],from_vs_series[from_index],
                                 to_series[to_index],to_vs_series[to_index],
                                 arg_min,base_type,is_terminated,precision,line_search,bound,iterations);
        if(from_index == 0) // cost evaluated at "from" space
            break;
        from_index = std::max<int>(0,from_index-1);
        to_index = std::max<int>(0,to_index-1);
        line_search = false;
    }
    return result;
}

}
}


#endif//IMAGE_REG_HPP
