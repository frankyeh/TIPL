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
        if(from_.size() > to_.size())
        {
            TransformType trans(transform);
            trans.inverse();
            return (*this)(to_,from_,trans);
        }
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
template<typename image_type1,typename image_type2,typename transform_type>
void get_bound(const image_type1& from,const image_type2& to,
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
            float range = std::max<float>(std::max<float>(from.shape()[index],to.shape()[index])*0.5f,
                                          std::fabs((float)from.shape()[index]-(float)to.shape()[index]));
            upper_trans[index] = range*bound[0];
            lower_trans[index] = range*bound[1];
        }
    }

    if (type & rotation)
    {
        for (unsigned int index = dimension; index < dimension + dimension; ++index)
        {
            upper_trans[index] += 3.14159265358979323846f*bound[2];
            lower_trans[index] += 3.14159265358979323846f*bound[3];
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
         typename transform_type,typename teminated_class>
float linear(const image_type1& from,const vs_type1& from_vs,
             const image_type2& to  ,const vs_type2& to_vs,
             transform_type& arg_min,
             reg_type base_type,
             teminated_class& terminated,
             double precision = 0.01,bool line_search = true,const float* bound = reg_bound)
{
    reg_type reg_list[4] = {translocation,rigid_body,rigid_scaling,affine};
    auto fun = make_functor<CostFunctionType>(from,from_vs,to,to_vs);
    auto optimal_value = fun(arg_min);
    for(int type = 0;type < 4 && reg_list[type] <= base_type && !terminated;++type)
    {
        if(!line_search && reg_list[type] != base_type)
            continue;
        transform_type upper,lower;
        tipl::reg::get_bound(from,to,transform_type(),upper,lower,reg_list[type],bound);
        if(line_search)
            tipl::optimization::line_search_mt(arg_min.begin(),arg_min.end(),
                                             upper.begin(),lower.begin(),fun,optimal_value,terminated);

        tipl::optimization::quasi_newtons_minimize_mt(arg_min.begin(),arg_min.end(),
                                                   upper.begin(),lower.begin(),fun,optimal_value,terminated,
                                                   precision);
    }
    return optimal_value;
}

template<typename CostFunctionType,typename image_type1,typename vs_type1,
         typename image_type2,typename vs_type2,
         typename transform_type,typename teminated_class>
float linear_mr(const image_type1& from,const vs_type1& from_vs,
                const image_type2& to  ,const vs_type2& to_vs,
                transform_type& arg_min,
                reg_type base_type,
                teminated_class& terminated,
                double precision = 0.01,
                const float* bound = reg_bound)
{
    // multi resolution
    bool line_search = true;
    if (*std::max_element(from.shape().begin(),from.shape().end()) > 64 &&
        *std::max_element(to.shape().begin(),to.shape().end()) > 64)
    {
        //downsampling
        image<image_type1::dimension,typename image_type1::value_type> from_r;
        image<image_type2::dimension,typename image_type2::value_type> to_r;
        tipl::vector<image_type1::dimension> from_vs_r(from_vs),to_vs_r(to_vs);
        downsample_with_padding(from,from_r);
        downsample_with_padding(to,to_r);
        from_vs_r *= 2.0;
        to_vs_r *= 2.0;
        transform_type arg_min_r(arg_min);
        arg_min_r.downsampling();
        linear_mr<CostFunctionType>(from_r,from_vs_r,to_r,to_vs_r,arg_min_r,base_type,terminated,precision,bound);
        arg_min_r.upsampling();
        arg_min = arg_min_r;
        if(terminated)
            return 0.0;
        line_search = false;
    }
    return linear<CostFunctionType>(from,from_vs,to,to_vs,arg_min,base_type,terminated,precision,line_search,bound);
}

template<typename CostFunctionType,typename image_type,typename vs_type,typename TransType,typename teminated_class>
float two_way_linear_mr(const image_type& from,const vs_type& from_vs,
                            const image_type& to,const vs_type& to_vs,
                            TransType& T,
                            reg_type base_type,
                            teminated_class& terminated,
                            tipl::affine_transform<typename TransType::value_type>* arg = nullptr,
                            const float* bound = reg_bound)
{
    tipl::affine_transform<typename TransType::value_type> arg1,arg2;
    if(arg)
        arg2.translocation[2] = -arg->translocation[2]*from_vs[2]/to_vs[2];

    tipl::par_for(2,[&](int i){
        if(i)
        {
            if(arg)
                linear_mr<CostFunctionType>(from,from_vs,to,to_vs,*arg,base_type,terminated,0.01,bound);
            else
                linear_mr<CostFunctionType>(from,from_vs,to,to_vs,arg1,base_type,terminated,0.01,bound);
        }
        else
            linear_mr<CostFunctionType>(to,to_vs,from,from_vs,arg2,base_type,terminated,0.01,bound);
    },2);


    TransType T1(arg == 0 ? arg1:*arg,from.shape(),from_vs,to.shape(),to_vs);
    TransType T2(arg2,to.shape(),to_vs,from.shape(),from_vs);
    T2.inverse();
    float cost = 0.0f;
    if(CostFunctionType()(from,to,T2) < CostFunctionType()(from,to,T1))
    {
        cost = linear<CostFunctionType>(to,to_vs,from,from_vs,arg2,base_type,terminated,0.001f,false,bound);
        TransType T22(arg2,to.shape(),to_vs,from.shape(),from_vs);
        T22.inverse();
        T = T22;
    }
    else
    {
        if(arg)
            cost = linear<CostFunctionType>(from,from_vs,to,to_vs,*arg,base_type,terminated,0.001f,false,bound);
        else
            cost = linear<CostFunctionType>(from,from_vs,to,to_vs,arg1,base_type,terminated,0.001f,false,bound);
        T = TransType(arg == 0 ? arg1:*arg,from.shape(),from_vs,to.shape(),to_vs);
    }
    return cost;
}


}
}


#endif//IMAGE_REG_HPP
