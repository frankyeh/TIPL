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

namespace tipl
{

namespace reg
{
struct square_error
{
    typedef double value_type;
    template<typename ImageType,typename TransformType>
    double operator()(const ImageType& Ifrom,const ImageType& Ito,const TransformType& transform)
    {
        const unsigned int dim = ImageType::dimension;
        tipl::shape<dim> geo(Ifrom.shape());
        double error = 0.0;
        tipl::vector<dim,double> pos;
        for (tipl::pixel_index<dim> index(geo);index < geo.size();++index)
        {
            transform(index,pos);
            double to_pixel = 0;
            if (estimate(Ito,pos,to_pixel) && to_pixel != 0)
                to_pixel -= Ifrom[index.index()];
            else
                to_pixel = Ifrom[index.index()];
            error += to_pixel*to_pixel;

        }
        return error;
    }
};
struct negative_product
{
    typedef double value_type;
    template<typename ImageType,typename TransformType>
    double operator()(const ImageType& Ifrom,const ImageType& Ito,const TransformType& transform)
    {
        const unsigned int dim = ImageType::dimension;
        tipl::shape<dim> geo(Ifrom.shape());
        double error = 0.0;
        tipl::vector<dim,double> pos;
        for (tipl::pixel_index<dim> index(geo);index < geo.size();++index)
        if(Ifrom[index.index()])
        {
            transform(index,pos);
            double to_pixel = 0;
            if (estimate(Ito,pos,to_pixel) && to_pixel != 0)
                error -= to_pixel*Ifrom[index.index()];
        }
        return error;
    }
};
struct correlation
{
    typedef double value_type;
    template<typename ImageType,typename TransformType>
    double operator()(const ImageType& Ifrom,const ImageType& Ito,const TransformType& transform)
    {
        tipl::shape<ImageType::dimension> geo(Ifrom.shape());
        tipl::image<ImageType::dimension,typename ImageType::value_type> y(geo);
        tipl::resample_mt(Ito,y,transform);
        float c = tipl::correlation(Ifrom.begin(),Ifrom.end(),y.begin());
        return -c*c;
    }
};

template<typename image_type,typename transform_type>
struct mt_correlation
{
    typedef double value_type;
    std::list<std::shared_ptr<std::future<void> > > threads;
    std::vector<unsigned char> status;
    const image_type* I1;
    const image_type* I2;
    image_type Y;
    transform_type T;
    double mean_from;
    double sd_from;
    bool end;
    mt_correlation(int){}
    mt_correlation(void):end(false),status(std::thread::hardware_concurrency()),I1(0)
    {

    }
    ~mt_correlation(void)
    {
        end = true;
        for(auto& i:threads)
            i->wait();
    }
    void evaluate(unsigned int id)
    {
        while(!end)
        {
            if(status[id] == 1)
            {
                unsigned int size = I1->size();
                unsigned int thread_size = (size/status.size())+1;
                unsigned int from_size = id*thread_size;
                unsigned int to_size = std::min<unsigned int>(size,(id+1)*thread_size);
                tipl::shape<image_type::dimension> geo(I1->shape());
                for (tipl::pixel_index<image_type::dimension> index(from_size,geo);
                     index < to_size;++index)
                {
                    tipl::vector<image_type::dimension,double> pos;
                    T(index,pos);
                    tipl::estimate(*I2,pos,Y[index.index()]);
                }
                status[id] = 2;
            }
            if(id == 0)
                return;
        }
    }

    double operator()(const image_type& Ifrom,const image_type& Ito,
                      const transform_type& transform)
    {
        if(!I1)
        {
            I1 = &Ifrom;
            I2 = &Ito;
            mean_from = tipl::mean(Ifrom.begin(),Ifrom.end());
            sd_from = tipl::standard_deviation(Ifrom.begin(),Ifrom.end(),mean_from);
            Y.resize(Ifrom.shape());
        }
        T = transform;
        image_type y(Ifrom.shape());
        Y.swap(y);
        std::fill(status.begin(),status.end(),1);
        if(threads.empty())
            for(unsigned int index = 1;index < status.size();++index)
                threads.push_back(std::make_shared<std::future<void> >(std::async(std::launch::async,
                                                                                  [this,index](){evaluate(index);})));
        evaluate(0);
        for(unsigned int index = 1;index < status.size();++index)
            if(status[index] == 1)
                --index;
        double mean_to = tipl::mean(Y.begin(),Y.end());
        double sd_to = tipl::standard_deviation(Y.begin(),Y.end(),mean_to);
        if(sd_from == 0 || sd_to == 0)
            return 0;
        float c = tipl::covariance(Ifrom.begin(),Ifrom.end(),Y.begin(),mean_from,mean_to)/sd_from/sd_to;
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
public:
    mutual_information(unsigned int band_width_ = 6):band_width(band_width_),his_bandwidth(1 << band_width_) {}
public:
    template<typename ImageType,typename TransformType>
    double operator()(const ImageType& from_,const ImageType& to_,const TransformType& transform)
    {
        if (from_hist.empty() || to_.size() != to.size() || from_.size() != from.size())
        {
            to.resize(to_.size());
            from.resize(from_.size());
            tipl::normalize(to_.begin(),to_.end(),to.begin(),his_bandwidth-1);
            tipl::normalize(from_.begin(),from_.end(),from.begin(),his_bandwidth-1);
            tipl::histogram(from,from_hist,0,his_bandwidth-1,his_bandwidth);
        }


        // obtain the histogram
        tipl::shape<ImageType::dimension> geo(from_.shape());
        unsigned int thread_count = std::thread::hardware_concurrency();


        std::vector<tipl::image<2,double> > mutual_hist(thread_count);
        std::vector<std::vector<double> > to_hist(thread_count);
        for(int i = 0;i < thread_count;++i)
        {
            mutual_hist[i].resize(tipl::shape<2>(his_bandwidth,his_bandwidth));
            to_hist[i].resize(his_bandwidth);
        }

        tipl::par_for(tipl::begin_index(geo),tipl::end_index(geo),
                       [&](const pixel_index<ImageType::dimension>& index,int id)
        {
            tipl::interpolator::linear<ImageType::dimension> interp;
            unsigned int from_index = ((unsigned int)from[index.index()]) << band_width;
            tipl::vector<ImageType::dimension,float> pos;
            transform(index,pos);
            if (!interp.get_location(to_.shape(),pos))
            {
                to_hist[id][0] += 1.0;
                mutual_hist[id][from_index] += 1.0;
            }
            else
                for (unsigned int i = 0; i < tipl::interpolator::linear<ImageType::dimension>::ref_count; ++i)
                {
                    double weighting = double(interp.ratio[i]);
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
            tipl::shape<2> geo(mutual_hist[0].shape());
            for (tipl::pixel_index<2> index(geo);index < geo.size();++index)
            {
                double mu = mutual_hist[0][index.index()];
                if (mu == 0.0)
                    continue;
                sum += mu*std::log(mu/double(from_hist[index.y()])/to_hist[0][index.x()]);
            }
            return -sum;
        }
    }
};

template<typename fun_type>
struct faster
{
    typedef typename fun_type::value_type value_type;
    fun_type fun;
    template<typename ImageType,typename TransformType>
    double operator()(const ImageType& Ifrom,const ImageType& Ito,const TransformType& T)
    {
        if(Ifrom.size() < Ito.size())
            return fun(Ifrom,Ito,T);
        else
        {
            TransformType iT = T;
            iT.inverse();
            return fun(Ito,Ifrom,iT);
        }
    }
};


template<typename image_type,
         typename vs_type,
         typename transform_type,
         typename fun_type>
class fun_adoptor{
public:
    const image_type& from;
    const image_type& to;
    const vs_type& from_vs;
    const vs_type& to_vs;
    fun_type fun;
    unsigned int count = 0;
    typedef typename fun_type::value_type value_type;
public:
    fun_adoptor(const image_type& from_,const vs_type& from_vs_,
                const image_type& to_,const vs_type& to_vs_):
        from(from_),to(to_),from_vs(from_vs_),to_vs(to_vs_){}
    template<typename param_type>
    float operator()(const param_type& new_param)
    {
        transform_type affine(new_param);
        tipl::transformation_matrix<typename transform_type::value_type> T(affine,from.shape(),from_vs,to.shape(),to_vs);
        ++count;
        return fun(from,to,T);
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



template<typename image_type,typename vs_type,typename transform_type,typename CostFunctionType,typename teminated_class>
float linear(const image_type& from,const vs_type& from_vs,
             const image_type& to  ,const vs_type& to_vs,
             transform_type& arg_min,
             reg_type base_type,
             CostFunctionType,
             teminated_class& terminated,
             double precision = 0.01,bool line_search = true,const float* bound = reg_bound)
{
    tipl::reg::fun_adoptor<image_type,vs_type,transform_type,CostFunctionType> fun(from,from_vs,to,to_vs);
    transform_type upper,lower;
    tipl::reg::get_bound(from,to,arg_min,upper,lower,base_type,bound);
    reg_type reg_list[4] = {translocation,rigid_body,rigid_scaling,affine};
    double optimal_value = fun(arg_min);
    for(int type = 0;type < 4 && reg_list[type] <= base_type && !terminated;++type)
    {
        tipl::reg::get_bound(from,to,arg_min,upper,lower,reg_list[type],bound);
        if(line_search)
            tipl::optimization::line_search(arg_min.begin(),arg_min.end(),
                                             upper.begin(),lower.begin(),fun,optimal_value,terminated);

        tipl::optimization::quasi_newtons_minimize(arg_min.begin(),arg_min.end(),
                                                    upper.begin(),lower.begin(),fun,optimal_value,terminated,
                                                    reg_list[type] == base_type ? precision/8.0f : precision);
    }
    return optimal_value;
}
/*
 *  This linear version use only gradient descent
 *
 */
template<typename image_type,typename vs_type,typename transform_type,typename CostFunctionType,typename teminated_class>
double linear2(const image_type& from,const vs_type& from_vs,
             const image_type& to  ,const vs_type& to_vs,
             transform_type& arg_min,
             tipl::reg::reg_type base_type,
             CostFunctionType,
             teminated_class& terminated,
             double precision = 0.001,const float* bound = tipl::reg::reg_bound)
{
    tipl::reg::fun_adoptor<image_type,vs_type,transform_type,CostFunctionType> fun(from,from_vs,to,to_vs);
    transform_type upper,lower;
    tipl::reg::get_bound(from,to,arg_min,upper,lower,base_type,bound);
    double optimal_value = fun(arg_min);
    tipl::optimization::line_search(arg_min.begin(),arg_min.end(),
                                         upper.begin(),lower.begin(),fun,optimal_value,terminated);
    tipl::optimization::gradient_descent(arg_min.begin(),arg_min.end(),
                                         upper.begin(),lower.begin(),fun,optimal_value,terminated,precision);
    return optimal_value;
}

template<typename image_type,typename vs_type,typename transform_type,typename CostFunctionType,typename teminated_class>
float linear_mr(const image_type& from,const vs_type& from_vs,
                const image_type& to  ,const vs_type& to_vs,
                transform_type& arg_min,
                reg_type base_type,
                CostFunctionType cost_type,
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
        image<image_type::dimension,typename image_type::value_type> from_r,to_r;
        tipl::vector<image_type::dimension> from_vs_r(from_vs),to_vs_r(to_vs);
        downsample_with_padding(from,from_r);
        downsample_with_padding(to,to_r);
        from_vs_r *= 2.0;
        to_vs_r *= 2.0;
        transform_type arg_min_r(arg_min);
        arg_min_r.downsampling();
        linear_mr(from_r,from_vs_r,to_r,to_vs_r,arg_min_r,base_type,cost_type,terminated,precision,bound);
        arg_min_r.upsampling();
        arg_min = arg_min_r;
        if(terminated)
            return 0.0;
        line_search = false;
    }
    return linear(from,from_vs,to,to_vs,arg_min,base_type,cost_type,terminated,precision,line_search,bound);
}

template<typename image_type,typename vs_type,typename TransType,typename CostFunctionType,typename teminated_class>
float two_way_linear_mr(const image_type& from,const vs_type& from_vs,
                            const image_type& to,const vs_type& to_vs,
                            TransType& T,
                            reg_type base_type,
                            CostFunctionType cost1,
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
            CostFunctionType cost2;
            if(arg)
                tipl::reg::linear_mr(from,from_vs,to,to_vs,*arg,base_type,cost2,terminated,0.01,bound);
            else
                tipl::reg::linear_mr(from,from_vs,to,to_vs,arg1,base_type,cost2,terminated,0.01,bound);
        }
        else
            tipl::reg::linear_mr(to,to_vs,from,from_vs,arg2,base_type,cost1,terminated,0.01,bound);
    },2);


    TransType T1(arg == 0 ? arg1:*arg,from.shape(),from_vs,to.shape(),to_vs);
    TransType T2(arg2,to.shape(),to_vs,from.shape(),from_vs);
    T2.inverse();
    float cost = 0.0f;
    if(CostFunctionType()(from,to,T2) < CostFunctionType()(from,to,T1))
    {
        cost = tipl::reg::linear(to,to_vs,from,from_vs,arg2,base_type,cost1,terminated,0.001f,false,bound);
        TransType T22(arg2,to.shape(),to_vs,from.shape(),from_vs);
        T22.inverse();
        T = T22;
    }
    else
    {
        if(arg)
            cost = tipl::reg::linear(from,from_vs,to,to_vs,*arg,base_type,cost1,terminated,0.001f,false,bound);
        else
            cost = tipl::reg::linear(from,from_vs,to,to_vs,arg1,base_type,cost1,terminated,0.001f,false,bound);
        T = TransType(arg == 0 ? arg1:*arg,from.shape(),from_vs,to.shape(),to_vs);
    }
    return cost;
}


}
}


#endif//IMAGE_REG_HPP
