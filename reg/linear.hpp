#ifndef IMAGE_REG_HPP
#define IMAGE_REG_HPP
#include <limits>
#include <future>
#include <list>
#include <memory>
#include <cstdlib>     /* srand, rand */
#include <ctime>
#include "tipl/numerical/interpolation.hpp"
#include "tipl/numerical/numerical.hpp"
#include "tipl/numerical/basic_op.hpp"
#include "tipl/numerical/transformation.hpp"
#include "tipl/numerical/optimization.hpp"
#include "tipl/numerical/statistics.hpp"
#include "tipl/numerical/resampling.hpp"
#include "tipl/segmentation/otsu.hpp"
#include "tipl/morphology/morphology.hpp"

namespace tipl
{

namespace reg
{
    struct square_error
    {
        typedef double value_type;
        template<class ImageType,class TransformType>
        double operator()(const ImageType& Ifrom,const ImageType& Ito,const TransformType& transform)
        {
            const unsigned int dim = ImageType::dimension;
            tipl::geometry<dim> geo(Ifrom.geometry());
            double error = 0.0;
            tipl::vector<dim,double> pos;
            for (tipl::pixel_index<dim> index(geo);index < geo.size();++index)
            {
                transform(index,pos);
                double to_pixel = 0;
                if (estimate(Ito,pos,to_pixel,tipl::linear) && to_pixel != 0)
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
        template<class ImageType,class TransformType>
        double operator()(const ImageType& Ifrom,const ImageType& Ito,const TransformType& transform)
        {
            const unsigned int dim = ImageType::dimension;
            tipl::geometry<dim> geo(Ifrom.geometry());
            double error = 0.0;
            tipl::vector<dim,double> pos;
            for (tipl::pixel_index<dim> index(geo);index < geo.size();++index)
            if(Ifrom[index.index()])
            {
                transform(index,pos);
                double to_pixel = 0;
                if (estimate(Ito,pos,to_pixel,tipl::linear) && to_pixel != 0)
                    error -= to_pixel*Ifrom[index.index()];
            }
            return error;
        }
    };

    struct correlation
    {
        typedef double value_type;
        template<class ImageType,class TransformType>
        double operator()(const ImageType& Ifrom,const ImageType& Ito,const TransformType& transform)
        {
            tipl::geometry<ImageType::dimension> geo(Ifrom.geometry());
            tipl::image<typename ImageType::value_type,ImageType::dimension> y(geo);
            tipl::resample(Ito,y,transform,tipl::linear);
            float c = tipl::correlation(Ifrom.begin(),Ifrom.end(),y.begin());
            return -c*c;
        }
    };

    template<class image_type,class transform_type>
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
            for(unsigned int index = 1;index < status.size();++index)
                threads.push_back(std::make_shared<std::future<void> >(std::async(std::launch::async,
                                                                                  [this,index](){evaluate(index);})));
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
                    tipl::geometry<image_type::dimension> geo(I1->geometry());
                    for (tipl::pixel_index<image_type::dimension> index(from_size,geo);
                         index < to_size;++index)
                    {
                        tipl::vector<image_type::dimension,double> pos;
                        T(index,pos);
                        tipl::estimate(*I2,pos,Y[index.index()],tipl::linear);
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
                Y.resize(Ifrom.geometry());
            }
            T = transform;
            image_type y(Ifrom.geometry());
            Y.swap(y);
            std::fill(status.begin(),status.end(),1);
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
        template<class ImageType,class TransformType>
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
            tipl::geometry<ImageType::dimension> geo(from_.geometry());
            unsigned int thread_count = std::thread::hardware_concurrency();


            std::vector<tipl::image<double,2> > mutual_hist(thread_count);
            std::vector<std::vector<double> > to_hist(thread_count);
            for(int i = 0;i < thread_count;++i)
            {
                mutual_hist[i].resize(tipl::geometry<2>(his_bandwidth,his_bandwidth));
                to_hist[i].resize(his_bandwidth);
            }

            tipl::make_image(&from[0],geo).for_each_mt2([&](unsigned char value,pixel_index<ImageType::dimension> index,int id)
            {
                tipl::interpolation<tipl::linear_weighting,ImageType::dimension> interp;
                unsigned int from_index = ((unsigned int)value) << band_width;
                tipl::vector<ImageType::dimension,float> pos;
                transform(index,pos);
                if (!interp.get_location(to_.geometry(),pos))
                {
                    to_hist[id][0] += 1.0;
                    mutual_hist[id][from_index] += 1.0;
                }
                else
                    for (unsigned int i = 0; i < tipl::interpolation<tipl::linear_weighting,ImageType::dimension>::ref_count; ++i)
                    {
                        float weighting = interp.ratio[i];
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
                float sum = 0.0;
                tipl::geometry<2> geo(mutual_hist[0].geometry());
                for (tipl::pixel_index<2> index(geo);index < geo.size();++index)
                {
                    float mu = mutual_hist[0][index.index()];
                    if (mu == 0.0)
                        continue;
                    sum += mu*std::log(mu/((float)from_hist[index.y()])/to_hist[0][index.x()]);
                }
                return -sum;
            }
        }
    };


    template<class image_type,
             typename vs_type,
             typename param_type,
             typename transform_type,
             typename fun_type>
    class fun_adoptor{
        const image_type& from;
        const image_type& to;
        const vs_type& from_vs;
        const vs_type& to_vs;
        param_type& param;
        fun_type fun;
    public:
        unsigned int cur_dim;
        unsigned int count;
        typedef typename fun_type::value_type value_type;
        typedef typename param_type::value_type param_value_type;
    public:
        fun_adoptor(const image_type& from_,const vs_type& from_vs_,
                    const image_type& to_,const vs_type& to_vs_,param_type& param_):
            from(from_),from_vs(from_vs_),
            to(to_),to_vs(to_vs_),
            param(param_),count(0),cur_dim(0){}
        float operator()(const param_type& new_param)
        {
            transform_type affine(new_param);
            tipl::transformation_matrix<typename transform_type::value_type> T(affine,from.geometry(),from_vs,to.geometry(),to_vs);
            ++count;
            return fun(from,to,T);
        }

        float operator()(param_value_type param_value)
        {
            transform_type affine(param);
            affine[cur_dim] = param_value;
            tipl::transformation_matrix<typename transform_type::value_type> T(affine,from.geometry(),from_vs,to.geometry(),to_vs);
            ++count;
            return fun(from,to,T);
        }
        float operator()(const param_value_type* param)
        {
            transform_type affine(&*param);
            tipl::transformation_matrix<typename transform_type::value_type> T(affine,from.geometry(),from_vs,to.geometry(),to_vs);
            ++count;
            return fun(from,to,T);
        }
    };

enum reg_type {none = 0,translocation = 1,rotation = 2,rigid_body = 3,scaling = 4,rigid_scaling = 7,tilt = 8,affine = 15};
enum cost_type{corr,mutual_info};

const float reg_bound[6] = {0.25f,-0.25f,2.0f,0.5f,0.2f,-0.2f};

template<class image_type1,class image_type2,class transform_type>
void get_bound(const image_type1& from,const image_type2& to,
               const transform_type& trans,
               transform_type& upper_trans,
               transform_type& lower_trans,
               reg_type type,const float* bound = reg_bound)
{
    typedef typename transform_type::value_type value_type;
    const unsigned int dimension = image_type1::dimension;
    upper_trans = trans;
    lower_trans = trans;
    if (type & translocation)
    {
        for (unsigned int index = 0; index < dimension; ++index)
        {
            upper_trans[index] = std::max<float>(std::max<float>(from.geometry()[index],to.geometry()[index])*0.5f,
                                                 std::fabs((float)from.geometry()[index]-(float)to.geometry()[index]));
            lower_trans[index] = -upper_trans[index];
        }
    }

    if (type & rotation)
    {
        for (unsigned int index = dimension; index < dimension + dimension; ++index)
        {
            upper_trans[index] = 3.14159265358979323846f*bound[0];
            lower_trans[index] = 3.14159265358979323846f*bound[1];
        }
    }

    if (type & scaling)
    {
        for (unsigned int index = dimension + dimension; index < dimension+dimension+dimension; ++index)
        {
            upper_trans[index] = bound[2];
            lower_trans[index] = bound[3];
        }
    }

    if (type & tilt)
    {
        for (unsigned int index = dimension + dimension + dimension; index < transform_type::total_size; ++index)
        {
            upper_trans[index] = bound[4];
            lower_trans[index] = bound[5];
        }
    }
}

template<class image_type,class vs_type,class transform_type,class CostFunctionType,class teminated_class>
float linear(const image_type& from,const vs_type& from_vs,
             const image_type& to  ,const vs_type& to_vs,
                transform_type& arg_min,
             reg_type base_type,
             CostFunctionType,
             teminated_class& terminated,
             double precision,int random_search_count = 0,const float* bound = reg_bound)
{
    reg_type reg_list[4] = {translocation,rigid_body,rigid_scaling,affine};
    transform_type upper,lower;
    tipl::reg::fun_adoptor<image_type,vs_type,transform_type,transform_type,CostFunctionType> fun(from,from_vs,to,to_vs,arg_min);
    double optimal_value = fun(arg_min[0]);
    tipl::reg::get_bound(from,to,arg_min,upper,lower,base_type,bound);
    std::default_random_engine gen;

    // translation search on translocation (x,y,z)
    std::uniform_int_distribution<int> un(0,2);
    tipl::par_for(std::thread::hardware_concurrency(),[&](int)
    {
        for(int j = 0;j < random_search_count && !terminated;++j)
        {
            transform_type this_arg_min = arg_min;
            int cur_dim = un(gen);
            if(upper[cur_dim] == lower[cur_dim])
                continue;
            float sd = std::max<float>(std::fabs(upper[cur_dim]-arg_min[cur_dim]),std::fabs(lower[cur_dim]-arg_min[cur_dim]))/2.0f;
            std::normal_distribution<double> distribution(arg_min[cur_dim],sd);
            this_arg_min[cur_dim] = distribution(gen);

            auto this_value = CostFunctionType()(from,to,
                tipl::transformation_matrix<typename transform_type::value_type>(this_arg_min,from.geometry(),from_vs,to.geometry(),to_vs));
            if(this_value < optimal_value)
            {
                arg_min = this_arg_min;
                optimal_value = this_value;
            }
        }
    });

    for(int type = 0;type < 4 && reg_list[type] <= base_type && !terminated;++type)
    {
            tipl::reg::get_bound(from,to,arg_min,upper,lower,reg_list[type]);
            tipl::optimization::gradient_descent(arg_min.begin(),arg_min.end(),
                                                 upper.begin(),lower.begin(),fun,optimal_value,terminated,precision);
    }

    if(!terminated)
        tipl::optimization::gradient_descent(arg_min.begin(),arg_min.end(),
                                         upper.begin(),lower.begin(),fun,optimal_value,terminated,precision*0.1f);
    return optimal_value;
}


template<class image_type,class vs_type,class transform_type,class CostFunctionType,class teminated_class>
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
    int random_search = 0;
    if (*std::max_element(from.geometry().begin(),from.geometry().end()) > 32 &&
        *std::max_element(to.geometry().begin(),to.geometry().end()) > 32)
    {
        //downsampling
        image<typename image_type::value_type,image_type::dimension> from_r,to_r;
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
    }
    else
        random_search = 800/(*std::max_element(from.geometry().begin(),from.geometry().end())+1);
    return linear(from,from_vs,to,to_vs,arg_min,base_type,cost_type,terminated,precision,random_search,bound);
}

template<class image_type,class vs_type,class TransType,class CostFunctionType,class teminated_class>
void two_way_linear_mr(const image_type& from,const vs_type& from_vs,
                            const image_type& to,const vs_type& to_vs,
                            TransType& T,
                            reg_type base_type,
                            CostFunctionType cost_type,
                            teminated_class& terminated,
                            unsigned int thread_count = std::thread::hardware_concurrency(),
                            tipl::affine_transform<typename TransType::value_type>* arg = 0,
                            const float* bound = reg_bound)
{
    tipl::affine_transform<typename TransType::value_type> arg1,arg2;
    tipl::par_for(2,[&](int i){
        if(i)
        {
            if(arg)
                tipl::reg::linear_mr(from,from_vs,to,to_vs,*arg,base_type,cost_type,terminated,0.1,bound);
            else
                tipl::reg::linear_mr(from,from_vs,to,to_vs,arg1,base_type,cost_type,terminated,0.1,bound);
        }
        else
            tipl::reg::linear_mr(to,to_vs,from,from_vs,arg2,base_type,cost_type,terminated,0.1);
    },thread_count);
    TransType T1(arg == 0 ? arg1:*arg,from.geometry(),from_vs,to.geometry(),to_vs);
    TransType T2(arg2,to.geometry(),to_vs,from.geometry(),from_vs);
    T2.inverse();
    if(CostFunctionType()(from,to,T2) < CostFunctionType()(from,to,T1))
    {
        tipl::reg::linear(to,to_vs,from,from_vs,arg2,base_type,cost_type,terminated,0.001f,0,bound);
        TransType T22(arg2,to.geometry(),to_vs,from.geometry(),from_vs);
        T22.inverse();
        T = T22;
    }
    else
    {
        if(arg)
            tipl::reg::linear(from,from_vs,to,to_vs,*arg,base_type,cost_type,terminated,0.001f,0,bound);
        else
            tipl::reg::linear(from,from_vs,to,to_vs,arg1,base_type,cost_type,terminated,0.001f,0,bound);
        T = TransType(arg == 0 ? arg1:*arg,from.geometry(),from_vs,to.geometry(),to_vs);
    }
}


}
}


#endif//IMAGE_REG_HPP
