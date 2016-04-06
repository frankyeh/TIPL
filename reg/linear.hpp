#ifndef IMAGE_REG_HPP
#define IMAGE_REG_HPP
#include <limits>
#include <future>
#include <list>
#include <memory>
#include <cstdlib>     /* srand, rand */
#include <ctime>
#include "image/numerical/interpolation.hpp"
#include "image/numerical/numerical.hpp"
#include "image/numerical/basic_op.hpp"
#include "image/numerical/transformation.hpp"
#include "image/numerical/optimization.hpp"
#include "image/numerical/statistics.hpp"
#include "image/numerical/resampling.hpp"
#include "image/segmentation/otsu.hpp"
#include "image/morphology/morphology.hpp"

namespace image
{

namespace reg
{

    template<class I_type>
    image::vector<3,double> center_of_mass(const I_type& Im)
    {
        image::basic_image<unsigned char,I_type::dimension> mask;
        image::segmentation::otsu(Im,mask);
        image::morphology::smoothing(mask);
        image::morphology::smoothing(mask);
        image::morphology::defragment(mask);
        image::vector<I_type::dimension,double> sum_mass;
        double total_w = 0.0;
        for(image::pixel_index<I_type::dimension> index;
            index.is_valid(mask.geometry());
            index.next(mask.geometry()))
            if(mask[index.index()])
            {
                total_w += 1.0;
                image::vector<3,double> pos(index);
                sum_mass += pos;
            }
        if(total_w == 0)
        {
            for(unsigned char dim = 0;dim < I_type::dimension;++dim)
                sum_mass[dim] = (double)Im.geometry()[dim]/2.0;
            return sum_mass;
        }
        sum_mass /= total_w;
        for(unsigned char dim = 0;dim < I_type::dimension;++dim)
            sum_mass[dim] -= (double)Im.geometry()[dim]/2.0;
        return sum_mass;
    }

    struct square_error
    {
        typedef double value_type;
        template<class ImageType,class TransformType>
        double operator()(const ImageType& Ifrom,const ImageType& Ito,const TransformType& transform)
        {
            const unsigned int dim = ImageType::dimension;
            image::geometry<dim> geo(Ifrom.geometry());
            double error = 0.0;
            image::vector<dim,double> pos;
            for (image::pixel_index<dim> index(geo);index < geo.size();++index)
            {
                transform(index,pos);
                double to_pixel = 0;
                if (estimate(Ito,pos,to_pixel,image::linear) && to_pixel != 0)
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
            image::geometry<dim> geo(Ifrom.geometry());
            double error = 0.0;
            image::vector<dim,double> pos;
            for (image::pixel_index<dim> index(geo);index < geo.size();++index)
            if(Ifrom[index.index()])
            {
                transform(index,pos);
                double to_pixel = 0;
                if (estimate(Ito,pos,to_pixel,image::linear) && to_pixel != 0)
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
            const unsigned int dim = ImageType::dimension;
            image::geometry<dim> geo(Ifrom.geometry());
            ImageType y(geo);
            image::resample(Ito,y,transform,image::linear);
            float c = image::correlation(Ifrom.begin(),Ifrom.end(),y.begin());
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
        mt_correlation(int dummy){}
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
                    image::geometry<image_type::dimension> geo(I1->geometry());
                    for (image::pixel_index<image_type::dimension> index(from_size,geo);
                         index < to_size;++index)
                    {
                        image::vector<image_type::dimension,double> pos;
                        T(index,pos);
                        image::estimate(*I2,pos,Y[index.index()],image::linear);
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
                mean_from = image::mean(Ifrom.begin(),Ifrom.end());
                sd_from = image::standard_deviation(Ifrom.begin(),Ifrom.end(),mean_from);
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
            double mean_to = image::mean(Y.begin(),Y.end());
            double sd_to = image::standard_deviation(Y.begin(),Y.end(),mean_to);
            if(sd_from == 0 || sd_to == 0)
                return 0;
            float c = image::covariance(Ifrom.begin(),Ifrom.end(),Y.begin(),mean_from,mean_to)/sd_from/sd_to;
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
            const unsigned int dimension = ImageType::dimension;
            if (from_hist.empty() || to_.size() != to.size() || from_.size() != from.size())
            {
                to.resize(to_.size());
                from.resize(from_.size());
                image::normalize(to_.begin(),to_.end(),to.begin(),his_bandwidth-1);
                image::normalize(from_.begin(),from_.end(),from.begin(),his_bandwidth-1);
                image::histogram(from,from_hist,0,his_bandwidth-1,his_bandwidth);
            }

            image::basic_image<double,2> mutual_hist((image::geometry<2>(his_bandwidth,his_bandwidth)));
            std::vector<double> to_hist(his_bandwidth);


            // obtain the histogram
            image::geometry<dimension> geo(from_.geometry());
            image::interpolation<image::linear_weighting,dimension> interp;
            image::vector<dimension,double> pos;
            for (image::pixel_index<dimension> index(geo);index < geo.size();++index)
            {
                unsigned int from_index = from[index.index()];
                transform(index,pos);
                if (!interp.get_location(to_.geometry(),pos))
                {
                    to_hist[0] += 1.0;
                    mutual_hist[from_index << band_width] += 1.0;
                }
                else
                    for (unsigned int i = 0; i < image::interpolation<image::linear_weighting,dimension>::ref_count; ++i)
                    {
                        float weighting = interp.ratio[i];
                        unsigned int to_index = to[interp.dindex[i]];
                        to_hist[to_index] += weighting;
                        mutual_hist[(from_index << band_width)+ to_index] += weighting;
                    }
            }

            // calculate the cost
            {
                float sum = 0.0;
                image::geometry<2> geo(mutual_hist.geometry());
                for (image::pixel_index<2> index(geo);index < geo.size();++index)
                {
                    float mu = mutual_hist[index.index()];
                    if (mu == 0.0)
                        continue;
                    sum += mu*std::log(mu/((float)from_hist[index.y()])/to_hist[index.x()]);
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
            image::transformation_matrix<typename transform_type::value_type> T(affine,from.geometry(),from_vs,to.geometry(),to_vs);
            ++count;
            return fun(from,to,T);
        }

        float operator()(param_value_type param_value)
        {
            transform_type affine(param);
            affine[cur_dim] = param_value;
            image::transformation_matrix<typename transform_type::value_type> T(affine,from.geometry(),from_vs,to.geometry(),to_vs);
            ++count;
            return fun(from,to,T);
        }
        float operator()(const param_value_type* param)
        {
            transform_type affine(&*param);
            image::transformation_matrix<typename transform_type::value_type> T(affine,from.geometry(),from_vs,to.geometry(),to_vs);
            ++count;
            return fun(from,to,T);
        }
    };

enum reg_type {none = 0,translocation = 1,rotation = 2,rigid_body = 3,scaling = 4,rigid_scaling = 7,tilt = 8,affine = 15};


template<class image_type1,class image_type2,class transform_type>
void get_bound(const image_type1& from,const image_type2& to,
               const transform_type& trans,
               transform_type& upper_trans,
               transform_type& lower_trans,
               reg_type type)
{
    typedef typename transform_type::value_type value_type;
    const unsigned int dimension = image_type1::dimension;
    upper_trans = trans;
    lower_trans = trans;
    if (type & translocation)
    {
        for (unsigned int index = 0; index < dimension; ++index)
        {
            upper_trans[index] = from.geometry()[index]*0.5;
            lower_trans[index] = -upper_trans[index];
        }
    }

    if (type & rotation)
    {
        for (unsigned int index = dimension; index < dimension + dimension; ++index)
        {
            upper_trans[index] = 3.14159265358979323846*0.25;
            lower_trans[index] = -3.14159265358979323846*0.25;
        }
    }

    if (type & scaling)
    {
        for (unsigned int index = dimension + dimension; index < dimension+dimension+dimension; ++index)
        {
            upper_trans[index] = 1.5;
            lower_trans[index] = 1.0/1.5;
        }
    }

    if (type & tilt)
    {
        for (unsigned int index = dimension + dimension + dimension; index < transform_type::total_size; ++index)
        {
            upper_trans[index] = 0.2;
            lower_trans[index] = -0.2;
        }
    }
}

template<class image_type,class vs_type,class transform_type,class CostFunctionType,class teminated_class>
float linear(const image_type& from,const vs_type& from_vs,
             const image_type& to  ,const vs_type& to_vs,
                    transform_type& arg_min,
                    reg_type base_type,
                    CostFunctionType,
                    teminated_class& terminated,double precision = 0.005)
{
    std::srand(0);
    reg_type reg_list[4] = {translocation,rigid_body,rigid_scaling,affine};
    double optimal_value = 0.0;
    transform_type upper,lower;
    image::reg::fun_adoptor<image_type,vs_type,transform_type,transform_type,CostFunctionType> fun(from,from_vs,to,to_vs,arg_min);
    std::vector<unsigned char> search_count(arg_min.size());
    unsigned int random_search_count = std::sqrt(1/precision);
    for(unsigned char type = 0;type < 4 && reg_list[type] <= base_type && !terminated;++type)
    {
        image::reg::get_bound(from,to,arg_min,upper,lower,reg_list[type]);
        if(type == 0)
            optimal_value = fun(arg_min[0]);
        while(!terminated)
        {
            bool running = false;
            for(fun.cur_dim = 0;fun.cur_dim < arg_min.size() && !terminated;++fun.cur_dim)
                if(upper[fun.cur_dim] != lower[fun.cur_dim] && search_count[fun.cur_dim] < random_search_count)
                {
                    image::optimization::rand_search2(arg_min[fun.cur_dim],
                                                            upper[fun.cur_dim],lower[fun.cur_dim],
                                                            optimal_value,fun);
                    ++search_count[fun.cur_dim];
                    running = true;
                }
            if(!running)
                break;
        }
        if(!terminated)
            image::optimization::graient_descent(arg_min.begin(),arg_min.end(),upper.begin(),lower.begin(),fun,optimal_value,terminated,precision);
        //std::cout << "type=" << reg_list[type] <<" count=" << fun.count << " cost=" << optimal_value << std::endl;
    }
    image::optimization::graient_descent(arg_min.begin(),arg_min.end(),upper.begin(),lower.begin(),fun,optimal_value,terminated,precision/10);
    //std::cout << "count=" << fun.count << " cost=" << optimal_value << std::endl;
    return optimal_value;
}


}
}


#endif//IMAGE_REG_HPP
