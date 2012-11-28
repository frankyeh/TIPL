#ifndef IMAGE_REG_HPP
#define IMAGE_REG_HPP
#include <limits>
#include "image/numerical/interpolation.hpp"
#include "image/numerical/numerical.hpp"
#include "image/numerical/basic_op.hpp"
#include "image/numerical/transformation.hpp"
#include "image/numerical/optimization.hpp"

namespace image
{

namespace reg
{


    struct square_error
    {
        template<typename ImageType,typename TransformType>
        double operator()(const ImageType& Ifrom,const ImageType& Ito,const TransformType& transform)
        {
            const unsigned int dim = ImageType::dimension;
            image::geometry<dim> geo(Ifrom.geometry());
            double error = 0.0;
            double pos[dim];
            for (image::pixel_index<dim> index; index.valid(geo); index.next(geo))
            {
                transform(index.begin(),pos);
                double to_pixel;
                if (linear_estimate(Ito,pos,to_pixel))
                {
                    to_pixel -= Ifrom[index.index()];
                    error += to_pixel*to_pixel;
                }
                else
                    error += Ifrom[index.index()]*Ifrom[index.index()];

            }
            return error;
        }
    };


    struct mutual_information
    {
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
            double pos[dimension];
            for (image::pixel_index<dimension> index; index.valid(geo); index.next(geo))
            {
                unsigned int from_index = from[index.index()];
                transform(index.begin(),pos);
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
                for (image::pixel_index<2> index; index.valid(geo); index.next(geo))
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


    template<typename geo_type,typename transform_type>
    void linear_get_trans(
            const geo_type& geo_from,
            const geo_type& geo_to,
            transform_type& T)
    {
        for(int i = 0,index = 0;i < geo_type::dimension;++i)
            for(int j = 0;j < geo_type::dimension;++j,++index)
                T.shift[i] -= T.scaling_rotation[index]*geo_from[j]/2.0;
        for(int i = 0;i < geo_type::dimension;++i)
            T.shift[i] += geo_to[i]/2.0;
    }

    template<typename ImageType,typename CostFunctionType>
    struct cost_function_adoptor
    {
        const ImageType& from;
        const ImageType& to;
        float sampling;
        CostFunctionType& cost_function;
        cost_function_adoptor(const ImageType& from_,const ImageType& to_,
                              CostFunctionType& cost_function_,
                              float sampling_):
            from(from_),to(to_),cost_function(cost_function_),sampling(sampling_){}
        template<typename transform_type>
        double operator()(const transform_type& trans)
        {
            const int dim = transform_type::dimension;
            transformation_matrix<dim,typename transform_type::value_type> T(trans);
            image::multiply_constant(T.shift,T.shift+dim,sampling);
            linear_get_trans(from.geometry(),to.geometry(),T);
            return cost_function(from,to,T);
        }
    };

enum coreg_type {translocation = 1,rotation = 2,scaling = 4,tilt = 8};
const int rigid_body = translocation | rotation;
const int rigid_scaling = translocation | rotation | scaling;
const int affine = translocation | rotation | scaling | tilt;

/**
 usage:
     image::reg::coregistration(image_from,image_to,transformation,
                        image::reg::translocation | image::reg::rotation,
                        image::reg::mutual_information<>(),terminated);

*/
template<typename image_type,typename transform_type,typename CostFunctionType,typename teminated_class>
void linear(const image_type& from,const image_type& to,
                    transform_type& trans,
                    int reg_type,
                    CostFunctionType cost_fun,
                    teminated_class& terminated,
                    float tol = 0.02,
                    float sampling = 1.0)

{
    if(terminated)
        return;
    typedef typename transform_type::value_type value_type;
    const unsigned int dimension = image_type::dimension;
    image::optimization::powell_method<image::optimization::enhanced_brent<value_type,value_type>,transform_type,value_type>
            opti_method(transform_type::total_size);
    for (int index = 0; index < transform_type::total_size; ++index)
            opti_method.search_methods[index].min = opti_method.search_methods[index].max = trans[index];
    if (reg_type & translocation)
        for (unsigned int index = 0; index < dimension; ++index)
        {
            opti_method.search_methods[index].max = from.geometry()[index]/2;
            opti_method.search_methods[index].min = from.geometry()[index]/-2;
        }

    if (reg_type & rotation)
        for (unsigned int index = dimension; index < dimension + dimension; ++index)
        {
            opti_method.search_methods[index].max = 3.14159265358979323846*0.2;
            opti_method.search_methods[index].min = -3.14159265358979323846*0.2;
        }

    if (reg_type & scaling)
        for (unsigned int index = dimension + dimension; index < dimension+dimension+dimension; ++index)
        {
            opti_method.search_methods[index].max = trans[index]*1.2;
            opti_method.search_methods[index].min = trans[index]/1.2;
        }

    if (reg_type & tilt)
        for (unsigned int index = dimension + dimension + dimension; index < transform_type::total_size; ++index)
        {
            opti_method.search_methods[index].max = 0.2;
            opti_method.search_methods[index].min = -0.2;
        }
    if(from.geometry()[0]*sampling > 32)
    {
        linear(from,to,trans,reg_type,cost_fun,terminated,tol,sampling*0.5);
        if(terminated)
            return;
    }
    if(sampling == 1.0)
    {
        cost_function_adoptor<image_type,CostFunctionType> cost_function(from,to,cost_fun,sampling);
        opti_method.minimize(cost_function,trans,terminated,tol);
    }
    else
    {
        image::basic_image<value_type,dimension> from_reduced,to_reduced;
        downsampling(from,from_reduced);
        downsampling(to,to_reduced);
        for(float s = sampling*2.0;s < 0.99;s *= 2.0)
        {
            downsampling(from_reduced);
            downsampling(to_reduced);
        }
        cost_function_adoptor<image::basic_image<value_type,dimension>,CostFunctionType> cost_function(from_reduced,to_reduced,cost_fun,sampling);
        opti_method.minimize(cost_function,trans,terminated,tol);

    }
}


template<typename image_type,typename transform_type,typename CostFunctionType,typename teminated_class>
void linear_seq(const image_type& from,const image_type& to,
                    transform_type& trans,
                    int reg_type,
                    CostFunctionType cost_fun,
                    teminated_class& terminated,
                    float tol = 0.02)
{
    image::reg::linear(from,to,trans,image::reg::translocation,cost_fun,terminated,tol);
    if(reg_type >= image::reg::rigid_body)
        image::reg::linear(from,to,trans,image::reg::rigid_body,cost_fun,terminated,tol);
    if(reg_type >= image::reg::rigid_scaling)
        image::reg::linear(from,to,trans,image::reg::rigid_scaling,cost_fun,terminated,tol);
    if(reg_type >= image::reg::affine)
        image::reg::linear(from,to,trans,image::reg::affine,cost_fun,terminated,tol);
}



}
}


#endif//IMAGE_REG_HPP
