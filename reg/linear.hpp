#ifndef IMAGE_REG_HPP
#define IMAGE_REG_HPP
#include <limits>
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

    template<typename I_type>
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
            mask.geometry().is_valid(index);
            index.next(mask.geometry()))
            if(mask[index.index()])
            {
                total_w += 1.0;
                image::vector<3,double> pos(index);
                sum_mass += pos;
            }
        sum_mass /= total_w;
        for(unsigned char dim = 0;dim < I_type::dimension;++dim)
            sum_mass[dim] -= (double)Im.geometry()[dim]/2.0;
        return sum_mass;
    }

    template<typename image_type1,typename image_type2>
    void align_center(const image_type1& from,const image_type2& to,image::affine_transform<3,float>& arg_min)
    {
        image::vector<3,double> mF = image::reg::center_of_mass(from);
        image::vector<3,double> mG = image::reg::center_of_mass(to);
        arg_min.translocation[0] = mG[0]-mF[0]*arg_min.scaling[0];
        arg_min.translocation[1] = mG[1]-mF[1]*arg_min.scaling[1];
        arg_min.translocation[2] = mG[2]-mF[2]*arg_min.scaling[2];
    }

    struct square_error
    {
        typedef double value_type;
        template<typename ImageType,typename TransformType>
        double operator()(const ImageType& Ifrom,const ImageType& Ito,const TransformType& transform)
        {
            const unsigned int dim = ImageType::dimension;
            image::geometry<dim> geo(Ifrom.geometry());
            double error = 0.0;
            image::vector<dim,double> pos;
            for (image::pixel_index<dim> index; index.valid(geo); index.next(geo))
            {
                transform(index,pos);
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


    struct square_error2
    {
        typedef double value_type;
        template<typename ImageType,typename TransformType>
        double operator()(const ImageType& Ifrom,const ImageType& Ito,const TransformType& transform)
        {
            TransformType inverse(transform);
            inverse.inverse();
            return square_error()(Ifrom,Ito,transform)+square_error()(Ito,Ifrom,inverse);
        }
    };

    struct correlation
    {
        typedef double value_type;
        template<typename ImageType,typename TransformType>
        double operator()(const ImageType& Ifrom,const ImageType& Ito,const TransformType& transform)
        {
            const unsigned int dim = ImageType::dimension;
            image::geometry<dim> geo(Ifrom.geometry());
            std::vector<double> y(geo.size());
            image::resample(Ito,y,transform);
            return image::correlation(Ifrom.begin(),Ifrom.end(),y.begin());
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
            for (image::pixel_index<dimension> index; index.valid(geo); index.next(geo))
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

    template<typename image_type,
             typename transform_type,
             typename fun_type>
    class fun_adoptor{
        const image_type& from;
        const image_type& to;
        fun_type fun;
    public:
        typedef typename fun_type::value_type value_type;
    public:
        fun_adoptor(const image_type& from_,const image_type& to_):from(from_),to(to_){}
        template<typename iterator_type>
        double operator()(iterator_type param)
        {
            transform_type affine(&*param);
            image::transformation_matrix<3,typename transform_type::value_type> T(affine,from.geometry(),to.geometry());
            return fun(from,to,T);
        }
    };

enum coreg_type {translocation = 1,rotation = 2,scaling = 4,tilt = 8};
const int rigid_body = translocation | rotation;
const int rigid_scaling = translocation | rotation | scaling;
const int affine = translocation | rotation | scaling | tilt;


template<typename image_type1,typename image_type2,typename transform_type>
void get_bound(const image_type1& from,const image_type2& to,
               const transform_type& trans,
               transform_type& upper_trans,
               transform_type& lower_trans,
               int reg_type)
{
    typedef typename transform_type::value_type value_type;
    const unsigned int dimension = image_type1::dimension;
    upper_trans = trans;
    lower_trans = trans;
    if (reg_type & translocation)
    {
        for (unsigned int index = 0; index < dimension; ++index)
        {
            upper_trans[index] = (to.geometry()[index]+from.geometry()[index]*trans[dimension*2+index])/2.0;
            lower_trans[index] = -upper_trans[index];
        }
    }

    if (reg_type & rotation)
    {
        for (unsigned int index = dimension; index < dimension + dimension; ++index)
        {
            upper_trans[index] = 3.14159265358979323846*0.25;
            lower_trans[index] = -3.14159265358979323846*0.25;
        }
    }

    if (reg_type & scaling)
    {
        for (unsigned int index = dimension + dimension; index < dimension+dimension+dimension; ++index)
        {
            upper_trans[index] = trans[index]*1.5;
            lower_trans[index] = trans[index]/1.5;
        }
    }

    if (reg_type & tilt)
    {
        for (unsigned int index = dimension + dimension + dimension; index < transform_type::total_size; ++index)
        {
            upper_trans[index] = 0.5;
            lower_trans[index] = -0.5;
        }
    }
}

template<typename image_type,typename transform_type,typename CostFunctionType,typename teminated_class>
void linear(const image_type& from,const image_type& to,
                    transform_type& arg_min,
                    int reg_type,
                    CostFunctionType,
                    teminated_class& terminated)
{
    transform_type upper,lower;
    image::reg::get_bound(from,to,arg_min,upper,lower,reg_type);
    image::reg::fun_adoptor<image_type,transform_type,CostFunctionType> fun(from,to);
    std::srand(0);
    double optimal_value = fun(arg_min.begin());
    image::optimization::graient_descent(arg_min.begin(),arg_min.end(),upper.begin(),lower.begin(),fun,optimal_value,terminated,0.001);
    for(unsigned int iter = 0;iter < arg_min.size()*10 && !terminated;++iter)
        if(image::optimization::rand_search(arg_min.begin(),arg_min.end(),
                                           upper.begin(),lower.begin(),
                                           optimal_value,fun,5))
        {
            image::optimization::graient_descent(arg_min.begin(),arg_min.end(),upper.begin(),lower.begin(),fun,optimal_value,terminated,0.001);
            iter = 0;
        }
}

}
}


#endif//IMAGE_REG_HPP
