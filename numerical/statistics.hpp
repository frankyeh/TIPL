#ifndef STATISTICS_HPP
#define STATISTICS_HPP
#include <cmath>
#include <algorithm>
#include <numeric>
#include <utility>

namespace image
{

class sample
{
protected:
    double sum;
    double sum2;
    unsigned int size_;
public:
    sample(void)
    {
        clear();
    }
public:
    template<typename vector_type>
    const sample& operator=(const vector_type& data)
    {
        sum = 0.0,sum2 = 0.0;
        size_ = data.size();
        double v;
        for (unsigned int index = 0;index < size_;++index)
        {
            v = data[index];
            sum += v;
            sum2 += v*v;
        }
        return *this;
    }

public:
    void clear(void)
    {
        sum = sum2 = 0;
        size_ = 0;
    }

    unsigned int size(void) const
    {
        return size_;
    }


    void push_back(double data)
    {
        ++size_;
        sum += data;
        data *= data;
        sum2 += data;
    }
    void pop_back(double data)
    {
        --size_;
        sum -= data;
        data *= data;
        sum2 -= data;
    }
    void replace(double old_data,double new_data)
    {
        sum += new_data-old_data;
        new_data*=new_data;
        old_data*=old_data;
        sum2 += new_data-old_data;
    }

public:
    // return (mean,variance)
    std::pair<double,double> get_mean_variance(void)
    {
        double mean = sum;
        mean /= (double)size_;
        double variance = sum2;
        variance /= (double)size_;
        variance -= mean*mean;
        return std::make_pair(mean,variance);
    }

};

template<typename value_type>
double gaussian_radial_basis(value_type dx,value_type sd)
{
	dx /= sd;
    dx *= dx;
    dx /= 2.0;
    return std::exp(-dx);
}


template<typename value_type>
double gaussian_distribution(value_type dx,value_type variance,value_type normalization_term)
{
    dx *= dx;
    dx /= variance*2.0;
    dx = std::exp(-dx);
    dx /= normalization_term;
    return dx;
}

template<typename value_type>
double gaussian_distribution(value_type x,value_type mean,value_type variance,value_type normalization_term)
{
    x -= mean;
    x *= x;
    x /= variance*2.0;
    x = std::exp(-x);
    x /= normalization_term;
    return x;
}

template<typename input_iterator>
double mean(input_iterator from,input_iterator to)
{
    return from == to ? 0.0 : std::accumulate(from,to,0.0)/((double)(to-from));
}
template<typename input_iterator>
std::pair<double,double> mean_variance(input_iterator from,input_iterator to)
{
    double sum = 0.0;
    double rms = 0.0;
    double num = to-from;
    while (from != to)
    {
        double t = *from;
        sum += t;
        rms += t*t;
        ++from;
    }
    sum /= num;
    rms /= num;
    return std::make_pair(sum,rms-sum*sum);
}

template<typename input_iterator>
double mean_square(input_iterator from,input_iterator to)
{
    double ms = 0.0;
    double num = to-from;
    while (from != to)
    {
        double t = *from;
        ms += t*t;
        ++from;
    }
    ms /= num;
    return ms;
}

template<typename input_iterator>
double root_mean_suqare(input_iterator from,input_iterator to)
{
    return std::sqrt(mean_square(from,to));
}

template<typename input_iterator>
double variance(input_iterator from,input_iterator to,double mean)
{
    return mean_square(from,to)-mean*mean;
}
template<typename input_iterator>
double standard_deviation(input_iterator from,input_iterator to,double mean)
{
    return std::sqrt(variance(from,to,mean));
}
template<typename input_iterator>
double standard_deviation(input_iterator from,input_iterator to)
{
    return standard_deviation(from,to,mean(from,to));
}

template<typename input_iterator1,typename input_iterator2>
double covariance(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from,double mean_x,double mean_y)
{
    double size = x_to-x_from;
    double co = 0.0;
    while (x_from != x_to)
    {
        co += *x_from*(*y_from);
        ++x_from;
        ++y_from;
    }
    co /= size;
    return co-mean_x*mean_y;
}

template<typename input_iterator1,typename input_iterator2>
double covariance(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from)
{
    return covariance(x_from,x_to,y_from,mean(x_from,x_to),mean(y_from,y_from+(x_to-x_from)));
}

template<typename input_iterator1,typename input_iterator2>
double correlation(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from,double mean_x,double mean_y)
{
    return covariance(x_from,x_to,y_from,mean_x,mean_y)
           /variance(x_from,x_to,mean_x)/variance(y_from,y_from+(x_to-x_from),mean_y);
}
template<typename input_iterator1,typename input_iterator2>
double correlation(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from)
{
    return correlation(x_from,x_to,y_from,mean(x_from,x_to),mean(y_from,y_from+(x_to-x_from)));
}

template<typename input_iterator>
double least_square_fitting_slop(input_iterator x_from,input_iterator x_to,
                                 input_iterator y_from,input_iterator y_to)
{
    double mean_x = mean(x_from,x_to);
    double mean_y = mean(y_from,y_to);
    double var_x = variance(x_from,x_to,mean_x);
    double size = x_to-x_from;
    double co = 0.0;
    while (x_from != x_to)
    {
        co += *x_from*(*y_from);
        ++x_from;
        ++y_from;
    }
    co /= size;
    return (co-mean_x*mean_y)/var_x;
}

// fitting equation y=ax+b
// return (a,b)
template<typename input_iterator>
std::pair<double,double> linear_regression(input_iterator x_from,input_iterator x_to,input_iterator y_from)
{
    double mean_x = mean(x_from,x_to);
    double mean_y = mean(y_from,y_from+(x_to-x_from));
    double x_var = variance(x_from,x_to,mean_x);
    if(x_var == 0.0)
        return std::pair<double,double>(0,0);
    double a = covariance(x_from,x_to,y_from,mean_x,mean_y)/x_var;
    double b = mean_y-a*mean_x;
    return std::pair<double,double>(a,b);
}

}
#endif
