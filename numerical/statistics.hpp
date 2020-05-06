#ifndef STATISTICS_HPP
#define STATISTICS_HPP
#include <cmath>
#include <algorithm>
#include <numeric>
#include <utility>
#include "tipl/numerical/numerical.hpp"
#include "tipl/numerical/matrix.hpp"

namespace tipl
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
    template<class vector_type>
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

template<class value_type>
double gaussian_radial_basis(value_type dx,value_type sd)
{
    if(sd == 0.0)
        return 0;
	dx /= sd;
    dx *= dx;
    dx /= 2.0;
    return std::exp(-dx);
}


template<class value_type>
double gaussian_distribution(value_type dx,value_type variance,value_type normalization_term)
{
    dx *= dx;
    if(variance == 0)
        return 0;
    dx /= variance*2.0;
    dx = std::exp(-dx);
    if(normalization_term == 0)
        return 0;
    dx /= normalization_term;
    return dx;
}

template<class value_type>
double gaussian_distribution(value_type x,value_type mean,value_type variance,value_type normalization_term)
{
    x -= mean;
    x *= x;
    if(variance == 0)
        return 0;
    x /= variance*2.0;
    x = std::exp(-x);
    if(normalization_term == 0)
        return 0;
    x /= normalization_term;
    return x;
}

template<class input_iterator>
double mean(input_iterator from,input_iterator to)
{
    return from == to ? 0.0 : std::accumulate(from,to,0.0)/((double)(to-from));
}

template<class image_type>
double mean(const image_type& I)
{
    return mean(I.begin(),I.end());
}

template<class input_iterator>
typename std::iterator_traits<input_iterator>::value_type minimum(input_iterator from,input_iterator to)
{
    auto m = *from;
    for(;from != to;++from)
        if(*from < m)
            m = *from;
    return m;
}

template<class image_type>
typename image_type::value_type minimum(const image_type& I)
{
    return minimum(I.begin(),I.end());
}

template<class input_iterator>
typename std::iterator_traits<input_iterator>::value_type maximum(input_iterator from,input_iterator to)
{
    auto m = *from;
    for(;from != to;++from)
        if(*from > m)
            m = *from;
    return m;
}

template<class image_type>
typename image_type::value_type maximum(const image_type& I)
{
    return maximum(I.begin(),I.end());
}

template <typename input_iterator>
typename std::iterator_traits<input_iterator>::value_type median(input_iterator begin, input_iterator end)
{
    auto size = std::distance(begin, end);
    size /= 2;
    std::nth_element(begin, begin + size,end);
    return *std::next(begin, size);
}

template<class input_iterator>
std::pair<double,double> mean_variance(input_iterator from,input_iterator to)
{
    double sum = 0.0;
    double rms = 0.0;
    unsigned int size = to-from;
    while (from != to)
    {
        double t = *from;
        sum += t;
        rms += t*t;
        ++from;
    }
    if(size)
    {
        sum /= size;
        rms /= size;
    }
    return std::make_pair(sum,rms-sum*sum);
}

template<class input_iterator>
double mean_square(input_iterator from,input_iterator to)
{
    double ms = 0.0;
    size_t size = to-from;
    while (from != to)
    {
        double t = *from;
        ms += t*t;
        ++from;
    }
    if(size)
        ms /= size;
    return ms;
}

template<class input_iterator>
double root_mean_suqare(input_iterator from,input_iterator to)
{
    return std::sqrt(mean_square(from,to));
}

template<class input_iterator,class input_iterator2>
double root_mean_suqare_error(input_iterator from,input_iterator to,input_iterator2 from2)
{
    double rmse = 0.0;
    unsigned int size = to-from;
    while (from != to)
    {
        double t = *from-*from2;
        rmse += t*t;
        ++from;
        ++from2;
    }
    if(size)
        rmse /= size;
    return std::sqrt(rmse);
}

template<class input_iterator>
double variance(input_iterator from,input_iterator to,double mean)
{
    return mean_square(from,to)-mean*mean;
}
template<class input_iterator,typename value_type>
value_type standard_deviation(input_iterator from,input_iterator to,value_type mean)
{
    return std::sqrt(std::max<value_type>(0,variance(from,to,mean)));
}
template<class input_iterator>
double standard_deviation(input_iterator from,input_iterator to)
{
    return standard_deviation(from,to,mean(from,to));
}


template<class input_iterator>
double median_absolute_deviation(input_iterator from,input_iterator to)
{
    auto size = std::distance(from,to);
    size /= 2;
    auto m = median(from,to);
    for(auto iter = from;iter != to;++iter)
    {
        auto v = *iter;
        *iter = v > m ? v-m : m-v;
    }
    return median(from,to);
}


template<class input_iterator>
double median_absolute_deviation(input_iterator from,input_iterator to,double median_value)
{
    auto size = std::distance(from,to);
    size /= 2;
    for(auto iter = from;iter != to;++iter)
    {
        auto v = *iter;
        *iter = v > median_value ? v-median_value : median_value-v;
    }
    return median(from,to);
}



template<class input_iterator1,class input_iterator2>
double covariance(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from,double mean_x,double mean_y)
{
    double co = 0.0;
    size_t size = x_to-x_from;
    while (x_from != x_to)
    {
        co += *x_from*(*y_from);
        ++x_from;
        ++y_from;
    }
    if(size)
        co /= size;
    return co-mean_x*mean_y;
}

template<class input_iterator1,class input_iterator2>
double covariance(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from)
{
    return covariance(x_from,x_to,y_from,mean(x_from,x_to),mean(y_from,y_from+(x_to-x_from)));
}

template<class input_iterator1,class input_iterator2>
double correlation(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from,double mean_x,double mean_y)
{
    double sd1 = standard_deviation(x_from,x_to,mean_x);
    double sd2 = standard_deviation(y_from,y_from+(x_to-x_from),mean_y);
    if(sd1 == 0 || sd2 == 0)
        return 0;
    return covariance(x_from,x_to,y_from,mean_x,mean_y)/sd1/sd2;
}
template<class input_iterator1,class input_iterator2>
double correlation(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from)
{
    return correlation(x_from,x_to,y_from,mean(x_from,x_to),mean(y_from,y_from+(x_to-x_from)));
}

template<class input_iterator1,class input_iterator2>
double t_statistics(input_iterator1 x_from,input_iterator1 x_to,input_iterator2 y_from,input_iterator2 y_to)
{
    double n1 = x_to-x_from;
    double n2 = y_to-x_from;
    double mean0 = tipl::mean(x_from,x_to);
    double mean1 = tipl::mean(y_from,y_to);
    double v = n1 + n2 - 2;
    double va1 = tipl::variance(x_from,x_to,mean0);
    double va2 = tipl::variance(y_from,y_to,mean1);
    // pooled variance:
    double sp = std::sqrt(((n1-1.0) * va1 + (n2-1.0) * va2) / v);
    // t-statistic:
    if(sp == 0.0)
        return 0;
    return (mean0-mean1) / (sp * std::sqrt(1.0 / n1 + 1.0 / n2));
}

template<class input_iterator>
double t_statistics(input_iterator x_from,input_iterator x_to)
{
    double n = x_to-x_from;
    double mean = tipl::mean(x_from,x_to);
    double var = tipl::variance(x_from,x_to,mean);
    if(var == 0.0)
        return 0.0;
    return mean* std::sqrt(double(n-1.0)/var);
}

template<class input_iterator>
double least_square_fitting_slop(input_iterator x_from,input_iterator x_to,
                                 input_iterator y_from,input_iterator y_to)
{
    double mean_x = mean(x_from,x_to);
    double mean_y = mean(y_from,y_to);
    double var_x = variance(x_from,x_to,mean_x);
    if(var_x == 0)
        return 0;
    double co = 0.0;
    while (x_from != x_to)
    {
        co += *x_from*(*y_from);
        ++x_from;
        ++y_from;
    }
    if(x_to != x_from)
        co /= x_to-x_from;
    return (co-mean_x*mean_y)/var_x;
}
// first nx elements are from distribution x, whereas the rest is from another
template<class input_iterator>
double permutation_test(input_iterator from,input_iterator to,unsigned int nx,unsigned int permutation_count = 2000)
{
    double m_dif = mean(from,from+nx)-mean(from+nx,to);
    unsigned int count = 0;
    for(unsigned int i = 0;i < permutation_count;++i)
    {
        std::random_shuffle(from,to);
        if(mean(from,from+nx)-mean(from+nx,to) > m_dif)
            count++;
    }
    return (double)count/(double)permutation_count;
}

// fitting equation y=ax+b
// return (a,b)
template<class input_iterator1,class input_iterator2>
std::pair<double,double> linear_regression(input_iterator1 x_from,input_iterator1 x_to,input_iterator2 y_from)
{
    double mean_x = mean(x_from,x_to);
    double mean_y = mean(y_from,y_from+(x_to-x_from));
    double x_var = variance(x_from,x_to,mean_x);
    if(x_var == 0.0)
        return std::pair<double,double>(0,mean_y);
    double a = covariance(x_from,x_to,y_from,mean_x,mean_y)/x_var;
    double b = mean_y-a*mean_x;
    return std::pair<double,double>(a,b);
}

template<typename input_iterator1,typename input_iterator2,typename value_type>
void linear_regression(input_iterator1 x_from,input_iterator1 x_to,input_iterator2 y_from,value_type& a,value_type& b,value_type& r2)
{
    value_type mean_x = mean(x_from,x_to);
    value_type mean_y = mean(y_from,y_from+(x_to-x_from));
    value_type x_var = variance(x_from,x_to,mean_x);
    value_type y_var = variance(y_from,y_from+(x_to-x_from),mean_y);
    if(x_var == 0 || y_var == 0)
    {
        a = 0;
        b = mean_y;
        r2 = 0;
        return;
    }
    value_type cov = covariance(x_from,x_to,y_from,mean_x,mean_y);
    a = cov/x_var;
    b = mean_y-a*mean_x;
    r2 = cov*cov/x_var/y_var;
}


/*
    float y[] = {1,2,2,2,3};
    float X[] = {1,1,1,
                  1,2,8,
                  1,3,27,
                  1,4,64,
                  1,5,125};

    float b[3]={0,0,0};
    float t[3]={0,0,0};
    // b = 0.896551724, 0.33646813, 0.002089864
    multiple_regression<float> m;
    m.set_variables(X,3,5);
    m.regress(y,b,t);
 */
template<class value_type>
class multiple_regression{
public:
    // the subject data are stored in each row
    std::vector<value_type> X,Xt,XtX;
    std::vector<value_type> X_cov;
    std::vector<int> piv;
    unsigned int feature_count;
    unsigned int subject_count;
public:
    multiple_regression(void){}
    template<class iterator>
    bool set_variables(iterator X_,
                       unsigned int feature_count_,
                       unsigned int subject_count_)
    {
        feature_count = feature_count_;
        subject_count = subject_count_;
        if(feature_count > subject_count)
            return false;
        X.resize(feature_count*subject_count);
        std::copy(X_,X_+X.size(),X.begin());
        Xt.resize(X.size());
        tipl::mat::transpose(&*X.begin(),&*Xt.begin(),tipl::dyndim(subject_count,feature_count));

        XtX.resize(feature_count*feature_count); // trans(x)*y    p by p
        tipl::mat::product_transpose(&*Xt.begin(),&*Xt.begin(),
                                         &*XtX.begin(),
                                         tipl::dyndim(feature_count,subject_count),
                                         tipl::dyndim(feature_count,subject_count));
        piv.resize(feature_count);
        tipl::mat::lu_decomposition(&*XtX.begin(),&*piv.begin(),tipl::dyndim(feature_count,feature_count));


        // calculate the covariance
        {
            X_cov = Xt;
            std::vector<value_type> c(feature_count),d(feature_count);
            if(!tipl::mat::lq_decomposition(&*X_cov.begin(),&*c.begin(),&*d.begin(),tipl::dyndim(feature_count,subject_count)))
                return false;
            tipl::mat::lq_get_l(&*X_cov.begin(),&*d.begin(),&*X_cov.begin(),
                                    tipl::dyndim(feature_count,subject_count));
        }


        // make l a squre matrix, get rid of the zero part
        for(unsigned int row = 1,pos = subject_count,pos2 = feature_count;row < feature_count;++row,pos += subject_count,pos2 += feature_count)
            std::copy(X_cov.begin() + pos,X_cov.begin() + pos + feature_count,X_cov.begin() + pos2);

        tipl::mat::inverse_lower(&*X_cov.begin(),tipl::dyndim(feature_count,feature_count));

        tipl::square(X_cov.begin(),X_cov.begin()+feature_count*feature_count);

        // sum column wise
        for(unsigned int row = 1,pos = feature_count;row < feature_count;++row,pos += feature_count)
            tipl::add(X_cov.begin(),X_cov.begin()+feature_count,X_cov.begin()+pos);
        tipl::square_root(X_cov.begin(),X_cov.begin()+feature_count);

        std::vector<value_type> new_X_cov(X_cov.begin(),X_cov.begin()+feature_count);
        new_X_cov.swap(X_cov);
        return true;
    }



    /*
     *       y0       x00 ...x0p
     *       y1       x10 ...x1p    b0
     *     [ :  ] = [  :        ][  :  ]
     *       :         :            bp
     *       yn       xn0 ...xnp
     *
     **/
    template<class iterator1,class iterator2,class iterator3>
    void regress(iterator1 y,iterator2 b,iterator3 t) const
    {
        regress(y,b);
        value_type rmse = get_rmse(y,b); // residual
        for(unsigned int index = 0;index < feature_count;++index)
            t[index] = b[index]/X_cov[index]/rmse;
    }
    // calculate residual
    template<class iterator1,class iterator2>
    void residual(iterator1 y,iterator2 b) const
    {
        std::vector<value_type> y_(subject_count);
        tipl::mat::left_vector_product(&*Xt.begin(),b,&*y_.begin(),tipl::dyndim(feature_count,subject_count));
        tipl::minus(y,y+subject_count,&*y_.begin());
    }

    template<class iterator1,class iterator2>
    value_type get_rmse(iterator1 y,iterator2 b) const
    {
        std::vector<value_type> y_(subject_count);
        tipl::mat::left_vector_product(&*Xt.begin(),b,&*y_.begin(),tipl::dyndim(feature_count,subject_count));
        tipl::minus(y_.begin(),y_.end(),y);
        tipl::square(y_);
        return std::sqrt(std::accumulate(y_.begin(),y_.end(),value_type(0.0))/(subject_count-feature_count));
    }


    template<class iterator1,class iterator2>
    void regress(iterator1 y,iterator2 b) const
    {
        std::vector<value_type> xty(feature_count); // trans(x)*y    p by 1
        tipl::mat::vector_product(&*Xt.begin(),y,&*xty.begin(),tipl::dyndim(feature_count,subject_count));
        tipl::mat::lu_solve(&*XtX.begin(),&*piv.begin(),&*xty.begin(),b,
                                tipl::dyndim(feature_count,feature_count));
    }

};



}
#endif
