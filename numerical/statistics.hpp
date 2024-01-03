#ifndef STATISTICS_HPP
#define STATISTICS_HPP
#include <cmath>
#include <algorithm>
#include <numeric>
#include <utility>
#include "../def.hpp"
#include "numerical.hpp"
#include "matrix.hpp"


namespace tipl
{

class sample
{
protected:
    double sum;
    double sum2;
    size_t size_;
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
        for (size_t index = 0;index < size_;++index)
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

    size_t size(void) const
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
    if(sd == 0.0)
        return 0;
    dx /= sd;
    dx *= dx;
    dx /= 2.0;
    return std::exp(-dx);
}


template<typename value_type>
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

template<typename value_type>
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

template<typename input_iterator,
         typename std::enable_if<
             std::is_floating_point<typename std::iterator_traits<input_iterator>::value_type>::value,bool>::type = true>
__INLINE__ double sum(input_iterator from,input_iterator to)
{
    double sum(0.0);
    for(;from != to;++from)
        sum += *from;
    return sum;
}

template<typename input_iterator,
         typename std::enable_if<
             std::is_integral<typename std::iterator_traits<input_iterator>::value_type>::value,bool>::type = true>
__INLINE__ size_t sum(input_iterator from,input_iterator to)
{
    size_t sum(0);
    for(;from != to;++from)
        sum += *from;
    return sum;
}

template<typename input_iterator,
         typename std::enable_if<
             std::is_class<typename std::iterator_traits<input_iterator>::value_type>::value,bool>::type = true>
__INLINE__ auto sum(input_iterator from,input_iterator to)
{
    typename std::iterator_traits<input_iterator>::value_type sum;
    for(;from != to;++from)
        sum += *from;
    return sum;
}


template<typename image_type>
__INLINE__ auto sum(const image_type& I)
{
    return sum(I.begin(),I.end());
}

template<typename input_iterator,
         typename std::enable_if<
             std::is_floating_point<typename std::iterator_traits<input_iterator>::value_type>::value,bool>::type = true>
__INLINE__ double sum_mt(input_iterator from,input_iterator to)
{
    if(from == to)
        return 0;
    size_t size = to-from;
    unsigned int thread_count = std::thread::hardware_concurrency();
    std::vector<double> ss(thread_count);
    tipl::par_for(thread_count,[&ss,thread_count,from,size](size_t sum_id)
    {
        double s(0.0);
        for(size_t i = sum_id;i < size;i += thread_count)
            s += from[i];
        ss[sum_id] = s;
    });
    return sum(ss);
}

template<typename input_iterator,
         typename std::enable_if<
             std::is_integral<typename std::iterator_traits<input_iterator>::value_type>::value,bool>::type = true>
__INLINE__ size_t sum_mt(input_iterator from,input_iterator to)
{
    if(from == to)
        return 0;
    size_t size = to-from;
    unsigned int thread_count = std::thread::hardware_concurrency();
    std::vector<size_t> ss(thread_count);
    tipl::par_for(thread_count,[&ss,thread_count,from,size](size_t sum_id)
    {
        size_t s(0);
        for(size_t i = sum_id;i < size;i += thread_count)
            s += from[i];
        ss[sum_id] = s;
    });
    return sum(ss);
}


template<typename input_iterator,
         typename std::enable_if<
             std::is_class<typename std::iterator_traits<input_iterator>::value_type>::value,bool>::type = true>
__INLINE__ auto sum_mt(input_iterator from,input_iterator to)
{
    if(from == to)
        return 0;
    size_t size = to-from;
    unsigned int thread_count = std::thread::hardware_concurrency();
    std::vector<typename std::iterator_traits<input_iterator>::value_type> ss(thread_count);
    tipl::par_for(thread_count,[&ss,thread_count,from,size](size_t sum_id)
    {
        size_t s;
        for(size_t i = sum_id;i < size;i += thread_count)
            s += from[i];
        ss[sum_id] = s;
    });
    return sum(ss);
}


template<typename image_type>
__INLINE__ auto sum_mt(const image_type& I)
{
    return sum_mt(I.begin(),I.end());
}

template<typename image_type1,typename image_type2>
__INLINE__ void sum_partial_mt(const image_type1& in,image_type2& out)
{
    size_t size = out.size();
    tipl::par_for(size,[&](size_t j)
    {
        auto v = out[j];
        for(size_t pos = j;pos < in.size();pos += size)
            v += in[pos];
        out[j] = v;
    });
}


template<typename input_iterator>
__INLINE__ double square_sum(input_iterator from,input_iterator to)
{
    double ss = 0.0;
    while (from != to)
    {
        double t = *from;
        ss += t*t;
        ++from;
    }
    return ss;
}

template<typename image_type>
__INLINE__ auto square_sum(const image_type& I)
{
    return square_sum(I.begin(),I.end());
}

template<typename input_iterator>
__INLINE__ double square_sum_mt(input_iterator from,input_iterator to)
{
    if(from == to)
        return 0.0;
    size_t size = to-from;
    unsigned int thread_count = std::thread::hardware_concurrency();
    std::vector<double> ss(thread_count);
    tipl::par_for(thread_count,[&ss,thread_count,from,size](size_t ss_id)
    {
        double sum = 0.0;
        for(size_t i = ss_id;i < size;i += thread_count)
        {
            double t = from[i];
            sum += t*t;
        }
        ss[ss_id] = sum;
    });
    return tipl::sum(ss);
}

template<typename image_type>
__INLINE__ auto square_sum_mt(const image_type& I)
{
    return square_sum_mt(I.begin(),I.end());
}

template<typename input_iterator>
__INLINE__ double mean(input_iterator from,input_iterator to)
{
    return (from == to) ? 0.0 :double(sum(from,to))/double(to-from);
}

template<typename input_iterator>
__INLINE__ double mean_mt(input_iterator from,input_iterator to)
{
    return (from == to) ? 0.0 :double(sum_mt(from,to))/double(to-from);
}

template<typename image_type>
__INLINE__ auto mean(const image_type& I)
{
    return mean(I.begin(),I.end());
}

template<typename image_type>
__INLINE__ auto mean_mt(const image_type& I)
{
    return mean_mt(I.begin(),I.end());
}

template <typename input_iterator>
auto median(input_iterator begin, input_iterator end)
{
    std::vector<typename std::iterator_traits<input_iterator>::value_type> tmp(begin,end);
    auto size = tmp.size()/2;
    std::nth_element(tmp.begin(),tmp.begin() + size,tmp.end());
    return tmp[size];
}
template<typename image_type>
auto median(const image_type& I)
{
    return median(I.begin(),I.end());
}
template<typename input_iterator>
std::pair<double,double> mean_variance(input_iterator from,input_iterator to)
{
    double sum = 0.0;
    double rms = 0.0;
    size_t size = to-from;
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
    return std::pair<double,double>(sum,rms-sum*sum);
}

template<typename input_iterator>
__INLINE__ double mean_square(input_iterator from,input_iterator to)
{
    if(from == to)
        return 0.0;
    return square_sum(from,to)/double(to-from);
}

template<typename input_iterator>
__INLINE__ double mean_square_mt(input_iterator from,input_iterator to)
{
    if(from == to)
        return 0.0;
    return square_sum_mt(from,to)/double(to-from);
}

template<typename input_iterator>
__INLINE__ double root_mean_suqare(input_iterator from,input_iterator to)
{
    return std::sqrt(mean_square(from,to));
}

template<typename input_iterator,typename input_iterator2>
__INLINE__ double root_mean_suqare_error(input_iterator from,input_iterator to,input_iterator2 from2)
{
    double rmse = 0.0;
    size_t size = to-from;
    while (from != to)
    {
        double t = *from;
        t -= double(*from2);
        rmse += t*t;
        ++from;
        ++from2;
    }
    if(size)
        rmse /= size;
    return std::sqrt(rmse);
}

template<typename input_iterator>
__INLINE__ double variance(input_iterator from,input_iterator to,double mean)
{
    return mean_square(from,to)-mean*mean;
}
template<typename input_iterator>
__INLINE__ double variance_mt(input_iterator from,input_iterator to,double mean)
{
    return mean_square_mt(from,to)-mean*mean;
}
template<typename input_iterator,typename value_type>
__INLINE__ value_type standard_deviation(input_iterator from,input_iterator to,value_type mean)
{
    auto var = variance(from,to,mean);
    return var > 0.0 ? std::sqrt(var) : 0.0;
}
template<typename input_iterator,typename value_type>
__INLINE__ value_type standard_deviation_mt(input_iterator from,input_iterator to,value_type mean)
{
    auto var = variance_mt(from,to,mean);
    return var > 0.0 ? std::sqrt(var) : 0.0;
}
template<typename input_iterator>
__INLINE__ double standard_deviation(input_iterator from,input_iterator to)
{
    return standard_deviation(from,to,mean(from,to));
}
template<typename input_iterator>
__INLINE__ double standard_deviation_mt(input_iterator from,input_iterator to)
{
    return standard_deviation_mt(from,to,mean_mt(from,to));
}

template<typename T>
__INLINE__ double standard_deviation(T& data)
{
    return standard_deviation(data.begin(),data.end(),mean(data.begin(),data.end()));
}
template<typename T>
__INLINE__ double standard_deviation_mt(T& data)
{
    return standard_deviation_mt(data.begin(),data.end(),mean(data.begin(),data.end()));
}

template<typename input_iterator>
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


template<typename input_iterator>
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



template<typename input_iterator1,typename input_iterator2>
__INLINE__ double covariance(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from,double mean_x,double mean_y)
{
    double co = 0.0;
    size_t size = x_to-x_from;
    while (x_from != x_to)
    {
        co += double(*x_from)*double(*y_from);
        ++x_from;
        ++y_from;
    }
    if(size)
        co /= size;
    return co-mean_x*mean_y;
}

template<typename input_iterator1,typename input_iterator2>
__INLINE__ double covariance_mt(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from,double mean_x,double mean_y)
{
    if(x_to == x_from)
        return 0.0;
    size_t size = x_to-x_from;
    unsigned int thread_count = std::thread::hardware_concurrency();
    std::vector<double> co(thread_count);
    tipl::par_for(thread_count,[&co,thread_count,x_from,y_from,size](size_t co_id)
    {
        double sum = 0.0;
        for(size_t i = co_id;i < size;i += thread_count)
            sum += double(x_from[i])*double(y_from[i]);
        co[co_id] = sum;
    });
    return sum(co)/double(size)-mean_x*mean_y;
}

template<typename input_iterator1,typename input_iterator2>
__INLINE__ double covariance(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from)
{
    return covariance(x_from,x_to,y_from,mean(x_from,x_to),mean(y_from,y_from+(x_to-x_from)));
}
template<typename input_iterator1,typename input_iterator2>
__INLINE__ double covariance_mt(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from)
{
    return covariance_mt(x_from,x_to,y_from,mean_mt(x_from,x_to),mean_mt(y_from,y_from+(x_to-x_from)));
}

template<typename T,typename U>
__INLINE__ double covariance(const T& x,const U& y)
{
    return covariance(x.begin(),x.end(),y.begin());
}

template<typename T,typename U>
__INLINE__ double covariance_mt(const T& x,const U& y)
{
    return covariance_mt(x.begin(),x.end(),y.begin());
}

template<typename input_iterator1,typename input_iterator2>
__INLINE__ double correlation(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from,double mean_x,double mean_y)
{
    double sd1 = standard_deviation(x_from,x_to,mean_x);
    double sd2 = standard_deviation(y_from,y_from+(x_to-x_from),mean_y);
    if(sd1 == 0 || sd2 == 0)
        return 0;
    return covariance(x_from,x_to,y_from,mean_x,mean_y)/sd1/sd2;
}

template<typename input_iterator1,typename input_iterator2>
__INLINE__ double correlation_mt(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from,double mean_x,double mean_y)
{
    double sd1 = standard_deviation_mt(x_from,x_to,mean_x);
    double sd2 = standard_deviation_mt(y_from,y_from+(x_to-x_from),mean_y);
    if(sd1 == 0 || sd2 == 0)
        return 0;
    return covariance_mt(x_from,x_to,y_from,mean_x,mean_y)/sd1/sd2;
}

template<typename input_iterator1,typename input_iterator2>
__INLINE__ double correlation(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from)
{
    return correlation(x_from,x_to,y_from,mean(x_from,x_to),mean(y_from,y_from+(x_to-x_from)));
}

template<typename input_iterator1,typename input_iterator2>
__INLINE__ double correlation_mt(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from)
{
    return correlation_mt(x_from,x_to,y_from,mean_mt(x_from,x_to),mean_mt(y_from,y_from+(x_to-x_from)));
}

template<typename input_iterator1,typename input_iterator2>
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

template<typename input_iterator>
double t_statistics(input_iterator x_from,input_iterator x_to)
{
    double n = x_to-x_from;
    double mean = tipl::mean(x_from,x_to);
    double var = tipl::variance(x_from,x_to,mean);
    if(var == 0.0)
        return 0.0;
    return mean* std::sqrt(double(n-1.0)/var);
}

template<typename input_iterator>
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
template<typename input_iterator>
double permutation_test(input_iterator from,input_iterator to,unsigned int nx,unsigned int permutation_count = 2000,unsigned int seed = std::random_device()())
{
    double m_dif = mean(from,from+nx)-mean(from+nx,to);
    unsigned int count = 0;
    std::mt19937 g(seed);
    for(unsigned int i = 0;i < permutation_count;++i)
    {
        std::shuffle(from,to,g);
        if(mean(from,from+nx)-mean(from+nx,to) > m_dif)
            count++;
    }
    return (double)count/(double)permutation_count;
}

// fitting equation y=ax+b
// return (a,b)
template<typename input_iterator1,typename input_iterator2>
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
__INLINE__ void linear_regression(input_iterator1 x_from,input_iterator1 x_to,input_iterator2 y_from,value_type& a,value_type& b,value_type& r2)
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
template<typename value_type>
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
    template<typename iterator>
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
        tipl::mat::transpose(&*X.begin(),&*Xt.begin(),tipl::shape<2>(subject_count,feature_count));

        XtX.resize(feature_count*feature_count); // trans(x)*y    p by p
        tipl::mat::product_transpose(&*Xt.begin(),&*Xt.begin(),
                                         &*XtX.begin(),
                                         tipl::shape<2>(feature_count,subject_count),
                                         tipl::shape<2>(feature_count,subject_count));
        piv.resize(feature_count);
        tipl::mat::lu_decomposition(&*XtX.begin(),&*piv.begin(),tipl::shape<2>(feature_count,feature_count));


        // calculate the covariance
        {
            X_cov = Xt;
            std::vector<value_type> c(feature_count),d(feature_count);
            if(!tipl::mat::lq_decomposition(&*X_cov.begin(),&*c.begin(),&*d.begin(),tipl::shape<2>(feature_count,subject_count)))
                return false;
            tipl::mat::lq_get_l(&*X_cov.begin(),&*d.begin(),&*X_cov.begin(),
                                    tipl::shape<2>(feature_count,subject_count));
        }


        // make l a squre matrix, get rid of the zero part
        for(unsigned int row = 1,pos = subject_count,pos2 = feature_count;row < feature_count;++row,pos += subject_count,pos2 += feature_count)
            std::copy(X_cov.begin() + pos,X_cov.begin() + pos + feature_count,X_cov.begin() + pos2);

        tipl::mat::inverse_lower(&*X_cov.begin(),tipl::shape<2>(feature_count,feature_count));

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
    template<typename iterator1,typename iterator2,typename iterator3>
    void regress(iterator1 y,iterator2 b,iterator3 t) const
    {
        regress(y,b);
        value_type rmse = get_rmse(y,b); // residual
        for(unsigned int index = 0;index < feature_count;++index)
            t[index] = b[index]/X_cov[index]/rmse;
    }
    // calculate residual
    template<typename iterator1,typename iterator2>
    void residual(iterator1 y,iterator2 b) const
    {
        std::vector<value_type> y_(subject_count);
        tipl::mat::left_vector_product(&*Xt.begin(),b,&*y_.begin(),tipl::shape<2>(feature_count,subject_count));
        tipl::minus(y,y+subject_count,&*y_.begin());
    }

    template<typename iterator1,typename iterator2>
    value_type get_rmse(iterator1 y,iterator2 b) const
    {
        std::vector<value_type> y_(subject_count);
        tipl::mat::left_vector_product(&*Xt.begin(),b,&*y_.begin(),tipl::shape<2>(feature_count,subject_count));
        tipl::minus(y_.begin(),y_.end(),y);
        tipl::square(y_);
        return std::sqrt(std::accumulate(y_.begin(),y_.end(),value_type(0.0))/(subject_count-feature_count));
    }


    template<typename iterator1,typename iterator2>
    void regress(iterator1 y,iterator2 b) const
    {
        std::vector<value_type> xty(feature_count); // trans(x)*y    p by 1
        tipl::mat::vector_product(&*Xt.begin(),y,&*xty.begin(),tipl::shape<2>(feature_count,subject_count));
        tipl::mat::lu_solve(&*XtX.begin(),&*piv.begin(),&*xty.begin(),b,
                                tipl::shape<2>(feature_count,feature_count));
    }

};



}
#endif
