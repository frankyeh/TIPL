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

template<typename T,typename Enable = void>
struct sum_result_type {
    using type = T;
};
template<typename T>
struct sum_result_type<T,typename std::enable_if<std::is_integral<T>::value>::type>{
    using type = int64_t;
};
template<typename T>
struct sum_result_type<T,typename std::enable_if<std::is_floating_point<T>::value>::type> {
    using type = double;
};
template<typename input_iterator,
         typename value_type = typename std::iterator_traits<input_iterator>::value_type,
         typename return_type = typename sum_result_type<value_type>::type>
__INLINE__ return_type sum(input_iterator from,input_iterator to)
{
    return_type sum = return_type();
    for(;from != to;++from)
        sum += *from;
    return sum;
}

template<typename T>
auto sum(const T& data)
{
    using value_type = typename T::value_type;
    using return_type = typename sum_result_type<value_type>::type;
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        return thrust::reduce(thrust::device,data.data(),data.data()+data.size(),return_type());
        #endif
    }
    else
    {
        if(data.size() < 1000 || max_thread_count < 2)
            return sum(data.begin(),data.end());
        std::mutex mutex;
        return_type sums = return_type();
        tipl::par_for<ranged>(data.begin(),data.end(),[&](auto beg,auto end)
        {
            auto v = sum(beg,end);
            std::lock_guard<std::mutex> lock(mutex);
            sums += v;
        });
        return sums;
    }

}

template<typename image_type1,typename image_type2>
__INLINE__ void sum_partial(const image_type1& in,image_type2& out)
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


template<typename input_iterator,
         typename value_type = typename std::iterator_traits<input_iterator>::value_type,
         typename return_type = typename sum_result_type<value_type>::type>
__INLINE__ return_type square_sum(input_iterator from,input_iterator to)
{
    return_type ss = return_type();
    while (from != to)
    {
        return_type t = *from;
        ss += t*t;
        ++from;
    }
    return ss;
}
#ifdef __CUDACC__
struct square_sum_imp {
    template<typename T>
    __device__ auto operator()(const T& v) const {
        typename sum_result_type<T>::type vv(v);
        vv *= v;
        return vv;
    }
};
#endif
template<typename T,
         typename value_type = typename T::value_type,
         typename return_type = typename sum_result_type<value_type>::type>
return_type square_sum(const T& data)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        return thrust::transform_reduce(thrust::device,
                                 data.data(),data.data()+data.size(),
                                 square_sum_imp(),return_type(),thrust::plus<return_type>());
        #endif
    }
    else
    {
        if(data.size() < 1000 || max_thread_count < 2)
            return square_sum(data.begin(),data.end());
        std::mutex mutex;
        return_type sums = return_type();
        tipl::par_for<ranged>(data.begin(),data.end(),[&](auto beg,auto end)
        {
            auto v = square_sum(beg,end);
            std::lock_guard<std::mutex> lock(mutex);
            sums += v;
        });
        return sums;
    }
}

template<typename input_iterator>
__INLINE__ double mean(input_iterator from,input_iterator to)
{
    return (from == to) ? 0.0 :double(sum(from,to))/double(to-from);
}

template<typename T>
inline double mean(const T& I)
{
    return (I.empty()) ? 0.0 :double(sum(I))/double(I.size());
}



template <typename input_iterator,
          typename value_type = typename std::iterator_traits<input_iterator>::value_type>
__INLINE__ auto median(input_iterator begin, input_iterator end)
    -> typename std::enable_if<!std::is_const<typename std::remove_reference<decltype(*begin)>::type>::value,typename std::iterator_traits<input_iterator>::value_type>::type
{
    auto size = std::distance(begin, end) / 2;
    std::nth_element(begin, begin + size, end);
    return *(begin + size);
}
template <typename input_iterator,
          typename value_type = typename std::iterator_traits<input_iterator>::value_type>
__INLINE__ auto median(input_iterator begin, input_iterator end)
    -> typename std::enable_if<std::is_const<typename std::remove_reference<decltype(*begin)>::type>::value,typename std::iterator_traits<input_iterator>::value_type>::type
{
    std::vector<value_type> tmp(begin, end);
    return median(tmp.begin(), tmp.end());
}

template<typename image_type,
         typename value_type = typename image_type::value_type>
value_type median(const image_type& I)
{
    const size_t chunk_count = 256;
    if(I.size() < 2048)
        return median(I.begin(),I.end());

    std::vector<value_type> chunk(chunk_count);
    size_t total_size = I.size();
    size_t chunk_size = total_size / chunk_count;
    size_t remainder = total_size % chunk_count;

    tipl::par_for(chunk.size(),[&](size_t i)
    {
        size_t start = i * chunk_size + std::min(i, remainder);
        size_t end = start + chunk_size + (i < remainder ? 1 : 0);
        chunk[i] = median(I.begin() + start, I.begin() + end);
    });
    return median(chunk.begin(),chunk.end());
}




template<typename input_iterator>
__INLINE__ double mean_square(input_iterator from,input_iterator to)
{
    if(from == to)
        return 0.0;
    return square_sum(from,to)/double(to-from);
}

template<typename container_type>
inline double mean_square(const container_type& data)
{
    if(data.empty())
        return 0.0;
    return square_sum(data)/double(data.size());
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
template<typename T>
inline double variance(T& data,double m)
{
    return mean_square(data)-m*m;
}
template<typename T>
inline double variance(T& data)
{
    return variance(data,mean(data));
}
template<typename input_iterator,typename value_type>
__INLINE__ double standard_deviation(input_iterator from,input_iterator to,value_type mean)
{
    auto var = variance(from,to,mean);
    return var > 0.0 ? std::sqrt(var) : 0.0;
}
template<typename input_iterator>
__INLINE__ double standard_deviation(input_iterator from,input_iterator to)
{
    return standard_deviation(from,to,mean(from,to));
}
template<typename T>
inline double standard_deviation(T& data,double m)
{
    auto var = variance(data,m);
    return var > 0.0 ? std::sqrt(var) : 0.0;
}
template<typename T>
inline double standard_deviation(T& data)
{
    return standard_deviation(data,mean(data));
}
template<typename input_iterator>
__INLINE__ auto median_absolute_deviation(input_iterator from,input_iterator to)
    -> typename std::enable_if<!std::is_const<typename std::remove_reference<decltype(*from)>::type>::value,typename std::iterator_traits<input_iterator>::value_type>::type
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
__INLINE__ auto median_absolute_deviation(input_iterator from,input_iterator to,double median_value)
    -> typename std::enable_if<!std::is_const<typename std::remove_reference<decltype(*from)>::type>::value,typename std::iterator_traits<input_iterator>::value_type>::type
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


template <typename input_iterator,
          typename value_type = typename std::iterator_traits<input_iterator>::value_type>
auto median_absolute_deviation(input_iterator from, input_iterator to,double median_value)
    -> typename std::enable_if<std::is_const<typename std::remove_reference<decltype(*from)>::type>::value,typename std::iterator_traits<input_iterator>::value_type>::type
{
    std::vector<value_type> tmp(from, to);
    return median_absolute_deviation(tmp.begin(), tmp.end(),median_value);
}

template<typename input_iterator1,typename input_iterator2,
         typename value_type = typename std::iterator_traits<input_iterator1>::value_type,
         typename return_type = typename sum_result_type<value_type>::type>
__INLINE__ return_type inner_product(input_iterator1 x_from,input_iterator1 x_to,
                                     input_iterator2 y_from)
{
    if(x_to == x_from)
        return return_type();
    return_type co = return_type();
    size_t size = x_to-x_from;
    while (x_from != x_to)
    {
        co += return_type(*x_from)*return_type(*y_from);
        ++x_from;
        ++y_from;
    }
    return co;
}
template<typename T,typename U,
         typename value_type = typename T::value_type,
         typename return_type = typename sum_result_type<value_type>::type>
__HOST__ auto inner_product(const T& x,const U& y)
{
    if(x.empty())
        return return_type();
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        device_vector<return_type> xy(x.size());
        thrust::transform(thrust::device,
                                 x.data(),x.data()+x.size(),
                                 y.data(),xy.data(),thrust::multiplies<return_type>());
        return thrust::reduce(thrust::device,xy.data(),xy.data()+xy.size(),return_type());
        #endif
    }
    else
    {
        std::mutex mutex;
        return_type sums = return_type();
        tipl::par_for<ranged>(x.size(),[&](auto from,auto to)
        {
            auto v = inner_product(x.begin()+from,x.begin()+to,y.begin()+from);
            std::lock_guard<std::mutex> lock(mutex);
            sums += v;
        });
        return sums;
    }
}

template<typename input_iterator1,typename input_iterator2>
__INLINE__ double covariance(input_iterator1 x_from,input_iterator1 x_to,
                             input_iterator2 y_from,double mean_x,double mean_y)
{
    if(x_to == x_from)
        return 0.0;
    auto size = x_to-x_from;
    return inner_product(x_from,x_to,y_from)/double(size)-mean_x*mean_y;
}
template<typename input_iterator1,typename input_iterator2>
__INLINE__ double covariance(input_iterator1 x_from,input_iterator1 x_to,
                             input_iterator2 y_from)
{
    if(x_to == x_from)
        return 0.0;
    auto size = x_to-x_from;
    return inner_product(x_from,x_to,y_from)/double(size)-mean(x_from,x_to)*mean(y_from,y_from+size);
}

template<typename T,typename U>
inline double covariance(const T& x,const U& y,double mean_x,double mean_y)
{
    if(x.empty())
        return 0.0;
    return inner_product(x,y)/double(x.size())-mean_x*mean_y;
}

template<typename T,typename U>
inline double covariance(const T& x,const U& y)
{
    return covariance(x,y,mean(x),mean(y));
}

template<typename input_iterator1,typename input_iterator2>
__INLINE__ double correlation(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from,double mean_x,double mean_y)
{
    auto sd1 = standard_deviation(x_from,x_to,mean_x);
    auto sd2 = standard_deviation(y_from,y_from+(x_to-x_from),mean_y);
    if(sd1 == 0 || sd2 == 0)
        return sd1;
    return covariance(x_from,x_to,y_from,mean_x,mean_y)/sd1/sd2;
}

template<typename T,typename U>
inline double correlation(const T& x,const U& y,double mean_x,double mean_y)
{
    auto sd1 = standard_deviation(x,mean_x);
    auto sd2 = standard_deviation(y,mean_y);
    if(sd1 == 0 || sd2 == 0)
        return 0;
    return covariance(x,y,mean_x,mean_y)/sd1/sd2;
}

template<typename input_iterator1,typename input_iterator2>
__INLINE__ double correlation(input_iterator1 x_from,input_iterator1 x_to,
                  input_iterator2 y_from)
{
    return correlation(x_from,x_to,y_from,mean(x_from,x_to),mean(y_from,y_from+(x_to-x_from)));
}

template<typename T,typename U>
inline double correlation(const T& x,const U& y)
{
    return correlation(x,y,mean(x),mean(y));
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
__INLINE__ std::pair<double,double> linear_regression(input_iterator1 x_from,input_iterator1 x_to,input_iterator2 y_from)
{
    auto mean_x = mean(x_from,x_to);
    auto mean_y = mean(y_from,y_from+(x_to-x_from));
    auto x_var = variance(x_from,x_to,mean_x);
    if(x_var == 0.0)
        return std::pair<double,double>(0,mean_y);
    auto a = covariance(x_from,x_to,y_from,mean_x,mean_y)/x_var;
    auto b = mean_y-a*mean_x;
    return std::pair<double,double>(a,b);
}

template<typename input_iterator1,typename input_iterator2,typename value_type>
__INLINE__ void linear_regression(input_iterator1 x_from,input_iterator1 x_to,input_iterator2 y_from,value_type& a,value_type& b,value_type& r2)
{
    auto mean_x = mean(x_from,x_to);
    auto mean_y = mean(y_from,y_from+(x_to-x_from));
    auto x_var = variance(x_from,x_to,mean_x);
    auto y_var = variance(y_from,y_from+(x_to-x_from),mean_y);
    if(x_var == 0 || y_var == 0)
    {
        a = 0;
        b = mean_y;
        r2 = 0;
        return;
    }
    auto cov = covariance(x_from,x_to,y_from,mean_x,mean_y);
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
