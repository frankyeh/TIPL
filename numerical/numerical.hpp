//---------------------------------------------------------------------------
#ifndef NUMERICAL_HPP
#define NUMERICAL_HPP
#include <random>
#include "../utility/basic_image.hpp"
#include "../utility/multi_thread.hpp"
#include "interpolation.hpp"


namespace tipl
{

template<typename T>
struct normal_dist{
    std::mt19937 gen;
    std::normal_distribution<T> dst;
    normal_dist(unsigned int seed = 0):gen(seed){}
    T operator()(void){
        return dst(gen);
    }
};

template<typename T>
struct uniform_dist{
    std::mt19937 gen;
    std::uniform_real_distribution<T> dst;
    uniform_dist(T min = T(0), T max = T(1),unsigned int seed = 0):gen(seed),dst(min, max){}
    T operator()(void){
        return dst(gen);
    }
};
template<>
struct uniform_dist<int>{
    std::mt19937 gen;
    std::uniform_int_distribution<int> dst;
    uniform_dist(int min, int max,unsigned int seed = 0):gen(seed),dst(min, max){}
    uniform_dist(int size, unsigned int seed = 0):gen(seed),dst(0, size-1){}
    uniform_dist(unsigned int seed = 0):gen(seed),dst(){}
    int operator()(void){
        return dst(gen);
    }
    int operator()(unsigned int size){
        std::uniform_int_distribution<int> temp_dst(0,size-1);
        return temp_dst(gen);
    }
    void reset(int seed = 0)
    {
        gen.seed(seed);
    }
};

struct bernoulli{
    float p;
    std::mt19937 gen;
    std::uniform_real_distribution<float> dst;
    bernoulli(float p_,unsigned int seed = 0):p(p_),gen(seed),dst(float(0), float(1)){}
    bool operator()(void){
        return dst(gen) <= p;
    }
};

//---------------------------------------------------------------------------
template<typename input_iterator,typename output_iterator>
inline void gradient(input_iterator src_from,input_iterator src_to,
                     output_iterator dest_from,unsigned int src_shift,unsigned int dest_shift)
{
    input_iterator src_from2 = src_from+src_shift;
    if(dest_shift)
    {
        output_iterator dest_from_beg = dest_from;
        dest_from += dest_shift;
        std::fill(dest_from_beg,dest_from,0);
    }
    for (; src_from2 < src_to; ++src_from,++src_from2,++dest_from)
        (*dest_from) = (*src_from2) - (*src_from);

    if(dest_shift < src_shift)
        std::fill(dest_from,dest_from+(src_shift-dest_shift),0);
}
//---------------------------------------------------------------------------
template<typename PixelImageType,typename VectorImageType>
inline void gradient(const PixelImageType& src,VectorImageType& dest_,unsigned int src_shift,unsigned int dest_shift)
{
    VectorImageType dest(src.shape());
    gradient(src.begin(),src.end(),dest.begin(),src_shift,dest_shift);
    dest.swap(dest_);
}
//---------------------------------------------------------------------------
template<typename PixelImageType,typename VectorImageType>
void gradient(const PixelImageType& src,VectorImageType& dest)
{
    dest.clear();
    dest.resize(src.shape());
    int shift = 1;
    for (unsigned int index = 0; index < PixelImageType::dimension; ++index)
    {
        typename PixelImageType::const_iterator in_from1 = src.begin();
        typename PixelImageType::const_iterator in_from2 = src.begin()+shift;
        typename VectorImageType::iterator out_from = dest.begin();
        typename PixelImageType::const_iterator in_to = src.end();
        for (; in_from2 != in_to; ++in_from1,++in_from2,++out_from)
            (*out_from)[index] = (*in_from2 - *in_from1);
        shift *= src.shape()[index];
    }
}
//---------------------------------------------------------------------------
template<typename PixelImageType,typename VectorImageType>
void gradient_sobel(const PixelImageType& src,VectorImageType& dest)
{
    dest.clear();
    dest.resize(src.shape());
    int shift = 1;
    for (unsigned int index = 0; index < PixelImageType::dimension; ++index)
    {
        typename PixelImageType::const_iterator in_from1 = src.begin();
        typename PixelImageType::const_iterator in_from2 = src.begin()+shift+shift;
        typename VectorImageType::iterator out_from = dest.begin()+shift;
        typename PixelImageType::const_iterator in_to = src.end();
        for (; in_from2 != in_to; ++in_from1,++in_from2,++out_from)
            (*out_from)[index] = (*in_from2 - *in_from1);
        shift *= src.shape()[index];
    }
}
//---------------------------------------------------------------------------
template<typename T1,typename T2,typename std::enable_if<T1::dimension==3,bool>::type = true>
void gradient_multiple_sampling(const T1& src,
                                T2& dest,
                                double line_interval = 1.0,
                                unsigned int line_sampling_num = 3,
                                unsigned int sampling_dir_num = 8)
{
    gradient(src,dest);
}
template<typename T1,typename T2,typename std::enable_if<T1::dimension==2,bool>::type = true>
void gradient_multiple_sampling(const T1& src,
                                T2& dest,
                                double line_interval = 1.0,
                                unsigned int line_sampling_num = 3,
                                unsigned int sampling_dir_num = 8)
{
    using vector_type = typename T2::value_type;
    std::vector<tipl::interpolation<tipl::linear_weighting,2> >
    interpo1(sampling_dir_num*line_sampling_num),interpo2(sampling_dir_num*line_sampling_num);
    std::vector<vector_type> offset_vector(sampling_dir_num*line_sampling_num);
    std::vector<double> offset_length(sampling_dir_num*line_sampling_num);

    {
        vector_type center(src.width() >> 1,src.height() >> 1);
        int center_index = (src.width() >> 1) + src.width()*(src.height() >> 1);
        for(unsigned int i = 0,index = 0; i < sampling_dir_num; ++i)
        {
            vector_type dist;
            dist[0] = std::cos(3.1415926*(double)i/(double)sampling_dir_num);
            dist[1] = std::sin(3.1415926*(double)i/(double)sampling_dir_num);
            for(unsigned int j = 1; j <= line_sampling_num; ++j,++index)
            {
                vector_type r = dist;
                r *= line_interval;
                r *= j;
                interpo1[index].get_location(src.shape(),center+r);
                interpo2[index].get_location(src.shape(),center-r);
                offset_vector[index] = dist;
                offset_length[index] = 2*line_interval*(double)j;
            }
        }
        for(unsigned int index = 0; index < interpo1.size(); ++index)
            for(unsigned int i = 0; i < 4; ++i)
            {
                interpo1[index].dindex[i] -= center_index;
                interpo2[index].dindex[i] -= center_index;
            }
    }

    dest.resize(src.shape());
    for (tipl::pixel_index<2> iter(src.shape()); iter < src.size(); ++iter)
    {
        vector_type dist;
        unsigned int total_dir = 0;
        for(unsigned int index = 0; index < interpo1.size(); ++index)
        {
            if(interpo1[index].dindex[0] >= 0 &&
               interpo1[index].dindex[3] < src.size() &&
               interpo2[index].dindex[0] >= 0 &&
               interpo2[index].dindex[3] < src.size())
            {
                double value1,value2;
                interpo1[index].estimate(src,value1);
                interpo2[index].estimate(src,value2);
                value1 -= value2;
                value1 /= offset_length[index];
                dist += offset_vector[index]*value1;
                ++total_dir;
            }
            for(unsigned int i = 0; i < 4; ++i)
            {
                ++(interpo1[index].dindex[i]);
                ++(interpo2[index].dindex[i]);
            }

        }
        if(total_dir)
            dest[iter.index()] = dist/(double)total_dir;
        else
            dest[iter.index()] = vector_type();
    }
}
//---------------------------------------------------------------------------
template<typename PixelImageType,typename GradientImageType>
void gradient(const PixelImageType& src,std::vector<GradientImageType>& dest)
{
    dest.resize(PixelImageType::dimension);
    unsigned int shift = 1;
    for (unsigned int index = 0; index < PixelImageType::dimension; ++index)
    {
        dest[index].resize(src.shape());
        gradient(src,dest[index],shift,0);
        shift *= src.shape()[index];
    }
}


//---------------------------------------------------------------------------
// implement -1 0 1
template<typename InputImageType,typename OutputImageType>
void gradient_2x(const InputImageType& in,OutputImageType& out)
{
    gradient(in,out,2,1);
}
//---------------------------------------------------------------------------
// implement -1
//            0
//            1
template<typename InputImageType,typename OutputImageType>
void gradient_2y(const InputImageType& in,OutputImageType& out)
{
    gradient(in,out,in.width() << 1,in.width());
}
//---------------------------------------------------------------------------
template<typename InputImageType,typename OutputImageType>
void gradient_2z(const InputImageType& in,OutputImageType& out)
{
    gradient(in,out,in.shape().plane_size() << 1,in.shape().plane_size());
}
//---------------------------------------------------------------------------
template<typename iterator1,typename iterator2>
void pdf2cdf(iterator1 lhs_from,iterator1 lhs_to,iterator2 out)
{
    iterator2 prev_out = out;
    *out = *lhs_from;
    ++out;
    for (++lhs_from; lhs_from != lhs_to; ++lhs_from,++out)
    {
        *out = *prev_out + *lhs_from;
        prev_out = out;
    }
}
//---------------------------------------------------------------------------
template<typename iterator1,typename iterator2>
void cdf2pdf(iterator1 lhs_from,iterator1 lhs_to,iterator2 out)
{
    iterator2 prev_out = out;
    *out = *lhs_from;
    ++out;
    for (++lhs_from; lhs_from != lhs_to; ++lhs_from,++out)
    {
        *out = *lhs_from-*prev_out;
        prev_out = out;
    }
}


template<typename LHType,typename RHType>
void assign_negate(LHType& lhs,const RHType& rhs)
{
    unsigned int total_size;
    total_size = (lhs.size() < rhs.size()) ? lhs.size() : rhs.size();
    typename LHType::iterator lhs_from = lhs.begin();
    typename LHType::iterator lhs_to = lhs_from + total_size;
    typename RHType::const_iterator rhs_from = rhs.begin();
    for (; lhs_from != lhs_to; ++lhs_from,++rhs_from)
        *lhs_from = -(*rhs_from);
}
//---------------------------------------------------------------------------
template<typename iterator,typename fun_type>
void apply_fun(iterator lhs_from,iterator lhs_to,fun_type fun)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from = fun(*lhs_from);
}
//---------------------------------------------------------------------------
template<typename iterator>
typename std::iterator_traits<iterator>::value_type norm2(iterator lhs_from,iterator lhs_to)
{
    typename std::iterator_traits<iterator>::value_type result(0);
    for (; lhs_from != lhs_to; ++lhs_from)
        result += (*lhs_from)*(*lhs_from);
    return std::sqrt(result);
}
//---------------------------------------------------------------------------
template<typename image_type>
typename image_type::value_type norm2(const image_type& I)
{
    return norm2(I.begin(),I.end());
}
//---------------------------------------------------------------------------
template<typename iterator>
void square(iterator lhs_from,iterator lhs_to)
{
    typename std::iterator_traits<iterator>::value_type tmp;
    for (; lhs_from != lhs_to; ++lhs_from)
    {
        tmp = *lhs_from;
        *lhs_from = tmp*tmp;
    }
}
//---------------------------------------------------------------------------
template<typename image_type>
void square(image_type& I)
{
    square(I.begin(),I.end());
}
//---------------------------------------------------------------------------
template<typename iterator>
void square_root(iterator lhs_from,iterator lhs_to)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from = std::sqrt(*lhs_from);
}
//---------------------------------------------------------------------------
template<typename image_type>
void square_root(image_type& I)
{
    square_root(I.begin(),I.end());
}
//---------------------------------------------------------------------------
template<typename iterator>
void zeros(iterator lhs_from,iterator lhs_to)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from = std::iterator_traits<iterator>::value_type(0);
}
//---------------------------------------------------------------------------
template<typename image_type>
void zeros(image_type& I)
{
    zeros(I.begin(),I.end());
}
//---------------------------------------------------------------------------
template<typename iterator>
void log(iterator lhs_from,iterator lhs_to)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from = std::log(*lhs_from);
}
//---------------------------------------------------------------------------
template<typename image_type>
void log(image_type& I)
{
    log(I.begin(),I.end());
}
//---------------------------------------------------------------------------
template<typename iterator>
void exp(iterator lhs_from,iterator lhs_to)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from = std::exp(*lhs_from);
}
//---------------------------------------------------------------------------
template<typename image_type>
void exp(image_type& I)
{
    exp(I.begin(),I.end());
}
//---------------------------------------------------------------------------
template<typename iterator>
void absolute_value(iterator lhs_from,iterator lhs_to)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from = std::abs(*lhs_from);
}
//---------------------------------------------------------------------------
template<typename image_type>
void abs(image_type& I)
{
    abs(I.begin(),I.end());
}
//---------------------------------------------------------------------------
template<typename iterator1,typename iterator2>
void add(iterator1 lhs_from,iterator1 lhs_to,iterator2 rhs_from)
{
    for (; lhs_from != lhs_to; ++lhs_from,++rhs_from)
        *lhs_from += *rhs_from;
}
//---------------------------------------------------------------------------
template<typename image_type1,typename image_type2>
void add(image_type1& I,const image_type2& I2)
{
    add(I.begin(),I.end(),I2.begin());
}
//---------------------------------------------------------------------------
template<typename image_type1,typename image_type2>
void add_mt(image_type1& I,const image_type2& I2)
{
    tipl::par_for(I.size(),[&I,&I2](int index){
       I[index] += I2[index];
    });
}
//---------------------------------------------------------------------------
template<typename iterator1,typename iterator2>
void minus(iterator1 lhs_from,iterator1 lhs_to,iterator2 rhs_from)
{
    for (; lhs_from != lhs_to; ++lhs_from,++rhs_from)
        *lhs_from -= *rhs_from;
}
//---------------------------------------------------------------------------
template<typename image_type1,typename image_type2>
void minus(image_type1& I,const image_type2& I2)
{
    minus(I.begin(),I.end(),I2.begin());
}
//---------------------------------------------------------------------------
template<typename image_type1,typename image_type2>
void minus_mt(image_type1& I,const image_type2& I2)
{
    tipl::par_for(I.size(),[&I,&I2](int index){
       I[index] -= I2[index];
    });
}
//---------------------------------------------------------------------------
template<typename iterator1,typename iterator2>
void multiply(iterator1 lhs_from,iterator1 lhs_to,iterator2 rhs_from)
{
    for (; lhs_from != lhs_to; ++lhs_from,++rhs_from)
        *lhs_from = typename std::iterator_traits<iterator1>::value_type((*lhs_from)*(*rhs_from));
}
//---------------------------------------------------------------------------
template<typename image_type1,typename image_type2>
void multiply(image_type1& I,const image_type2& I2)
{
    multiply(I.begin(),I.end(),I2.begin());
}
template<typename image_type1,typename image_type2>
void multiply_mt(image_type1& I,const image_type2& I2)
{
    tipl::par_for(I.size(),[&I,&I2](int index){
       I[index] *= I2[index];
    });
}
//---------------------------------------------------------------------------
template<typename iterator1,typename iterator2>
void divide(iterator1 lhs_from,iterator1 lhs_to,iterator2 rhs_from)
{
    for (; lhs_from != lhs_to; ++lhs_from,++rhs_from)
        *lhs_from /= *rhs_from;
}
//---------------------------------------------------------------------------
template<typename image_type1,typename image_type2>
void divide(image_type1& I,const image_type2& I2)
{
    divide(I.begin(),I.end(),I2.begin());
}
//---------------------------------------------------------------------------
template<typename iterator1,typename value_type>
void add_constant(iterator1 lhs_from,iterator1 lhs_to,value_type value)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from += value;
}
//---------------------------------------------------------------------------
template<typename image_type,typename value_type>
void add_constant(image_type& I,value_type value)
{
    add_constant(I.begin(),I.end(),value);
}
//---------------------------------------------------------------------------
template<typename iterator1,typename value_type>
void mod_constant(iterator1 lhs_from,iterator1 lhs_to,value_type value)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from %= value;
}
//---------------------------------------------------------------------------
template<typename image_type,typename value_type>
void mod_constant(image_type& I,value_type value)
{
    mod_constant(I.begin(),I.end(),value);
}
//---------------------------------------------------------------------------
template<typename iterator1,typename value_type>
void minus_constant(iterator1 lhs_from,iterator1 lhs_to,value_type value)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from -= value;
}
//---------------------------------------------------------------------------
template<typename image_type,typename value_type>
void minus_constant(image_type& I,value_type value)
{
    minus_constant(I.begin(),I.end(),value);
}
//---------------------------------------------------------------------------
template<typename image_type,typename value_type>
void minus_constant_mt(image_type& I,value_type value)
{
    tipl::par_for(I.size(),[&I,value](int index)
    {
       I[index] -= value;
    });
}
//---------------------------------------------------------------------------
template<typename iterator1,typename value_type>
void multiply_constant(iterator1 lhs_from,iterator1 lhs_to,value_type value)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from *= value;
}
//---------------------------------------------------------------------------
template<typename image_type,typename value_type>
void multiply_constant(image_type& I,value_type value)
{
    multiply_constant(I.begin(),I.end(),value);
}
template<typename image_type,typename value_type>
void multiply_constant_mt(image_type& I,value_type value)
{
    tipl::par_for(I.size(),[&I,value](int index){
       I[index] *= value;
    });
}

//---------------------------------------------------------------------------
template<typename iterator1,typename value_type>
void divide_constant(iterator1 lhs_from,iterator1 lhs_to,value_type value)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from /= value;
}
//---------------------------------------------------------------------------
template<typename image_type,typename value_type>
void divide_constant(image_type& I,value_type value)
{
    divide_constant(I.begin(),I.end(),value);
}
//---------------------------------------------------------------------------
template<typename image_type,typename value_type>
void divide_constant_mt(image_type& I,value_type value)
{
    tipl::par_for(I.size(),[&I,value](int index){
       I[index] /= value;
    });
}
//---------------------------------------------------------------------------
// perform x <- x*pow(2,y)
template<typename iterator1,typename iterator2>
void multiply_pow(iterator1 lhs_from,iterator1 lhs_to,iterator2 rhs_from)
{
    typename std::iterator_traits<iterator1>::value_type value;
    typename std::iterator_traits<iterator1>::value_type pow;
    for (; lhs_from != lhs_to; ++lhs_from,++rhs_from)
    {
        value = *lhs_from;
        pow = *rhs_from;
        if(!value || !pow)
            continue;
        if(pow < 0)
        {
            pow = -pow;
            if(value >= 0)
                *lhs_from = (value >> pow);
            else
            {
                value = -value;
                *lhs_from = -(value >> pow);
            }
        }
        else
        {
            if(value >= 0)
                *lhs_from = (value << pow);
            else
            {
                value = -value;
                *lhs_from = -(value << pow);
            }
        }
    }
}
//---------------------------------------------------------------------------
template<typename image_type1,typename image_type2>
void multiply_pow(image_type1& I,const image_type2& I2)
{
    multiply_pow(I.begin(),I.end(),I2.begin());
}
//---------------------------------------------------------------------------
// perform x <- x*2^pow
template<typename iterator1,typename value_type>
void multiply_pow_constant(iterator1 lhs_from,iterator1 lhs_to,value_type pow)
{
    typename std::iterator_traits<iterator1>::value_type value;
    if(pow == 0)
        return;
    if(pow > 0)
    {
        for (; lhs_from != lhs_to; ++lhs_from)
        {
            value = *lhs_from;
            if(value >= 0)
                *lhs_from = (value << pow);
            else
            {
                value = -value;
                *lhs_from = -(value << pow);
            }
        }
    }
    else
    {
        pow = -pow;
        for (; lhs_from != lhs_to; ++lhs_from)
        {
            value = *lhs_from;
            if(value >= 0)
                *lhs_from = (value >> pow);
            else
            {
                value = -value;
                *lhs_from = -(value >> pow);
            }
        }
    }
}
//---------------------------------------------------------------------------
template<typename image_type,typename value_type>
void multiply_pow_constant(image_type& I,value_type value)
{
    multiply_pow_constant(I.begin(),I.end(),value);
}
//---------------------------------------------------------------------------
// perform x <- x/pow(2,y)
template<typename iterator1,typename iterator2>
void divide_pow(iterator1 lhs_from,iterator1 lhs_to,iterator2 rhs_from)
{
    typename std::iterator_traits<iterator1>::value_type value;
    typename std::iterator_traits<iterator1>::value_type pow;
    for (; lhs_from != lhs_to; ++lhs_from,++rhs_from)
    {
        value = *lhs_from;
        pow = *rhs_from;
        if(!value || !pow)
            continue;
        if(pow < 0)
        {
            pow = -pow;
            if(value >= 0)
                *lhs_from = (value << pow);
            else
            {
                value = -value;
                *lhs_from = -(value << pow);
            }
        }
        else
        {
            if(value >= 0)
                *lhs_from = (value >> pow);
            else
            {
                value = -value;
                *lhs_from = -(value >> pow);
            }
        }
    }
}
//---------------------------------------------------------------------------
template<typename image_type1,typename image_type2>
void divide_pow(image_type1& I,const image_type2& I2)
{
    divide_pow(I.begin(),I.end(),I2.begin());
}
//---------------------------------------------------------------------------
// perform x <- x/2^pow
template<typename iterator1,typename value_type>
void divide_pow_constant(iterator1 lhs_from,iterator1 lhs_to,value_type pow)
{
    typename std::iterator_traits<iterator1>::value_type value;
    if(pow == 0)
        return;
    if(pow > 0)
    {
        for (; lhs_from != lhs_to; ++lhs_from)
        {
            value = *lhs_from;
            if(value >= 0)
                *lhs_from = (value >> pow);
            else
            {
                value = -value;
                *lhs_from = -(value >> pow);
            }
        }
    }
    else
    {
        pow = -pow;
        for (; lhs_from != lhs_to; ++lhs_from)
        {
            value = *lhs_from;
            if(value >= 0)
                *lhs_from = (value << pow);
            else
            {
                value = -value;
                *lhs_from = -(value << pow);
            }
        }
    }
}
//---------------------------------------------------------------------------
template<typename image_type,typename value_type>
void divide_pow_constant(image_type& I,value_type value)
{
    divide_pow_constant(I.begin(),I.end(),value);
}
//---------------------------------------------------------------------------
template<typename container_type>
typename container_type::value_type max_abs_value(const container_type& image)
{
    typename container_type::value_type max_value = 0;
    auto from = image.begin();
    auto to = image.end();
    for (; from != to; ++from)
    {
        float value = *from;
        if (value > max_value)
            max_value = value;
        else if (-value > max_value)
            max_value = -value;
    }
    return max_value;
}


template<typename iterator_type>
std::pair<typename std::iterator_traits<iterator_type>::value_type,typename std::iterator_traits<iterator_type>::value_type>
min_max_value(iterator_type iter,iterator_type end)
{
    if(iter == end)
        return std::make_pair(0,0);
    typename std::iterator_traits<iterator_type>::value_type min_value = *iter;
    typename std::iterator_traits<iterator_type>::value_type max_value = *iter;
    typename std::iterator_traits<iterator_type>::value_type value;
    for(++iter; iter != end; ++iter)
    {
        value = *iter;
        if(value > max_value)
            max_value = value;
        else if(value < min_value)
            min_value = value;
    }
    return std::make_pair(min_value,max_value);
}

//---------------------------------------------------------------------------
template<typename iterator_type>
std::pair<typename std::iterator_traits<iterator_type>::value_type,typename std::iterator_traits<iterator_type>::value_type>
min_max_value_mt(iterator_type iter,iterator_type end)
{
    if(iter == end)
        return std::make_pair(0,0);
    auto n = std::thread::hardware_concurrency();
    size_t size = size_t(end-iter);
    typedef typename std::iterator_traits<iterator_type>::value_type value_type;
    std::vector<value_type> max_v(n,*iter),min_v(n,*iter);
    tipl::par_for(n,[&](uint16_t thread)
    {
        value_type& max_value = max_v[thread];
        value_type& min_value = min_v[thread];
        for(size_t i = 0;i < size;i += n)
        {
            auto value = iter[i];
            if(value > max_value)
                max_value = value;
            else if(value < min_value)
                min_value = value;
        }
    });
    return std::make_pair(*std::min_element(min_v.begin(),min_v.end()),
                          *std::max_element(max_v.begin(),max_v.end()));
}


template<typename InputIter,typename OutputIter>
void normalize(InputIter from,InputIter to,OutputIter out,float upper_limit = 255.0)
{
    typedef typename std::iterator_traits<InputIter>::value_type value_type;
    std::pair<value_type,value_type> min_max(min_max_value(from,to));
    value_type range = min_max.second-min_max.first;
    if(range == 0)
        return;
    upper_limit /= range;
    for(;from != to;++from,++out)
        *out = (*from-min_max.first)*upper_limit;
}
//---------------------------------------------------------------------------
template<typename InputIter,typename OutputIter,typename value_type>
void upper_threshold(InputIter from,InputIter to,OutputIter out,value_type upper)
{
    for(;from != to;++from,++out)
        *out = std::min<value_type>(*from,upper);
}
//---------------------------------------------------------------------------
template<typename InputIter,typename value_type>
void upper_threshold(InputIter from,InputIter to,value_type upper)
{
    for(;from != to;++from)
        *from = std::min<value_type>(*from,upper);
}
//---------------------------------------------------------------------------
template<typename image_type,typename value_type>
void upper_threshold(image_type& I,value_type value)
{
    upper_threshold(I.begin(),I.end(),value);
}
//---------------------------------------------------------------------------
template<typename InputIter,typename OutputIter,typename value_type>
void lower_threshold(InputIter from,InputIter to,OutputIter out,value_type lower)
{
    for(;from != to;++from,++out)
        *out = std::max<value_type>(*from,lower);
}
//---------------------------------------------------------------------------
template<typename InputIter,typename value_type>
void lower_threshold(InputIter from,InputIter to,value_type lower)
{
    for(;from != to;++from)
        *from = std::max<value_type>(*from,lower);
}
//---------------------------------------------------------------------------
template<typename image_type,typename value_type>
void lower_threshold(image_type& I,value_type value)
{
    lower_threshold(I.begin(),I.end(),value);
}
//---------------------------------------------------------------------------
template<typename InputIter,typename OutputIter,typename value_type>
void upper_lower_threshold(InputIter from,InputIter to,OutputIter out,value_type lower,value_type upper)
{
    for(;from != to;++from,++out)
        *out = std::min<value_type>(std::max<value_type>(*from,lower),upper);
}

template<typename ImageType,typename value_type>
void upper_lower_threshold(ImageType& data,value_type lower,value_type upper)
{
    typename ImageType::iterator from = data.begin();
    typename ImageType::iterator to = data.end();
    for(;from != to;++from)
        *from = std::min<value_type>(std::max<value_type>(*from,lower),upper);
}

template<typename ImageType>
ImageType& normalize(ImageType& I,float upper_limit = 255)
{
    if(I.empty())
        return I;
    auto m = (*std::max_element(I.begin(),I.end()));
    if( m != 0)
        multiply_constant(I.begin(),I.end(),upper_limit/m);
    return I;
}
template<typename ImageType>
ImageType& normalize_abs(ImageType& I,float upper_limit = 1.0f)
{
    if(I.empty())
        return I;
    auto minmax = std::minmax_element(I.begin(),I.end());
    auto scale = std::max(-*minmax.first,*minmax.second);
    if(scale != 0)
    multiply_constant(I.begin(),I.end(),upper_limit/scale);
    return I;
}

template<typename ImageType1,typename ImageType2>
ImageType2& normalize(const ImageType1& image1,ImageType2& image2,float upper_limit = 255.0)
{
    image2.resize(image1.shape());
    normalize(image1.begin(),image1.end(),image2.begin(),upper_limit);
    return image2;
}

template<typename container_type,typename index_type>
void get_sort_index(const container_type& c,std::vector<index_type>& idx)
{
    idx.resize(c.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),[&c](size_t i, size_t j){return c[i] < c[j];});
}

template<typename container_type,typename index_type>
void apply_sort_index(container_type& c,const std::vector<index_type>& idx)
{
    container_type new_c(c.size());
    for(size_t i = 0;i < idx.size();++i)
        new_c[i] = c[idx[i]];
    c.swap(new_c);
}

template<typename I_type>
tipl::vector<I_type::dimension,float> center_of_mass(const I_type& Im)
{
    tipl::vector<I_type::dimension,float> sum_mass;
    double total_w = 0.0;
    Im.for_each([&](typename I_type::value_type v,
                    tipl::pixel_index<I_type::dimension> index)
    {
        total_w += v;
        tipl::vector<I_type::dimension,float> pos(index);
        pos *= v;
        sum_mass += pos;
    });
    sum_mass /= total_w;
    return sum_mass;
}



}
#endif
