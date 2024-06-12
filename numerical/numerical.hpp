//---------------------------------------------------------------------------
#ifndef NUMERICAL_HPP
#define NUMERICAL_HPP
#include <random>
#include <string>
#include <stdexcept>
#include "../mt.hpp"
#include "interpolation.hpp"

#ifdef __CUDACC__
#include "../cu.hpp"
#endif

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
    size_t shift = 1;
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
    size_t shift = 1;
    for (unsigned int index = 0; index < PixelImageType::dimension; ++index)
    {
        auto in_from1 = src.begin();
        auto in_from2 = src.begin()+shift+shift;
        auto out_from = dest.begin()+shift;
        auto in_to = src.end();
        for (; in_from2 != in_to; ++in_from1,++in_from2,++out_from)
            (*out_from)[index] = (*in_from2 - *in_from1);
        shift *= src.shape()[index];
    }
}
//---------------------------------------------------------------------------
template<typename PixelImageType,typename GradientImageType>
void gradient(const PixelImageType& src,std::vector<GradientImageType>& dest)
{
    dest.resize(PixelImageType::dimension);
    size_t shift = 1;
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
//---------------------------------------------------------------------------
template<typename iterator>
__INLINE__ auto norm2(iterator lhs_from,iterator lhs_to)
{
    typename std::iterator_traits<iterator>::value_type result(0);
    for (; lhs_from != lhs_to; ++lhs_from)
        result += (*lhs_from)*(*lhs_from);
    return std::sqrt(result);
}
//---------------------------------------------------------------------------
template<typename image_type>
__INLINE__ auto norm2(const image_type& I)
{
    return norm2(I.begin(),I.end());
}
//---------------------------------------------------------------------------
template<typename iterator>
__INLINE__ void square(iterator lhs_from,iterator lhs_to)
{
    for (; lhs_from != lhs_to; ++lhs_from)
    {
        auto tmp = *lhs_from;
        *lhs_from = tmp*tmp;
    }
}
//---------------------------------------------------------------------------
template<typename image_type>
__INLINE__ void square(image_type& I)
{
    square(I.begin(),I.end());
}
//---------------------------------------------------------------------------
template<typename iterator>
__INLINE__ void square_root(iterator lhs_from,iterator lhs_to)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from = std::sqrt(*lhs_from);
}
//---------------------------------------------------------------------------
template<typename image_type>
__INLINE__ void square_root(image_type& I)
{
    square_root(I.begin(),I.end());
}
//---------------------------------------------------------------------------
template<typename iterator>
__INLINE__ void log(iterator lhs_from,iterator lhs_to)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from = std::log(*lhs_from);
}
//---------------------------------------------------------------------------
template<typename image_type>
__INLINE__ void log(image_type& I)
{
    log(I.begin(),I.end());
}
//---------------------------------------------------------------------------
template<typename iterator>
__INLINE__ void exp(iterator lhs_from,iterator lhs_to)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from = std::exp(*lhs_from);
}
//---------------------------------------------------------------------------
template<typename image_type>
__INLINE__ void exp(image_type& I)
{
    exp(I.begin(),I.end());
}
//---------------------------------------------------------------------------
template<typename iterator>
__INLINE__ void absolute_value(iterator lhs_from,iterator lhs_to)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from = std::abs(*lhs_from);
}
//---------------------------------------------------------------------------
template<typename image_type>
__INLINE__ void abs(image_type& I)
{
    abs(I.begin(),I.end());
}
//---------------------------------------------------------------------------
template<typename iterator1,typename iterator2>
__INLINE__ void add(iterator1 lhs_from,iterator1 lhs_to,iterator2 rhs_from)
{
    for (; lhs_from != lhs_to; ++lhs_from,++rhs_from)
        *lhs_from += *rhs_from;
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T,typename U>
__global__ void add_cuda_kernel(T I,U I2)
{
    TIPL_FOR(index,I.size())
        I[index] += I2[index];
}
#endif
//---------------------------------------------------------------------------
template<typename T,typename U>
void add(T& I,const U& I2)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(add_cuda_kernel,I.size())
            (tipl::make_shared(I),tipl::make_shared(I2));
        #endif
    }
    else
        tipl::par_for(I.size(),[&I,&I2](size_t index)
        {
            I[index] += I2[index];
        });
}
//---------------------------------------------------------------------------
template<typename iterator1,typename iterator2>
__INLINE__ void minus(iterator1 lhs_from,iterator1 lhs_to,iterator2 rhs_from)
{
    for (; lhs_from != lhs_to; ++lhs_from,++rhs_from)
        *lhs_from -= *rhs_from;
}
//---------------------------------------------------------------------------
template<typename image_type1,typename image_type2>
inline void minus(image_type1& I,const image_type2& I2)
{
    tipl::par_for(I.size(),[&I,&I2](size_t index)
    {
        I[index] -= I2[index];
    });
}
template<typename iterator1,typename iterator2>
__INLINE__ void multiply(iterator1 lhs_from,iterator1 lhs_to,iterator2 rhs_from)
{
    for (; lhs_from != lhs_to; ++lhs_from,++rhs_from)
        *lhs_from = typename std::iterator_traits<iterator1>::value_type((*lhs_from)*(*rhs_from));
}

template<typename image_type1,typename image_type2>
inline void multiply(image_type1& I,const image_type2& I2)
{
    tipl::par_for(I.size(),[&I,&I2](size_t index)
    {
        I[index] *= I2[index];
    });
}
template<typename iterator1,typename iterator2>
__INLINE__ void divide(iterator1 lhs_from,iterator1 lhs_to,iterator2 rhs_from)
{
    for (; lhs_from != lhs_to; ++lhs_from,++rhs_from)
        *lhs_from /= *rhs_from;
}
//---------------------------------------------------------------------------
template<typename image_type1,typename image_type2>
inline void divide(image_type1& I,const image_type2& I2)
{
    tipl::par_for(I.size(),[&I,&I2](size_t index)
    {
        I[index] /= I2[index];
    });
}
//---------------------------------------------------------------------------
template<typename image_type1,typename image_type2>
void greater(image_type1& I,const image_type2& I2)
{
    tipl::par_for(I.size(),[&I,&I2](size_t index)
    {
        I[index] = (I[index] > I2[index] ? 1 : 0);
    });
}
//---------------------------------------------------------------------------
template<typename image_type1,typename image_type2>
inline void lesser(image_type1& I,const image_type2& I2)
{
    tipl::par_for(I.size(),[&I,&I2](size_t index)
    {
        I[index] = (I[index] < I2[index] ? 1 : 0);
    });
}
//---------------------------------------------------------------------------
template<typename image_type1,typename image_type2>
inline void equal(image_type1& I,const image_type2& I2)
{
    tipl::par_for(I.size(),[&I,&I2](size_t index)
    {
        I[index] = (I[index] == I2[index] ? 1 : 0);
    });
}
//---------------------------------------------------------------------------
template<typename iterator1,typename value_type>
__INLINE__ void add_constant(iterator1 lhs_from,iterator1 lhs_to,value_type value)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from += value;
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T,typename U>
__global__ void add_constant_cuda_kernel(T I,U v)
{
    TIPL_FOR(index,I.size())
        I[index] += v;
}
#endif
//---------------------------------------------------------------------------
template<typename T,typename U>
void add_constant(T& I,U v)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(add_constant_cuda_kernel,I.size())
            (tipl::make_shared(I),v);
        #endif
    }
    else
        tipl::par_for(I.size(),[&I,v](size_t index)
        {
           I[index] += v;
        });
}
//---------------------------------------------------------------------------
template<typename iterator1,typename value_type>
__INLINE__ void mod_constant(iterator1 lhs_from,iterator1 lhs_to,value_type value)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from %= value;
}
//---------------------------------------------------------------------------
template<typename image_type,typename value_type>
inline void mod_constant(image_type& I,value_type value)
{
    tipl::par_for(I.size(),[&I,value](size_t index)
    {
        I[index] %= value;
    });
}
//---------------------------------------------------------------------------
template<typename iterator1,typename value_type>
__INLINE__ void minus_constant(iterator1 lhs_from,iterator1 lhs_to,value_type value)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from -= value;
}
//---------------------------------------------------------------------------
template<typename image_type,typename value_type>
inline void minus_constant(image_type& I,value_type value)
{
    tipl::par_for(I.size(),[&I,value](size_t index)
    {
        I[index] -= value;
    });
}
//---------------------------------------------------------------------------
template<typename iterator1,typename value_type>
__INLINE__ void minus_by_constant(iterator1 lhs_from,iterator1 lhs_to,value_type value)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from = value - *lhs_from;
}
//---------------------------------------------------------------------------
template<typename image_type,typename value_type>
inline void minus_by_constant(image_type& I,value_type value)
{
    tipl::par_for(I.size(),[&I,value](size_t index)
    {
        I[index] = value - I[index];
    });
}

//---------------------------------------------------------------------------
template<typename iterator1,typename value_type>
__INLINE__ void multiply_constant(iterator1 lhs_from,iterator1 lhs_to,value_type value)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from *= value;
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T,typename U>
__global__ void multiply_constant_cuda_kernel(T I,U v)
{
    TIPL_FOR(index,I.size())
        I[index] *= v;
}
#endif
//---------------------------------------------------------------------------
template<typename T,typename U>
void multiply_constant(T& I,U value)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(multiply_constant_cuda_kernel,I.size())
            (tipl::make_shared(I),value);
        #endif
    }
    else
        tipl::par_for(I.size(),[&I,value](size_t index){
           I[index] *= value;
        });
}

template<typename iterator1,typename value_type>
__INLINE__ void divide_constant(iterator1 lhs_from,iterator1 lhs_to,value_type value)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from /= value;
}

template<typename image_type,typename value_type>
inline void divide_constant(image_type& I,value_type value)
{
    tipl::par_for(I.size(),[&I,value](size_t index)
    {
        I[index] /= value;
    });
}
template<typename iterator1,typename value_type>
inline void divide_by_constant(iterator1 lhs_from,iterator1 lhs_to,value_type value)
{
    for (; lhs_from != lhs_to; ++lhs_from)
        *lhs_from = value/(*lhs_from);
}
template<typename image_type,typename value_type>
inline void divide_by_constant(image_type& I,value_type value)
{
    tipl::par_for(I.size(),[&I,value](size_t index)
    {
        I[index] = value/I[index];
    });
}
template<typename image_type,typename value_type>
inline void greater_constant(image_type& I,value_type value)
{
    tipl::par_for(I.size(),[&I,value](size_t index)
    {
       I[index] = (I[index] > value ? 1 : 0);
    });
}
template<typename image_type,typename value_type>
inline void lesser_constant(image_type& I,value_type value)
{
    tipl::par_for(I.size(),[&I,value](size_t index)
    {
        I[index] = (I[index] < value ? 1 : 0);
    });
}
template<typename image_type,typename value_type>
inline void equal_constant(image_type& I,value_type value)
{
    tipl::par_for(I.size(),[&I,value](size_t index)
    {
        I[index] = (I[index] == value ? 1 : 0);
    });
}
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
inline void multiply_pow(image_type1& I,const image_type2& I2)
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

template<typename image_type,typename value_type>
void divide_pow_constant(image_type& I,value_type value)
{
    divide_pow_constant(I.begin(),I.end(),value);
}

template<typename input_iterator>
__INLINE__ auto min_value(input_iterator from,input_iterator to)
{
    if(from == to)
        return typename std::iterator_traits<input_iterator>::value_type(0);
    auto m = *from;
    for(;from != to;++from)
        if(*from < m)
            m = *from;
    return m;
}

template<typename T>
auto min_value(const T& data)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
            return device_eval(thrust::min_element(thrust::device,data.data(),data.data()+data.size()));
        #endif
    }
    else
    {
        if(data.size() < 10000 || available_thread_count() < 2)
            return min_value(data.begin(),data.end());

        std::mutex mutex;
        auto min_v = data[0];
        tipl::par_for<ranged>(data.begin(),data.end(),[&](auto beg,auto end)
        {
            auto v = min_value(beg,end);
            std::lock_guard<std::mutex> lock(mutex);
            if(v < min_v)
                min_v = v;
        });
        return min_v;
    }
}

template<typename input_iterator>
__INLINE__ auto max_value(input_iterator from,input_iterator to)
{
    if(from == to)
        return typename std::iterator_traits<input_iterator>::value_type();
    auto m = *from;
    for(;from != to;++from)
        if(*from > m)
            m = *from;
    return m;
}

template<typename T>
auto max_value(const T& data)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
            return device_eval(thrust::max_element(thrust::device,data.data(),data.data()+data.size()));
        #endif
    }
    else
    {
        if(data.size() < 10000 || available_thread_count() < 2)
            return max_value(data.begin(),data.end());
        std::mutex mutex;
        auto max_v = data[0];

        tipl::par_for<ranged>(data.begin(),data.end(),[&](auto beg,auto end)
        {
            auto v = max_value(beg,end);
            std::lock_guard<std::mutex> lock(mutex);
            if(v > max_v)
                max_v = v;
        });
        return max_v;
    }
}



template<typename container_type>
__INLINE__ typename container_type::value_type max_abs_value(const container_type& image)
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
__INLINE__
std::pair<typename std::iterator_traits<iterator_type>::value_type,typename std::iterator_traits<iterator_type>::value_type>
minmax_value(iterator_type iter,iterator_type end)
{
    if(iter == end)
        return std::make_pair(0,0);
    auto min_value = *iter;
    auto max_value = *iter;
    for(++iter; iter != end; ++iter)
    {
        auto value = *iter;
        if(value > max_value)
            max_value = value;
        else if(value < min_value)
            min_value = value;
    }
    return std::make_pair(min_value,max_value);
}

template<typename T>
__HOST__
std::pair<typename T::value_type,typename T::value_type>
minmax_value(const T& data)
{
    if(data.empty())
        return std::make_pair(0,0);
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        auto result = thrust::minmax_element(thrust::device,
                                     data.data(),data.data()+data.size());
        return std::make_pair(device_eval(result.first),device_eval(result.second));
        #endif
    }
    else
    {
        if(data.size() < 10000 || available_thread_count() < 2)
            return minmax_value(data.begin(),data.end());

        std::mutex mutex;
        auto min_v = data[0];
        auto max_v = data[0];
        tipl::par_for<ranged>(data.begin(),data.end(),[&](auto beg,auto end)
        {
            auto result = minmax_value(beg,end);
            std::lock_guard<std::mutex> lock(mutex);
            if(result.first < min_v)
                min_v = result.first;
            if(result.second > max_v)
                max_v = result.second;
        });
        return std::make_pair(min_v,max_v);
    }
}
//---------------------------------------------------------------------------
template<typename iterator1,typename iterator2>
__INLINE__ void masking(iterator1 lhs_from,iterator1 lhs_to,iterator2 rhs_from)
{
    for (; lhs_from != lhs_to; ++lhs_from,++rhs_from)
        if(*rhs_from)
            *lhs_from = 0;
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T,typename U>
__global__ void masking_kernel(T I,U I2)
{
    TIPL_FOR(index,I.size())
        if(I2[index])
            I[index] = 0;
}
#endif
//---------------------------------------------------------------------------
template<typename T,typename U>
void masking(T& I,const U& I2)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(masking_kernel,I.size())
            (tipl::make_shared(I),tipl::make_shared(I2));
        #endif
    }
    else
        tipl::par_for(I.size(),[&I,&I2](size_t index){
            if(I2[index])
                I[index] = 0;
        });
}
//---------------------------------------------------------------------------
template<typename iterator1,typename iterator2>
__INLINE__ void preserve(iterator1 lhs_from,iterator1 lhs_to,iterator2 rhs_from)
{
    for (; lhs_from != lhs_to; ++lhs_from,++rhs_from)
        if(!*rhs_from)
            *lhs_from = 0;
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T,typename U>
__global__ void preserve_kernel(T I,U I2)
{
    TIPL_FOR(index,I.size())
        if(!I2[index])
            I[index] = 0;
}
#endif
//---------------------------------------------------------------------------
template<typename T,typename U>
void preserve(T& I,const U& I2)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(preserve_kernel,I.size())
            (tipl::make_shared(I),tipl::make_shared(I2));
        #endif
    }
    else
        tipl::par_for(I.size(),[&I,&I2](size_t index){
            if(!I2[index])
                I[index] = 0;
        });
}
//---------------------------------------------------------------------------
template<typename InputIter,typename value_type>
__INLINE__ void upper_threshold(InputIter from,InputIter to,value_type upper)
{
    for(;from != to;++from)
        if(*from > upper)
            *from = upper;
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T1,typename U>
__global__ void upper_threshold_cuda_kernel(T1 I,U v)
{
    TIPL_FOR(index,I.size())
        if(I[index] > v)
            I[index] = v;
}
#endif
//---------------------------------------------------------------------------
template<typename T,typename V>
void upper_threshold(T& I,V value)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(upper_threshold_cuda_kernel,I.size())
            (tipl::make_shared(I),value);
        #endif
    }
    else
    {
        if(I.size() < 1000 || available_thread_count() < 2)
        {
            upper_threshold(I.begin(),I.end(),value);
            return;
        }
        tipl::par_for<ranged>(I.begin(),I.end(),[&](auto beg,auto end)
        {
            upper_threshold(beg,end,value);
        });
    }
}
//---------------------------------------------------------------------------
template<typename InputIter,typename value_type>
__INLINE__ void lower_threshold(InputIter from,InputIter to,value_type lower)
{
    for(;from != to;++from)
        if(*from < lower)
            *from = lower;
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T1,typename U>
__global__ void lower_threshold_cuda_kernel(T1 I,U v)
{
    TIPL_FOR(index,I.size())
        if(I[index] < v)
            I[index] = v;
}
#endif
//---------------------------------------------------------------------------
template<typename T,typename V>
void lower_threshold(T& I,V value)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(lower_threshold_cuda_kernel,I.size())
            (tipl::make_shared(I),value);
        #endif
    }
    else
    {
        if(I.size() < 1000 || available_thread_count() < 2)
        {
            lower_threshold(I.begin(),I.end(),value);
            return;
        }
        tipl::par_for<ranged>(I.begin(),I.end(),[&](auto beg,auto end)
        {
            lower_threshold(beg,end,value);
        });
    }
}
//---------------------------------------------------------------------------
template<typename T,typename V>
__INLINE__ void upper_lower_threshold(T from,T to,V lower,V upper)
{
    for(;from != to;++from)
    {
        auto v = *from;
        if(v > upper)
            *from = upper;
        else
            if(v < lower)
                *from = lower;
    }
}
//---------------------------------------------------------------------------
template<typename T,typename U,typename V>
__INLINE__ void upper_lower_threshold2(T from,T to,U out,V lower,V upper)
{
    for(;from != to;++from,++out)
    {
        auto v = *from;
        if(v > upper)
            *out = upper;
        else
            if(v < lower)
                *out = lower;
    }
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T1,typename U>
__global__ void upper_lower_threshold_cuda_kernel(T1 I,U lower,U upper)
{
    TIPL_FOR(index,I.size())
    {
        if(I[index] > upper)
            I[index] = upper;
        else
            if(I[index] < lower)
                I[index] = lower;
    }
}
#endif
template<typename T>
void upper_lower_threshold(T& I,typename T::value_type lower,typename T::value_type upper)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(upper_lower_threshold_cuda_kernel,I.size())
            (tipl::make_shared(I),lower,upper);
        #endif
    }
    else
    {
        if(I.size() < 1000 || available_thread_count() < 2)
        {
            upper_lower_threshold(I.begin(),I.end(),lower,upper);
            return;
        }
        tipl::par_for<ranged>(I.begin(),I.end(),[&](auto beg,auto end)
        {
            upper_lower_threshold(beg,end,lower,upper);
        });
    }
}

template<typename InputIter,typename OutputIter>
__INLINE__ void normalize_upper_lower2(InputIter from,InputIter to,OutputIter out,float upper_limit = 255.0)
{
		using MinMaxType = typename std::iterator_traits<InputIter>::value_type;

    std::pair<MinMaxType,MinMaxType> min_max(minmax_value(from,to));
    auto range = min_max.second-min_max.first;
    if(range == 0)
        return;
    upper_limit /= range;
    for(;from != to;++from,++out)
        *out = (*from-min_max.first)*upper_limit;
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T1, typename T2,typename value_type>
__global__  void normalize_upper_lower2_cuda_kernel(T1 in,T2 out,value_type min,value_type coef)
{
    TIPL_FOR(index,in.size())
        out[index] = value_type(in[index]-min)*coef;
}
#endif
//---------------------------------------------------------------------------
template<typename T,typename U>
void normalize_upper_lower2(const T& in,U& out,float upper_limit = 255.0)
{
    auto min_max = minmax_value(in);
    auto range = min_max.second-min_max.first;
    if(range == 0)
        return;
    upper_limit /= range;
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(normalize_upper_lower2_cuda_kernel,in.size())
            (tipl::make_shared(in),tipl::make_shared(out),float(min_max.first),upper_limit);
        #endif
    }
    else
    {
        par_for(in.size(),[&](size_t i)
        {
            out[i] = (in[i]-min_max.first)*upper_limit;
        });
    }
}

template<typename ImageType>
inline void normalize_upper_lower(ImageType& I,float upper_limit = 255.0)
{
    normalize_upper_lower2(I.begin(),I.end(),I.begin(),upper_limit);
}

template<typename ImageType>
void normalize(ImageType& I,float upper_limit = 1.0f)
{
    if(I.empty())
        return;
    auto m = max_value(I);
    if(m != 0)
        multiply_constant(I,upper_limit/m);
}



template<typename ImageType>
void normalize_abs(ImageType& I,float upper_limit = 1.0f)
{
    if(I.empty())
        return;
    auto minmax = minmax_value(I);
    auto scale = std::max(-*minmax.first,*minmax.second);
    if(scale != 0)
        multiply_constant(I,upper_limit/scale);
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
tipl::vector<I_type::dimension,float> center_of_mass_weighted(const I_type& Im)
{
    std::vector<tipl::vector<I_type::dimension,float> > sum_mass(std::thread::hardware_concurrency());
    std::vector<double> total_w(std::thread::hardware_concurrency());
    tipl::par_for(tipl::begin_index(Im.shape()),tipl::end_index(Im.shape()),
                        [&](const tipl::pixel_index<I_type::dimension>& index,size_t id)
    {
        auto v = Im[index.index()];
        total_w[id] += v;
        tipl::vector<I_type::dimension,float> pos(index);
        pos *= v;
        sum_mass[id] += pos;
    });
    for(size_t i = 1;i < sum_mass.size();++i)
    {
        sum_mass[0] += sum_mass[i];
        total_w[0] += total_w[i];
    }
    if(total_w[0] != 0.0)
        sum_mass[0] /= total_w[0];
    return sum_mass[0];
}

template<typename I_type>
auto center_of_mass_binary(const I_type& Im)
{
    std::vector<tipl::vector<I_type::dimension> > sum_mass(std::thread::hardware_concurrency());
    std::vector<size_t> total_w(std::thread::hardware_concurrency());
    tipl::par_for(tipl::begin_index(Im.shape()),tipl::end_index(Im.shape()),
                        [&](const tipl::pixel_index<I_type::dimension>& index,size_t id)
    {
        if(Im[index.index()])
        {
            ++total_w[id];
            sum_mass[id] += tipl::vector<I_type::dimension>(index);
        }
    });
    for(size_t i = 1;i < sum_mass.size();++i)
    {
        sum_mass[0] += sum_mass[i];
        total_w[0] += total_w[i];
    }
    if(total_w[0] != 0.0)
        sum_mass[0] /= float(total_w[0]);
    return sum_mass[0];
}

}
#endif
