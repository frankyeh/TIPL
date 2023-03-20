//---------------------------------------------------------------------------
#ifndef SHAPE_HPP
#define SHAPE_HPP
#include <functional>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "../def.hpp"
//---------------------------------------------------------------------------
namespace tipl
{

template<int d_>
class shape
{
public:
    static constexpr int dimension = d_;
    unsigned int dim[dimension];
public:

public:
    __INLINE__ shape(void)
    {
        clear();
    }
    template<typename T,typename U>
    __INLINE__ shape(T x,T y,T z,U t)
    {
        dim[0] = uint32_t(x);
        dim[1] = uint32_t(y);
        dim[2] = uint32_t(z);
        dim[3] = uint32_t(t);
    }
    __INLINE__ shape(const shape<dimension>& rhs)
    {
        *this = rhs;
    }
    template<typename pointer_type>
    __INLINE__ explicit shape(const pointer_type* rhs)
    {
        *this = rhs;
    }
    __INLINE__ const shape<dimension>& operator=(const shape<dimension>& rhs)
    {
	if(this == &rhs)
            return *this;
        std::copy(rhs.dim,rhs.dim+dimension,dim);
        return *this;
    }
    template<typename pointer_type>
    __INLINE__ const shape<dimension>& operator=(const pointer_type* rhs)
    {
        std::copy(rhs,rhs+dimension,dim);
        return *this;
    }
public:
    __INLINE__ size_t size(void) const
    {
        size_t prod = dim[0];
        for (int index = 1;index < dimension;++index)
            prod *= size_t(dim[index]);
        return prod;
    }
    __INLINE__ size_t plane_size(void) const
    {
        return size_t(dim[0])*size_t(dim[1]);
    }
    __INLINE__ void clear(void)
    {
        std::fill(dim,dim+dimension,0);
    }
    __INLINE__ void swap(shape<dimension>& rhs) noexcept
    {
        for (int index = 1;index < dimension;++index)
            std::swap(dim[index],rhs.dim[index]);
    }
public:
    __INLINE__ const unsigned int* begin(void) const
    {
        return dim;
    }
    __INLINE__ const unsigned int* end(void) const
    {
        return dim+dimension;
    }
    __INLINE__ unsigned int operator[](int index) const
    {
        return dim[index];
    }
    template<typename rhs_type>
    __INLINE__ shape operator*(rhs_type value) const
    {
        shape new_shape = *this;
        for(unsigned int i = 0;i < dimension;++i)
            new_shape.dim[i] *= value;
        return new_shape;
    }
    __INLINE__ unsigned int* begin(void)
    {
        return dim;
    }
    __INLINE__ unsigned int* end(void)
    {
        return dim+dimension;
    }
    template<typename index_type>
    __INLINE__ unsigned int& operator[](index_type index)
    {
        return dim[index];
    }
public:
    template<typename T>
    __INLINE__ bool is_valid(const T& pos) const
    {
        for (int index = 0;index < dimension;++index)
            if (pos[index] >= 0 || pos[index] < dim[index])
                return false;
        return true;
    }
    template<typename T>
    __INLINE__ bool is_edge(const T& pos) const
    {
        for (int index = 0;index < dimension;++index)
            if (pos[index] == 0 || pos[index]+1 == dim[index])
                return true;
        return false;
    }
public:
    __INLINE__ unsigned int width(void) const
    {
        return dim[0];
    }
    __INLINE__ unsigned int height(void) const
    {
        return dim[1];
    }
    __INLINE__ unsigned int depth(void) const
    {
        return dim[2];
    }
public:
    __INLINE__ bool operator==(const shape<dimension>& rhs) const
    {
        for (int index = 0;index < dimension;++index)
            if (dim[index] != rhs.dim[index])
                return false;
        return true;
    }
    __INLINE__ bool operator!=(const shape<3>& rhs) const
    {
        return !(*this == rhs);
    }

};



template<>
class shape<1>
{
    union
    {
        unsigned int dim[1];
        struct
        {
            unsigned int w;
        };
    };
public:
    static constexpr int dimension = 1;
public:
    __INLINE__ shape(void):w(0) {}
    __INLINE__ shape(unsigned int w_):w(w_) {}

    __INLINE__ shape(const shape<1>& rhs)
    {
        *this = rhs;
    }
    template<typename pointer_type>
    __INLINE__ explicit shape(const pointer_type* rhs)
    {
        *this = rhs;
    }
    __INLINE__ const shape<1>& operator=(const shape<1>& rhs)
    {
        w = rhs.w;
        return *this;
    }
    template<typename pointer_type>
    __INLINE__ const shape<1>& operator=(const pointer_type* rhs)
    {
        w = rhs[0];
        return *this;
    }
    __INLINE__ const shape<1>& operator=(const unsigned short* rhs)
    {
        w = rhs[0];
        return *this;
    }
public:
    __INLINE__ size_t size(void) const
    {
        return size_t(w);
    }
    __INLINE__ void clear(void)
    {
        w = 0;
    }
    __INLINE__ void swap(shape<1>& rhs)
    {
        std::swap(w,rhs.w);
    }
public:
    __INLINE__ const unsigned int* begin(void) const
    {
        return dim;
    }
    __INLINE__ const unsigned int* end(void) const
    {
        return dim+1;
    }
    template<typename index_type>
    __INLINE__ const unsigned int& operator[](index_type index) const
    {
        return dim[index];
    }
    __INLINE__ unsigned int* begin(void)
    {
        return dim;
    }
    __INLINE__ unsigned int* end(void)
    {
        return dim+1;
    }
    __INLINE__ unsigned int& operator[](size_t index)
    {
        return dim[index];
    }
public:
    template<typename T>
    __INLINE__ bool is_valid(T x) const
    {
        return x >= 0 && x < dim[0];
    }
    template<typename T>
    __INLINE__ bool is_edge(T x) const
    {
        return x == 0 || x+1 == dim[0];
    }
    template<typename T>
    __INLINE__ bool is_valid(T& pos) const
    {
        return pos[0] >= 0 && pos[0]+1 <= dim[0];
    }
    template<typename T>
    __INLINE__ bool is_edge(T& pos) const
    {
        return pos[0] == 0 || pos[0]+1 == dim[0];
    }
public:
    __INLINE__ unsigned int width(void) const
    {
        return w;
    }
    __INLINE__ size_t plane_size(void) const
    {
        return size_t(w);
    }
public:
    __INLINE__ bool operator==(const shape<1>& rhs) const
    {
        return dim[0] == rhs.dim[0];
    }
    __INLINE__ bool operator!=(const shape<1>& rhs) const
    {
        return !(*this == rhs);
    }

};

template<>
class shape<2>
{
    union
    {
        unsigned int dim[2];
        struct
        {
            unsigned int w;
            unsigned int h;
        };
    };
public:
    static constexpr int dimension = 2;
public:
    __INLINE__ shape(void):w(0),h(0) {}
    __INLINE__ shape(unsigned int w_,unsigned int h_):w(w_),h(h_) {}
    template<typename T>
    __INLINE__ shape(std::initializer_list<T> arg):w(*arg.begin()),h(*(arg.begin()+1)){}
    __INLINE__ shape(const shape<2>& rhs):w(rhs.w),h(rhs.h){}
    template<typename pointer_type>
    __INLINE__ explicit shape(const pointer_type* rhs)
    {
        *this = rhs;
    }
    __INLINE__ const shape<2>& operator=(const shape<2>& rhs)
    {
        w = rhs.w;
        h = rhs.h;
        return *this;
    }
    template<typename pointer_type>
    __INLINE__ const shape<2>& operator=(const pointer_type* rhs)
    {
        w = rhs[0];
        h = rhs[1];
        return *this;
    }
public:
    __INLINE__ size_t size(void) const
    {
        return size_t(w)*size_t(h);
    }
    __INLINE__ void clear(void)
    {
        w = h = 0;
    }
    __INLINE__ void swap(shape<2>& rhs)
    {
        std::swap(w,rhs.w);
        std::swap(h,rhs.h);
    }
public:
    __INLINE__ const unsigned int* begin(void) const
    {
        return dim;
    }
    __INLINE__ const unsigned int* end(void) const
    {
        return dim+2;
    }
    template<typename index_type>
    __INLINE__ const unsigned int& operator[](index_type index) const
    {
        return dim[index];
    }
    __INLINE__ unsigned int* begin(void)
    {
        return dim;
    }
    __INLINE__ unsigned int* end(void)
    {
        return dim+2;
    }
    template<typename index_type>
    __INLINE__ unsigned int& operator[](index_type index)
    {
        return dim[index];
    }
public:
    template<typename T>
    __INLINE__ bool is_valid(T x,T y) const
    {
        return x >= 0 && y >= 0 && x < dim[0] && y < dim[1];
    }
    template<typename T>
    __INLINE__ bool is_edge(T x,T y) const
    {
        return x == 0 || y == 0 || x+1 == dim[0] || y+1 == dim[1];
    }
    template<typename T>
    __INLINE__ bool is_valid(const T& pos) const
    {
        return pos[0] >= 0 && pos[1] >= 0 && pos[0]+1 <= dim[0] && pos[1]+1 <= dim[1];
    }
    template<typename T>
    __INLINE__ bool is_edge(const T& pos) const
    {
        return pos[0] == 0 || pos[1] == 0 || pos[0]+1 == dim[0] || pos[1]+1 == dim[1];
    }
public:
    __INLINE__ unsigned int width(void) const
    {
        return w;
    }
    __INLINE__ unsigned int height(void) const
    {
        return h;
    }
    __INLINE__ unsigned int depth(void) const
    {
        return 1;
    }
    __INLINE__ size_t plane_size(void) const
    {
        return size();
    }
public:
    __INLINE__ bool operator==(const shape<2>& rhs) const
    {
        return dim[0] == rhs.dim[0] && dim[1] == rhs.dim[1];
    }
    __INLINE__ bool operator!=(const shape<2>& rhs) const
    {
        return !(*this == rhs);
    }
    friend std::ostream& operator<<(std::ostream& out,const shape<2>& dim)
    {
        out << int(dim[0]) << " " << int(dim[1]);
        return out;
    }
};
template<>
class shape<3>
{
public:
    union
    {
        unsigned int dim[3];
        struct
        {
            unsigned int w;
            unsigned int h;
            unsigned int d;
        };
    };
    mutable size_t wh = 0;
    mutable size_t size_ = 0;
public:
    static constexpr int dimension = 3;
public:
    __INLINE__ shape(void):w(0),h(0),d(0){}
    template<typename T>
    __INLINE__ shape(std::initializer_list<T> arg):w(*arg.begin()),h(*(arg.begin()+1)),d(*(arg.begin()+2)){}
    __INLINE__ shape(unsigned int w_,unsigned int h_,unsigned int d_):w(w_),h(h_),d(d_){}
    __INLINE__ shape(const shape& rhs)
    {
        *this = rhs;
    }
    template<typename pointer_type>
    __INLINE__ explicit shape(const pointer_type* rhs)
    {
        *this = rhs;
    }
    __INLINE__ const shape& operator=(const shape& rhs)
    {
        w = rhs.w;
        h = rhs.h;
        d = rhs.d;
        wh = rhs.wh;
        size_ = rhs.size_;
        return *this;
    }
    template<typename pointer_type>
    __INLINE__ const shape& operator=(const pointer_type* rhs)
    {
        w = rhs[0];
        h = rhs[1];
        d = rhs[2];
        wh = size_ = 0;
        return *this;
    }
public:
    __INLINE__ size_t size(void) const
    {
        if(w && !size_)
            size_ = plane_size()*size_t(d);
        return size_;
    }
    __INLINE__ void clear(void)
    {
        w = h = d = 0;
        wh = size_ = 0;
    }
    __INLINE__ void swap(shape<3>& rhs)
    {
        std::swap(w,rhs.w);
        std::swap(h,rhs.h);
        std::swap(d,rhs.d);
        std::swap(wh,rhs.wh);
        std::swap(size_,rhs.size_);
    }
public:
    enum axis_type {x=0,y=1,z=2};
    auto multiply(axis_type axis,unsigned int v) const
    {
        auto new_shape = *this;
        new_shape.dim[axis] *= v;
        new_shape.wh = new_shape.size_ = 0;
        return new_shape;
    }
    auto add(axis_type axis,unsigned int v) const
    {
        auto new_shape = *this;
        new_shape.dim[axis] += v;
        new_shape.wh = new_shape.size_ = 0;
        return new_shape;
    }
    auto minus(axis_type axis,unsigned int v) const
    {
        auto new_shape = *this;
        new_shape.dim[axis] -= v;
        new_shape.wh = new_shape.size_ = 0;
        return new_shape;
    }
public:
    __INLINE__ const unsigned int* begin(void) const
    {
        return dim;
    }
    __INLINE__ const unsigned int* end(void) const
    {
        return dim+3;
    }
    template<typename index_type>
    __INLINE__ const unsigned int& operator[](index_type index) const
    {
        return dim[index];
    }
    template<typename rhs_type>
    __INLINE__ shape operator*(rhs_type value) const
    {
        return shape(uint32_t(w*value),uint32_t(h*value),uint32_t(d*value));
    }
    __INLINE__ unsigned int* begin(void)
    {
        wh = size_ = 0;
        return dim;
    }
    __INLINE__ unsigned int* end(void)
    {
        wh = size_ = 0;
        return dim+3;
    }
    template<typename index_type>
    __INLINE__ unsigned int& operator[](index_type index)
    {
        wh = size_ = 0;
        return dim[index];
    }

public:
    template<typename T>
    __INLINE__ bool is_valid(T x,T y,T z) const
    {
        return x >= 0 && y >= 0 && z >= 0 && x < w && y < h && z < d;
    }
    template<typename T>
    __INLINE__ bool is_valid(const T& pos) const
    {
        return pos[0] >= 0 && pos[1] >= 0 && pos[2] >= 0 &&
               pos[0] <= dim[0]-1 && pos[1] <= dim[1]-1 && pos[2] <= dim[2]-1;
    }
    template<typename T>
    __INLINE__ bool is_edge(T x,T y,T z) const
    {
        return x == 0 || y == 0 || z == 0 || x+1 == dim[0] || y+1 == dim[1] || z+1 == dim[2];
    }
    template<typename T>
    __INLINE__ bool is_edge(const T& pos) const
    {
        return pos[0] == 0 || pos[1] == 0 || pos[2] == 0 ||
               pos[0]+1 == dim[0] || pos[1]+1 == dim[1] || pos[2]+1 == dim[2];
    }
public:
    __INLINE__ unsigned int width(void) const
    {
        return w;
    }
    __INLINE__ unsigned int height(void) const
    {
        return h;
    }
    __INLINE__ unsigned int depth(void) const
    {
        return d;
    }
    __INLINE__ size_t plane_size(void) const
    {
        if (w && !wh)
            wh = size_t(w)*size_t(h);
        return wh;
    }
public:
    template<typename T>
    __INLINE__ shape<3> operator*(T r)
    {
        return shape<3>(dim[0]*r,dim[1]*r,dim[2]*r);
    }
public:
    __INLINE__ bool operator==(const shape& rhs) const
    {
        return dim[0] == rhs.dim[0] && dim[1] == rhs.dim[1] && dim[2] == rhs.dim[2];
    }
    __INLINE__ bool operator<(const shape& rhs) const
    {
        return dim[2] < rhs.dim[2] || dim[1] < rhs.dim[1] || dim[0] < rhs.dim[0];
    }
    __INLINE__ bool operator!=(const shape& rhs) const
    {
        return !(*this == rhs);
    }
    friend std::ostream& operator<<(std::ostream& out,const shape& dim)
    {
        out << "(" << dim[0] << "," << dim[1] << "," << dim[2] << ")";
        return out;
    }

};

template<typename value_type>
__INLINE__ tipl::shape<3> s(value_type x,value_type y,value_type z)
{
    return tipl::shape<3>(x,y,z);
}
template<typename value_type>
__INLINE__ tipl::shape<2> s(value_type x,value_type y)
{
    return tipl::shape<2>(x,y);
}


}
#endif
