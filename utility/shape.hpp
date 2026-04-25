//---------------------------------------------------------------------------
#ifndef SHAPE_HPP
#define SHAPE_HPP
#include <functional>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cstdint>
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
private:
    size_t precomputed_size = 0;
    size_t precomputed_plane_size = 0;
    __INLINE__ void update(void)
    {
        precomputed_size = dim[0];
        for(int index = 1;index < dimension;++index)
            precomputed_size *= size_t(dim[index]);
        precomputed_plane_size = size_t(dim[0]) * size_t(dim[1]);
    }
public:
    __INLINE__ shape(void)
    {
        clear();
    }
    template<typename T,typename U,typename V,typename W>
    __INLINE__ shape(T x,U y,V z,W t)
    {
        dim[0] = uint32_t(x),dim[1] = uint32_t(y),dim[2] = uint32_t(z),dim[3] = uint32_t(t),update();
    }
    template<typename pointer_type>
    __INLINE__ explicit shape(const pointer_type* rhs)
    {
        *this = rhs;
    }
    template<typename pointer_type>
    __INLINE__ const shape<dimension>& operator=(const pointer_type* rhs)
    {
        return std::copy_n(rhs,dimension,dim),update(),*this;
    }
public:
    __INLINE__ size_t size(void) const
    {
        return precomputed_size;
    }
    __INLINE__ size_t plane_size(void) const
    {
        return precomputed_plane_size;
    }
    __INLINE__ void clear(void)
    {
        std::fill_n(dim,dimension,0),precomputed_size = 0,precomputed_plane_size = 0;
    }
    __INLINE__ void swap(shape<dimension>& rhs) noexcept
    {
        std::swap_ranges(dim,dim+dimension,rhs.dim),std::swap(precomputed_size,rhs.precomputed_size),std::swap(precomputed_plane_size,rhs.precomputed_plane_size);
    }
    __INLINE__ void set_dim(int index,unsigned int value)
    {
        dim[index] = value,update();
    }
    __INLINE__ shape swap_dim(int index1,int index2) const
    {
        shape new_shape = *this;
        return std::swap(new_shape.dim[index1],new_shape.dim[index2]),new_shape.update(),new_shape;
    }
public:
    __INLINE__ const unsigned int* data(void) const
    {
        return dim;
    }
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
        return new_shape.update(),new_shape;
    }
    __INLINE__ const unsigned int* begin(void)
    {
        return dim;
    }
    __INLINE__ const unsigned int* end(void)
    {
        return dim+dimension;
    }
    template<typename index_type>
    __INLINE__ unsigned int operator[](index_type index) const
    {
        return dim[index];
    }
public:
    template<typename T>
    __INLINE__ bool is_valid(const T& pos) const
    {
        for(int index = 0;index < dimension;++index)
            if(pos[index] < 0 || pos[index] >= dim[index])
                return false;
        return true;
    }
    template<typename T>
    __INLINE__ bool is_edge(const T& pos) const
    {
        for(int index = 0;index < dimension;++index)
            if(pos[index] == 0 || pos[index]+1 == dim[index])
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
        for(int index = 0;index < dimension;++index)
            if(dim[index] != rhs.dim[index])
                return false;
        return true;
    }
    __INLINE__ bool operator!=(const shape<dimension>& rhs) const
    {
        return !(*this == rhs);
    }
    friend std::ostream& operator<<(std::ostream& out,const shape& dim)
    {
        for(int i = 0;i < dimension;++i)
            out << (i ? " " : "") << dim[i];
        return out;
    }
};

template<>
class shape<1>
{
public:
    union
    {
        unsigned int dim[1];
        struct
        {
            unsigned int w;
        };
    };
    static constexpr int dimension = 1;
public:
    __INLINE__ shape(void):w(0) {}
    __INLINE__ shape(unsigned int w_):w(w_) {}

    template<typename pointer_type>
    __INLINE__ explicit shape(const pointer_type* rhs)
    {
        w = rhs[0];
    }
    template<typename pointer_type>
    __INLINE__ const shape<1>& operator=(const pointer_type* rhs)
    {
        return w = rhs[0],*this;
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
    __INLINE__ void set_dim(int index,unsigned int value)
    {
        dim[index] = value;
    }
    __INLINE__ shape swap_dim(int index1,int index2) const
    {
        shape new_shape = *this;
        return std::swap(new_shape.dim[index1],new_shape.dim[index2]),new_shape;
    }
public:
    __INLINE__ const unsigned int* data(void) const
    {
        return dim;
    }
    __INLINE__ const unsigned int* begin(void) const
    {
        return dim;
    }
    __INLINE__ const unsigned int* end(void) const
    {
        return dim+1;
    }
    __INLINE__ unsigned int operator[](int index) const
    {
        return dim[index];
    }
    __INLINE__ const unsigned int* begin(void)
    {
        return dim;
    }
    __INLINE__ const unsigned int* end(void)
    {
        return dim+1;
    }
    template<typename index_type>
    __INLINE__ unsigned int operator[](index_type index) const
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
    __INLINE__ bool is_valid(const T& pos) const
    {
        return pos[0] >= 0 && pos[0] < dim[0];
    }
    template<typename T>
    __INLINE__ bool is_edge(const T& pos) const
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
public:
    union
    {
        unsigned int dim[2];
        struct
        {
            unsigned int w;
            unsigned int h;
        };
    };
    static constexpr int dimension = 2;
private:
    size_t precomputed_size = 0;
    __INLINE__ void update(void)
    {
        precomputed_size = size_t(w) * h;
    }
public:
    __INLINE__ shape(void):w(0),h(0) {}
    __INLINE__ shape(unsigned int w_,unsigned int h_):w(w_),h(h_)
    {
        update();
    }
    template<typename T>
    __INLINE__ shape(std::initializer_list<T> arg):w(*arg.begin()),h(*(arg.begin()+1))
    {
        update();
    }

    template<typename pointer_type>
    __INLINE__ explicit shape(const pointer_type* rhs)
    {
        w = rhs[0],h = rhs[1],update();
    }
    template<typename pointer_type>
    __INLINE__ const shape<2>& operator=(const pointer_type* rhs)
    {
        return w = rhs[0],h = rhs[1],update(),*this;
    }
public:
    __INLINE__ size_t size(void) const
    {
        return precomputed_size;
    }
    __INLINE__ void clear(void)
    {
        w = 0,h = 0,precomputed_size = 0;
    }
    __INLINE__ void swap(shape<2>& rhs)
    {
        std::swap(w,rhs.w),std::swap(h,rhs.h),std::swap(precomputed_size,rhs.precomputed_size);
    }
    __INLINE__ void set_dim(int index,unsigned int value)
    {
        dim[index] = value,update();
    }
    __INLINE__ shape swap_dim(int index1,int index2) const
    {
        shape new_shape = *this;
        return std::swap(new_shape.dim[index1],new_shape.dim[index2]),new_shape.update(),new_shape;
    }
public:
    __INLINE__ const unsigned int* data(void) const
    {
        return dim;
    }
    __INLINE__ const unsigned int* begin(void) const
    {
        return dim;
    }
    __INLINE__ const unsigned int* end(void) const
    {
        return dim+2;
    }
    __INLINE__ unsigned int operator[](int index) const
    {
        return dim[index];
    }
    __INLINE__ const unsigned int* begin(void)
    {
        return dim;
    }
    __INLINE__ const unsigned int* end(void)
    {
        return dim+2;
    }
    template<typename index_type>
    __INLINE__ unsigned int operator[](index_type index) const
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
        return pos[0] >= 0 && pos[1] >= 0 && pos[0] < dim[0] && pos[1] < dim[1];
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
        return precomputed_size;
    }
    __INLINE__ auto expand(unsigned int t) const;
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
    friend std::istream& operator>>(std::istream& in,shape& rhs)
    {
        return in >> rhs.dim[0] >> rhs.dim[1],rhs.update(),in;
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
public:
    static constexpr int dimension = 3;
private:
    size_t precomputed_size = 0;
    size_t precomputed_plane_size = 0;
    __INLINE__ void update(void)
    {
        precomputed_plane_size = size_t(w) * h;
        precomputed_size = precomputed_plane_size * d;
    }
public:
    __INLINE__ shape(void):w(0),h(0),d(0) {}
    template<typename T>
    __INLINE__ shape(std::initializer_list<T> arg):w(*arg.begin()),h(*(arg.begin()+1)),d(*(arg.begin()+2))
    {
        update();
    }
    __INLINE__ shape(unsigned int w_,unsigned int h_,unsigned int d_):w(w_),h(h_),d(d_)
    {
        update();
    }

    template<typename pointer_type>
    __INLINE__ explicit shape(const pointer_type* rhs)
    {
        w = rhs[0],h = rhs[1],d = rhs[2],update();
    }
    template<typename pointer_type>
    __INLINE__ const shape& operator=(const pointer_type* rhs)
    {
        return w = rhs[0],h = rhs[1],d = rhs[2],update(),*this;
    }
public:
    __INLINE__ size_t size(void) const
    {
        return precomputed_size;
    }
    __INLINE__ size_t plane_size(void) const
    {
        return precomputed_plane_size;
    }
    __INLINE__ void clear(void)
    {
        w = 0,h = 0,d = 0,precomputed_size = 0,precomputed_plane_size = 0;
    }
    __INLINE__ void swap(shape<3>& rhs)
    {
        std::swap(w,rhs.w),std::swap(h,rhs.h),std::swap(d,rhs.d),std::swap(precomputed_size,rhs.precomputed_size),std::swap(precomputed_plane_size,rhs.precomputed_plane_size);
    }
    __INLINE__ void set_dim(int index,unsigned int value)
    {
        dim[index] = value,update();
    }
    __INLINE__ shape swap_dim(int index1,int index2) const
    {
        shape new_shape = *this;
        return std::swap(new_shape.dim[index1],new_shape.dim[index2]),new_shape.update(),new_shape;
    }
public:
    enum axis_type {x=0,y=1,z=2};
    auto multiply(axis_type axis,unsigned int v) const
    {
        auto new_shape = *this;
        return new_shape.dim[axis] *= v,new_shape.update(),new_shape;
    }
    auto add(axis_type axis,unsigned int v) const
    {
        auto new_shape = *this;
        return new_shape.dim[axis] += v,new_shape.update(),new_shape;
    }
    auto divide(axis_type axis,unsigned int v) const
    {
        auto new_shape = *this;
        return new_shape.dim[axis] /= v,new_shape.update(),new_shape;
    }
    auto minus(axis_type axis,unsigned int v) const
    {
        auto new_shape = *this;
        return new_shape.dim[axis] -= v,new_shape.update(),new_shape;
    }
public:
    __INLINE__ const unsigned int* data(void) const
    {
        return dim;
    }
    __INLINE__ const unsigned int* begin(void) const
    {
        return dim;
    }
    __INLINE__ const unsigned int* end(void) const
    {
        return dim+3;
    }
    __INLINE__ unsigned int operator[](int index) const
    {
        return dim[index];
    }
    template<typename rhs_type>
    __INLINE__ shape operator*(rhs_type value) const
    {
        shape new_shape(uint32_t(w * value),uint32_t(h * value),uint32_t(d * value));
        return new_shape;
    }

    __INLINE__ const unsigned int* begin(void)
    {
        return dim;
    }
    __INLINE__ const unsigned int* end(void)
    {
        return dim+3;
    }
    template<typename index_type>
    __INLINE__ unsigned int operator[](index_type index) const
    {
        return dim[index];
    }
    __INLINE__ shape<4> expand(unsigned int t) const
    {
        return shape<4>(w,h,d,t);
    }
    __INLINE__ shape<2> reduce(void) const
    {
        return shape<2>(w,h);
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
               pos[0] < dim[0] && pos[1] < dim[1] && pos[2] < dim[2];
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
public:
    __INLINE__ bool operator==(const shape& rhs) const
    {
        return dim[0] == rhs.dim[0] && dim[1] == rhs.dim[1] && dim[2] == rhs.dim[2];
    }
    __INLINE__ bool operator<(const shape& rhs) const
    {
        if(dim[2] != rhs.dim[2]) return dim[2] < rhs.dim[2];
        if(dim[1] != rhs.dim[1]) return dim[1] < rhs.dim[1];
        return dim[0] < rhs.dim[0];
    }
    __INLINE__ bool operator!=(const shape& rhs) const
    {
        return !(*this == rhs);
    }
    friend std::ostream& operator<<(std::ostream& out,const shape& dim)
    {
        out << dim[0] << " " << dim[1] << " " << dim[2];
        return out;
    }
    friend std::istream& operator>>(std::istream& in,shape& rhs)
    {
        return in >> rhs.dim[0] >> rhs.dim[1] >> rhs.dim[2],rhs.update(),in;
    }
};

__INLINE__ auto shape<2>::expand(unsigned int t) const
{
    return shape<3>(w,h,t);
}

__INLINE__ tipl::shape<3> s(unsigned int x,unsigned int y,unsigned int z)
{
    return tipl::shape<3>(x,y,z);
}
__INLINE__ tipl::shape<2> s(unsigned int x,unsigned int y)
{
    return tipl::shape<2>(x,y);
}

}
#endif
