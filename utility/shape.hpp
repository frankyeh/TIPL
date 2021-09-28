//---------------------------------------------------------------------------
#ifndef SHAPE_HPP
#define SHAPE_HPP
#include <functional>
#include <numeric>
#include <algorithm>
#include <iostream>
//---------------------------------------------------------------------------
namespace tipl
{

template<int Dim>
class shape
{
public:
    unsigned int dim[Dim];
public:
    static const int dimension = Dim;
public:
    shape(void)
    {
        clear();
    }
    shape(unsigned int x,unsigned int y,unsigned int z,unsigned int t)
    {
        dim[0] = x;
        dim[1] = y;
        dim[2] = z;
        dim[3] = t;
    }
    shape(const shape<Dim>& rhs)
    {
        *this = rhs;
    }
    template<typename pointer_type>
    explicit shape(const pointer_type* rhs)
    {
        *this = rhs;
    }
    const shape<Dim>& operator=(const shape<Dim>& rhs)
    {
	if(this == &rhs)
            return *this;
        std::copy(rhs.dim,rhs.dim+Dim,dim);
        return *this;
    }
    template<typename pointer_type>
    const shape<Dim>& operator=(const pointer_type* rhs)
    {
        std::copy(rhs,rhs+Dim,dim);
        return *this;
    }
public:
    size_t size(void) const
    {
        size_t prod = dim[0];
        for (int index = 1;index < Dim;++index)
            prod *= size_t(dim[index]);
        return prod;
    }
    size_t plane_size(void) const
    {
        return size_t(dim[0])*size_t(dim[1]);
    }
    void clear(void)
    {
        std::fill(dim,dim+Dim,0);
    }
    void swap(shape<Dim>& rhs)
    {
        for (int index = 1;index < Dim;++index)
            std::swap(dim[index],rhs.dim[index]);
    }
public:
    const unsigned int* begin(void) const
    {
        return dim;
    }
    const unsigned int* end(void) const
    {
        return dim+Dim;
    }
    unsigned int operator[](int index) const
    {
        return dim[index];
    }
    unsigned int* begin(void)
    {
        return dim;
    }
    unsigned int* end(void)
    {
        return dim+Dim;
    }
    template<typename index_type>
    unsigned int& operator[](index_type index)
    {
        return dim[index];
    }
public:
    template<typename T>
    bool is_valid(const T& pos) const
    {
        for (int index = 0;index < Dim;++index)
            if (pos[index] >= 0 || pos[index] < dim[index])
                return false;
        return true;
    }
    template<typename T>
    bool is_edge(const T& pos) const
    {
        for (int index = 0;index < Dim;++index)
            if (pos[index] == 0 || pos[index]+1 == dim[index])
                return true;
        return false;
    }
public:
    unsigned int width(void) const
    {
        return dim[0];
    }
    unsigned int height(void) const
    {
        return dim[1];
    }
    unsigned int depth(void) const
    {
        return dim[2];
    }

public:
    bool operator==(const shape<Dim>& rhs) const
    {
        for (int index = 0;index < Dim;++index)
            if (dim[index] != rhs.dim[index])
                return false;
        return true;
    }
    bool operator!=(const shape<3>& rhs) const
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
    static const int dimension = 1;
public:
    shape(void):w(0) {}
    shape(unsigned int w_):w(w_) {}

    shape(const shape<1>& rhs)
    {
        *this = rhs;
    }
    template<typename pointer_type>
    explicit shape(const pointer_type* rhs)
    {
        *this = rhs;
    }
    const shape<1>& operator=(const shape<1>& rhs)
    {
        w = rhs.w;
        return *this;
    }
    template<typename pointer_type>
    const shape<1>& operator=(const pointer_type* rhs)
    {
        w = rhs[0];
        return *this;
    }
    const shape<1>& operator=(const unsigned short* rhs)
    {
        w = rhs[0];
        return *this;
    }
public:
    size_t size(void) const
    {
        return size_t(w);
    }
    void clear(void)
    {
        w = 0;
    }
    void swap(shape<1>& rhs)
    {
        std::swap(w,rhs.w);
    }
public:
    const unsigned int* begin(void) const
    {
        return dim;
    }
    const unsigned int* end(void) const
    {
        return dim+1;
    }
    template<typename index_type>
    const unsigned int& operator[](index_type index) const
    {
        return dim[index];
    }
    unsigned int* begin(void)
    {
        return dim;
    }
    unsigned int* end(void)
    {
        return dim+1;
    }
    unsigned int& operator[](size_t index)
    {
        return dim[index];
    }
public:
    template<typename T>
    bool is_valid(T x) const
    {
        return x >= 0 && x < dim[0];
    }
    template<typename T>
    bool is_edge(T x) const
    {
        return x == 0 || x+1 == dim[0];
    }
    template<typename T>
    bool is_valid(T& pos) const
    {
        return pos[0] >= 0 && pos[0]+1 <= dim[0];
    }
    template<typename T>
    bool is_edge(T& pos) const
    {
        return pos[0] == 0 || pos[0]+1 == dim[0];
    }
public:
    unsigned int width(void) const
    {
        return w;
    }
    size_t plane_size(void) const
    {
        return size_t(w);
    }
public:
    bool operator==(const shape<1>& rhs) const
    {
        return dim[0] == rhs.dim[0];
    }
    bool operator!=(const shape<1>& rhs) const
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
    static const int dimension = 2;
public:
    shape(void):w(0),h(0) {}
    shape(unsigned int w_,unsigned int h_):w(w_),h(h_) {}
    template<typename T>
    shape(const std::initializer_list<T>& arg):w(*arg.begin()),h(*(arg.begin()+1)){}
    shape(const shape<2>& rhs)
    {
        *this = rhs;
    }
    template<typename pointer_type>
    explicit shape(const pointer_type* rhs)
    {
        *this = rhs;
    }
    const shape<2>& operator=(const shape<2>& rhs)
    {
        w = rhs.w;
        h = rhs.h;
        return *this;
    }
    template<typename pointer_type>
    const shape<2>& operator=(const pointer_type* rhs)
    {
        w = rhs[0];
        h = rhs[1];
        return *this;
    }
public:
    size_t size(void) const
    {
        return size_t(w)*size_t(h);
    }
    void clear(void)
    {
        w = h = 0;
    }
    void swap(shape<2>& rhs)
    {
        std::swap(w,rhs.w);
        std::swap(h,rhs.h);
    }
public:
    const unsigned int* begin(void) const
    {
        return dim;
    }
    const unsigned int* end(void) const
    {
        return dim+2;
    }
    template<typename index_type>
    const unsigned int& operator[](index_type index) const
    {
        return dim[index];
    }
    unsigned int* begin(void)
    {
        return dim;
    }
    unsigned int* end(void)
    {
        return dim+2;
    }
    template<typename index_type>
    unsigned int& operator[](index_type index)
    {
        return dim[index];
    }

public:
    template<typename T>
    bool is_valid(T x,T y) const
    {
        return x >= 0 && y >= 0 && x < dim[0] && y < dim[1];
    }
    template<typename T>
    bool is_edge(T x,T y) const
    {
        return x == 0 || y == 0 || x+1 == dim[0] || y+1 == dim[1];
    }
    template<typename T>
    bool is_valid(const T& pos) const
    {
        return pos[0] >= 0 && pos[1] >= 0 && pos[0]+1 <= dim[0] && pos[1]+1 <= dim[1];
    }
    template<typename T>
    bool is_edge(const T& pos) const
    {
        return pos[0] == 0 || pos[1] == 0 || pos[0]+1 == dim[0] || pos[1]+1 == dim[1];
    }
public:
    unsigned int width(void) const
    {
        return w;
    }
    unsigned int height(void) const
    {
        return h;
    }
    unsigned int depth(void) const
    {
        return 1;
    }
    size_t plane_size(void) const
    {
        return size();
    }
public:
    bool operator==(const shape<2>& rhs) const
    {
        return dim[0] == rhs.dim[0] && dim[1] == rhs.dim[1];
    }
    bool operator!=(const shape<2>& rhs) const
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
    static const int dimension = 3;
public:
    shape(void):w(0),h(0),d(0){}
    template<typename T>
    shape(const std::initializer_list<T>& arg):w(*arg.begin()),h(*(arg.begin()+1)),d(*(arg.begin()+2)){}
    shape(unsigned int w_,unsigned int h_,unsigned int d_):w(w_),h(h_),d(d_){}
    shape(const shape<3>& rhs)
    {
        *this = rhs;
    }
    template<typename pointer_type>
    explicit shape(const pointer_type* rhs)
    {
        *this = rhs;
    }
    const shape<3>& operator=(const shape<3>& rhs)
    {
        w = rhs.w;
        h = rhs.h;
        d = rhs.d;
        wh = rhs.wh;
        size_ = rhs.size_;
        return *this;
    }
    template<typename pointer_type>
    const shape<3>& operator=(const pointer_type* rhs)
    {
        w = rhs[0];
        h = rhs[1];
        d = rhs[2];
        wh = size_ = 0;
        return *this;
    }
public:
    size_t size(void) const
    {
        if(w && !size_)
            size_ = plane_size()*size_t(d);
        return size_;
    }
    void clear(void)
    {
        w = h = d = 0;
        wh = size_ = 0;
    }
    void swap(shape<3>& rhs)
    {
        std::swap(w,rhs.w);
        std::swap(h,rhs.h);
        std::swap(d,rhs.d);
        std::swap(wh,rhs.wh);
        std::swap(size_,rhs.size_);
    }
public:
    const unsigned int* begin(void) const
    {
        return dim;
    }
    const unsigned int* end(void) const
    {
        return dim+3;
    }
    template<typename index_type>
    const unsigned int& operator[](index_type index) const
    {
        return dim[index];
    }
    unsigned int* begin(void)
    {
        wh = size_ = 0;
        return dim;
    }
    unsigned int* end(void)
    {
        wh = size_ = 0;
        return dim+3;
    }
    template<typename index_type>
    unsigned int& operator[](index_type index)
    {
        wh = size_ = 0;
        return dim[index];
    }

public:
    template<typename T>
    bool is_valid(T x,T y,T z) const
    {
        return x >= 0 && y >= 0 && z >= 0 && x < w && y < h && z < d;
    }
    template<typename T>
    bool is_valid(const T& pos) const
    {
        return pos[0] >= 0 && pos[1] >= 0 && pos[2] >= 0 &&
               pos[0] <= dim[0]-1 && pos[1] <= dim[1]-1 && pos[2] <= dim[2]-1;
    }
    template<typename T>
    bool is_edge(T x,T y,T z) const
    {
        return x == 0 || y == 0 || z == 0 || x+1 == dim[0] || y+1 == dim[1] || z+1 == dim[2];
    }
    template<typename T>
    bool is_edge(const T& pos) const
    {
        return pos[0] == 0 || pos[1] == 0 || pos[2] == 0 ||
               pos[0]+1 == dim[0] || pos[1]+1 == dim[1] || pos[2]+1 == dim[2];
    }
public:
    unsigned int width(void) const
    {
        return w;
    }
    unsigned int height(void) const
    {
        return h;
    }
    unsigned int depth(void) const
    {
        return d;
    }
    size_t plane_size(void) const
    {
        if (w && !wh)
            wh = size_t(w)*size_t(h);
        return wh;
    }
public:
    bool operator==(const shape<3>& rhs) const
    {
        return dim[0] == rhs.dim[0] && dim[1] == rhs.dim[1] && dim[2] == rhs.dim[2];
    }
    bool operator!=(const shape<3>& rhs) const
    {
        return !(*this == rhs);
    }
    friend std::ostream& operator<<(std::ostream& out,const shape<3>& dim)
    {
        out << "(" << dim[0] << "," << dim[1] << "," << dim[2] << ")";
        return out;
    }

};

}
#endif
