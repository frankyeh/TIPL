#ifndef PIXEL_INDEX_HPP
#define PIXEL_INDEX_HPP
#include <vector>
#include <deque>
#include <algorithm>
#include <iosfwd>
#include <cmath>
#include "../def.hpp"
#include "../mt.hpp"
#include "shape.hpp"
namespace tipl
{

template<typename ptr_type,typename stype,
         typename std::enable_if<stype::dimension==3,bool>::type = true>
__INLINE__ size_t voxel2index(const ptr_type* offset,const stype& geo)
{
    return (size_t(offset[2])*size_t(geo.height()) + size_t(offset[1]))*size_t(geo.width())+size_t(offset[0]);
}

template<typename vtype,typename stype,
         typename std::enable_if<stype::dimension==3,bool>::type = true>
__INLINE__ size_t voxel2index(vtype x,vtype y,vtype z,const stype& geo)
{
    return (size_t(z)*size_t(geo.height()) + size_t(y))*size_t(geo.width())+size_t(x);
}

template<unsigned int dim>
class pixel_index;

template<>
class pixel_index<2>
{
public:
    static const unsigned int dimension = 2;
protected:
    union
    {
        int offset_[2];
        struct { int x_, y_; };
    };
    int index_;
    int w;
public:
    __INLINE__ pixel_index():x_(0),y_(0),index_(0),w(0){}
    __INLINE__ pixel_index(const shape<2>& geo):x_(0),y_(0),index_(0),w(geo.width()){}

    template<typename vtype>
    __INLINE__ pixel_index(vtype x,vtype y,vtype index,const shape<2>& geo):
            x_(int(x)),y_(int(y)),index_(index),w(geo.width()){}
    template<typename vtype>
    __INLINE__ pixel_index(vtype x,vtype y,const shape<2>& geo):
            x_(int(x)),y_(int(y)),index_(int(y)*geo.width()+int(x)),w(geo.width()){}
    template<typename vtype>
    __INLINE__ pixel_index(vtype* offset,const shape<2>& geo):
            x_(offset[0]),y_(offset[1]),index_(offset[1]*geo.width()+offset[0]),w(geo.width()){}
    template<typename vtype>
    __INLINE__ pixel_index(vtype y,const shape<2>& geo):
            x_(y % geo.width()),y_(y / geo.width()),index_(y),w(geo.width()){}

    template<typename rhs_type>
    __INLINE__ const pixel_index<2>& operator=(const rhs_type& rhs)
    {
        x_ = rhs[0]; y_ = rhs[1]; return *this;
    }

public:
    __INLINE__ int x() const { return x_; }
    __INLINE__ int y() const { return y_; }
    __INLINE__ int index() const { return index_; }
    __INLINE__ int& index() { return index_; }

    __INLINE__ const int* begin() const { return offset_; }
    __INLINE__ const int* end() const { return offset_+2; }
    __INLINE__ int* begin() { return offset_; }
    __INLINE__ int* end() { return offset_+2; }

    __INLINE__ int operator[](unsigned int index) const { return offset_[index]; }
    __INLINE__ int& operator[](unsigned int index) { return offset_[index]; }

public:
    __INLINE__ bool operator<(const pixel_index& rhs) const { return index_ < rhs.index_; }
    __INLINE__ bool operator==(const pixel_index& rhs) const { return index_ == rhs.index_; }
    __INLINE__ bool operator!=(const pixel_index& rhs) const { return index_ != rhs.index_; }

    template<typename T> __INLINE__ bool operator<(T rhs) const { return index_ < rhs; }
    template<typename T> __INLINE__ bool operator==(T rhs) const { return index_ == rhs; }
    template<typename T> __INLINE__ bool operator!=(T rhs) const { return index_ != rhs; }

public:
    template<typename T>
    __INLINE__ pixel_index<2> operator+(T value) const
    {
        pixel_index<2> result = *this;
        return result += value;
    }
    template<typename T>
    __INLINE__ pixel_index<2>& operator+=(T value)
    {
        index_ += value;
        x_ = index_ % w;
        y_ = index_ / w;
        return *this;
    }

    __INLINE__ int64_t operator-(const pixel_index& rhs) const
    {
        return int64_t(index_)-int64_t(rhs.index_);
    }

    __INLINE__ pixel_index<2>& operator++()
    {
        ++offset_[0];
        ++index_;
        if (offset_[0] < w) return *this;
        offset_[0] = 0;
        ++offset_[1];
        return *this;
    }

    template<typename stream_type>
    friend stream_type& operator>>(stream_type& in,pixel_index& rhs)
    {
        in >> rhs.x_ >> rhs.y_; return in;
    }

public:
    operator size_t() const { return index_; }
};

template<>
class pixel_index<3>
{
public:
    static const unsigned int dimension = 3;
protected:
    union
    {
        int offset_[3];
        struct { int x_, y_, z_; };
    };
    size_t index_;
    int w,h;
public:
    __INLINE__ pixel_index():x_(0),y_(0),z_(0),index_(0),w(0),h(0){}
    __INLINE__ pixel_index(const shape<3>& geo):x_(0),y_(0),z_(0),index_(0),w(int(geo.width())),h(int(geo.height())){}

    template<typename vtype>
    __INLINE__ pixel_index(vtype x,vtype y,vtype z,size_t i,const shape<3>& geo):x_(int(x)),y_(int(y)),z_(int(z)),index_(i),w(int(geo.width())),h(int(geo.height())){}
    template<typename vtype>
    __INLINE__ pixel_index(vtype x,vtype y,vtype z,const shape<3>& geo):
            x_(int(x)),y_(int(y)),z_(int(z)),index_(voxel2index(x,y,z,geo)),w(int(geo.width())),h(int(geo.height())){}
    template<typename vtype>
    __INLINE__ pixel_index(const vtype* offset,const shape<3>& geo):
            x_(offset[0]),y_(offset[1]),z_(offset[2]),
            index_(voxel2index(offset,geo)),
            w(int(geo.width())),h(int(geo.height())){}
    __INLINE__ pixel_index(size_t index,const shape<3>& geo):index_(index),w(int(geo.width())),h(int(geo.height()))
    {
        x_ = int(index % geo.width());
        index /= geo.width();
        y_ = int(index % geo.height());
        z_ = int(index / geo.height());
    }

    template<typename rhs_type>
    __INLINE__ const pixel_index<3>& operator=(const rhs_type& rhs)
    {
        x_ = rhs[0]; y_ = rhs[1]; z_ = rhs[2]; return *this;
    }
public:
    __INLINE__ int x() const { return x_; }
    __INLINE__ int y() const { return y_; }
    __INLINE__ int z() const { return z_; }
    __INLINE__ size_t index() const { return index_; }
    __INLINE__ size_t& index() { return index_; }

    __INLINE__ const int* begin() const { return offset_; }
    __INLINE__ const int* end() const { return offset_+3; }
    __INLINE__ int* begin() { return offset_; }
    __INLINE__ int* end() { return offset_+3; }

    __INLINE__ int operator[](unsigned int index) const { return offset_[index]; }
    __INLINE__ int& operator[](unsigned int index) { return offset_[index]; }

public:
    __INLINE__ bool operator<(const pixel_index& rhs) const { return index_ < rhs.index_; }
    __INLINE__ bool operator==(const pixel_index& rhs) const { return index_ == rhs.index_; }
    __INLINE__ bool operator!=(const pixel_index& rhs) const { return index_ != rhs.index_; }

    template<typename value_type> __INLINE__ bool operator<(value_type rhs) const { return index_ < rhs; }
    template<typename value_type> __INLINE__ bool operator==(value_type rhs) const { return index_ == rhs; }
    template<typename value_type> __INLINE__ bool operator!=(value_type rhs) const { return index_ != rhs; }

public:
    template<typename T>
    __INLINE__ pixel_index<3> operator+(T value) const
    {
        pixel_index<3> result = *this;
        return result += value;
    }
    template<typename T>
    __INLINE__ pixel_index<3>& operator+=(T value)
    {
        index_ += value;
        x_ = int(index_ % w);
        auto lines = index_ / w;
        y_ = int(lines % h);
        z_ = int(lines / h);
        return *this;
    }

    __INLINE__ pixel_index<3>& operator++()
    {
        ++offset_[0];
        ++index_;
        if (offset_[0] < w) return *this;
        offset_[0] = 0;
        ++offset_[1];
        if (offset_[1] < h) return *this;
        offset_[1] = 0;
        ++offset_[2];
        return *this;
    }

    __INLINE__ int64_t operator-(const pixel_index& rhs) const
    {
        return int64_t(index_)-int64_t(rhs.index_);
    }
    __INLINE__ pixel_index<3> operator++(int)
    {
        auto old = *this; operator++(); return old;
    }

    template<typename stream_type>
    friend stream_type& operator>>(stream_type& in,pixel_index& rhs)
    {
        in >> rhs.x_ >> rhs.y_ >> rhs.z_; return in;
    }

public:
    operator size_t() const { return index_; }
};

template<int dim,typename data_type = float>
class vector;

template<typename data_type>
class vector<2,data_type>
{
public:
    static const unsigned int dimension = 2;
protected:
    union
    {
        data_type data_[2];
        struct { data_type x_, y_; };
    };
public:
    __INLINE__ vector():x_(0),y_(0) {}
    __INLINE__ vector(data_type x,data_type y):x_(x),y_(y){}
    template<typename rhs_type,typename std::enable_if<std::is_class<rhs_type>::value,bool>::type = true>
    __INLINE__ vector(const rhs_type& rhs):x_(rhs[0]),y_(rhs[1]){}
    template<typename rhs_type,typename std::enable_if<std::is_fundamental<rhs_type>::value,bool>::type = true>
    __INLINE__ vector(const rhs_type* rhs):x_(rhs[0]),y_(rhs[1]){}

    vector(std::initializer_list<data_type> rhs)
    {
        auto it = rhs.begin(); x_ = *it++; y_ = *it++;
    }

    template<typename rhs_type,typename std::enable_if<std::is_fundamental<rhs_type>::value,bool>::type = true>
    __INLINE__ vector& operator=(const rhs_type* rhs)
    {
        x_ = rhs[0]; y_ = rhs[1]; return *this;
    }
    template<typename  rhs_type>
    __INLINE__ vector<2,data_type>& operator=(const rhs_type& rhs)
    {
        x_ = rhs[0]; y_ = rhs[1]; return *this;
    }
    vector& operator=(std::initializer_list<data_type> rhs)
    {
        auto it = rhs.begin(); x_ = *it++; y_ = *it++; return *this;
    }

public:
    __INLINE__ data_type operator[](unsigned int index) const { return data_[index]; }
    __INLINE__ data_type& operator[](unsigned int index) { return data_[index]; }
    __INLINE__ data_type* begin() { return data_; }
    __INLINE__ data_type* end() { return data_+2; }
    __INLINE__ const data_type* data() const { return data_; }
    __INLINE__ const data_type* begin() const { return data_; }
    __INLINE__ const data_type* end() const { return data_+2; }
    __INLINE__ size_t size() const { return 2; }

public:
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto& operator+=(const T* rhs) { x_+=rhs[0]; y_+=rhs[1]; return *this; }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto& operator-=(const T* rhs) { x_-=rhs[0]; y_-=rhs[1]; return *this; }
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto& operator+=(const T& rhs) { x_+=rhs[0]; y_+=rhs[1]; return *this; }
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto& operator-=(const T& rhs) { x_-=rhs[0]; y_-=rhs[1]; return *this; }

    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto& operator+=(T r) { x_+=r; y_+=r; return *this; }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto& operator-=(T r) { x_-=r; y_-=r; return *this; }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto& operator*=(T r) { x_*=r; y_*=r; return *this; }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto& operator/=(T r) { x_/=r; y_/=r; return *this; }

    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator+(const T& rhs) const { return vector<2,data_type>(*this)+=rhs; }
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator-(const T& rhs) const { return vector<2,data_type>(*this)-=rhs; }

    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator+(T rhs) const { return vector<2,data_type>(*this)+=rhs; }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator-(T rhs) const { return vector<2,data_type>(*this)-=rhs; }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator*(T rhs) const { return vector<2,data_type>(*this)*=rhs; }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator/(T rhs) const { return vector<2,data_type>(*this)/=rhs; }

    __INLINE__ auto operator-() const { return vector<2,data_type>(-x_,-y_); }

public:
    __INLINE__ void floor() { x_ = std::floor(x_); y_ = std::floor(y_); }
    __INLINE__ void round() { x_ = std::round(x_); y_ = std::round(y_); }
    __INLINE__ void abs() { x_ = std::abs(x_); y_ = std::abs(y_); }

    template<typename rhs_type>
    __INLINE__ data_type operator*(const vector<2,rhs_type>& rhs) const { return x_*rhs.x_+y_*rhs.y_; }

    __INLINE__ data_type length2() const { return x_*x_+y_*y_; }
    __INLINE__ double length() const { return std::sqrt(x_*x_+y_*y_); }

    __INLINE__ data_type normalize()
    {
        data_type r = std::sqrt(length2());
        if (r == (data_type)0) return 0;
        x_ /= r; y_ /= r;
        return r;
    }

public:
    __INLINE__ double project_length(const vector<2,data_type>& rhs) { return *this*rhs/length(); }
    __INLINE__ vector<2,data_type> project(const vector<2,data_type>& rhs) { return *this*(*this*rhs/length2()); }

    template<typename tran_type>
    __INLINE__ void to(const tran_type& m)
    {
        data_type x__(x_),y__(y_);
        x_ = x__*m[0] + y__*m[1] + m[2];
        y_ = x__*m[3] + y__*m[4] + m[5];
    }
    template<typename tran_type>
    __INLINE__ void to(const tran_type* m)
    {
        data_type x__(x_),y__(y_);
        x_ = x__*m[0] + y__*m[1] + m[2];
        y_ = x__*m[3] + y__*m[4] + m[5];
    }
    template<typename tran_type>
    __INLINE__ void rotate(const tran_type& m)
    {
        data_type x__(x_),y__(y_);
        x_ = x__*m[0] + y__*m[1];
        y_ = x__*m[2] + y__*m[3];
    }
    template<typename tran_type>
    __INLINE__ void rotate(const tran_type* m)
    {
        data_type x__(x_),y__(y_);
        x_ = x__*m[0] + y__*m[1];
        y_ = x__*m[2] + y__*m[3];
    }

public:
    __INLINE__ bool operator<(const vector<2,data_type>& rhs) const
    {
        if (y_ != rhs.y_) return y_ < rhs.y_; return x_ < rhs.x_;
    }
    __INLINE__ bool operator>(const vector<2,data_type>& rhs) const
    {
        if (y_ != rhs.y_) return y_ > rhs.y_; return x_ > rhs.x_;
    }
    __INLINE__ bool operator==(const vector<2,data_type>& rhs) const { return x_ == rhs.x_ && y_ == rhs.y_; }
    __INLINE__ bool operator!=(const vector<2,data_type>& rhs) const { return x_ != rhs.x_ || y_ != rhs.y_; }

    friend std::istream& operator>>(std::istream& in,vector<2,data_type>& point)
    {
        in >> point.x_ >> point.y_; return in;
    }
    friend std::ostream& operator<<(std::ostream& out,const vector<2,data_type>& point)
    {
        out << point.x_ << " " << point.y_ << " "; return out;
    }

public:
    __INLINE__ data_type x() const { return x_; }
    __INLINE__ data_type y() const { return y_; }
};


template<typename data_type>
class vector<3,data_type>
{
public:
    static const unsigned int dimension = 3;
protected:
    union
    {
        data_type data_[3];
        struct { data_type x_, y_, z_; };
    };
public:
    using value_type = data_type;
public:
    __INLINE__ vector():x_(0),y_(0),z_(0) {}
    __INLINE__ vector(data_type x,data_type y,data_type z):x_(x),y_(y),z_(z){}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ vector(const T& rhs):x_(data_type(rhs[0])),y_(data_type(rhs[1])),z_(data_type(rhs[2])){}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ vector(const T* rhs):x_(data_type(rhs[0])),y_(data_type(rhs[1])),z_(data_type(rhs[2])){}

    vector(std::initializer_list<data_type>  rhs)
    {
        auto it = rhs.begin(); x_ = *it++; y_ = *it++; z_ = *it++;
    }

    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ vector& operator=(T rhs)
    {
        x_=rhs; y_=rhs; z_=rhs; return *this;
    }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ vector& operator=(const T* rhs)
    {
        x_=rhs[0]; y_=rhs[1]; z_=rhs[2]; return *this;
    }
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ vector& operator=(const T& rhs)
    {
        x_=rhs[0]; y_=rhs[1]; z_=rhs[2]; return *this;
    }
    vector& operator=(std::initializer_list<data_type> rhs)
    {
        auto it = rhs.begin(); x_ = *it++; y_ = *it++; z_ = *it++; return *this;
    }

public:
    __INLINE__ const data_type& operator[](unsigned int index) const { return data_[index]; }
    __INLINE__ data_type& operator[](unsigned int index) { return data_[index]; }
    __INLINE__ data_type* begin() { return data_; }
    __INLINE__ data_type* end() { return data_+3; }
    __INLINE__ const data_type* data() const { return data_; }
    __INLINE__ const data_type* begin() const { return data_; }
    __INLINE__ const data_type* end() const { return data_+3; }
    __INLINE__ size_t size() const { return 3; }

public:
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto& operator+=(const T* rhs) { x_+=rhs[0]; y_+=rhs[1]; z_+=rhs[2]; return *this; }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto& operator-=(const T* rhs) { x_-=rhs[0]; y_-=rhs[1]; z_-=rhs[2]; return *this; }
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto& operator+=(const T& rhs) { x_+=rhs[0]; y_+=rhs[1]; z_+=rhs[2]; return *this; }
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto& operator-=(const T& rhs) { x_-=rhs[0]; y_-=rhs[1]; z_-=rhs[2]; return *this; }

    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto& operator+=(T r) { x_+=r; y_+=r; z_+=r; return *this; }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto& operator-=(T r) { x_-=r; y_-=r; z_-=r; return *this; }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto& operator*=(T r) { x_*=r; y_*=r; z_*=r; return *this; }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto& operator/=(T r) { x_/=r; y_/=r; z_/=r; return *this; }

    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator+(const T& rhs) const { return vector<3,data_type>(*this)+=rhs; }
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator-(const T& rhs) const { return vector<3,data_type>(*this)-=rhs; }

    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator+(T rhs) const { return vector<3,data_type>(*this)+=rhs; }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator-(T rhs) const { return vector<3,data_type>(*this)-=rhs; }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator*(T rhs) const { return vector<3,data_type>(*this)*=rhs; }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator/(T rhs) const { return vector<3,data_type>(*this)/=rhs; }

    __INLINE__ auto operator-() const { return vector<3,data_type>(-x_,-y_,-z_); }

    __INLINE__ vector<3,data_type> cross_product(const vector<3,data_type>& rhs) const
    {
        return vector<3,data_type>(y_*rhs.z_-rhs.y_*z_,z_*rhs.x_-rhs.z_*x_,x_*rhs.y_-rhs.x_*y_);
    }

    __INLINE__ auto normal() const
    {
        vector<3,data_type> result;
        if(std::abs(y_) > std::abs(x_))
            result = cross_product(vector<3,data_type>(data_type(1),0,0));
        else
            result = cross_product(vector<3,data_type>(0,data_type(1),0));
        result.normalize();
        return result;
    }
    __INLINE__ auto normal(const vector<3,data_type>& rhs) const
    {
        vector<3,data_type> result = cross_product(rhs);
        result.normalize();
        return result;
    }

    template<typename function_type>
    __INLINE__ void apply(function_type& function)
    {
        x_ = function(x_); y_ = function(y_); z_ = function(z_);
    }
    __INLINE__ void floor() { x_ = std::floor(x_); y_ = std::floor(y_); z_ = std::floor(z_); }
    __INLINE__ void round() { x_ = std::round(x_); y_ = std::round(y_); z_ = std::round(z_); }
    __INLINE__ void abs() { x_ = std::abs(x_); y_ = std::abs(y_); z_ = std::abs(z_); }
    __INLINE__ void ceil() { x_ = std::ceil(x_); y_ = std::ceil(y_); z_ = std::ceil(z_); }

    template<typename T,typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = true>
    __INLINE__ void ceil(T cx,T cy,T cz)
    {
        if(x_ > cx) x_ = cx; if(y_ > cy) y_ = cy; if(z_ > cz) z_ = cz;
    }
    template<typename T,typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = true>
    __INLINE__ void floor(T fx,T fy,T fz)
    {
        if(x_ < fx) x_ = fx; if(y_ < fy) y_ = fy; if(z_ < fz) z_ = fz;
    }

    __INLINE__ data_type operator*(const vector<3,data_type>& rhs) const { return x_*rhs.x_+y_*rhs.y_+z_*rhs.z_; }

    __INLINE__ auto& elem_mul(const vector<3,data_type>& rhs)
    {
        x_*=rhs.x_; y_*=rhs.y_; z_*=rhs.z_; return *this;
    }
    __INLINE__ auto& elem_div(const vector<3,data_type>& rhs)
    {
        x_/=rhs.x_; y_/=rhs.y_; z_/=rhs.z_; return *this;
    }

    template<typename rhs_type>
    __INLINE__ data_type operator*(const vector<3,rhs_type>& rhs) const { return x_*rhs[0]+y_*rhs[1]+z_*rhs[2]; }

    __INLINE__ data_type length2() const { return x_*x_+y_*y_+z_*z_; }
    __INLINE__ double length() const { return std::sqrt(double(x_*x_+y_*y_+z_*z_)); }

    __INLINE__ data_type normalize()
    {
        data_type r = std::sqrt(length2());
        if (r == data_type(0)) return 0;
        x_ /= r; y_ /= r; z_ /= r;
        return r;
    }

public:
    __INLINE__ data_type distance2(const vector<3,data_type>& rhs)
    {
        data_type sum = 0, t = x_-rhs.x_; sum += t*t;
        t = y_-rhs.y_; sum += t*t;
        t = z_-rhs.z_; sum += t*t;
        return sum;
    }
    template<typename pointer_type>
    __INLINE__ data_type distance2(const pointer_type* rhs)
    {
        data_type sum = 0, t = x_-rhs[0]; sum += t*t;
        t = y_-rhs[1]; sum += t*t;
        t = z_-rhs[2]; sum += t*t;
        return sum;
    }

    __INLINE__ double distance(const vector<3,data_type>& rhs) { return std::sqrt(distance2(rhs)); }
    template<typename pointer_type>
    __INLINE__ double distance(const pointer_type* rhs) { return std::sqrt(distance2(rhs)); }

    template<typename tran_type>
    __INLINE__ void to(const tran_type& m)
    {
        data_type x__(x_),y__(y_),z__(z_);
        x_ = x__*m[0] + y__*m[1] + z__*m[2] + m[3];
        y_ = x__*m[4] + y__*m[5] + z__*m[6] + m[7];
        z_ = x__*m[8] + y__*m[9] + z__*m[10] + m[11];
    }
    template<typename tran_type>
    __INLINE__ void to(const tran_type* m)
    {
        data_type x__(x_),y__(y_),z__(z_);
        x_ = x__*m[0] + y__*m[1] + z__*m[2] + m[3];
        y_ = x__*m[4] + y__*m[5] + z__*m[6] + m[7];
        z_ = x__*m[8] + y__*m[9] + z__*m[10] + m[11];
    }
    template<typename tran_type>
    __INLINE__ void rotate(const tran_type& m)
    {
        data_type x__(x_),y__(y_),z__(z_);
        x_ = x__*m[0] + y__*m[1] + z__*m[2];
        y_ = x__*m[3] + y__*m[4] + z__*m[5];
        z_ = x__*m[6] + y__*m[7] + z__*m[8];
    }
    template<typename tran_type>
    __INLINE__ void rotate(const tran_type* m)
    {
        data_type x__(x_),y__(y_),z__(z_);
        x_ = x__*m[0] + y__*m[1] + z__*m[2];
        y_ = x__*m[3] + y__*m[4] + z__*m[5];
        z_ = x__*m[6] + y__*m[7] + z__*m[8];
    }

public:
    __INLINE__ bool operator<(const vector<3,data_type>& rhs) const
    {
        if (z_ != rhs.z_) return z_ < rhs.z_;
        if (y_ != rhs.y_) return y_ < rhs.y_;
        return x_ < rhs.x_;
    }
    __INLINE__ bool operator>(const vector<3,data_type>& rhs) const
    {
        if (z_ != rhs.z_) return z_ > rhs.z_;
        if (y_ != rhs.y_) return y_ > rhs.y_;
        return x_ > rhs.x_;
    }
    __INLINE__ bool operator==(const vector<3,data_type>& rhs) const { return x_ == rhs.x_ && y_ == rhs.y_ && z_ == rhs.z_; }
    __INLINE__ bool operator!=(const vector<3,data_type>& rhs) const { return x_ != rhs.x_ || y_ != rhs.y_ || z_ != rhs.z_; }

    friend std::istream& operator>>(std::istream& in,vector<3,data_type>& point)
    {
        in >> point.x_ >> point.y_ >> point.z_; return in;
    }
    friend std::ostream& operator<<(std::ostream& out,const vector<3,data_type>& point)
    {
        out << point.x_ << " " << point.y_ << " " << point.z_ << " "; return out;
    }
public:
    __INLINE__ data_type x() const { return x_; }
    __INLINE__ data_type y() const { return y_; }
    __INLINE__ data_type z() const { return z_; }
};

template <par_for_type type = dynamic, int dim, typename Func>
inline void par_for(const shape<dim>& s, Func&& f, int tc = max_thread_count) {
    par_for<type>(pixel_index<dim>(s),pixel_index<dim>(s.size(),s), std::forward<Func>(f), tc);
}

template<typename value_type, typename std::enable_if<std::is_fundamental<value_type>::value,bool>::type = true>
__INLINE__ auto v(value_type x,value_type y,value_type z) { return vector<3,value_type>(x,y,z); }

template<typename value_type, typename std::enable_if<std::is_fundamental<value_type>::value,bool>::type = true>
__INLINE__ auto v(value_type x,value_type y) { return vector<2,value_type>(x,y); }

template<typename T, typename std::enable_if<std::is_class<T>::value,bool>::type = true>
__INLINE__ auto v(const T& data)
{
    return vector<T::dimension, typename std::remove_const<typename std::remove_reference<decltype(data[0])>::type>::type>(data.begin());
}

}
#endif
