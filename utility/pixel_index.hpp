#ifndef PIXEL_INDEX_HPP
#define PIXEL_INDEX_HPP
#include <vector>
#include <deque>
#include <algorithm>
#include <iosfwd>
#include <cmath>
#include "shape.hpp"

namespace tipl
{
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
        struct
        {
            int x_;
            int y_;
        };
    };
    int index_;
    int w;
public:
    pixel_index(const shape<2>& geo):x_(0),y_(0),index_(0),w(geo[0]){}
    pixel_index(const pixel_index& rhs)
    {
        *this = rhs;
    }
    template<typename vtype>
    pixel_index(vtype x,vtype y,vtype index,const shape<2>& geo):
            x_(int(x)),y_(int(y)),index_(index),w(geo[0]){}
    template<typename vtype>
    pixel_index(vtype x,vtype y,const shape<2>& geo):
            x_(int(x)),y_(int(y)),index_(int(y)*geo.width()+int(x)),w(geo[0]){}
    template<typename vtype>
    pixel_index(vtype* offset,const shape<2>& geo):
            x_(offset[0]),y_(offset[1]),index_(offset[1]*geo.width()+offset[0]),w(geo[0]){}
    template<typename vtype>
    pixel_index(vtype y,const shape<2>& geo):
            x_(y % geo.width()),y_(y / geo.width()),index_(y),w(geo[0]){}

    const pixel_index& operator=(const pixel_index<2>& rhs)
    {
        x_ = rhs.x_;
        y_ = rhs.y_;
        index_ = rhs.index_;
        return *this;
    }

    template<typename rhs_type>
    const pixel_index<2>& operator=(const rhs_type& rhs)
    {
        x_ = rhs[0];
        y_ = rhs[1];
        return *this;
    }

public:
    int x(void) const
    {
        return x_;
    }
    int y(void) const
    {
        return y_;
    }
    int index(void) const
    {
        return index_;
    }
    int& index(void)
    {
        return index_;
    }
public:
    const int* begin(void) const
    {
        return offset_;
    }
    const int* end(void) const
    {
        return offset_+2;
    }
    int* begin(void)
    {
        return offset_;
    }
    int* end(void)
    {
        return offset_+2;
    }
    int operator[](unsigned int index) const
    {
        return offset_[index];
    }
    int& operator[](unsigned int index)
    {
        return offset_[index];
    }
public:
    bool operator<(const pixel_index& rhs) const
    {
        return index_ < rhs.index_;
    }
    bool operator==(const pixel_index& rhs) const
    {
        return index_ == rhs.index_;
    }
    bool operator!=(const pixel_index& rhs) const
    {
        return index_ != rhs.index_;
    }
    template<typename T>
    bool operator<(T rhs) const
    {
        return index_ < rhs;
    }
    template<typename T>
    bool operator==(T rhs) const
    {
        return index_ == rhs;
    }
    template<typename T>
    bool operator!=(T rhs) const
    {
        return index_ != rhs;
    }
public:
    pixel_index<2>& operator++(void)
    {
        ++offset_[0];
        ++index_;
        if (offset_[0] < w)
            return *this;
        offset_[0] = 0;
        ++offset_[1];
        return *this;
    }
    template<typename stream_type>
    friend stream_type& operator>>(stream_type& in,pixel_index& rhs)
    {
        in >> rhs.x_ >> rhs.y_;
        return in;
    }
    bool is_valid(const shape<2>& geo) const
    {
        return offset_[1] < geo[1];
    }
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
        struct
        {
            int x_;
            int y_;
            int z_;
        };
    };
    size_t index_;
    int w,h;
public:
    pixel_index(void):x_(0),y_(0),z_(0),index_(0),w(0),h(0){}
    pixel_index(const shape<3>& geo):x_(0),y_(0),z_(0),index_(0),w(int(geo[0])),h(int(geo[1])){}
    pixel_index(const pixel_index& rhs)
    {
        *this = rhs;
    }
    template<typename vtype>
    pixel_index(vtype x,vtype y,vtype z,size_t i,const shape<3>& geo):x_(int(x)),y_(int(y)),z_(int(z)),index_(i),w(int(geo[0])),h(int(geo[1])){}
    template<typename vtype>
    pixel_index(vtype x,vtype y,vtype z,const shape<3>& geo):
            x_(int(x)),y_(int(y)),z_(int(z)),index_(voxel2index(x,y,z,geo)),w(int(geo[0])),h(int(geo[1])){}
    template<typename vtype>
    pixel_index(const vtype* offset,const shape<3>& geo):
            x_(offset[0]),y_(offset[1]),z_(offset[2]),
            index_(voxel2index(offset,geo)),
            w(int(geo[0])),h(int(geo[1])){}
    pixel_index(size_t index,const shape<3>& geo):index_(index),w(int(geo[0])),h(int(geo[1]))
    {
        x_ = int(index % geo.width());
        index /= geo.width();
        y_ = int(index % geo.height());
        z_ = int(index / geo.height());
    }
public:
    template<typename ptr_type>
    static size_t voxel2index(const ptr_type* offset,const shape<3>& geo)
    {
        return (size_t(offset[2])*size_t(geo.height()) + size_t(offset[1]))*size_t(geo.width())+size_t(offset[0]);
    }
    template<typename vtype>
    static size_t voxel2index(vtype x,vtype y,vtype z,const shape<3>& geo)
    {
        return (size_t(z)*size_t(geo.height()) + size_t(y))*size_t(geo.width())+size_t(x);
    }
    const pixel_index<3>& operator=(const pixel_index<3>& rhs)
    {
        x_ = rhs.x_;
        y_ = rhs.y_;
        z_ = rhs.z_;
        index_ = rhs.index_;
        return *this;
    }
    template<typename rhs_type>
    const pixel_index<3>& operator=(const rhs_type& rhs)
    {
        x_ = rhs[0];
        y_ = rhs[1];
        z_ = rhs[2];
        return *this;
    }
public:
    int x(void) const
    {
        return x_;
    }
    int y(void) const
    {
        return y_;
    }
    int z(void) const
    {
        return z_;
    }
    size_t index(void) const
    {
        return index_;
    }
    size_t& index(void)
    {
        return index_;
    }
public:
    const int* begin(void) const
    {
        return offset_;
    }
    const int* end(void) const
    {
        return offset_+3;
    }
    int* begin(void)
    {
        return offset_;
    }
    int* end(void)
    {
        return offset_+3;
    }
    int operator[](unsigned int index) const
    {
        return offset_[index];
    }
    int& operator[](unsigned int index)
    {
        return offset_[index];
    }
public:
    bool operator<(const pixel_index& rhs) const
    {
        return index_ < rhs.index_;
    }
    bool operator==(const pixel_index& rhs) const
    {
        return index_ == rhs.index_;
    }
    bool operator!=(const pixel_index& rhs) const
    {
        return index_ != rhs.index_;
    }
    template<typename value_type>
    bool operator<(value_type rhs) const
    {
        return index_ < rhs;
    }
    template<typename value_type>
    bool operator==(value_type rhs) const
    {
        return index_ == rhs;
    }
    template<typename value_type>
    bool operator!=(value_type rhs) const
    {
        return index_ != rhs;
    }
public:
    pixel_index<3>& operator++(void)
    {
        ++offset_[0];
        ++index_;
        if (offset_[0] < w)
            return *this;
        offset_[0] = 0;
        ++offset_[1];
        if (offset_[1] < h)
            return *this;
        offset_[1] = 0;
        ++offset_[2];
        return *this;
    }
    bool is_valid(const shape<3>& geo) const
    {
        return offset_[2] < int(geo[2]);
    }
    template<typename stream_type>
    friend stream_type& operator>>(stream_type& in,pixel_index& rhs)
    {
        in >> rhs.x_ >> rhs.y_ >> rhs.z_;
        return in;
    }
};


template<int dim,typename data_type = float>
class vector;

template<typename data_type>
class vector<2,data_type>
{
protected:
    union
    {
        data_type data_[2];
        struct
        {
            data_type x_;
            data_type y_;
        };
    };
public:
    vector(void):x_(0),y_(0)				{}
    vector(data_type x,data_type y):x_(x),y_(y){}
    template<typename rhs_type>
    vector(const rhs_type& rhs):x_(rhs[0]),y_(rhs[1]){}
    template<typename rhs_type>
    vector(const rhs_type* rhs):x_(rhs[0]),y_(rhs[1]){}
    template<typename rhs_type>
    vector& operator=(const rhs_type* rhs)
    {
        x_ = rhs[0];
        y_ = rhs[1];
        return *this;
    }
    template<typename  rhs_type>
    vector<2,data_type>& operator=(const rhs_type& rhs)
    {
        x_ = rhs[0];
        y_ = rhs[1];
        return *this;
    }

public:
    data_type operator[](unsigned int index) const
    {
        return data_[index];
    }
    data_type& operator[](unsigned int index)
    {
        return data_[index];
    }
    data_type* begin(void)
    {
        return data_;
    }
    data_type* end(void)
    {
        return data_+2;
    }
    const data_type* begin(void)	const
    {
        return data_;
    }
    const data_type* end(void)	const
    {
        return data_+2;
    }
    size_t size(void) const{return 2;}
public:
    template<typename func>
    void for_each(func)
    {
        std::for_each(data_,data_+2,func());
    }
    template<typename rhs_type>
    vector<2,data_type>& operator+=(const rhs_type* rhs)
    {
        x_ += rhs[0];
        y_ += rhs[1];
        return *this;
    }
    template<typename rhs_type>
    vector<2,data_type>& operator-=(const rhs_type* rhs)
    {
        x_ -= rhs[0];
        y_ -= rhs[1];
        return *this;
    }
    template<typename rhs_type>
    vector<2,data_type>& operator+=(const vector<2,rhs_type>& rhs)
    {
        x_ += rhs[0];
        y_ += rhs[1];
        return *this;
    }
    template<typename rhs_type>
    vector<2,data_type>& operator-=(const vector<2,rhs_type>& rhs)
    {
        x_ -= rhs[0];
        y_ -= rhs[1];
        return *this;
    }
    vector<2,data_type>& operator+=(data_type r)
    {
        x_ += r;
        y_ += r;
        return *this;
    }
    vector<2,data_type>& operator-=(data_type r)
    {
        x_ -= r;
        y_ -= r;
        return *this;
    }
    template<typename rhs_type>
    vector<2,data_type>& operator*=(rhs_type r)
    {
        x_ *= r;
        y_ *= r;
        return *this;
    }
    template<typename rhs_type>
    vector<2,data_type>& operator/=(rhs_type r)
    {
        x_ /= r;
        y_ /= r;
        return *this;
    }
    template<typename rhs_type>
    vector<2,data_type> operator+(const rhs_type& rhs) const
    {
        return vector<2,data_type>(*this)+=rhs;
    }
    template<typename rhs_type>
    vector<2,data_type> operator-(const rhs_type& rhs) const
    {
        return vector<2,data_type>(*this)-=rhs;
    }
    vector<2,data_type> operator+(data_type rhs) const
    {
        return vector<2,data_type>(*this)+=rhs;
    }
    vector<2,data_type> operator-(data_type rhs) const
    {
        return vector<2,data_type>(*this)-=rhs;
    }
    template<typename rhs_type>
    vector<2,data_type> operator*(rhs_type rhs) const
    {
        return vector<2,data_type>(*this)*=rhs;
    }
    template<typename rhs_type>
    vector<2,data_type> operator/(rhs_type rhs) const
    {
        return vector<2,data_type>(*this)/=rhs;
    }
    vector<2,data_type> operator-(void) const
    {
        return vector<2,data_type>(-x_,-y_);
    }
    void floor(void)
    {
        x_ = std::floor(x_);
        y_ = std::floor(y_);
    }
    void round(void)
    {
        x_ = std::round(x_);
        y_ = std::round(y_);
    }

    void abs(void)
    {
        x_ = std::abs(x_);
        y_ = std::abs(y_);
    }
    template<typename rhs_type>
    data_type operator*(const vector<2,rhs_type>& rhs) const
    {
        return x_*rhs.x_+y_*rhs.y_;
    }

    data_type length2(void)	const
    {
        return x_*x_+y_*y_;
    }
    double length(void)	const
    {
        return std::sqrt(x_*x_+y_*y_);
    }

    data_type normalize(void)
    {
        data_type r = std::sqrt(length2());
        if (r == (data_type)0)
            return 0;
        x_ /= r;
        y_ /= r;
        return r;
    }
public:
    double project_length(const vector<2,data_type>& rhs)
    {
        return *this*rhs/length();
    }
    vector<2,data_type> project(const vector<2,data_type>& rhs)
    {
        vector<2,data_type> proj = *this;
        return *this*(*this*rhs/length2());
    }
    template<typename tran_type>
    void to(const tran_type& m)
    {
        data_type x__(x_),y__(y_);
        x_ = x__*m[0] + y__*m[1] + m[2];
        y_ = x__*m[3] + y__*m[4] + m[5];
    }
    template<typename tran_type>
    void to(const tran_type* m)
    {
        data_type x__(x_),y__(y_);
        x_ = x__*m[0] + y__*m[1] + m[2];
        y_ = x__*m[3] + y__*m[4] + m[5];
    }
    template<typename tran_type>
    void rotate(const tran_type& m)
    {
        data_type x__(x_),y__(y_);
        x_ = x__*m[0] + y__*m[1];
        y_ = x__*m[2] + y__*m[3];
    }
    template<typename tran_type>
    void rotate(const tran_type* m)
    {
        data_type x__(x_),y__(y_);
        x_ = x__*m[0] + y__*m[1];
        y_ = x__*m[2] + y__*m[3];
    }
public:
    bool operator<(const vector<2,data_type>& rhs) const
    {
        if (y_ != rhs.y_)
            return y_ < rhs.y_;
        return x_ < rhs.x_;
    }
    bool operator>(const vector<2,data_type>& rhs) const
    {
        if (y_ != rhs.y_)
            return y_ > rhs.y_;
        return x_ > rhs.x_;
    }
    bool operator==(const vector<2,data_type>& rhs) const
    {
        return x_ == rhs.x_ && y_ == rhs.y_;
    }
    bool operator!=(const vector<2,data_type>& rhs) const
    {
        return x_ != rhs.x_ || y_ != rhs.y_;
    }
    friend std::istream& operator>>(std::istream& in,vector<2,data_type>& point)
    {
        in >> point.x_ >> point.y_;
        return in;
    }
    friend std::ostream& operator<<(std::ostream& out,const vector<2,data_type>& point)
    {
        out << point.x_ << " " << point.y_ << " ";
        return out;
    }


public:
    data_type x(void) const
    {
        return x_;
    }
    data_type y(void) const
    {
        return y_;
    }
};


template<typename data_type>
class vector<3,data_type>
{
protected:
    union
    {
        data_type data_[3];
        struct
        {
            data_type x_;
            data_type y_;
            data_type z_;
        };
    };
public:
    using value_type = data_type;
public:
    vector(void):x_(0),y_(0),z_(0)				{}
    template<typename T>
    vector(T x,T y,T z):x_(data_type(x)),y_(data_type(y)),z_(data_type(z)){}
    template<typename T>
    vector(const T& rhs):x_(data_type(rhs[0])),y_(data_type(rhs[1])),z_(data_type(rhs[2])){}
    template<typename T>
    vector(const T* rhs):x_(data_type(rhs[0])),y_(data_type(rhs[1])),z_(data_type(rhs[2])){}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    vector& operator=(T rhs)
    {
        x_ = rhs;
        y_ = rhs;
        z_ = rhs;
        return *this;
    }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    vector& operator=(const T* rhs)
    {
        x_ = rhs[0];
        y_ = rhs[1];
        z_ = rhs[2];
        return *this;
    }
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    vector& operator=(const T& rhs)
    {
        x_ = rhs[0];
        y_ = rhs[1];
        z_ = rhs[2];
        return *this;
    }
public:
    const data_type& operator[](unsigned int index) const
    {
        return data_[index];
    }
    data_type& operator[](unsigned int index)
    {
        return data_[index];
    }
    data_type* begin(void)
    {
        return data_;
    }
    data_type* end(void)
    {
        return data_+3;
    }
    const data_type* begin(void)	const
    {
        return data_;
    }
    const data_type* end(void)	const
    {
        return data_+3;
    }
    size_t size(void) const{return 3;}
public:
    template<typename func>
    void for_each(func)
    {
        std::for_each(data_,data_+3,func());
    }
    template<typename rhs_type>
    vector<3,data_type>& operator+=(const rhs_type* rhs)
    {
        x_ += rhs[0];
        y_ += rhs[1];
        z_ += rhs[2];
        return *this;
    }
    template<typename rhs_type>
    vector<3,data_type>& operator-=(const rhs_type* rhs)
    {
        x_ -= rhs[0];
        y_ -= rhs[1];
        z_ -= rhs[2];
        return *this;
    }
    template<typename rhs_type>
    vector<3,data_type>& operator+=(const vector<3,rhs_type>& rhs)
    {
        x_ += rhs[0];
        y_ += rhs[1];
        z_ += rhs[2];
        return *this;
    }
    template<typename rhs_type>
    vector<3,data_type>& operator-=(const vector<3,rhs_type>& rhs)
    {
        x_ -= rhs[0];
        y_ -= rhs[1];
        z_ -= rhs[2];
        return *this;
    }
    vector<3,data_type>& operator+=(data_type r)
    {
        x_ += r;
        y_ += r;
        z_ += r;
        return *this;
    }
    vector<3,data_type>& operator-=(data_type r)
    {
        x_ -= r;
        y_ -= r;
        z_ -= r;
        return *this;
    }
    template<typename rhs_type>
    vector<3,data_type>& operator*=(rhs_type r)
    {
        x_ *= r;
        y_ *= r;
        z_ *= r;
        return *this;
    }
    template<typename rhs_type>
    vector<3,data_type>& operator/=(rhs_type r)
    {
        x_ /= r;
        y_ /= r;
        z_ /= r;
        return *this;
    }
    template<typename rhs_type>
    vector<3,data_type> operator+(const rhs_type& rhs) const
    {
        vector<3,data_type> result(*this);result += rhs;
        return result;
    }
    template<typename rhs_type>
    vector<3,data_type> operator-(const rhs_type& rhs) const
    {
        vector<3,data_type> result(*this);result -= rhs;
        return result;
    }

    vector<3,data_type> operator+(data_type rhs) const
    {
        vector<3,data_type> result(*this);result += rhs;
        return result;
    }
    vector<3,data_type> operator-(data_type rhs) const
    {
        vector<3,data_type> result(*this);result -= rhs;
        return result;
    }
    template<typename rhs_type>
    vector<3,data_type> operator*(rhs_type rhs) const
    {
        vector<3,data_type> result(*this);result *= rhs;
        return result;
    }
    template<typename rhs_type>
    vector<3,data_type> operator/(rhs_type rhs) const
    {
        vector<3,data_type> result(*this);result /= rhs;
        return result;
    }
    vector<3,data_type> operator-(void) const
    {
        return vector<3,data_type>(-x_,-y_,-z_);
    }
    vector<3,data_type> cross_product(const vector<3,data_type>& rhs) const
    {
        return vector<3,data_type>(y_*rhs.z_-rhs.y_*z_,z_*rhs.x_-rhs.z_*x_,x_*rhs.y_-rhs.x_*y_);
    }
        vector<3,data_type> normal(void) const
    {
                vector<3,data_type> result;
        if(std::abs(y_) > std::abs(x_))
                        result = cross_product(vector<3,data_type>(1.0,0,0));
		else
                        result = cross_product(vector<3,data_type>(0,1.0,0));
		result.normalize();
		return result;
    }
        vector<3,data_type> normal(const vector<3,data_type>& rhs) const
    {
        vector<3,data_type> result;
		result = cross_product(rhs);
		result.normalize();
		return result;
    }
	//apply(std::ptr_fun(static_cast <float(*)(float)>(&std::floor)));
    template<typename function_type>
	void apply(function_type& function)
	{
		x_ = function(x_);
        y_ = function(y_);
        z_ = function(z_);
	}

    void floor(void)
    {
        x_ = std::floor(x_);
        y_ = std::floor(y_);
        z_ = std::floor(z_);
    }
    void round(void)
    {
        x_ = std::round(x_);
        y_ = std::round(y_);
        z_ = std::round(z_);
    }
    void abs(void)
    {
        x_ = std::abs(x_);
        y_ = std::abs(y_);
        z_ = std::abs(z_);
    }

    void ceil(void)
    {
        x_ = std::ceil(x_);
        y_ = std::ceil(y_);
        z_ = std::ceil(z_);
    }

    data_type operator*(const vector<3,data_type>& rhs) const
    {
        return x_*rhs.x_+y_*rhs.y_+z_*rhs.z_;
    }

    template<typename rhs_type>
    data_type operator*(const vector<3,rhs_type>& rhs) const
    {
        return x_*rhs[0]+y_*rhs[1]+z_*rhs[2];
    }

    data_type length2(void)	const
    {
        return x_*x_+y_*y_+z_*z_;
    }
    double length(void)	const
    {
        return std::sqrt(double(x_*x_+y_*y_+z_*z_));
    }

    data_type normalize(void)
    {
        data_type r = std::sqrt(length2());
        if (r == data_type(0))
            return 0;
        x_ /= r;
        y_ /= r;
        z_ /= r;
        return r;
    }
public:
        data_type distance2(const vector<3,data_type>& rhs)
	{
		data_type sum = 0;
		data_type t = x_-rhs.x_;
		sum += t*t;
		t = y_-rhs.y_;
		sum += t*t;
		t = z_-rhs.z_;
		sum += t*t;
		return sum;
	}
    template<typename pointer_type>
	data_type distance2(const pointer_type* rhs)
	{
		data_type sum = 0;
		data_type t = x_-rhs[0];
		sum += t*t;
		t = y_-rhs[1];
		sum += t*t;
		t = z_-rhs[2];
		sum += t*t;
		return sum;
	}
        data_type distance(const vector<3,data_type>& rhs)
	{
		return std::sqrt(distance2(rhs));
	}
    template<typename pointer_type>
	data_type distance(const pointer_type* rhs)
	{
		return std::sqrt(distance2(rhs));
	}
        template<typename tran_type>
        void to(const tran_type& m)
        {
            data_type x__(x_),y__(y_),z__(z_);
            x_ = x__*m[0] + y__*m[1] + z__*m[2] + m[3];
            y_ = x__*m[4] + y__*m[5] + z__*m[6] + m[7];
            z_ = x__*m[8] + y__*m[9] + z__*m[10] + m[11];
        }
        template<typename tran_type>
        void to(const tran_type* m)
        {
            data_type x__(x_),y__(y_),z__(z_);
            x_ = x__*m[0] + y__*m[1] + z__*m[2] + m[3];
            y_ = x__*m[4] + y__*m[5] + z__*m[6] + m[7];
            z_ = x__*m[8] + y__*m[9] + z__*m[10] + m[11];
        }
        template<typename tran_type>
        void rotate(const tran_type& m)
        {
            data_type x__(x_),y__(y_),z__(z_);
            x_ = x__*m[0] + y__*m[1] + z__*m[2];
            y_ = x__*m[3] + y__*m[4] + z__*m[5];
            z_ = x__*m[6] + y__*m[7] + z__*m[8];
        }
        template<typename tran_type>
        void rotate(const tran_type* m)
        {
            data_type x__(x_),y__(y_),z__(z_);
            x_ = x__*m[0] + y__*m[1] + z__*m[2];
            y_ = x__*m[3] + y__*m[4] + z__*m[5];
            z_ = x__*m[6] + y__*m[7] + z__*m[8];
        }
public:
    bool operator<(const vector<3,data_type>& rhs) const
    {
        if (z_ != rhs.z_)
            return z_ < rhs.z_;
        if (y_ != rhs.y_)
            return y_ < rhs.y_;
        return x_ < rhs.x_;
    }
    bool operator>(const vector<3,data_type>& rhs) const
    {
        if (z_ != rhs.z_)
            return z_ > rhs.z_;
        if (y_ != rhs.y_)
            return y_ > rhs.y_;
        return x_ > rhs.x_;
    }
    bool operator==(const vector<3,data_type>& rhs) const
    {
        return x_ == rhs.x_ && y_ == rhs.y_ && z_ == rhs.z_;
    }
    bool operator!=(const vector<3,data_type>& rhs) const
    {
        return x_ != rhs.x_ || y_ != rhs.y_ || z_ != rhs.z_;
    }
    friend std::istream& operator>>(std::istream& in,vector<3,data_type>& point)
    {
        in >> point.x_ >> point.y_ >> point.z_;
        return in;
    }
    friend std::ostream& operator<<(std::ostream& out,const vector<3,data_type>& point)
    {
        out << point.x_ << " " << point.y_ << " " << point.z_ << " ";
        return out;
    }
public:
    data_type x(void) const
    {
        return x_;
    }
    data_type y(void) const
    {
        return y_;
    }
    data_type z(void) const
    {
        return z_;
    }
};

template<typename value_type>
inline tipl::vector<3,value_type> v(value_type x,value_type y,value_type z)
{
    return tipl::vector<3>(x,y,z);
}
template<typename value_type>
inline tipl::vector<2,value_type> v(value_type x,value_type y)
{
    return tipl::vector<2>(x,y);
}
template<typename value_type>
inline tipl::shape<3> s(value_type x,value_type y,value_type z)
{
    return tipl::shape<3>(x,y,z);
}
template<typename value_type>
inline tipl::shape<2> s(value_type x,value_type y)
{
    return tipl::shape<2>(x,y);
}


}
#endif
