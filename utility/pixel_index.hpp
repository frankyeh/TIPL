#ifndef PIXEL_INDEX_HPP
#define PIXEL_INDEX_HPP
#include <vector>
#include <deque>
#include <algorithm>
#include <iosfwd>
#include <cmath>
#include "geometry.hpp"

namespace image
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
    pixel_index(const geometry<2>& geo):x_(0),y_(0),index_(0),w(geo[0]){}
    pixel_index(const pixel_index& rhs)
    {
        *this = rhs;
    }
    template<class vtype>
    pixel_index(vtype x,vtype y,vtype index,const geometry<2>& geo):
            x_(int(x)),y_(int(y)),index_(index),w(geo[0]){}
    template<class vtype>
    pixel_index(vtype x,vtype y,const geometry<2>& geo):
            x_(int(x)),y_(int(y)),index_(int(y)*geo.width()+int(x)),w(geo[0]){}
    template<class vtype>
    pixel_index(vtype* offset,const geometry<2>& geo):
            x_(offset[0]),y_(offset[1]),index_(offset[1]*geo.width()+offset[0]),w(geo[0]){}
    template<class vtype>
    pixel_index(vtype y,const geometry<2>& geo):
            x_(y % geo.width()),y_(y / geo.width()),index_(y),w(geo[0]){}

    const pixel_index& operator=(const pixel_index<2>& rhs)
    {
        x_ = rhs.x_;
        y_ = rhs.y_;
        index_ = rhs.index_;
        return *this;
    }

    template<class rhs_type>
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
    bool operator<(int rhs) const
    {
        return index_ < rhs;
    }
    bool operator==(int rhs) const
    {
        return index_ == rhs;
    }
    bool operator!=(int rhs) const
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
    template<class stream_type>
    friend stream_type& operator>>(stream_type& in,pixel_index& rhs)
    {
        in >> rhs.x_ >> rhs.y_;
        return in;
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
    int index_;
    int w,h;
public:
    pixel_index(const geometry<3>& geo):x_(0),y_(0),z_(0),index_(0),w(geo[0]),h(geo[1]){}
    pixel_index(const pixel_index& rhs)
    {
        *this = rhs;
    }
    template<class vtype>
    pixel_index(vtype x,vtype y,vtype z,int i,const geometry<3>& geo):x_(int(x)),y_(int(y)),z_(int(z)),index_(i),w(geo[0]),h(geo[1]){}
    template<class vtype>
    pixel_index(vtype x,vtype y,vtype z,const geometry<3>& geo):
            x_(int(x)),y_(int(y)),z_(int(z)),index_((int(z)*geo.height() + int(y))*geo.width()+int(x)),w(geo[0]),h(geo[1]){}
    template<class vtype>
    pixel_index(vtype* offset,const geometry<3>& geo):
            x_(offset[0]),y_(offset[1]),z_(offset[2]),
            index_((offset[2]*geo.height() + offset[1])*geo.width()+offset[0]),
            w(geo[0]),h(geo[1]){}
    template<class vtype>
    pixel_index(vtype i,const geometry<3>& geo):index_(i),w(geo[0]),h(geo[1])
    {
        x_ = i % geo.width();
        i /= geo.width();
        y_ = i % geo.height();
        z_ = i / geo.height();
    }

    const pixel_index<3>& operator=(const pixel_index<3>& rhs)
    {
        x_ = rhs.x_;
        y_ = rhs.y_;
        z_ = rhs.z_;
        index_ = rhs.index_;
        return *this;
    }

        template<class rhs_type>
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
    bool operator<(int rhs) const
    {
        return index_ < rhs;
    }
    bool operator==(int rhs) const
    {
        return index_ == rhs;
    }
    bool operator!=(int rhs) const
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
    bool is_valid(const geometry<3>& geo) const
    {
        return offset_[2] < geo[2];
    }
    template<class stream_type>
    friend stream_type& operator>>(stream_type& in,pixel_index& rhs)
    {
        in >> rhs.x_ >> rhs.y_ >> rhs.z_;
        return in;
    }
};


template<int dim,class data_type = float>
class vector;

template<class data_type>
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
    vector(const vector& rhs)
    {
        *this = rhs;
    }
    vector(const geometry<2>& rhs)
    {
        *this = rhs;
    }

    template<class rhs_type>
    explicit vector(const rhs_type& rhs):x_(rhs[0]),y_(rhs[1]){}
    template<class rhs_type>
    explicit vector(const rhs_type* rhs):x_(rhs[0]),y_(rhs[1]){}


    template<class rhs_type>
    const vector<2,data_type>& operator=(const rhs_type* rhs)
    {
        x_ = rhs[0];
        y_ = rhs[1];
        return *this;
    }
    template<class rhs_type>
    const vector<2,data_type>& operator=(const rhs_type& rhs)
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
public:
    template<class func>
    void for_each(func)
    {
        std::for_each(data_,data_+2,func());
    }
    template<class rhs_type>
    vector<2,data_type>& operator+=(const rhs_type* rhs)
    {
        x_ += rhs[0];
        y_ += rhs[1];
        return *this;
    }
    template<class rhs_type>
    vector<2,data_type>& operator-=(const rhs_type* rhs)
    {
        x_ -= rhs[0];
        y_ -= rhs[1];
        return *this;
    }
    template<class rhs_type>
    vector<2,data_type>& operator+=(const vector<2,rhs_type>& rhs)
    {
        x_ += rhs[0];
        y_ += rhs[1];
        return *this;
    }
    template<class rhs_type>
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
    vector<2,data_type>& operator*=(data_type r)
    {
        x_ *= r;
        y_ *= r;
        return *this;
    }
    vector<2,data_type>& operator/=(data_type r)
    {
        x_ /= r;
        y_ /= r;
        return *this;
    }
    template<class rhs_type>
    vector<2,data_type> operator+(const rhs_type& rhs) const
    {
        return vector<2,data_type>(*this)+=rhs;
    }
    template<class rhs_type>
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
    vector<2,data_type> operator*(data_type rhs) const
    {
        return vector<2,data_type>(*this)*=rhs;
    }
    vector<2,data_type> operator/(data_type rhs) const
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
    template<class rhs_type>
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
    template<class tran_type>
    void to(const tran_type& m)
    {
        data_type x__(x_),y__(y_);
        x_ = x__*m[0] + y__*m[1] + m[2];
        y_ = x__*m[3] + y__*m[4] + m[5];
    }
    template<class tran_type>
    void to(const tran_type* m)
    {
        data_type x__(x_),y__(y_);
        x_ = x__*m[0] + y__*m[1] + m[2];
        y_ = x__*m[3] + y__*m[4] + m[5];
    }
    template<class tran_type>
    void rotate(const tran_type& m)
    {
        data_type x__(x_),y__(y_);
        x_ = x__*m[0] + y__*m[1];
        y_ = x__*m[2] + y__*m[3];
    }
    template<class tran_type>
    void rotate(const tran_type* m)
    {
        data_type x__(x_),y__(y_);
        x_ = x__*m[0] + y__*m[1];
        y_ = x__*m[2] + y__*m[3];
    }
public:
    bool operator<(const vector<2,data_type>& rhs) const
    {
        if (y_ != rhs.y)
            return y_ < rhs.y;
        return x_ < rhs.x_;
    }
    bool operator>(const vector<2,data_type>& rhs) const
    {
        if (y_ != rhs.y)
            return y_ > rhs.y;
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


template<class data_type>
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
    vector(void):x_(0),y_(0),z_(0)				{}
    vector(data_type x,data_type y,data_type z):x_(x),y_(y),z_(z){}
    vector(const vector<3,data_type>& rhs)
    {
        *this = rhs;
    }
    vector(const geometry<3>& rhs)
    {
        *this = rhs;
    }
    template<class rhs_type>
    explicit vector(const rhs_type& rhs):x_(rhs[0]),y_(rhs[1]),z_(rhs[2]){}
    template<class rhs_type>
    explicit vector(const rhs_type* rhs):x_(rhs[0]),y_(rhs[1]),z_(rhs[2]){}

    template<class rhs_type>
    const vector<3,data_type>& operator=(const rhs_type* rhs)
    {
        x_ = rhs[0];
        y_ = rhs[1];
        z_ = rhs[2];
        return *this;
    }
        template<class rhs_type>
    const vector<3,data_type>& operator=(const rhs_type& rhs)
    {
        x_ = rhs[0];
        y_ = rhs[1];
        z_ = rhs[2];
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
public:
    template<class func>
    void for_each(func)
    {
        std::for_each(data_,data_+3,func());
    }
    template<class rhs_type>
    vector<3,data_type>& operator+=(const rhs_type* rhs)
    {
        x_ += rhs[0];
        y_ += rhs[1];
        z_ += rhs[2];
        return *this;
    }
    template<class rhs_type>
    vector<3,data_type>& operator-=(const rhs_type* rhs)
    {
        x_ -= rhs[0];
        y_ -= rhs[1];
        z_ -= rhs[2];
        return *this;
    }
    template<class rhs_type>
    vector<3,data_type>& operator+=(const vector<3,rhs_type>& rhs)
    {
        x_ += rhs[0];
        y_ += rhs[1];
        z_ += rhs[2];
        return *this;
    }
    template<class rhs_type>
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
    vector<3,data_type>& operator*=(data_type r)
    {
        x_ *= r;
        y_ *= r;
        z_ *= r;
        return *this;
    }
    vector<3,data_type>& operator/=(data_type r)
    {
        x_ /= r;
        y_ /= r;
        z_ /= r;
        return *this;
    }
    template<class rhs_type>
    vector<3,data_type> operator+(const rhs_type& rhs) const
    {
        vector<3,data_type> result(*this);result += rhs;
        return result;
    }
    template<class rhs_type>
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

    vector<3,data_type> operator*(data_type rhs) const
    {
        vector<3,data_type> result(*this);result *= rhs;
        return result;
    }
    vector<3,data_type> operator/(data_type rhs) const
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
    template<class function_type>
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

    template<class rhs_type>
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
        return std::sqrt((double)(x_*x_+y_*y_+z_*z_));
    }

    data_type normalize(void)
    {
        data_type r = std::sqrt(length2());
        if (r == (data_type)0)
            return 0;
        x_ /= r;
        y_ /= r;
        z_ /= r;
        return r;
    }
public:
    data_type project_length(const vector<3,data_type>& rhs)
    {
        return *this*rhs/length();
    }
    vector<3,data_type> project(const vector<3,data_type>& rhs)
    {
        vector<3,data_type> proj = *this;
        return *this*(*this*rhs/length2());
    }
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
    template<class pointer_type>
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
    template<class pointer_type>
	data_type distance(const pointer_type* rhs)
	{
		return std::sqrt(distance2(rhs));
	}
        template<class tran_type>
        void to(const tran_type& m)
        {
            data_type x__(x_),y__(y_),z__(z_);
            x_ = x__*m[0] + y__*m[1] + z__*m[2] + m[3];
            y_ = x__*m[4] + y__*m[5] + z__*m[6] + m[7];
            z_ = x__*m[8] + y__*m[9] + z__*m[10] + m[11];
        }
        template<class tran_type>
        void to(const tran_type* m)
        {
            data_type x__(x_),y__(y_),z__(z_);
            x_ = x__*m[0] + y__*m[1] + z__*m[2] + m[3];
            y_ = x__*m[4] + y__*m[5] + z__*m[6] + m[7];
            z_ = x__*m[8] + y__*m[9] + z__*m[10] + m[11];
        }
        template<class tran_type>
        void rotate(const tran_type& m)
        {
            data_type x__(x_),y__(y_),z__(z_);
            x_ = x__*m[0] + y__*m[1] + z__*m[2];
            y_ = x__*m[3] + y__*m[4] + z__*m[5];
            z_ = x__*m[6] + y__*m[7] + z__*m[8];
        }
        template<class tran_type>
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



}
#endif
