#ifndef PIXEL_INDEX_HPP
#define PIXEL_INDEX_HPP
#include <vector>
#include <deque>
#include <algorithm>
#include <iosfwd>
#include "geometry.hpp"

namespace image
{
template<unsigned int dim>
class pixel_index;


template<>
class pixel_index<2>
{
public:
    typedef int value_type;
    static const unsigned int dimension = 2;
protected:
    union
    {
        value_type offset_[2];
        struct
        {
            value_type x_;
            value_type y_;
        };
    };
    value_type index_;
public:
    pixel_index(void):x_(0),y_(0),index_(0){}
    pixel_index(const pixel_index& rhs)
    {
        *this = rhs;
    }
    pixel_index(value_type x,value_type y,unsigned int i):
            x_(x),y_(y),index_(i){}
    pixel_index(value_type x,value_type y,const geometry<2>& geo):
            x_(x),y_(y),index_(y*geo.width()+x){}
    pixel_index(value_type* offset,const geometry<2>& geo):
            x_(offset[0]),y_(offset[1]),index_(offset[1]*geo.width()+offset[0]){}
    pixel_index(value_type y,const geometry<2>& geo):
            x_(y % geo.width()),y_(y / geo.width()),index_(y){}

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
    value_type x(void) const
    {
        return x_;
    }
    value_type y(void) const
    {
        return y_;
    }
    unsigned int index(void) const
    {
        return index_;
    }
public:
    const value_type* begin(void) const
    {
        return offset_;
    }
    const value_type* end(void) const
    {
        return offset_+2;
    }
    value_type* begin(void)
    {
        return offset_;
    }
    value_type* end(void)
    {
        return offset_+2;
    }
    value_type operator[](unsigned int index) const
    {
        return offset_[index];
    }
    value_type& operator[](unsigned int index)
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
public:
    void next(const geometry<2>& geo)
    {
        ++offset_[0];
        ++index_;
        if (offset_[0] < geo[0])
            return;
        offset_[0] = 0;
        ++offset_[1];
    }
    bool valid(const geometry<2>& geo) const
    {
        return offset_[1] < geo[1];
    }
    template<typename stream_type>
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
    typedef int value_type;
    static const unsigned int dimension = 3;
protected:
    union
    {
        value_type offset_[3];
        struct
        {
            value_type x_;
            value_type y_;
            value_type z_;
        };
    };
    value_type index_;
public:
    pixel_index(void):x_(0),y_(0),z_(0),index_(0){}
    pixel_index(const pixel_index& rhs)
    {
        *this = rhs;
    }
    pixel_index(value_type x,value_type y,value_type z,value_type i):x_(x),y_(y),z_(z),index_(i){}
    pixel_index(value_type x,value_type y,value_type z,const geometry<3>& geo):
            x_(x),y_(y),z_(z),index_((z*geo.height() + y)*geo.width()+x){}
    pixel_index(value_type* offset,const geometry<3>& geo):
            x_(offset[0]),y_(offset[1]),z_(offset[2]),
				index_((offset[2]*geo.height() + offset[1])*geo.width()+offset[0]){}
    pixel_index(value_type i,const geometry<3>& geo):index_(i)
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

        template<typename rhs_type>
    const pixel_index<3>& operator=(const rhs_type& rhs)
    {
        x_ = rhs[0];
        y_ = rhs[1];
        z_ = rhs[2];
        return *this;
    }
public:
    value_type x(void) const
    {
        return x_;
    }
    value_type y(void) const
    {
        return y_;
    }
    value_type z(void) const
    {
        return z_;
    }
    unsigned int index(void) const
    {
        return index_;
    }
public:
    const value_type* begin(void) const
    {
        return offset_;
    }
    const value_type* end(void) const
    {
        return offset_+3;
    }
    value_type* begin(void)
    {
        return offset_;
    }
    value_type* end(void)
    {
        return offset_+3;
    }
    value_type operator[](unsigned int index) const
    {
        return offset_[index];
    }
    value_type& operator[](unsigned int index)
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
public:
    void next(const geometry<3>& geo)
    {
        ++offset_[0];
        ++index_;
        if (offset_[0] < geo[0])
            return;
        offset_[0] = 0;
        ++offset_[1];
        if (offset_[1] < geo[1])
            return;
        offset_[1] = 0;
        ++offset_[2];
    }
    bool valid(const geometry<3>& geo) const
    {
        return offset_[2] < geo[2];
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
    vector(const vector& rhs)
    {
        *this = rhs;
    }
    vector(const geometry<2>& rhs)
    {
        *this = rhs;
    }

	template<typename rhs_type>
    explicit vector(const rhs_type& rhs):x_(rhs[0]),y_(rhs[1]){}
    template<typename rhs_type>
    explicit vector(const rhs_type* rhs):x_(rhs[0]),y_(rhs[1]){}


    template<typename rhs_type>
    const vector<2,data_type>& operator=(const rhs_type* rhs)
    {
        x_ = rhs[0];
        y_ = rhs[1];
        return *this;
    }
    template<typename rhs_type>
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
public:
    bool operator<(const vector<2,data_type>& rhs) const
    {
        if (y_ != rhs.y)
            return y < rhs.y;
        return x_ < rhs.x_;
    }
    bool operator>(const vector<2,data_type>& rhs) const
    {
        if (y_ != rhs.y)
            return y > rhs.y;
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
    template<typename rhs_type>
    explicit vector(const rhs_type& rhs):x_(rhs[0]),y_(rhs[1]),z_(rhs[2]){}
    template<typename rhs_type>
    explicit vector(const rhs_type* rhs):x_(rhs[0]),y_(rhs[1]),z_(rhs[2]){}

    template<typename rhs_type>
    const vector<3,data_type>& operator=(const rhs_type* rhs)
    {
        x_ = rhs[0];
        y_ = rhs[1];
        z_ = rhs[2];
        return *this;
    }
        template<typename rhs_type>
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
    template<typename rhs_type>
    vector<3,data_type> operator+(const rhs_type& rhs) const
    {
        return vector<3,data_type>(*this)+=rhs;
    }
    template<typename rhs_type>
    vector<3,data_type> operator-(const rhs_type& rhs) const
    {
        return vector<3,data_type>(*this)-=rhs;
    }

    vector<3,data_type> operator+(data_type rhs) const
    {
        return vector<3,data_type>(*this)+=rhs;
    }
    vector<3,data_type> operator-(data_type rhs) const
    {
        return vector<3,data_type>(*this)-=rhs;
    }

    vector<3,data_type> operator*(data_type rhs) const
    {
        return vector<3,data_type>(*this)*=rhs;
    }
    vector<3,data_type> operator/(data_type rhs) const
    {
        return vector<3,data_type>(*this)/=rhs;
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
        return std::sqrt(x_*x_+y_*y_+z_*z_);
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
