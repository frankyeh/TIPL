//---------------------------------------------------------------------------
#ifndef basic_imageH
#define basic_imageH
#include <vector>
#include "geometry.hpp"
#include "pixel_value.hpp"
//---------------------------------------------------------------------------
namespace image
{

template <typename pixel_type>
class pointer_memory
{
public:
    typedef pixel_type value_type;
    typedef pixel_type* iterator;
    typedef const pixel_type* const_iterator;
protected:
    pixel_type* from;
    pixel_type* to;
    unsigned int size_;
public:
    pointer_memory(void):from(0),to(0),size_(0){}
    template<typename iterator_type>
    pointer_memory(iterator_type from_,iterator_type to_):
            from(&*from_),to(0),size_(to_-from_){to = from + size_;}
    pointer_memory(const pointer_memory& rhs):from(rhs.from),to(rhs.to),size_(rhs.size_){}
    const pointer_memory& operator=(const pointer_memory& rhs)
    {
        if (this == &rhs)
            return *this;
        from = rhs.from;
        to = rhs.to;
        size_ = rhs.size_;
        return *this;
    }
    const pointer_memory& operator=(std::vector<pixel_type>& rhs)
    {
        if((size_ = rhs.size()))
        {
            from = &rhs[0];
            to = from + size_;
        }
        else
            from = to = 0;
    }
    const pointer_memory& operator=(pixel_type* rhs)
    {
        from = rhs;
        to = rhs + size_;
        return *this;
    }
public:
    pixel_type operator[](unsigned int index) const
    {
        return from[index];
    }
    pixel_type& operator[](unsigned int index)
    {
        return from[index];
    }
    const_iterator begin(void) const
    {
        return from;
    }
    const_iterator end(void) const
    {
        return to;
    }
    iterator begin(void)
    {
        return from;
    }
    iterator end(void)
    {
        return to;
    }

    unsigned int size(void)            const
    {
        return size_;
    }

    bool empty(void) const
    {
        return size == 0;
    }
public:
    void swap(pointer_memory& rhs)
    {
        std::swap(from,rhs.from);
        std::swap(to,rhs.to);
        std::swap(size_,rhs.size_);
    }
    void resize(unsigned int new_size) 
    {
        size_ = new_size;
        to = from + size_;
    }
};

template <typename pixel_type>
class const_pointer_memory
{
public:
    typedef pixel_type value_type;
    typedef const pixel_type* iterator;
    typedef const pixel_type* const_iterator;
protected:
    const pixel_type* from;
    const pixel_type* to;
    unsigned int size_;
public:
    const_pointer_memory(void):from(0),to(0),size_(0){}
    template<typename iterator_type>
    const_pointer_memory(iterator_type from_,iterator_type to_):
            from(&*from_),to(0),size_(to_-from_){to = from + size_;}
    const_pointer_memory(const const_pointer_memory& rhs):from(rhs.from),to(rhs.to),size_(rhs.size_){}
    const const_pointer_memory& operator=(const const_pointer_memory& rhs)
    {
        if (this == &rhs)
            return *this;
        from = rhs.from;
        to = rhs.to;
        size_ = rhs.size_;
        return *this;
    }
    const const_pointer_memory& operator=(const std::vector<pixel_type>& rhs)
    {
        if((size_ = rhs.size()))
        {
            from = &rhs[0];
            to = from + size_;
        }
        else
            from = to = 0;
        return *this;
    }
    const const_pointer_memory& operator=(const pixel_type* rhs)
    {
        from = rhs;
        to = rhs + size_;
        return *this;
    }
public:
    pixel_type operator[](unsigned int index) const
    {
        return from[index];
    }
    const_iterator begin(void) const
    {
        return from;
    }
    const_iterator end(void) const
    {
        return to;
    }
    unsigned int size(void)            const
    {
        return size_;
    }

    bool empty(void) const
    {
        return size_ == 0;
    }
public:
    void swap(const_pointer_memory& rhs)
    {
        std::swap(from,rhs.from);
        std::swap(to,rhs.to);
        std::swap(size_,rhs.size_);
    }
    void resize(unsigned int new_size) 
    {
        size_ = new_size;
        to = from + size_;
    }
};



template <typename pixel_type,unsigned int Dim = 2,typename storage_type = std::vector<pixel_type> >
class basic_image : public storage_type
{
public:
    typedef pixel_type value_type;
    typedef typename storage_type::iterator iterator;
    typedef typename storage_type::const_iterator const_iterator;
    typedef image::basic_image<pixel_type,Dim-1,pointer_memory<pixel_type> > slice_type;
    typedef image::basic_image<pixel_type,Dim-1,const_pointer_memory<pixel_type> > const_slice_type;
    typedef image::geometry<Dim> geometry_type;
    static const unsigned int dimension = Dim;
protected:
    geometry_type geo;
public:
    const geometry_type& geometry(void) const
    {
        return geo;
    }
    unsigned int width(void) const
    {
        return geo.width();
    }
    unsigned int height(void) const
    {
        return geo.height();
    }
    unsigned int depth(void) const
    {
        return geo.depth();
    }
    unsigned int plane_size(void) const
    {
        return geo.plane_size();
    }
public:
    pixel_type at(unsigned int x,unsigned int y) const
    {
        return (*this)[y*geo[0]+x];
    }
    pixel_type& at(unsigned int x,unsigned int y)
    {
        return (*this)[y*geo[0]+x];
    }

    pixel_type at(unsigned int x,unsigned int y,unsigned int z) const
    {
        return (*this)[(z*geo[1]+y)*geo[0]+x];
    }
    pixel_type& at(unsigned int x,unsigned int y,unsigned int z)
    {
        return (*this)[(z*geo[1]+y)*geo[0]+x];
    }
public:
    basic_image(void) {}
    template<typename image_type>
    basic_image(const image_type& rhs):storage_type(rhs.begin(),rhs.end()),geo(rhs.geometry()) {}
    basic_image(const geometry_type& geo_):storage_type(geo_.size()),geo(geo_) {}
    basic_image(pixel_type* pointer,const geometry_type& geo_):storage_type(pointer,pointer+geo_.size()),geo(geo_) {}
    basic_image(const pixel_type* pointer,const geometry_type& geo_):storage_type(pointer,pointer+geo_.size()),geo(geo_) {}
public:
    template <typename rhs_pixel_type,typename rhs_storage_type>
    const basic_image& operator=(const basic_image<rhs_pixel_type,Dim,rhs_storage_type>& rhs)
    {
        storage_type::operator=(rhs);
        geo = rhs.geometry();
        return *this;
    }
    const basic_image& operator=(const pixel_type* rhs)
    {
        storage_type::operator=(rhs);
        return *this;
    }
    const basic_image& operator=(pixel_type* rhs)
    {
        storage_type::operator=(rhs);
        return *this;
    }
public:
    void swap(basic_image& rhs)
    {
        storage_type::swap(rhs);
        geo.swap(rhs.geo);
    }
    void resize(const image::geometry<Dim>& geo_)
    {
        geo = geo_;
        storage_type::resize(geo.size());
    }
    void clear(void)
    {
        storage_type::clear();
        std::fill(geo.begin(),geo.end(),0);
    }
public:
    slice_type slice_at(unsigned int pos)
    {
        image::geometry<Dim-1> slice_geo(geo.begin());
        return slice_type(&*this->begin()+pos*slice_geo.size(),slice_geo);
    }
    const_slice_type slice_at(unsigned int pos) const
    {
        image::geometry<Dim-1> slice_geo(geo.begin());
        return const_slice_type(&*this->begin()+pos*slice_geo.size(),slice_geo);
    }
public:
    template<typename value_type>
    const basic_image& operator+=(value_type value)
    {
        iterator end_iter = storage_type::end();
        for(iterator iter = storage_type::begin();iter != end_iter;++iter)
            *iter += value;
        return *this;
    }
    template<typename value_type>
    const basic_image& operator-=(value_type value)
    {
        iterator end_iter = storage_type::end();
        for(iterator iter = storage_type::begin();iter != end_iter;++iter)
            *iter -= value;
        return *this;
    }
    template<typename value_type>
    const basic_image& operator*=(value_type value)
    {
        iterator end_iter = storage_type::end();
        for(iterator iter = storage_type::begin();iter != end_iter;++iter)
            *iter *= value;
        return *this;
    }
    template<typename value_type>
    const basic_image& operator/=(value_type value)
    {
        iterator end_iter = storage_type::end();
        for(iterator iter = storage_type::begin();iter != end_iter;++iter)
            *iter /= value;
        return *this;
    }
    template <typename rhs_pixel_type,typename rhs_storage_type>
    const basic_image& operator+=(const basic_image<rhs_pixel_type,Dim,rhs_storage_type>& rhs)
    {
        iterator end_iter = storage_type::end();
        typename basic_image<rhs_pixel_type,Dim,rhs_storage_type>::const_iterator iter2 = rhs.begin();
        for(iterator iter = storage_type::begin();iter != end_iter;++iter,++iter2)
            *iter += *iter2;
        return *this;
    }
    template <typename rhs_pixel_type,typename rhs_storage_type>
    const basic_image& operator-=(const basic_image<rhs_pixel_type,Dim,rhs_storage_type>& rhs)
    {
        iterator end_iter = storage_type::end();
        typename basic_image<rhs_pixel_type,Dim,rhs_storage_type>::const_iterator iter2 = rhs.begin();
        for(iterator iter = storage_type::begin();iter != end_iter;++iter,++iter2)
            *iter -= *iter2;
        return *this;
    }
    template <typename rhs_pixel_type,typename rhs_storage_type>
    const basic_image& operator*=(const basic_image<rhs_pixel_type,Dim,rhs_storage_type>& rhs)
    {
        iterator end_iter = storage_type::end();
        typename basic_image<rhs_pixel_type,Dim,rhs_storage_type>::const_iterator iter2 = rhs.begin();
        for(iterator iter = storage_type::begin();iter != end_iter;++iter,++iter2)
            *iter *= *iter2;
        return *this;
    }
    template <typename rhs_pixel_type,typename rhs_storage_type>
    const basic_image& operator/=(const basic_image<rhs_pixel_type,Dim,rhs_storage_type>& rhs)
    {
        iterator end_iter = storage_type::end();
        typename basic_image<rhs_pixel_type,Dim,rhs_storage_type>::const_iterator iter2 = rhs.begin();
        for(iterator iter = storage_type::begin();iter != end_iter;++iter,++iter2)
            *iter /= *iter2;
        return *this;
    }
public:
    template<typename format_type>
    void save_to_file(const char* file_name)
    {
        format_type out;
        out.load_from_image(*this);
        out.save_to_file(file_name);
    }
    template<typename format_type>
    void load_from_file(const char* file_name)
    {
        format_type out;
        out.load_from_file(file_name);
        out.save_to_image(*this);
    }
};

typedef basic_image<rgb_color,2> color_image;
typedef basic_image<unsigned char,2> grayscale_image;



}
#endif
