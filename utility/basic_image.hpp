//---------------------------------------------------------------------------
#ifndef basic_imageH
#define basic_imageH
#include <vector>
#include <thread>
#include <future>
#include "geometry.hpp"
#include "pixel_value.hpp"
#include "pixel_index.hpp"

//---------------------------------------------------------------------------
namespace tipl
{

template <class pixel_type>
class pointer_container
{
public:
    typedef pixel_type value_type;
    typedef pixel_type* iterator;
    typedef pixel_type* const_iterator;
    typedef pixel_type& reference;
protected:
    iterator from,to;
    size_t size_;
public:
    pointer_container(void):from(0),to(0),size_(0){}
    pointer_container(size_t size_):from(0),to(0),size_(size_){}
    template<class any_iterator_type>
    pointer_container(any_iterator_type from_,any_iterator_type to_):
        from(&*from_),to(0),size_(to_-from_){to = from + size_;}
    pointer_container(const pointer_container<pixel_type>& rhs){operator=(rhs);}
    pointer_container(std::vector<pixel_type>& rhs){operator=(rhs);}
public:
    const pointer_container& operator=(const pointer_container& rhs)
    {
        if (this == &rhs)
            return *this;
        from = rhs.from;
        to = rhs.to;
        size_ = rhs.size_;
        return *this;
    }
    const pointer_container& operator=(std::vector<pixel_type>& rhs)
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
public:
    template<typename value_type>
    const reference operator[](value_type index) const
    {
        return from[index];
    }
    template<typename value_type>
    reference operator[](value_type index)
    {
        return from[index];
    }
    iterator begin(void) const
    {
        return from;
    }
    iterator end(void) const
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

    size_t size(void)            const
    {
        return size_;
    }

    bool empty(void) const
    {
        return size_ == 0;
    }
    void clear(void)
    {
        size_ = 0;
        to = from;
    }

public:
    void swap(pointer_container& rhs)
    {
        std::swap(from,rhs.from);
        std::swap(to,rhs.to);
        std::swap(size_,rhs.size_);
    }
    void resize(size_t new_size)
    {
        size_ = new_size;
        to = from + size_;
    }
};

template <class pixel_type>
class const_pointer_container
{
public:
    typedef pixel_type value_type;
    typedef const pixel_type* iterator;
    typedef const pixel_type* const_iterator;
    typedef const pixel_type& reference;
protected:
    const_iterator from,to;
    size_t size_;
public:
    const_pointer_container(void):from(0),to(0),size_(0){}
    const_pointer_container(size_t size_):from(0),to(0),size_(size_){}
    template<class any_iterator_type>
    const_pointer_container(any_iterator_type from_,any_iterator_type to_):
        from(&*from_),to(0),size_(to_-from_){to = from + size_;}
    const_pointer_container(const const_pointer_container<pixel_type>& rhs){operator=(rhs);}
    const_pointer_container(const pointer_container<pixel_type>& rhs){operator=(rhs);}
    const_pointer_container(const std::vector<pixel_type>& rhs){operator=(rhs);}
public:
    template<class other_type>
    const const_pointer_container& operator=(const other_type& rhs)
    {
        if (this == &rhs)
            return *this;
        from = rhs.begin();
        to = rhs.end();
        size_ = rhs.size();
        return *this;
    }
    const const_pointer_container& operator=(const std::vector<pixel_type>& rhs)
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
public:
    template<typename value_type>
    reference operator[](value_type index) const
    {
        return from[index];
    }
    template<typename value_type>
    reference operator[](value_type index)
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
    const_iterator begin(void)
    {
        return from;
    }
    const_iterator end(void)
    {
        return to;
    }

    size_t size(void) const
    {
        return size_;
    }

    bool empty(void) const
    {
        return size_ == 0;
    }
public:
    void swap(const_pointer_container<pixel_type>& rhs)
    {
        std::swap(from,rhs.from);
        std::swap(to,rhs.to);
        std::swap(size_,rhs.size_);
    }
    void resize(size_t new_size)
    {
        size_ = new_size;
        to = from + size_;
    }
};


template <class pixel_type,unsigned int dim,class storage_type = std::vector<pixel_type> >
class image
{
public:
    typedef pixel_type value_type;
    typedef typename storage_type::iterator iterator;
    typedef typename storage_type::const_iterator const_iterator;
    typedef typename storage_type::reference reference;
    typedef tipl::image<pixel_type,dim-1,pointer_container<pixel_type> > slice_type;
    typedef tipl::image<pixel_type,dim-1,const_pointer_container<pixel_type> > const_slice_type;
    typedef tipl::geometry<dim> geometry_type;
    static const unsigned int dimension = dim;
protected:
    storage_type data;
    geometry_type geo;
public:
    const geometry_type& geometry(void) const
    {
        return geo;
    }
    int width(void) const
    {
        return geo.width();
    }
    int height(void) const
    {
        return geo.height();
    }
    int depth(void) const
    {
        return geo.depth();
    }
    size_t plane_size(void) const
    {
        return geo.plane_size();
    }
public:
    pixel_type at(unsigned int x,unsigned int y) const
    {
        return data[y*geo[0]+x];
    }
    reference at(unsigned int x,unsigned int y)
    {
        return data[y*geo[0]+x];
    }

    pixel_type at(unsigned int x,unsigned int y,unsigned int z) const
    {
        return data[size_t(z*geo[1]+y)*geo[0]+x];
    }
    reference at(unsigned int x,unsigned int y,unsigned int z)
    {
        return data[size_t(z*geo[1]+y)*geo[0]+x];
    }
public:
    image(void) {}
    image(const image& rhs){operator=(rhs);}
    image(image&& rhs){operator=(rhs);}
    template <class rhs_pixel_type,class rhs_storage_type>
    image(const image<rhs_pixel_type,dim,rhs_storage_type>& rhs){operator=(rhs);}
    image(const geometry_type& geo_):data(geo_.size()),geo(geo_) {}
    template <typename any_pixel_type>
    image(any_pixel_type* pointer,const geometry_type& geo_):data(pointer,pointer+geo_.size()),geo(geo_) {}
    template <typename any_pixel_type>
    image(const any_pixel_type* pointer,const geometry_type& geo_):data(pointer,pointer+geo_.size()),geo(geo_) {}
public:
    template <class rhs_pixel_type,class rhs_storage_type>
    const image& operator=(const image<rhs_pixel_type,dim,rhs_storage_type>& rhs)
    {
        storage_type new_data(rhs.begin(),rhs.end());
        data.swap(new_data);
        geo = rhs.geometry();
        return *this;
    }
    const image& operator=(const image& rhs)
    {
        storage_type new_data(rhs.begin(),rhs.end());
        data.swap(new_data);
        geo = rhs.geometry();
        return *this;
    }
    const image& operator=(image&& rhs)
    {
        data.swap(rhs.data);
        geo.swap(rhs.geo);
        return *this;
    }
public:
    void swap(image& rhs)
    {
        data.swap(rhs.data);
        geo.swap(rhs.geo);
    }
    void resize(const tipl::geometry<dim>& geo_)
    {
        geo = geo_;
        data.resize(geo.size());
    }
    void clear(void)
    {
        data.clear();
        std::fill(geo.begin(),geo.end(),0);
    }
    size_t size(void) const
    {
        return data.size();
    }
    bool empty(void) const
    {
        return data.empty();
    }
    pixel_type front(void) const
    {
        return data.front();
    }
    pixel_type back(void) const
    {
        return data.back();
    }
    template<typename value_type>
    const pixel_type& operator[](value_type index) const
    {
        return data[index];
    }
    template<typename value_type>
    reference operator[](value_type index)
    {
        return data[index];
    }
    const_iterator begin(void) const
    {
        return data.begin();
    }
    const_iterator end(void) const
    {
        return data.end();
    }

    iterator begin(void)
    {
        return data.begin();
    }
    iterator end(void)
    {
        return data.end();
    }
public:
    slice_type slice_at(unsigned int pos)
    {
        tipl::geometry<dim-1> slice_geo(geo.begin());
        return slice_type(&*data.begin()+pos*slice_geo.size(),slice_geo);
    }
    const_slice_type slice_at(unsigned int pos) const
    {
        tipl::geometry<dim-1> slice_geo(geo.begin());
        return const_slice_type(&*data.begin()+pos*slice_geo.size(),slice_geo);
    }
public:
    template<class value_type>
    const image& operator+=(value_type value)
    {
        iterator end_iter = data.end();
        for(iterator iter = data.begin();iter != end_iter;++iter)
            *iter += value;
        return *this;
    }
    template<class value_type>
    const image& operator-=(value_type value)
    {
        iterator end_iter = data.end();
        for(iterator iter = data.begin();iter != end_iter;++iter)
            *iter -= value;
        return *this;
    }
    template<class value_type>
    const image& operator*=(value_type value)
    {
        iterator end_iter = data.end();
        for(iterator iter = data.begin();iter != end_iter;++iter)
            *iter *= value;
        return *this;
    }
    template<class value_type>
    const image& operator/=(value_type value)
    {
        iterator end_iter = data.end();
        for(iterator iter = data.begin();iter != end_iter;++iter)
            *iter /= value;
        return *this;
    }
    template <class rhs_pixel_type,class rhs_storage_type>
    const image& operator+=(const image<rhs_pixel_type,dim,rhs_storage_type>& rhs)
    {
        iterator end_iter = data.end();
        typename image<rhs_pixel_type,dim,rhs_storage_type>::const_iterator iter2 = rhs.begin();
        for(iterator iter = data.begin();iter != end_iter;++iter,++iter2)
            *iter += *iter2;
        return *this;
    }
    template <class rhs_pixel_type,class rhs_storage_type>
    const image& operator-=(const image<rhs_pixel_type,dim,rhs_storage_type>& rhs)
    {
        iterator end_iter = data.end();
        typename image<rhs_pixel_type,dim,rhs_storage_type>::const_iterator iter2 = rhs.begin();
        for(iterator iter = data.begin();iter != end_iter;++iter,++iter2)
            *iter -= *iter2;
        return *this;
    }
    template <class rhs_pixel_type,class rhs_storage_type>
    const image& operator*=(const image<rhs_pixel_type,dim,rhs_storage_type>& rhs)
    {
        iterator end_iter = data.end();
        typename image<rhs_pixel_type,dim,rhs_storage_type>::const_iterator iter2 = rhs.begin();
        for(iterator iter = data.begin();iter != end_iter;++iter,++iter2)
            *iter *= *iter2;
        return *this;
    }
    template <class rhs_pixel_type,class rhs_storage_type>
    const image& operator/=(const image<rhs_pixel_type,dim,rhs_storage_type>& rhs)
    {
        iterator end_iter = data.end();
        typename image<rhs_pixel_type,dim,rhs_storage_type>::const_iterator iter2 = rhs.begin();
        for(iterator iter = data.begin();iter != end_iter;++iter,++iter2)
            *iter /= *iter2;
        return *this;
    }
public:
    template<class format_type>
    bool save_to_file(const char* file_name) const
    {
        format_type out;
        out.load_from_image(*this);
        return out.save_to_file(file_name);
    }
    template<class format_type,class voxel_type>
    bool save_to_file(const char* file_name,voxel_type vs) const
    {
        format_type out;
        out.load_from_image(*this);
        out.set_voxel_size(vs);
        return out.save_to_file(file_name);
    }
    template<class format_type>
    bool load_from_file(const char* file_name)
    {
        format_type out;
        if(!out.load_from_file(file_name))
            return false;
        out.save_to_image(*this);
        return true;
    }
    template<typename Func>
    void for_each(Func f)
    {
        for(pixel_index<dim> index(geometry());index.index() < data.size();++index)
            f(data[index.index()],index);
    }
    template<typename Func>
    void for_each(Func f) const
    {
        for(pixel_index<dim> index(geometry());index.index() < data.size();++index)
            f(data[index.index()],index);
    }
    template<typename Func>
    void for_each_mt(Func f, int thread_count = std::thread::hardware_concurrency())
    {
        if(thread_count < 1)
            thread_count = 1;
        size_t block_size = data.size()/thread_count;

        std::vector<std::future<void> > futures;
        size_t pos = 0;
        for(int id = 1; id < thread_count; id++)
        {
            size_t end = pos + block_size;
            futures.push_back(std::move(std::async(std::launch::async, [this,f,pos,end]
            {
                for(pixel_index<dim> index(pos,geometry());index.index() < end;++index)
                    f(data[index.index()],index);
            })));
            pos = end;
        }
        for(pixel_index<dim> index(pos,geometry());index.index() < data.size();++index)
            f(data[index.index()],index);
        for(auto &future : futures)
            future.wait();
    }
    template<typename Func>
    void for_each_mt(Func f, int thread_count = std::thread::hardware_concurrency()) const
    {
        if(thread_count < 1)
            thread_count = 1;
        size_t block_size = data.size()/thread_count;

        std::vector<std::future<void> > futures;
        size_t pos = 0;
        for(int id = 1; id < thread_count; id++)
        {
            size_t end = pos + block_size;
            futures.push_back(std::move(std::async(std::launch::async, [this,f,pos,end]
            {
                for(pixel_index<dim> index(pos,geometry());index.index() < end;++index)
                    f(data[index.index()],index);
            })));
            pos = end;
        }
        for(pixel_index<dim> index(pos,geometry());index.index() < data.size();++index)
            f(data[index.index()],index);
        for(auto &future : futures)
            future.wait();
    }
    template<typename Func>
    void for_each_mt2(Func f, int thread_count = std::thread::hardware_concurrency())
    {
        if(thread_count < 1)
            thread_count = 1;
        size_t block_size = data.size()/thread_count;

        std::vector<std::future<void> > futures;
        size_t pos = 0;
        for(int id = 1; id < thread_count; id++)
        {
            size_t end = pos + block_size;
            futures.push_back(std::move(std::async(std::launch::async, [this,id,f,pos,end]
            {
                for(pixel_index<dim> index(pos,geometry());index.index() < end;++index)
                    f(data[index.index()],index,id);
            })));
            pos = end;
        }
        for(pixel_index<dim> index(pos,geometry());index.index() < data.size();++index)
            f(data[index.index()],index,0);
        for(auto &future : futures)
            future.wait();
    }
};


typedef image<rgb,2> color_image;
typedef image<unsigned char,2> grayscale_image;

template <class pixel_type,unsigned int dim>
class pointer_image : public image<pixel_type,dim,pointer_container<pixel_type> >
{
public:
    typedef pixel_type value_type;
    typedef image<pixel_type,dim,pointer_container<pixel_type> > base_type;
    typedef typename base_type::iterator iterator;
    typedef typename base_type::iterator const_iterator;
    static const unsigned int dimension = dim;
public:
    pointer_image(void) {}
    pointer_image(const pointer_image& rhs):base_type(){operator=(rhs);}
    template<class rhs_storage_type>
    pointer_image(image<pixel_type,dim,rhs_storage_type>& rhs):base_type(&*rhs.begin(),rhs.geometry()) {}
    pointer_image(pixel_type* pointer,const tipl::geometry<dim>& geo_):base_type(pointer,geo_) {}
public:
    const pointer_image& operator=(const pointer_image& rhs)
    {
        base_type::data = rhs.data;
        base_type::geo = rhs.geometry();
        return *this;
    }
};

template <class pixel_type,unsigned int dim>
class const_pointer_image : public image<pixel_type,dim,const_pointer_container<pixel_type> >
{
public:
    typedef pixel_type value_type;
    typedef image<pixel_type,dim,const_pointer_container<pixel_type> > base_type;
    typedef typename base_type::iterator iterator;
    typedef typename base_type::const_iterator const_iterator;
    static const unsigned int dimension = dim;
public:
    const_pointer_image(void) {}
    const_pointer_image(const const_pointer_image& rhs):base_type()
    {
        base_type::data = rhs.data;
        base_type::geo = rhs.geometry();
    }
    template<class rhs_storage_type>
    const_pointer_image(const image<pixel_type,dim,rhs_storage_type>& rhs):base_type(&*rhs.begin(),rhs.geometry()) {}
    const_pointer_image(const pixel_type* pointer,const tipl::geometry<dim>& geo_):base_type(pointer,geo_){}
};


template<class value_type,class geometry_type>
pointer_image<value_type,geometry_type::dimension>
    make_image(value_type* pointer,const geometry_type& geo)
{
    return pointer_image<value_type,geometry_type::dimension>(pointer,geo);
}

template<class value_type,class geometry_type>
const_pointer_image<value_type,geometry_type::dimension>
    make_image(const value_type* pointer,const geometry_type& geo)
{
    return const_pointer_image<value_type,geometry_type::dimension>(pointer,geo);
}

}
#endif
