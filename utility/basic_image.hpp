//---------------------------------------------------------------------------
#ifndef basic_imageH
#define basic_imageH
#include <vector>
#include <thread>
#include <future>
#include "shape.hpp"
#include "pixel_value.hpp"
#include "pixel_index.hpp"

//---------------------------------------------------------------------------
namespace tipl
{

template<typename vtype>
class pointer_container
{
public:
    using value_type        = vtype;
    using iterator          = vtype*;
    using const_iterator    = vtype*;
    using reference         = vtype&;
protected:
    iterator from,to;
    size_t size_;
public:
    pointer_container(void):from(0),to(0),size_(0){}
    pointer_container(size_t size_):from(0),to(0),size_(size_){}
    template<typename any_iterator_type>
    pointer_container(any_iterator_type from_,any_iterator_type to_):
        from(&*from_),to(0),size_(to_-from_){to = from + size_;}
    template<typename any_container_type>
    pointer_container(const any_container_type& rhs){operator=(rhs);}
    pointer_container(std::vector<value_type>& rhs)
    {
        if((size_ = rhs.size()))
        {
            from = &rhs[0];
            to = from + size_;
        }
        else
            from = to = 0;
    }
public:
    template<typename any_container_type>
    pointer_container& operator=(const any_container_type& rhs)
    {
        if (this == &rhs)
            return *this;
        from = rhs.from;
        to = rhs.to;
        size_ = rhs.size_;
    }
public:
    template<typename index_type>
    reference operator[](index_type index) const
    {
        return from[index];
    }
    template<typename index_type>
    reference operator[](index_type index)
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

template<typename vtype>
class const_pointer_container
{
public:
    using value_type        = vtype;
    using iterator          = const vtype*;
    using const_iterator    = const vtype*;
    using reference         = const vtype&;
protected:
    const_iterator from,to;
    size_t size_;
public:
    const_pointer_container(void):from(0),to(0),size_(0){}
    const_pointer_container(size_t size_):from(0),to(0),size_(size_){}
    template<typename any_iterator_type>
    const_pointer_container(any_iterator_type from_,any_iterator_type to_):
        from(&*from_),to(0),size_(to_-from_){to = from + size_;}
    template<typename any_container_type>
    const_pointer_container(const any_container_type& rhs){operator=(rhs);}
    const_pointer_container(const std::vector<value_type>& rhs)
    {
        if((size_ = rhs.size()))
        {
            from = &rhs[0];
            to = from + size_;
        }
        else
            from = to = 0;
    }
public:
    template<typename any_container_type>
    const_pointer_container& operator=(const any_container_type& rhs)
    {
        if (this == &rhs)
            return *this;
        from = rhs.begin();
        to = rhs.end();
        size_ = rhs.size();
        return *this;
    }
public:
    template<typename index_type>
    reference operator[](index_type index) const
    {
        return from[index];
    }
    template<typename index_type>
    reference operator[](index_type index)
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
    void swap(const_pointer_container<value_type>& rhs)
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


template <int dim,typename vtype = float,typename storage_type = std::vector<vtype> >
class image
{
public:
    using value_type        = vtype;
    using iterator          = typename storage_type::iterator;
    using const_iterator    = typename storage_type::const_iterator ;
    using reference         = typename storage_type::reference ;
    using slice_type        = tipl::image<dim-1,value_type,pointer_container<value_type> > ;
    using const_slice_type  = tipl::image<dim-1,value_type,const_pointer_container<value_type> >;
    using shape_type        = tipl::shape<dim>;
    static const int dimension = dim;
protected:
    storage_type data;
    shape_type geo;
public:
    const shape_type& shape(void) const
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
    value_type at(unsigned int x,unsigned int y) const
    {
        return data[y*geo[0]+x];
    }
    reference at(unsigned int x,unsigned int y)
    {
        return data[y*geo[0]+x];
    }

    value_type at(unsigned int x,unsigned int y,unsigned int z) const
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
    template<typename T>
    image(const std::initializer_list<T>& rhs):geo(rhs){data.resize(geo.size());}
    template<typename rhs_value_type,typename rhs_storage_type>
    image(const image<dim,rhs_value_type,rhs_storage_type>& rhs){operator=(rhs);}
    image(const shape_type& geo_):data(geo_.size()),geo(geo_) {}
    template <typename any_value_type>
    image(any_value_type* pointer,const shape_type& geo_):data(pointer,pointer+geo_.size()),geo(geo_) {}
    template <typename any_value_type>
    image(const any_value_type* pointer,const shape_type& geo_):data(pointer,pointer+geo_.size()),geo(geo_) {}
public:
    template<typename rhs_value_type,typename rhs_storage_type>
    const image& operator=(const image<dim,rhs_value_type,rhs_storage_type>& rhs)
    {
        storage_type new_data(rhs.begin(),rhs.end());
        data.swap(new_data);
        geo = rhs.shape();
        return *this;
    }
    const image& operator=(const image& rhs)
    {
        storage_type new_data(rhs.begin(),rhs.end());
        data.swap(new_data);
        geo = rhs.shape();
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
    void resize(const shape_type& geo_)
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
    value_type front(void) const
    {
        return data.front();
    }
    value_type back(void) const
    {
        return data.back();
    }
    template<typename index_type>
    const value_type& operator[](index_type index) const
    {
        return data[index];
    }
    template<typename index_type>
    reference operator[](index_type index)
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
        tipl::shape<dim-1> slice_geo(geo.begin());
        return slice_type(&*data.begin()+pos*slice_geo.size(),slice_geo);
    }
    const_slice_type slice_at(unsigned int pos) const
    {
        tipl::shape<dim-1> slice_geo(geo.begin());
        return const_slice_type(&*data.begin()+pos*slice_geo.size(),slice_geo);
    }
public:
    template<typename value_type>
    const image& operator+=(value_type value)
    {
        iterator end_iter = data.end();
        for(iterator iter = data.begin();iter != end_iter;++iter)
            *iter += value;
        return *this;
    }
    template<typename value_type>
    const image& operator-=(value_type value)
    {
        iterator end_iter = data.end();
        for(iterator iter = data.begin();iter != end_iter;++iter)
            *iter -= value;
        return *this;
    }
    template<typename value_type>
    const image& operator*=(value_type value)
    {
        iterator end_iter = data.end();
        for(iterator iter = data.begin();iter != end_iter;++iter)
            *iter *= value;
        return *this;
    }
    template<typename value_type>
    const image& operator/=(value_type value)
    {
        iterator end_iter = data.end();
        for(iterator iter = data.begin();iter != end_iter;++iter)
            *iter /= value;
        return *this;
    }
    template<typename rhs_value_type,typename rhs_storage_type>
    const image& operator+=(const image<dim,rhs_value_type,rhs_storage_type>& rhs)
    {
        iterator end_iter = data.end();
        typename image<dim,rhs_value_type,rhs_storage_type>::const_iterator iter2 = rhs.begin();
        for(iterator iter = data.begin();iter != end_iter;++iter,++iter2)
            *iter += *iter2;
        return *this;
    }
    template<typename rhs_value_type,typename rhs_storage_type>
    const image& operator-=(const image<dim,rhs_value_type,rhs_storage_type>& rhs)
    {
        iterator end_iter = data.end();
        typename image<dim,rhs_value_type,rhs_storage_type>::const_iterator iter2 = rhs.begin();
        for(iterator iter = data.begin();iter != end_iter;++iter,++iter2)
            *iter -= *iter2;
        return *this;
    }
    template<typename rhs_value_type,typename rhs_storage_type>
    const image& operator*=(const image<dim,rhs_value_type,rhs_storage_type>& rhs)
    {
        iterator end_iter = data.end();
        typename image<dim,rhs_value_type,rhs_storage_type>::const_iterator iter2 = rhs.begin();
        for(iterator iter = data.begin();iter != end_iter;++iter,++iter2)
            *iter *= *iter2;
        return *this;
    }
    template<typename rhs_value_type,typename rhs_storage_type>
    const image& operator/=(const image<dim,rhs_value_type,rhs_storage_type>& rhs)
    {
        iterator end_iter = data.end();
        typename image<dim,rhs_value_type,rhs_storage_type>::const_iterator iter2 = rhs.begin();
        for(iterator iter = data.begin();iter != end_iter;++iter,++iter2)
            *iter /= *iter2;
        return *this;
    }
public:
    template<typename format_type>
    bool save_to_file(const char* file_name) const
    {
        format_type out;
        out.load_from_image(*this);
        return out.save_to_file(file_name);
    }
    template<typename format_type,typename voxel_type>
    bool save_to_file(const char* file_name,voxel_type vs) const
    {
        format_type out;
        out.load_from_image(*this);
        out.set_voxel_size(vs);
        return out.save_to_file(file_name);
    }
    template<typename format_type>
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
        for(pixel_index<dim> index(shape());index.index() < data.size();++index)
            f(data[index.index()],index);
    }
    template<typename Func>
    void for_each(Func f) const
    {
        for(pixel_index<dim> index(shape());index.index() < data.size();++index)
            f(data[index.index()],index);
    }
    template<typename Func>
    void for_each_mt(Func f, unsigned int thread_count = std::thread::hardware_concurrency())
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
                for(pixel_index<dim> index(pos,shape());index.index() < end;++index)
                    f(data[index.index()],index);
            })));
            pos = end;
        }
        for(pixel_index<dim> index(pos,shape());index.index() < data.size();++index)
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
                for(pixel_index<dim> index(pos,shape());index.index() < end;++index)
                    f(data[index.index()],index);
            })));
            pos = end;
        }
        for(pixel_index<dim> index(pos,shape());index.index() < data.size();++index)
            f(data[index.index()],index);
        for(auto &future : futures)
            future.wait();
    }
    template<typename Func>
    void for_each_mt2(Func f,unsigned int thread_count = std::thread::hardware_concurrency())
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
                for(pixel_index<dim> index(pos,shape());index.index() < end;++index)
                    f(data[index.index()],index,id);
            })));
            pos = end;
        }
        for(pixel_index<dim> index(pos,shape());index.index() < data.size();++index)
            f(data[index.index()],index,0);
        for(auto &future : futures)
            future.wait();
    }
};


typedef image<2,rgb> color_image;
typedef image<2,unsigned char> grayscale_image;

template<int dim,typename vtype = float>
class pointer_image : public image<dim,vtype,pointer_container<vtype> >
{
public:
    using base_type         =   image<dim,vtype,pointer_container<vtype> >;
    using iterator          =   typename base_type::iterator;
    using const_iterator    = typename base_type::iterator;
    static const int dimension = dim;
public:
    pointer_image(void) {}
    pointer_image(const pointer_image& rhs):base_type(){operator=(rhs);}
    template<typename rhs_storage_type>
    pointer_image(image<dim,vtype,rhs_storage_type>& rhs):base_type(&*rhs.begin(),rhs.shape()) {}
    pointer_image(vtype* pointer,const tipl::shape<dim>& geo_):base_type(pointer,geo_) {}
public:
    pointer_image& operator=(const pointer_image& rhs)
    {
        base_type::data = rhs.data;
        base_type::geo = rhs.shape();
        return *this;
    }
};

template<int dim,typename vtype = float>
class const_pointer_image : public image<dim,vtype,const_pointer_container<vtype> >
{
public:
    using value_type        =   vtype;
    using base_type         =   image<dim,value_type,const_pointer_container<value_type> >;
    using iterator          =   typename base_type::iterator;
    using const_iterator    =   typename base_type::const_iterator;
    static const int dimension = dim;
public:
    const_pointer_image(void) {}
    const_pointer_image(const const_pointer_image& rhs):base_type(){operator=(rhs);}
    template<typename rhs_storage_type>
    const_pointer_image(const image<dim,vtype,rhs_storage_type>& rhs):base_type(&*rhs.begin(),rhs.shape()) {}
    const_pointer_image(const vtype* pointer,const tipl::shape<dim>& geo_):base_type(pointer,geo_){}
public:
    const_pointer_image& operator=(const const_pointer_image& rhs)
    {
        base_type::data = rhs.data;
        base_type::geo = rhs.shape();
        return *this;
    }

};


template<typename value_type,typename shape_type>
pointer_image<shape_type::dimension,value_type>
    make_image(value_type* pointer,const shape_type& geo)
{
    return pointer_image<shape_type::dimension,value_type>(pointer,geo);
}

template<typename value_type,typename shape_type>
const_pointer_image<shape_type::dimension,value_type>
    make_image(const value_type* pointer,const shape_type& geo)
{
    return const_pointer_image<shape_type::dimension,value_type>(pointer,geo);
}

}
#endif
