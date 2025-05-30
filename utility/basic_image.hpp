//---------------------------------------------------------------------------
#ifndef basic_imageH
#define basic_imageH
#include <vector>
#include "../def.hpp"
#include "shape.hpp"

//---------------------------------------------------------------------------
namespace tipl
{


template<typename vtype>
class pointer_container
{
public:
    using value_type        = vtype;
    using iterator          = vtype*;
    using const_iterator    = const vtype*;
    using reference         = vtype&;
    using const_reference   = const vtype&;
protected:
    iterator bg = nullptr;
    size_t sz = 0;
public:
    pointer_container(void){}
    template<typename any_iterator_type>
    pointer_container(any_iterator_type bg_,any_iterator_type ed_):
        bg(bg_),sz(ed_-bg_){}
    template<typename any_container_type>
    pointer_container(any_container_type& rhs){operator=(rhs);}
    template<typename any_container_type>
    pointer_container& operator=(any_container_type& rhs)
    {
        sz = rhs.size();
        if (sz)
            bg = rhs.data();
        return *this;
    }
public:
    pointer_container(const pointer_container& rhs):bg(rhs.bg),sz(rhs.sz){}
    pointer_container& operator=(const pointer_container& rhs)
    {
        if (this == &rhs)
            return *this;
        bg = rhs.bg;
        sz = rhs.sz;
        return *this;
    }
public:
    template<typename index_type>
    __INLINE__ const_reference operator[](index_type index) const    {return bg[index];}
    __INLINE__ const_iterator begin(void)                   const    {return bg;}
    const_iterator end(void)                     const    {return bg+sz;}
    const_iterator data(void)                     const    {return bg;}
public:
    template<typename index_type>
    __INLINE__ reference operator[](index_type index)                {return bg[index];}
    __INLINE__ iterator begin(void)                                  {return bg;}
    iterator end(void)                                    {return bg+sz;}
    iterator data(void)                                    {return bg;}
public:
    __INLINE__ size_t size(void)                            const    {return sz;}
    bool empty(void)                             const    {return sz == 0;}
    void clear(void)                                      {bg = nullptr;sz = 0;}
public:
    void swap(pointer_container& rhs)
    {
        iterator temp_bg = rhs.bg;
        size_t temp_sz = rhs.sz;
        rhs.bg = bg;
        rhs.sz = sz;
        bg = temp_bg;
        sz = temp_sz;
    }
    void resize(size_t new_size)                          {sz = new_size;}
};



template<typename vtype>
class const_pointer_container
{
public:
    using value_type        = vtype;
    using iterator          = const vtype*;
    using const_iterator    = const vtype*;
    using reference         = const vtype&;
    using const_reference   = const vtype&;
protected:
    iterator bg = nullptr;
    size_t sz = 0;
public:
    const_pointer_container(void){}
    template<typename any_iterator_type>
    const_pointer_container(any_iterator_type bg_,any_iterator_type ed_):
        bg(bg_),sz(ed_-bg_){}
public:
    template<typename any_container_type>
    const_pointer_container(const any_container_type& rhs){operator=(rhs);}
    template<typename any_container_type>
    const_pointer_container& operator=(const any_container_type& rhs)
    {
        sz = rhs.size();
        if (sz)
            bg = rhs.data();
        return *this;
    }
public:
    const_pointer_container(const const_pointer_container& rhs):bg(rhs.bg),sz(rhs.sz){}
    const_pointer_container& operator=(const const_pointer_container& rhs)
    {
        if (this == &rhs)
            return *this;
        bg = rhs.bg;
        sz = rhs.sz;
        return *this;
    }
public:
    const_pointer_container(const pointer_container<value_type>& rhs):bg(rhs.begin()),sz(rhs.sz){}
    const_pointer_container& operator=(const pointer_container<value_type>& rhs)
    {
        bg = rhs.begin();
        sz = rhs.sz;
        return *this;
    }
public:
    template<typename index_type>
    __INLINE__ const_reference operator[](index_type index) const    {return bg[index];}
    __INLINE__ const_iterator begin(void)                   const    {return bg;}
    __INLINE__ const_iterator end(void)                     const    {return bg+sz;}
    const_iterator data(void)                     const    {return bg;}
public:
    template<typename index_type>
    __INLINE__ reference operator[](index_type index)                {return bg[index];}
    iterator begin(void)                                  {return bg;}
    iterator end(void)                                    {return bg+sz;}
    iterator data(void)                                    {return bg;}
public:
    __INLINE__ size_t size(void)                            const    {return sz;}
    bool empty(void)                             const    {return sz == 0;}
public:
    void swap(const_pointer_container& rhs)
    {
        iterator temp_bg = rhs.bg;
        size_t temp_sz = rhs.sz;
        rhs.bg = bg;
        rhs.sz = sz;
        bg = temp_bg;
        sz = temp_sz;
    }
};

template<typename vtype>
class buffer_container
{
public:
    using value_type        = vtype;
    using iterator          = vtype*;
    using const_iterator    = const vtype*;
    using reference         = vtype&;
    using const_reference   = const vtype&;
protected:
    std::vector<unsigned char> buffer;
    iterator beg = nullptr;
    size_t sz = 0;
    void update_beg(void)
    {
        beg = (buffer.empty() ? nullptr : reinterpret_cast<iterator>(buffer.data()));
        sz = buffer.size()/sizeof(value_type);
    }
public:
    buffer_container(void){}
    buffer_container(size_t new_size)
    {
        resize(new_size);
    }
    template<typename any_iterator_type>
    buffer_container(any_iterator_type from,any_iterator_type to)
    {
        resize(to-from);
        if(beg)
            std::copy(from,to,beg);
    }
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    buffer_container(T& rhs){operator=(rhs);}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    buffer_container(T&& rhs){operator=(std::move(rhs));}

public:
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    buffer_container& operator=(T& rhs)
    {
        resize(rhs.size());
        if(beg)
            std::copy(rhs.begin(),rhs.end(),beg);
        return *this;
    }
    buffer_container& operator=(const buffer_container& rhs)
    {
        buffer = rhs.buffer;
        update_beg();
        return *this;
    }
    buffer_container& operator=(buffer_container&& rhs)
    {
        buffer.swap(rhs.buffer);
        update_beg();
        return *this;
    }
    buffer_container& operator=(std::vector<unsigned char>&& buffer_)
    {
        buffer.swap(buffer_);
        update_beg();
        return *this;
    }
public:
    template<typename index_type>
    __INLINE__ const_reference operator[](index_type index) const    {return beg[index];}
    __INLINE__ const_iterator begin(void)                   const    {return beg;}
    __INLINE__ const_iterator end(void)                     const    {return beg+sz;}
    const_iterator data(void)                     const   {return beg;}
public:
    template<typename index_type>
    __INLINE__ reference operator[](index_type index)                {return beg[index];}
    __INLINE__ iterator begin(void)                                  {return beg;}
    __INLINE__ iterator end(void)                                    {return beg+sz;}
    iterator data(void)                                   {return beg;}
public:
    auto& buf(void) {return buffer;}
public:
    __INLINE__ size_t size(void)                            const    {return sz;}
    bool empty(void)                             const    {return buffer.empty();}
    void clear(void)                                      {buffer.clear();beg = nullptr;sz = 0;}
public:
    void swap(buffer_container& rhs)
    {
        buffer.swap(rhs.buffer);
        update_beg();
    }
    void swap(std::vector<unsigned char>& buffer_)
    {
        buffer.swap(buffer_);
        update_beg();
    }
    void resize(size_t new_size)
    {
        buffer.resize(new_size*sizeof(value_type));
        update_beg();
    }
};

template <int dim,typename vtype = float,template <typename...> typename stype = std::vector>
class image
{
public:
    using value_type        = vtype;
    using storage_type      = stype<vtype>;
    using iterator          = typename storage_type::iterator;
    using const_iterator    = typename storage_type::const_iterator ;
    using reference         = typename storage_type::reference ;
    using slice_type        = tipl::image<dim-1,value_type,pointer_container> ;
    using const_slice_type  = tipl::image<dim-1,value_type,const_pointer_container>;
    using shape_type        = tipl::shape<dim>;
    using buffer_type       = image<dim,vtype,stype>;
    static constexpr int dimension = dim;
protected:
    storage_type alloc;
    shape_type sp;
public:
    __INLINE__ const shape_type& shape(void)    const   {return sp;}
    __INLINE__ int width(void)                  const   {return sp.width();}
    __INLINE__ int height(void)                 const   {return sp.height();}
    __INLINE__ int depth(void)                  const   {return sp.depth();}
    __INLINE__ size_t plane_size(void)          const   {return sp.plane_size();}
public:
    value_type at(unsigned int x,unsigned int y) const   {return alloc[size_t(y)*sp[0]+x];}
    reference at(unsigned int x,unsigned int y)          {return alloc[size_t(y)*sp[0]+x];}
    value_type at(unsigned int x,unsigned int y,unsigned int z) const    {return alloc[size_t(z*sp[1]+y)*sp[0]+x];}
    reference at(unsigned int x,unsigned int y,unsigned int z)           {return alloc[size_t(z*sp[1]+y)*sp[0]+x];}
    template<typename T,typename std::enable_if<T::dimension==2,bool>::type = true>
    value_type at(const T& pos) const    {return alloc[size_t(pos[1])*sp[0]+size_t(pos[0])];}
    template<typename T,typename std::enable_if<T::dimension==2,bool>::type = true>
    reference at(const T& pos)           {return alloc[size_t(pos[1])*sp[0]+size_t(pos[0])];}
    template<typename T,typename std::enable_if<T::dimension==3,bool>::type = true>
    value_type at(const T& pos) const    {return alloc[size_t(size_t(pos[2])*sp[1]+size_t(pos[1]))*sp[0]+size_t(pos[0])];}
    template<typename T,typename std::enable_if<T::dimension==3,bool>::type = true>
    reference at(const T& pos)           {return alloc[size_t(size_t(pos[2])*sp[1]+size_t(pos[1]))*sp[0]+size_t(pos[0])];}
public:
    image(void) {}
    template<typename T,typename std::enable_if<T::dimension==dimension && !std::is_same<storage_type,typename T::storage_type>::value,bool>::type = true>
    image(const T& rhs)              {operator=(rhs);}
    image(const image& rhs)          {operator=(rhs);}
    image(image&& rhs) noexcept      {operator=(std::move(rhs));}
public:
    template<typename T>
    image(std::initializer_list<T> rhs):sp(rhs)      {alloc.resize(sp.size());}
    image(const shape_type& sp_):alloc(sp_.size()),sp(sp_){}
    image(const shape_type& sp_,value_type v):alloc(sp_.size(),v),sp(sp_){}
public:
    template<typename T>
    image(T* pointer,const shape_type& sp_):alloc(pointer,pointer+sp_.size()),sp(sp_)         {}
    template<typename T>
    image(const T* pointer,const shape_type& sp_):alloc(pointer,pointer+sp_.size()),sp(sp_)   {}
public:
    template<typename T,typename std::enable_if<T::dimension==dimension && !std::is_same<storage_type,typename T::storage_type>::value,bool>::type = true>
    image& operator=(const T& rhs)
    {
        using U = typename T::const_iterator;
        // Casting is needed here, because if rhs is a device vector, rhs.begin() is a void * and this
        // will later lead to a reference to void
        storage_type new_alloc(static_cast<U>(rhs.begin()),static_cast<U>(rhs.end()));
        alloc.swap(new_alloc);
        sp = rhs.shape();
        return *this;
    }
    image& operator=(const image& rhs)
    {
        alloc = rhs.alloc;
        sp = rhs.shape();
        return *this;
    }
    image& operator=(image&& rhs) noexcept
    {
        alloc.swap(rhs.alloc);
        sp.swap(rhs.sp);
        return *this;
    }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    image& operator=(T value)
    {
        iterator end_iter = alloc.end();
        for(iterator iter = alloc.begin();iter != end_iter;++iter)
            *iter = value;
        return *this;
    }
public:
    auto& buf(void){return alloc;}
    const auto& buf(void)const{return alloc;}
    auto data(void){return alloc.data();}
    auto data(void)const{return alloc.data();}
    void swap(image& rhs)
    {
        alloc.swap(rhs.alloc);
        sp.swap(rhs.sp);
    }
    void resize(const shape_type& sp_)
    {
        sp = sp_;
        alloc.resize(sp.size());
    }
    void clear(void)
    {
        alloc.clear();
        sp.clear();
    }
    __INLINE__ size_t size(void)    const    {return alloc.size();}
    bool empty(void)     const   {return alloc.empty();}

    template<typename index_type,typename std::enable_if<std::is_integral<index_type>::value,bool>::type = true>
    __INLINE__ const value_type& operator[](index_type index)   const   {return alloc[index];}
    __INLINE__ auto begin(void)                    const   {return alloc.begin();}
    __INLINE__ auto end(void)                      const   {return alloc.end();}

    template<typename index_type,typename std::enable_if<std::is_integral<index_type>::value,bool>::type = true>
    __INLINE__ reference operator[](index_type index)           {return alloc[index];}
    __INLINE__ auto begin(void)                            {return alloc.begin();}
    __INLINE__ auto end(void)                              {return alloc.end();}
public:
    auto slice_at(unsigned int pos)
    {
        tipl::shape<dim-1> slice_sp(sp.begin());
        if constexpr(std::is_same<storage_type,const_pointer_container<vtype> >::value)
            return const_slice_type(alloc.data()+pos*slice_sp.size(),slice_sp);
        else
            return slice_type(alloc.data()+pos*slice_sp.size(),slice_sp);
    }
    auto slice_at(unsigned int pos) const
    {
        tipl::shape<dim-1> slice_sp(sp.begin());
        return const_slice_type(alloc.data()+pos*slice_sp.size(),slice_sp);
    }
    template<typename shape_type>
    auto alias(size_t offset,const shape_type& new_shape)
    {
        if constexpr(std::is_same<storage_type,const_pointer_container<vtype> >::value)
            return tipl::image<shape_type::dimension,value_type,const_pointer_container>(alloc.data()+offset,new_shape);
        else
            return tipl::image<shape_type::dimension,value_type,pointer_container>(alloc.data()+offset,new_shape);
    }
    template<typename shape_type>
    auto alias(size_t offset,const shape_type& new_shape) const
    {
        return tipl::image<shape_type::dimension,value_type,const_pointer_container>(alloc.data()+offset,new_shape);
    }
    auto alias(void)
    {
        if constexpr(std::is_same<storage_type,const_pointer_container<vtype> >::value)
            return tipl::image<dim,value_type,const_pointer_container>(alloc.data(),sp);
        else
            return tipl::image<dim,value_type,pointer_container>(alloc.data(),sp);
    }
    auto alias(void) const
    {
        return tipl::image<dim,value_type,const_pointer_container>(alloc.data(),sp);
    }
public:
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    auto operator+=(T value)
    {auto ed = end();for(auto bg = begin();bg != ed;++bg)*bg += value;
        return *this;}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    auto operator-=(T value)
    {auto ed = end();for(auto bg = begin();bg != ed;++bg)*bg -= value;
        return *this;}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    auto operator*=(T value)
    {auto ed = end();for(auto bg = begin();bg != ed;++bg)*bg *= value;
        return *this;}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    auto operator/=(T value)
    {auto ed = end();for(auto bg = begin();bg != ed;++bg)*bg /= value;
        return *this;}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    auto operator+=(const T& rhs)
    {auto ed = end();auto bg2 = rhs.begin();
     for(auto bg = begin();bg != ed;++bg,++bg2)*bg += *bg2;
     return *this;}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    auto operator-=(const T& rhs)
    {auto ed = end();auto bg2 = rhs.begin();
     for(auto bg = begin();bg != ed;++bg,++bg2)*bg -= *bg2;
     return *this;}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    auto operator*=(const T& rhs)
    {auto ed = end();auto bg2 = rhs.begin();
     for(auto bg = begin();bg != ed;++bg,++bg2)*bg *= *bg2;
     return *this;}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    auto operator/=(const T& rhs)
    {auto ed = end();auto bg2 = rhs.begin();
     for(auto bg = begin();bg != ed;++bg,++bg2)*bg /= *bg2;
     return *this;}
public:
    template<typename format_type>
    bool save_to_file(const std::string& file_name) const
    {
        format_type out;
        out.load_from_image(*this);
        return out.save_to_file(file_name);
    }
    template<typename format_type,typename voxel_type>
    bool save_to_file(const std::string& file_name,voxel_type vs) const
    {
        format_type out;
        out.load_from_image(*this);
        out.set_voxel_size(vs);
        return out.save_to_file(file_name);
    }
    template<typename format_type>
    bool load_from_file(const std::string& file_name)
    {
        format_type out;
        if(!out.load_from_file(file_name))
            return false;
        out.save_to_image(*this);
        return true;
    }
};

template<typename V,typename T>
inline V extract_pointer(T* p){return V(p);}
template<typename V,typename T>
inline V extract_pointer(T p){return V(&*p);}

template<int dim,typename vtype = float>
class pointer_image : public image<dim,vtype,pointer_container>
{
public:
    using value_type        = vtype;
    using base_type         = image<dim,vtype,pointer_container>;
    using iterator          = typename base_type::iterator;
    using const_iterator    = typename base_type::const_iterator;
    using storage_type      = typename image<dim,vtype,pointer_container>::storage_type;
    using buffer_type       = image<dim,vtype>;
    static const int dimension = dim;
public:
    pointer_image(void) {}
    pointer_image(const pointer_image& rhs):base_type(){operator=(rhs);}
    template<typename T>
    pointer_image(T& rhs):base_type(extract_pointer<vtype*>(rhs.begin()),rhs.shape()) {}
    pointer_image(vtype* pointer,const tipl::shape<dim>& sp_):base_type(pointer,sp_) {}
public:
    pointer_image& operator=(const pointer_image& rhs)
    {
        base_type::alloc = rhs.alloc;
        base_type::sp = rhs.sp;
        return *this;
    }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    pointer_image& operator=(T value)
    {
        base_type::operator=(value);
        return *this;
    }
};

template<typename V,typename T>
inline V extract_const_pointer(const T* p){return V(p);}
template<typename V,typename T>
inline V extract_const_pointer(T p){return V(&*p);}

template<int dim,typename vtype = float>
class const_pointer_image : public image<dim,vtype,const_pointer_container>
{
public:
    using value_type        =   vtype;
    using base_type         =   image<dim,value_type,const_pointer_container>;
    using iterator          =   typename base_type::iterator;
    using const_iterator    =   typename base_type::const_iterator;
    using storage_type      =   typename image<dim,vtype,const_pointer_container>::storage_type;
    using buffer_type       =   image<dim,vtype>;
    static const int dimension = dim;
public:
    const_pointer_image(void) {}
    const_pointer_image(const const_pointer_image& rhs):base_type(){operator=(rhs);}
    template<typename T,typename std::enable_if<T::dimension==dimension && !std::is_same<storage_type,typename T::storage_type>::value,bool>::type = true>
    const_pointer_image(const T& rhs):base_type(extract_const_pointer<const vtype*>(rhs.begin()),rhs.shape()) {}
    const_pointer_image(const vtype* pointer,const tipl::shape<dim>& sp_):base_type(pointer,sp_){}
public:
    const_pointer_image& operator=(const const_pointer_image& rhs)
    {
        base_type::alloc = rhs.alloc;
        base_type::sp = rhs.sp;
        return *this;
    }

};


template<typename T>
inline auto make_shared(T& I)
{
    return pointer_image<T::dimension,typename T::value_type>(I);
}
template<typename T>
inline auto make_shared(const T& I)
{
    return const_pointer_image<T::dimension,typename T::value_type>(I);
}


template<typename value_type,typename shape_type>
inline auto make_image(value_type* pointer,const shape_type& sp)
{
    return pointer_image<shape_type::dimension,value_type>(pointer,sp);
}

template<typename value_type,typename shape_type>
inline auto make_image(const value_type* pointer,const shape_type& sp)
{
    return const_pointer_image<shape_type::dimension,value_type>(pointer,sp);
}

}
#endif
