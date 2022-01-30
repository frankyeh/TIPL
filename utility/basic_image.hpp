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
    iterator from = nullptr;
    size_t size_ = 0;
public:
    __INLINE__ pointer_container(void){}
    template<typename any_iterator_type>
    __INLINE__ pointer_container(any_iterator_type from_,any_iterator_type to_):
        from(from_),size_(to_-from_){}
    template<typename any_container_type>
    __INLINE__ pointer_container(any_container_type& rhs){operator=(rhs);}
    template<typename any_container_type>
    __INLINE__ pointer_container& operator=(any_container_type& rhs)
    {
        size_ = rhs.size();
        if (size_)
            from = &rhs[0];
        return *this;
    }
public:
    __INLINE__ pointer_container(const pointer_container& rhs):from(rhs.from),size_(rhs.size_){}
    __INLINE__ pointer_container& operator=(const pointer_container& rhs)
    {
        if (this == &rhs)
            return *this;
        from = rhs.from;
        size_ = rhs.size_;
        return *this;
    }
public:
    template<typename index_type>
    __INLINE__ const_reference operator[](index_type index) const    {return from[index];}
    __INLINE__ const_iterator begin(void)                   const    {return from;}
    __INLINE__ const_iterator end(void)                     const    {return from+size_;}
    __INLINE__ const_iterator get(void)                     const    {return from;}
public:
    template<typename index_type>
    __INLINE__ reference operator[](index_type index)                {return from[index];}
    __INLINE__ iterator begin(void)                                  {return from;}
    __INLINE__ iterator end(void)                                    {return from+size_;}
    __INLINE__ iterator get(void)                                    {return from;}
public:
    __INLINE__ size_t size(void)                            const    {return size_;}
    __INLINE__ bool empty(void)                             const    {return size_ == 0;}
    __INLINE__ void clear(void)                                      {size_ = 0;}
public:
    __INLINE__ void swap(pointer_container& rhs)                     {std::swap(from,rhs.from);std::swap(size_,rhs.size_);}
    __INLINE__ void resize(size_t new_size)                          {size_ = new_size;}
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
    iterator from = nullptr;
    size_t size_ = 0;
public:
    __INLINE__ const_pointer_container(void){}
    template<typename any_iterator_type>
    __INLINE__ const_pointer_container(any_iterator_type from_,any_iterator_type to_):
        from(from_),size_(to_-from_){}
public:
    template<typename any_container_type>
    __INLINE__ const_pointer_container(const any_container_type& rhs){operator=(rhs);}
    template<typename any_container_type>
    __INLINE__ const_pointer_container& operator=(const any_container_type& rhs)
    {
        size_ = rhs.size();
        if (size_)
            from = &rhs[0];
        return *this;
    }
public:
    __INLINE__ const_pointer_container(const const_pointer_container& rhs):from(rhs.from),size_(rhs.size_){}
    __INLINE__ const_pointer_container& operator=(const const_pointer_container& rhs)
    {
        if (this == &rhs)
            return *this;
        from = rhs.from;
        size_ = rhs.size_;
        return *this;
    }
public:
    __INLINE__ const_pointer_container(const pointer_container<value_type>& rhs):from(rhs.begin()),size_(rhs.size_){}
    __INLINE__ const_pointer_container& operator=(const pointer_container<value_type>& rhs)
    {
        from = rhs.begin();
        size_ = rhs.size_;
        return *this;
    }
public:
    template<typename index_type>
    __INLINE__ const_reference operator[](index_type index) const    {return from[index];}
    __INLINE__ const_iterator begin(void)                   const    {return from;}
    __INLINE__ const_iterator end(void)                     const    {return from+size_;}
    __INLINE__ const_iterator get(void)                     const    {return from;}
public:
    template<typename index_type>
    __INLINE__ reference operator[](index_type index)                {return from[index];}
    __INLINE__ iterator begin(void)                                  {return from;}
    __INLINE__ iterator end(void)                                    {return from+size_;}
    __INLINE__ iterator get(void)                                    {return from;}
public:
    __INLINE__ size_t size(void)                            const    {return size_;}
    __INLINE__ bool empty(void)                             const    {return size_ == 0;}
public:
    __INLINE__ void swap(const_pointer_container& rhs)               {std::swap(from,rhs.from);std::swap(size_,rhs.size_);}
};

template<typename Fun>
class operation{
    size_t size;
    Fun f;
    bool done = false;
public:
    operation(size_t size_,Fun&& f_):size(size_),f(std::move(f_)) {}
public:
    template<typename RhsFun>
    __INLINE__ static auto make_operation(size_t size,RhsFun&& f)
    {return operation<RhsFun>(size,std::move(f));}
public:
    ~operation(void)
    {
        if(done)return;
        for(size_t i = 0;i < size;++i)
            f(i);
    }
    template<typename RhsFun>
    auto operator>>(operation<RhsFun>&& rhs)
    {
        done = true;
        rhs.done = true;
        return make_operation(size,[this,&rhs](size_t i){f(i);rhs.f(i);});
    }
    template<typename T>
    void operator>>(T&& backend)
    {
        backend(size,std::move(f));
        done = true;
    }
};
template<typename Fun>
__INLINE__ static auto make_operation(size_t size,Fun&& f)
{return operation<Fun>(size,std::move(f));}

template<typename EvaluationType,typename ConditionType>
struct selection{
    size_t size;
    EvaluationType v;
    ConditionType f;
    __INLINE__ selection(size_t size_,EvaluationType&& v_,ConditionType&& f_):
        size(size_),v(std::move(v_)),f(std::move(f_)){}
public:
    template<typename T,typename U>
    __INLINE__ static auto make_selection(size_t size,T&& v,U&& f)
    {return selection<T,U>(size,std::move(v),std::move(f));}
    template<typename RhsSelectionType,typename U>
    __INLINE__ auto combine_evaluation(RhsSelectionType& sel,U&& new_v)
    {return make_selection(size < sel.size? size:sel.size,std::move(new_v),
                [this,&sel](size_t i){return f(i) && sel.f(i);});}
public:
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator=(T rhs)
    {return make_operation(size,[this,rhs](size_t i){if(f(i))v(i)=rhs;});}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator+=(T rhs)
    {return make_operation(size,[this,rhs](size_t i){if(f(i))v(i)+=rhs;});}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator-=(T rhs)
    {return make_operation(size,[this,rhs](size_t i){if(f(i))v(i)-=rhs;});}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator*=(T rhs)
    {return make_operation(size,[this,rhs](size_t i){if(f(i))v(i)*=rhs;});}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator/=(T rhs)
    {return make_operation(size,[this,rhs](size_t i){if(f(i))v(i)/=rhs;});}
public:
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator+(const T& sel)
    {return combine_evaluation(sel,[this,&sel](size_t i){return v(i)+sel.v(i);});}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator-(const T& sel)
    {return combine_evaluation(sel,[this,&sel](size_t i){return v(i)-sel.v(i);});}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator*(const T& sel)
    {return combine_evaluation(sel,[this,&sel](size_t i){return v(i)*sel.v(i);});}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator/(const T& sel)
    {return combine_evaluation(sel,[this,&sel](size_t i){return v(i)/sel.v(i);});}
};
template<typename EvaluationType,typename ConditionType>
__INLINE__ static auto make_selection(size_t s,EvaluationType&& v,ConditionType&& f)
{return selection<EvaluationType,ConditionType>(s,std::move(v),std::move(f));}


template<typename Func>
struct condition{
    Func f;
    __INLINE__ condition(Func&& f_):f(f_){}
public:
    template<typename NewFunc>
    __INLINE__ static auto make_condition(NewFunc&& f)
    {return condition<NewFunc>(std::move(f));}
public:
    __INLINE__ bool operator()(size_t i) const{return f(i);}
public:
    template<typename T>
    __INLINE__ auto operator&&(const T& rhs) const
    {return make_condition([this,&rhs](size_t i){return f(i) && rhs(i);});}
    template<typename T>
    __INLINE__ auto operator||(const T& rhs) const
    {return make_condition([this,&rhs](size_t i){return f(i) || rhs(i);});}
};
template<typename NewFunc>
__INLINE__ static auto make_condition(NewFunc&& f)
{return condition<NewFunc>(std::move(f));}

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
    static constexpr int dimension = dim;
protected:
    storage_type data;
    shape_type geo;
public:
    __INLINE__ const shape_type& shape(void)    const   {return geo;}
    __INLINE__ int width(void)                  const   {return geo.width();}
    __INLINE__ int height(void)                 const   {return geo.height();}
    __INLINE__ int depth(void)                  const   {return geo.depth();}
    __INLINE__ size_t plane_size(void)          const   {return geo.plane_size();}
public:
    __INLINE__ value_type at(unsigned int x,unsigned int y) const   {return data[size_t(y)*geo[0]+x];}
    __INLINE__ reference at(unsigned int x,unsigned int y)          {return data[size_t(y)*geo[0]+x];}
    __INLINE__ value_type at(unsigned int x,unsigned int y,unsigned int z) const    {return data[size_t(z*geo[1]+y)*geo[0]+x];}
    __INLINE__ reference at(unsigned int x,unsigned int y,unsigned int z)           {return data[size_t(z*geo[1]+y)*geo[0]+x];}
public:
    __INLINE__ image(void) {}
    template<typename T,typename std::enable_if<T::dimension==dimension && !std::is_same<storage_type,typename T::storage_type>::value,bool>::type = true>
    __INLINE__ image(const T& rhs)      {operator=(rhs);}
    __INLINE__ image(const image& rhs)  {operator=(rhs);}
    __INLINE__ image(image&& rhs)       {operator=(rhs);}
public:
    template<typename T>
    __INLINE__ image(const std::initializer_list<T>& rhs):geo(rhs)      {data.resize(geo.size());}
    __INLINE__ image(const shape_type& geo_):data(geo_.size()),geo(geo_){}
public:
    template<typename T>
    __INLINE__ image(T* pointer,const shape_type& geo_):data(pointer,pointer+geo_.size()),geo(geo_)         {}
    template<typename T>
    __INLINE__ image(const T* pointer,const shape_type& geo_):data(pointer,pointer+geo_.size()),geo(geo_)   {}
public:
    template<typename T,typename std::enable_if<T::dimension==dimension && !std::is_same<storage_type,typename T::storage_type>::value,bool>::type = true>
    __INLINE__ image& operator=(const T& rhs)
    {
        storage_type new_data(rhs.begin(),rhs.end());
        data.swap(new_data);
        geo = rhs.shape();
        return *this;
    }
    __INLINE__ image& operator=(const image& rhs)
    {
        storage_type new_data(rhs.begin(),rhs.end());
        data.swap(new_data);
        geo = rhs.shape();
        return *this;
    }
    __INLINE__ image& operator=(image&& rhs)
    {
        data.swap(rhs.data);
        geo.swap(rhs.geo);
        return *this;
    }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ image& operator=(T value)
    {
        iterator end_iter = data.end();
        for(iterator iter = data.begin();iter != end_iter;++iter)
            *iter = value;
        return *this;
    }
public:
    __INLINE__ typename storage_type::iterator get(void){return data.get();}
    __INLINE__ typename storage_type::const_iterator get(void)const{return data.get();}
    __INLINE__ void swap(image& rhs)
    {
        data.swap(rhs.data);
        geo.swap(rhs.geo);
    }
    __INLINE__ void resize(const shape_type& geo_)
    {
        geo = geo_;
        data.resize(geo.size());
    }
    __INLINE__ void clear(void)
    {
        data.clear();
        geo.clear();
    }
    __INLINE__ size_t size(void)    const    {return data.size();}
    __INLINE__ bool empty(void)     const   {return data.empty();}

    template<typename index_type,typename std::enable_if<std::is_integral<index_type>::value,bool>::type = true>
    __INLINE__ const value_type& operator[](index_type index)   const   {return data[index];}
    __INLINE__ auto begin(void)                    const   {return data.begin();}
    __INLINE__ auto end(void)                      const   {return data.end();}

    template<typename index_type,typename std::enable_if<std::is_integral<index_type>::value,bool>::type = true>
    __INLINE__ reference operator[](index_type index)           {return data[index];}
    __INLINE__ auto begin(void)                            {return data.begin();}
    __INLINE__ auto end(void)                              {return data.end();}
public:
    __INLINE__ slice_type slice_at(unsigned int pos)
    {
        tipl::shape<dim-1> slice_geo(geo.begin());
        return slice_type(&*data.begin()+pos*slice_geo.size(),slice_geo);
    }
    __INLINE__ const_slice_type slice_at(unsigned int pos) const
    {
        tipl::shape<dim-1> slice_geo(geo.begin());
        return const_slice_type(&*data.begin()+pos*slice_geo.size(),slice_geo);
    }    
public:
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto  operator<(T rhs) const {return make_condition([this,rhs](size_t i){return data[i] < rhs;});}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto  operator>(T rhs) const {return make_condition([this,rhs](size_t i){return data[i] > rhs;});}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator<=(T rhs) const {return make_condition([this,rhs](size_t i){return data[i] <= rhs;});}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator>=(T rhs) const {return make_condition([this,rhs](size_t i){return data[i] >= rhs;});}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator==(T rhs) const {return make_condition([this,rhs](size_t i){return data[i] == rhs;});}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator!=(T rhs) const {return make_condition([this,rhs](size_t i){return data[i] != rhs;});}
public:
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto  operator<(const T& rhs) const {return make_condition([this,&rhs](size_t i){return data[i] < rhs[i];});}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto  operator>(const T& rhs) const {return make_condition([this,&rhs](size_t i){return data[i] > rhs[i];});}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator<=(const T& rhs) const {return make_condition([this,&rhs](size_t i){return data[i] <= rhs[i];});}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator>=(const T& rhs) const {return make_condition([this,&rhs](size_t i){return data[i] >= rhs[i];});}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator==(const T& rhs) const {return make_condition([this,&rhs](size_t i){return data[i] == rhs[i];});}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator!=(const T& rhs) const {return make_condition([this,&rhs](size_t i){return data[i] != rhs[i];});}
public:
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator+=(T value)
    {return make_operation(size(),[this,value](size_t i){data[i] += value;});}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator-=(T value)
    {return make_operation(size(),[this,value](size_t i){data[i] -= value;});}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator*=(T value)
    {return make_operation(size(),[this,value](size_t i){data[i] *= value;});}
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ auto operator/=(T value)
    {return make_operation(size(),[this,value](size_t i){data[i] /= value;});}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator+=(const T& rhs)
    {return make_operation(size(),[this,&rhs](size_t i){data[i] += rhs[i];});}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator-=(const T& rhs)
    {return make_operation(size(),[this,&rhs](size_t i){data[i] -= rhs[i];});}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator*=(const T& rhs)
    {return make_operation(size(),[this,&rhs](size_t i){data[i] *= rhs[i];});}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator/=(const T& rhs)
    {return make_operation(size(),[this,&rhs](size_t i){data[i] /= rhs[i];});}
public:
    template<typename T,typename std::enable_if<std::is_integral<T>::value,bool>::type = true>
    __INLINE__ auto operator[](const image<dimension,T>& mask)
    {return make_selection(size(),[this](size_t i)->reference{return data[i];},[&mask](size_t i){return mask[i];});}
    template<typename T,typename std::enable_if<std::is_integral<T>::value,bool>::type = true>
    __INLINE__ auto operator[](const image<dimension,T>& mask) const
    {return make_selection(size(),[this](size_t i){return data[i];},[&mask](size_t i){return mask[i];});}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator[](condition<T>&& condition)
    {return make_selection(size(),[this](size_t i)->reference{return data[i];},std::move(condition));}
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    __INLINE__ auto operator[](condition<T>&& condition) const
    {return make_selection(size(),[this](size_t i){return data[i];},std::move(condition));}
    template<typename EvaluationType,typename SelectionType>
    __INLINE__ auto operator=(const selection<EvaluationType,SelectionType>& sel)
    {return make_operation(size(),[this,&sel](size_t i){if(sel.f(i))(*this)[i] = sel.v(i);});}
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
};

template<typename V,typename T>
inline V extract_pointer(T* p){return V(p);}
template<typename V,typename T>
inline V extract_pointer(T p){return V(&*p);}

template<int dim,typename vtype = float>
class pointer_image : public image<dim,vtype,pointer_container>
{
public:
    using base_type         = image<dim,vtype,pointer_container>;
    using iterator          = typename base_type::iterator;
    using const_iterator    = typename base_type::iterator;
    using storage_type      = typename image<dim,vtype,pointer_container>::storage_type;
    static const int dimension = dim;
public:
    __INLINE__ pointer_image(void) {}
    __INLINE__ pointer_image(const pointer_image& rhs):base_type(){operator=(rhs);}
    template<typename T,typename std::enable_if<T::dimension==dimension && !std::is_same<storage_type,typename T::storage_type>::value,bool>::type = true>
    __INLINE__ pointer_image(T& rhs):base_type(extract_pointer<vtype*>(rhs.begin()),rhs.shape()) {}
    __INLINE__ pointer_image(vtype* pointer,const tipl::shape<dim>& geo_):base_type(pointer,geo_) {}
public:
    __INLINE__ pointer_image& operator=(const pointer_image& rhs)
    {
        base_type::data = rhs.data;
        base_type::geo = rhs.shape();
        return *this;
    }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    __INLINE__ pointer_image& operator=(T value)
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
    static const int dimension = dim;
public:
    __INLINE__ const_pointer_image(void) {}
    __INLINE__ const_pointer_image(const const_pointer_image& rhs):base_type(){operator=(rhs);}
    template<typename T,typename std::enable_if<T::dimension==dimension && !std::is_same<storage_type,typename T::storage_type>::value,bool>::type = true>
    __INLINE__ const_pointer_image(const T& rhs):base_type(extract_const_pointer<const vtype*>(rhs.begin()),rhs.shape()) {}
    __INLINE__ const_pointer_image(const vtype* pointer,const tipl::shape<dim>& geo_):base_type(pointer,geo_){}
public:
    __INLINE__ const_pointer_image& operator=(const const_pointer_image& rhs)
    {
        base_type::data = rhs.data;
        base_type::geo = rhs.shape();
        return *this;
    }

};


template<typename T>
__INLINE__ auto make_alias(T& I)
{
    return pointer_image<T::dimension,typename T::value_type>(I.begin(),I.shape());
}
template<typename T>
__INLINE__ auto make_alias(const T& I)
{
    return const_pointer_image<T::dimension,typename T::value_type>(I.begin(),I.shape());
}


template<typename value_type,typename shape_type>
__INLINE__ auto make_image(value_type* pointer,const shape_type& geo)
{
    return pointer_image<shape_type::dimension,value_type>(pointer,geo);
}

template<typename value_type,typename shape_type>
__INLINE__ auto make_image(const value_type* pointer,const shape_type& geo)
{
    return const_pointer_image<shape_type::dimension,value_type>(pointer,geo);
}

}
#endif
