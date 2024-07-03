#ifndef CU_HPP
#define CU_HPP
#include "utility/basic_image.hpp"
#include "def.hpp"
#include <iterator>
#include <type_traits>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cuda.h>
#include <cuda_runtime.h>




template<typename T>
__global__ void device_vector_fill(T* buf,size_t size,T v)
{
    TIPL_FOR(index,size)
        buf[index] = v;
}

namespace tipl {



template<typename T,typename Fun>
__global__ void cuda_for_kernel(size_t size,T from,Fun f)
{
    TIPL_FOR(index,size)
        f(from+index);
}

template <typename T,typename Func,typename std::enable_if<
              std::is_integral<T>::value ||
              std::is_class<T>::value,bool>::type = true>
inline void cuda_for(T from,T to,Func&& f,unsigned int thread_count = 256)
{
    if(to == from)
        return;
    size_t size = to-from;
    size_t grid_size = (size+thread_count-1)/thread_count;
    cuda_for_kernel<<<(grid_size > thread_count ? thread_count:grid_size),thread_count>>>(size,from,f);
    if(cudaPeekAtLastError() != cudaSuccess)
        throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
}

template <typename T,typename Func,typename std::enable_if<std::is_integral<T>::value,bool>::type = true>
inline void cuda_for(T size, Func&& f, unsigned int thread_count = 256)
{
    if(!size)
        return;
    cuda_for(T(),size,std::move(f),thread_count);
}

template <typename T,typename Func,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
inline void cuda_for(T& c, Func&& f, unsigned int thread_count = 256)
{
    cuda_for(c.begin(),c.end(),std::move(f),thread_count);
}



template<typename vtype>
class device_vector{
    public:
        using value_type = vtype;
        using iterator          = value_type*;
        using const_iterator    = const value_type*;
        using reference         = value_type&;
    private:
        value_type* buf = nullptr;
        size_t buf_size = 0;
        size_t s = 0;
    public:
        device_vector(size_t new_size,bool init = true)                {resize(new_size,init);}
        device_vector(device_vector&& rhs)noexcept                     {swap(rhs);}
        device_vector(void){}
        template<typename T,typename std::enable_if<std::is_class_v<T>,bool>::type = true>
        device_vector(const T& rhs)                                    {copy_from(rhs);}
        template<typename T,typename std::enable_if<std::is_class_v<T>,bool>::type = true>
        device_vector(T from,T to)
        {
            copy_from(&*from,to-from);
        }
        template<typename T>
        device_vector(const T* from,const T* to)
        {
            if constexpr(std::is_same_v<T,void>)
                copy_from(from,(to-from)/sizeof(value_type));
            else
                copy_from(from,to-from);
        }
        ~device_vector(void)
        {
            if(buf)
            {
                cudaFree(buf);
                buf = nullptr;
                buf_size = 0;
                s = 0;
            }
        }
    public:
        device_vector& operator=(const device_vector& rhs)  {copy_from(rhs);return *this;}
        template<typename T>
        device_vector& operator=(const T& rhs)  {copy_from(rhs);return *this;}
        void clear(void)
        {
            s = 0;
        }
        template<typename T>
        void copy_from(const T* from,size_t size)
        {
            resize(size,false);
            enum cudaMemcpyKind device_host_setting;
            if constexpr(std::is_same_v<T,void>)
                device_host_setting = cudaMemcpyDeviceToDevice;
            else
                device_host_setting = cudaMemcpyHostToDevice;
            if(s && cudaMemcpy(buf,from,size*sizeof(value_type),device_host_setting) != cudaSuccess)
                throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
        }
        template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
        void copy_from(T from,size_t size)
        {
            copy_from(&*from,size);
        }
        template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
        void copy_from(const T& rhs)
        {
            copy_from(rhs.begin(),rhs.size());
        }
        void copy_to(device_vector& rhs)
        {
            rhs.copy_from(*this);
        }
        template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
        void copy_to(T& rhs)
        {
            if(s)
            {
                if(cudaMemcpy(rhs.data(), buf, s*sizeof(value_type), cudaMemcpyDeviceToHost) != cudaSuccess)
                    throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
            }
        }
        void resize(size_t new_s,bool init = true)
        {
            if(s == new_s)
                return;
            if(new_s > buf_size) // need reallocation
            {
                value_type* new_buf;
                if(cudaMalloc(&new_buf,sizeof(value_type)*new_s) != cudaSuccess)
                    throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
                if(s)
                {
                    if(cudaMemcpy(new_buf,buf,s*sizeof(value_type),cudaMemcpyDeviceToDevice) != cudaSuccess)
                    {
                        cudaFree(new_buf);
                        throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
                    }
                    cudaFree(buf);
                }
                buf = new_buf;
                buf_size = new_s;
            }
            if(new_s > s && init)
            {
                size_t added_s = new_s-s;
                if constexpr(std::is_integral<value_type>::value ||
                             std::is_pointer<value_type>::value)
                {
                    if(cudaMemset(buf+s,0,added_s*sizeof(value_type)) != cudaSuccess)
                        throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
                }
                else
                {
                    #ifdef __CUDACC__
                    if constexpr(std::is_class<value_type>::value)
                        TIPL_RUN(device_vector_fill,added_s)
                            (buf+s,added_s,value_type());
                    else
                    if constexpr(std::is_floating_point<value_type>::value)
                        TIPL_RUN(device_vector_fill,added_s)
                            (buf+s,added_s,value_type(0));
                    #endif//__CUDACC__
                }
            }
            s = new_s;
        }
    public:
        __INLINE__ void swap(device_vector& rhs)
        {
            std::swap(buf,rhs.buf);
            std::swap(buf_size,rhs.buf_size);
            std::swap(s,rhs.s);
        }
        __INLINE__ size_t size(void)    const       {return s;}
        __INLINE__ bool empty(void)     const       {return s==0;}
    public: // only in device memory
        __INLINE__ iterator data(void)                                       {return buf;}
        __INLINE__ const_iterator data(void) const                           {return buf;}
        __INLINE__ const void* begin(void)                          const   {return buf;}
        __INLINE__ const void* end(void)                            const   {return buf+s;}
        __INLINE__ void* begin(void)                                        {return buf;}
        __INLINE__ void* end(void)                                          {return buf+s;}
};

template<typename vtype>
struct shared_device_vector{

public:
    using value_type = vtype;
    using iterator          = value_type*;
    using const_iterator    = const value_type*;
    using reference         = value_type&;
    using const_reference   = const value_type&;
private:
    union{
        iterator buf;
        const_iterator const_buf;
    };
    size_t s = 0;
public:
    __INLINE__ shared_device_vector(void){}
    __INLINE__ shared_device_vector(device_vector<value_type>& rhs):buf(rhs.data()),s(rhs.size())                    {}
    __INLINE__ shared_device_vector(const device_vector<value_type>& rhs):const_buf(rhs.data()),s(rhs.size())        {}

    __INLINE__ shared_device_vector(shared_device_vector<value_type>& rhs):buf(rhs.buf),s(rhs.s)                    {}
    __INLINE__ shared_device_vector(const shared_device_vector<value_type>& rhs):const_buf(rhs.const_buf),s(rhs.s)  {}

    __INLINE__ shared_device_vector(iterator buf_,size_t s_):buf(buf_),s(s_)                                        {}
    __INLINE__ shared_device_vector(const_iterator buf_,size_t s_):const_buf(buf_),s(s_)                            {}
public:
    __INLINE__ const shared_device_vector& operator=(const shared_device_vector<value_type>& rhs) const{const_buf = rhs.const_buf;s=rhs.s;return *this;}
    __INLINE__ shared_device_vector& operator=(shared_device_vector<value_type>& rhs) {buf = rhs.buf;s=rhs.s;return *this;}
public:
    __INLINE__ size_t size(void)    const       {return s;}
    __INLINE__ bool empty(void)     const       {return s==0;}
public:
    __INLINE__ const_iterator begin(void)                          const   {return const_buf;}
    __INLINE__ const_iterator end(void)                            const   {return const_buf+s;}
    __INLINE__ const_iterator data(void)                           const   {return const_buf;}
    template<typename index_type>
    __INLINE__ const_reference operator[](index_type index)        const   {return const_buf[index];}
public:
    __INLINE__ iterator begin(void)                                        {return buf;}
    __INLINE__ iterator end(void)                                          {return buf+s;}
    __INLINE__ iterator data(void)                                         {return buf;}
    template<typename index_type>
    __INLINE__ reference operator[](index_type index)                      {return buf[index];}

};

template<typename T>
__HOST__ auto device_eval(const T* p)
{
    T v;
    cudaMemcpy(&v,p,sizeof(T),cudaMemcpyDeviceToHost);
    return v;
}

template<typename T>
__INLINE__ auto make_shared(device_vector<T>& I)
{
    return shared_device_vector<T>(I);
}
template<typename T>
__INLINE__ const auto make_shared(const device_vector<T>& I)
{
    return shared_device_vector<T>(I);
}

template<typename vtype>
class host_vector{
    public:
        using value_type = vtype;
        using iterator          = value_type*;
        using const_iterator    = const value_type*;
        using reference         = value_type&;
    private:
        value_type* buf = nullptr;
        size_t buf_size = 0;
        size_t s = 0;
    private:
        template<typename iter_type,typename std::enable_if<
                     std::is_same<value_type, typename std::iterator_traits<iter_type>::value_type>::value,bool>::type = true>
        void copy_from(iter_type from,iter_type to)
        {
            resize(to-from,false);
            if(s)
            {
                if(cudaMemcpy(buf, &*from, s*sizeof(value_type),cudaMemcpyHostToHost) != cudaSuccess)
                    throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
            }
        }
        void copy_from(const void* from,const void* to)
        {
            size_t size_in_byte = reinterpret_cast<const char*>(to)-reinterpret_cast<const char*>(from);
            resize(size_in_byte/sizeof(value_type),false);
            if(s)
            {
                if(cudaMemcpy(buf, from, s*sizeof(value_type),cudaMemcpyDeviceToHost) != cudaSuccess)
                    throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
            }
        }
    public:
        template<typename T,typename std::enable_if<std::is_class<T>::value &&
                                                    std::is_same<typename T::value_type,value_type>::value,bool>::type = true>
        host_vector(const T& rhs)                                    {copy_from(rhs.begin(),rhs.end());}
        host_vector(size_t new_size,bool init = true)                {resize(new_size,init);}
        host_vector(host_vector&& rhs)noexcept                       {swap(rhs);}
        host_vector(void){}
        template<typename iter_type>
        host_vector(iter_type from,iter_type to)                     {copy_from(from,to);}
        ~host_vector(void)
        {
            if(buf)
            {
                cudaFreeHost(buf);
                buf = nullptr;
                buf_size = 0;
                s = 0;
            }
        }
    public:
        template<typename T>
        host_vector& operator=(const T& rhs)  {copy_from(rhs);return *this;}
        void clear(void)
        {
            s = 0;
        }
        template<typename T,typename std::enable_if<
                     std::is_class<T>::value && !std::is_same<T,device_vector<value_type> >::value,bool>::type = true>
        void copy_from(const T& rhs)
        {
            copy_from(rhs.begin(),rhs.end());
        }
        void copy_from(const device_vector<value_type>& rhs)
        {
            resize(rhs.size(),false);
            if(s)
            {
                if(cudaMemcpy(buf, rhs.begin(), s*sizeof(value_type),cudaMemcpyDeviceToHost) != cudaSuccess)
                    throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
            }
        }
        void resize(size_t new_s, bool init = true)
        {
            if(new_s > buf_size) // need reallocation
            {
                iterator new_buf;
                if(cudaMallocHost(&new_buf,sizeof(value_type)*new_s) != cudaSuccess)
                    throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
                if(s)
                {
                    if(cudaMemcpy(new_buf, buf, s*sizeof(value_type), cudaMemcpyHostToHost) != cudaSuccess)
                    {
                        cudaFreeHost(new_buf);
                        throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
                    }
                    cudaFreeHost(buf);
                }
                buf = new_buf;
            }
            if(new_s > s && init)
                std::fill(buf+s,buf+new_s,0);
            s = new_s;
        }
    public:
        __INLINE__ void swap(host_vector& rhs)
        {
            std::swap(buf,rhs.buf);
            std::swap(s,rhs.s);
        }
        __INLINE__ size_t size(void)    const       {return s;}
        __INLINE__ bool empty(void)     const       {return s==0;}

    public: // only available in host memory
        template<typename index_type>
        __INLINE__ const value_type& operator[](index_type index)   const   {return buf[index];}
        template<typename index_type>
        __INLINE__ reference operator[](index_type index)                   {return buf[index];}
        __INLINE__ const_iterator begin(void)                       const   {return buf;}
        __INLINE__ const_iterator end(void)                         const   {return buf+s;}
        __INLINE__ iterator begin(void)                                     {return buf;}
        __INLINE__ iterator end(void)                                       {return buf+s;}
        __INLINE__ const_iterator data(void)                       const   {return buf;}
        __INLINE__ iterator data(void)                                     {return buf;}
};


template<typename vtype>
class const_pointer_device_container
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
    __INLINE__ const_pointer_device_container(void){}
    __INLINE__ const_pointer_device_container(const const_pointer_device_container& rhs):bg(rhs.bg),sz(rhs.sz){}
    __INLINE__ const_pointer_device_container(iterator from,iterator to):bg(from),sz(to-from){}
    __INLINE__ const_pointer_device_container& operator=(const const_pointer_device_container& rhs)
    {
        bg = rhs.bg;sz = rhs.sz;
        return *this;
    }
public:
    __INLINE__ const void* begin(void)                   const    {return bg;}
    __INLINE__ const void* end(void)                     const    {return bg+sz;}
    __INLINE__ const_iterator data(void)                     const    {return bg;}
public:
    __INLINE__ size_t size(void)                            const    {return sz;}
    __INLINE__ bool empty(void)                             const    {return sz == 0;}
};

template<int dim,typename vtype = float>
class const_pointer_device_image : public image<dim,vtype,const_pointer_device_container>
{
public:
    using value_type        =   vtype;
    using base_type         =   image<dim,value_type,const_pointer_device_container>;
    using iterator          =   typename base_type::iterator;
    using const_iterator    =   typename base_type::const_iterator;
    using storage_type      =   typename image<dim,vtype,const_pointer_device_container>::storage_type;
    using buffer_type       =   image<dim,vtype,device_vector>;
    static constexpr int dimension = dim;
public:
    __INLINE__ const_pointer_device_image(void) {}
    __INLINE__ const_pointer_device_image(const const_pointer_device_image& rhs):base_type(){operator=(rhs);}
    __INLINE__ const_pointer_device_image(const image<dimension,vtype,device_vector>& rhs):
                base_type(reinterpret_cast<const vtype*>(rhs.begin()),rhs.shape()){}
public:
    __INLINE__ const_pointer_device_image& operator=(const const_pointer_device_image& rhs)
    {
        base_type::alloc = rhs.alloc;
        base_type::sp = rhs.sp;
        return *this;
    }
};


template<int dim,typename vtype = float>
using device_image = image<dim,vtype,device_vector>;

template<int dim,typename vtype>
__INLINE__ auto make_device_shared(const device_image<dim,vtype>& I)
{
    return const_pointer_device_image<dim,vtype>(I);
}

template<int dim,typename vtype = float>
using host_image = tipl::image<dim,vtype,host_vector>;


template<int dim,typename vtype>
struct memory_location<device_image<dim,vtype> >{
    static constexpr memory_location_type at = CUDA;
};
template<int dim,typename vtype>
struct memory_location<const_pointer_device_image<dim,vtype> >{
    static constexpr memory_location_type at = CUDA;
};

template<typename vtype>
struct memory_location<device_vector<vtype> >{
    static constexpr memory_location_type at = CUDA;
};


}//namespace tipl
#endif//MEM_HPP
