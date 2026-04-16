#ifndef CU_HPP
#define CU_HPP

#include "utility/basic_image.hpp"
#include "def.hpp"
#include <iterator>
#include <type_traits>
#include <stdexcept>
#include <cstring>
#include <algorithm>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace tipl {

// -------------------------------------------------------------------------
// DECLARATIONS FOR CUDA API WRAPPERS
// These tell g++ they exist elsewhere, preventing syntax/linking errors.
// -------------------------------------------------------------------------

template<typename T> void cu_malloc(T** ptr, size_t count);
template<typename T> void cu_free(T* ptr);
template<typename T> void cu_malloc_host(T** ptr, size_t count);
template<typename T> void cu_free_host(T* ptr);
template<typename Dest, typename Src> void cu_copy_d2d(Dest* dest, const Src* src, size_t count);
template<typename Dest, typename Src> void cu_copy_h2d(Dest* dest, const Src* src, size_t count);
template<typename Dest, typename Src> void cu_copy_d2h(Dest* dest, const Src* src, size_t count);
template<typename Dest, typename Src> void cu_copy_h2h(Dest* dest, const Src* src, size_t count);
template<typename T> void cu_memset(T* dest, int val, size_t count);
template<typename T> void cu_fill(T* dest, size_t count, T val);
template<typename T> T cu_eval(const T* ptr);

template<typename vtype>
class device_vector{
    public:
        using value_type = vtype;
        using iterator          = value_type*;
        using const_iterator    = const value_type*;
        using reference         = value_type;
    private:
        value_type* buf = nullptr;
        size_t buf_size = 0;
        size_t s = 0;
    public:
        device_vector(size_t new_size,bool init = true)                {resize(new_size,init);}
        device_vector(device_vector&& rhs)noexcept                     {swap(rhs);}
        device_vector(const device_vector& rhs)                        {copy_from(rhs);}
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
                if constexpr (tipl::use_cuda) cu_free(buf);
                else delete[] buf;

                buf = nullptr;
                buf_size = 0;
                s = 0;
            }
        }
    public:
        device_vector& operator=(const device_vector& rhs)  {copy_from(rhs);return *this;}
        template<typename T>
        device_vector& operator=(const T& rhs)  {copy_from(rhs);return *this;}
        device_vector& operator=(device_vector&& rhs) noexcept
        {
            swap(rhs);
            return *this;
        }

        void clear(void)
        {
            s = 0;
        }
        template<typename T>
        void copy_from(const T* from,size_t size)
        {
            resize(size,false);
            if(s) {
                if constexpr (tipl::use_cuda) {
                    if constexpr(std::is_same_v<T,void>)
                        cu_copy_d2d(buf, from, size);
                    else
                        cu_copy_h2d(buf, from, size);
                } else {
                    std::copy_n(reinterpret_cast<const value_type*>(from), size, buf);
                }
            }
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
            if(s) {
                if constexpr (tipl::use_cuda) {
                    cu_copy_d2h(rhs.data(), buf, s);
                } else {
                    std::copy_n(buf, s, rhs.data());
                }
            }
        }
        void resize(size_t new_s,bool init = true)
        {
            if(s == new_s)
                return;
            if(new_s > buf_size) // need reallocation
            {
                value_type* new_buf;
                if constexpr (tipl::use_cuda) {
                    cu_malloc(&new_buf, new_s);
                    if(s) cu_copy_d2d(new_buf, buf, s);
                    if(buf) cu_free(buf);
                } else {
                    new_buf = new value_type[new_s];
                    if(s) std::copy_n(buf, s, new_buf);
                    if(buf) delete[] buf;
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
                    if constexpr(tipl::use_cuda) cu_memset(buf+s, 0, added_s);
                    else std::memset(buf+s, 0, added_s*sizeof(value_type));
                }
                else
                {
                    if constexpr(tipl::use_cuda) {
                        if constexpr(std::is_class<value_type>::value)
                            cu_fill(buf+s, added_s, value_type());
                        else if constexpr(std::is_floating_point<value_type>::value)
                            cu_fill(buf+s, added_s, value_type(0));
                    } else {
                        if constexpr(std::is_class<value_type>::value)
                            std::fill(buf+s, buf+new_s, value_type());
                        else if constexpr(std::is_floating_point<value_type>::value)
                            std::fill(buf+s, buf+new_s, value_type(0));
                    }
                }
            }
            s = new_s;
        }
    public:
        void swap(device_vector& rhs)
        {
            std::swap(buf,rhs.buf);
            std::swap(buf_size,rhs.buf_size);
            std::swap(s,rhs.s);
        }
        __INLINE__ size_t size(void)    const       {return s;}
        __INLINE__ bool empty(void)     const       {return s==0;}
    public: // only in device memory
        template<typename index_type>
        value_type operator[](index_type index) const
        {
            value_type result;
            if constexpr(tipl::use_cuda) cu_copy_d2h(&result, &buf[index], 1);
            else result = buf[index];
            return result;
        }
    public:
        __INLINE__ iterator data(void)                                       {return buf;}
        __INLINE__ const_iterator data(void) const                           {return buf;}
        __INLINE__ const void* begin(void)                          const   {return buf;}
        __INLINE__ const void* end(void)                            const   {return buf+s;}
        __INLINE__ void* begin(void)                                        {return buf;}
        __INLINE__ void* end(void)                                          {return buf+s;}
};

// ... (shared_device_vector remains fully header inline because it only manages pointers)

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
    shared_device_vector(void){}
    shared_device_vector(device_vector<value_type>& rhs):buf(rhs.data()),s(rhs.size())                    {}
    shared_device_vector(const device_vector<value_type>& rhs):const_buf(rhs.data()),s(rhs.size())        {}

    shared_device_vector(shared_device_vector<value_type>& rhs):buf(rhs.buf),s(rhs.s)                    {}
    shared_device_vector(const shared_device_vector<value_type>& rhs):const_buf(rhs.const_buf),s(rhs.s)  {}

    shared_device_vector(iterator buf_,size_t s_):buf(buf_),s(s_)                                        {}
    shared_device_vector(const_iterator buf_,size_t s_):const_buf(buf_),s(s_)                            {}
public:
    shared_device_vector& operator=(const shared_device_vector<value_type>& rhs) const{const_buf = rhs.const_buf;s=rhs.s;return *this;}
    shared_device_vector& operator=(shared_device_vector<value_type>& rhs) {buf = rhs.buf;s=rhs.s;return *this;}
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
    if constexpr(tipl::use_cuda) return cu_eval(p);
    else return *p;
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

// -------------------------------------------------------------------------
// HOST VECTOR
// -------------------------------------------------------------------------

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
                if constexpr(tipl::use_cuda) cu_copy_h2h(buf, &*from, s);
                else std::copy_n(&*from, s, buf);
            }
        }
        void copy_from(const void* from,const void* to)
        {
            size_t size_in_byte = reinterpret_cast<const char*>(to)-reinterpret_cast<const char*>(from);
            size_t count = size_in_byte/sizeof(value_type);
            resize(count,false);
            if(s)
            {
                if constexpr(tipl::use_cuda) cu_copy_d2h(buf, from, count);
                else std::memcpy(buf, from, size_in_byte);
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
                if constexpr(tipl::use_cuda) cu_free_host(buf);
                else delete[] buf;
                buf = nullptr;
                buf_size = 0;
                s = 0;
            }
        }
    public:
        template<typename T>
        host_vector& operator=(const T& rhs)  {copy_from(rhs);return *this;}
        host_vector& operator=(host_vector&& rhs) noexcept
        {
            swap(rhs);
            return *this;
        }
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
                if constexpr(tipl::use_cuda) cu_copy_d2h(buf, rhs.begin(), s);
                else std::copy_n(rhs.begin(), s, buf);
            }
        }
        void resize(size_t new_s, bool init = true)
        {
            if(new_s > buf_size) // need reallocation
            {
                iterator new_buf;
                if constexpr(tipl::use_cuda) {
                    cu_malloc_host(&new_buf, new_s);
                    if(s) cu_copy_h2h(new_buf, buf, s);
                    if(buf) cu_free_host(buf);
                } else {
                    new_buf = new value_type[new_s];
                    if(s) std::copy_n(buf, s, new_buf);
                    if(buf) delete[] buf;
                }

                buf = new_buf;
                buf_size = new_s;
            }
            if(new_s > s && init)
                std::fill(buf+s,buf+new_s,0);
            s = new_s;
        }
    public:
        void swap(host_vector& rhs)
        {
            std::swap(buf,rhs.buf);
            std::swap(buf_size,rhs.buf_size);
            std::swap(s,rhs.s);
        }
        size_t size(void)    const       {return s;}
        bool empty(void)     const       {return s==0;}

    public: // only available in host memory
        template<typename index_type>
        const value_type& operator[](index_type index)   const   {return buf[index];}
        template<typename index_type>
        reference operator[](index_type index)                   {return buf[index];}
        const_iterator begin(void)                       const   {return buf;}
        const_iterator end(void)                         const   {return buf+s;}
        iterator begin(void)                                     {return buf;}
        iterator end(void)                                       {return buf+s;}
        const_iterator data(void)                       const   {return buf;}
        iterator data(void)                                     {return buf;}
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
    const_pointer_device_container(void){}
    const_pointer_device_container(const const_pointer_device_container& rhs):bg(rhs.bg),sz(rhs.sz){}
    const_pointer_device_container(iterator from,iterator to):bg(from),sz(to-from){}
    const_pointer_device_container& operator=(const const_pointer_device_container& rhs)
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
    __INLINE__ void clear(void)                             {sz = 0;}
    __INLINE__ void resize(size_t s)                        {sz = s;}

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
    const_pointer_device_image(void) {}
    const_pointer_device_image(const const_pointer_device_image& rhs):base_type(){operator=(rhs);}
    const_pointer_device_image(const image<dimension,vtype,device_vector>& rhs):
                base_type(reinterpret_cast<const vtype*>(rhs.begin()),rhs.shape()){}
public:
    const_pointer_device_image& operator=(const const_pointer_device_image& rhs)
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


} //namespace tipl


// -------------------------------------------------------------------------
// CUDA EXPLICIT IMPLEMENTATIONS & INSTANTIATIONS
// -------------------------------------------------------------------------
#ifdef __CUDACC__

namespace tipl {

template<typename T>
__global__ void device_vector_fill(T* buf,size_t size,T v)
{
    TIPL_FOR(index,size)
        buf[index] = v;
}

// Wrapper Implementations
template<typename T> void cu_malloc(T** ptr, size_t count) {
    if(cudaMalloc(ptr, sizeof(T) * count) != cudaSuccess) throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
}
template<typename T> void cu_free(T* ptr) {
    if(ptr) cudaFree(ptr);
}
template<typename T> void cu_malloc_host(T** ptr, size_t count) {
    if(cudaMallocHost(ptr, sizeof(T) * count) != cudaSuccess) throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
}
template<typename T> void cu_free_host(T* ptr) {
    if(ptr) cudaFreeHost(ptr);
}
template<typename Dest, typename Src> void cu_copy_d2d(Dest* dest, const Src* src, size_t count) {
    if(cudaMemcpy(dest, src, count * sizeof(Dest), cudaMemcpyDeviceToDevice) != cudaSuccess) throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
}
template<typename Dest, typename Src> void cu_copy_h2d(Dest* dest, const Src* src, size_t count) {
    if(cudaMemcpy(dest, src, count * sizeof(Dest), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
}
template<typename Dest, typename Src> void cu_copy_d2h(Dest* dest, const Src* src, size_t count) {
    if(cudaMemcpy(dest, src, count * sizeof(Dest), cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
}
template<typename Dest, typename Src> void cu_copy_h2h(Dest* dest, const Src* src, size_t count) {
    if(cudaMemcpy(dest, src, count * sizeof(Dest), cudaMemcpyHostToHost) != cudaSuccess) throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
}
template<typename T> void cu_memset(T* dest, int val, size_t count) {
    if(cudaMemset(dest, val, count * sizeof(T)) != cudaSuccess) throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
}
template<typename T> void cu_fill(T* dest, size_t count, T val) {
    TIPL_RUN(device_vector_fill, count)(dest, count, val);
}
template<typename T> T cu_eval(const T* ptr) {
    T v;
    if(cudaMemcpy(&v, ptr, sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
    return v;
}

} // namespace tipl

// Explicit template instantiations to solve ODR when linked to external object files
#define INSTANTIATE_CU_WRAPPERS(T) \
    template void tipl::cu_malloc<T>(T**, size_t); \
    template void tipl::cu_free<T>(T*); \
    template void tipl::cu_malloc_host<T>(T**, size_t); \
    template void tipl::cu_free_host<T>(T*); \
    template void tipl::cu_copy_d2d<T, void>(T*, const void*, size_t); \
    template void tipl::cu_copy_d2d<T, T>(T*, const T*, size_t); \
    template void tipl::cu_copy_h2d<T, void>(T*, const void*, size_t); \
    template void tipl::cu_copy_h2d<T, T>(T*, const T*, size_t); \
    template void tipl::cu_copy_d2h<T, void>(T*, const void*, size_t); \
    template void tipl::cu_copy_d2h<T, T>(T*, const T*, size_t); \
    template void tipl::cu_copy_h2h<T, void>(T*, const void*, size_t); \
    template void tipl::cu_copy_h2h<T, T>(T*, const T*, size_t); \
    template void tipl::cu_memset<T>(T*, int, size_t); \
    template void tipl::cu_fill<T>(T*, size_t, T); \
    template T tipl::cu_eval<T>(const T*);

INSTANTIATE_CU_WRAPPERS(float)
INSTANTIATE_CU_WRAPPERS(double)
INSTANTIATE_CU_WRAPPERS(int)
INSTANTIATE_CU_WRAPPERS(unsigned int)
INSTANTIATE_CU_WRAPPERS(unsigned char)
INSTANTIATE_CU_WRAPPERS(short)
INSTANTIATE_CU_WRAPPERS(size_t)

#endif // __CUDACC__

#endif//CU_HPP
