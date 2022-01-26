#ifndef MEM_HPP
#define MEM_HPP
#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <type_traits>


namespace tipl {


template<typename vtype>
class device_memory{
    public:
        using value_type = vtype;
        using iterator          = value_type*;
        using const_iterator    = const value_type*;
        using reference         = value_type&;
    private:
        value_type* buf = nullptr;
        size_t s = 0;
    public:
        template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
        device_memory(const T& rhs)                                    {copy_from(rhs);}
        device_memory(size_t new_size)                                 {resize(new_size);}
        device_memory(device_memory&& rhs)                             {swap(rhs);}
        device_memory(void){}
        template<typename iter_type,typename std::enable_if<std::is_same<value_type,std::iterator_traits<iter_type>::value_type>::value,bool>::type = true>
        device_memory(iter_type from,iter_type to)
        {
            resize(to-from);
            cudaMemcpy(buf, &*from, s*sizeof(value_type),cudaMemcpyHostToDevice);
        }
        ~device_memory(void){clear();}
    public:
        template<typename T>
        device_memory& operator=(const T& rhs)  {copy_from(rhs);return *this;}
        void clear(void)
        {
            if(buf)
            {
                cudaFree(buf);
                buf = nullptr;
                s = 0;
            }
        }
        template<typename T>
        void copy_from(const T& rhs)
        {
            resize(rhs.size());
            if(s)
                cudaMemcpy(buf, &rhs[0], s*sizeof(value_type), cudaMemcpyHostToDevice);
        }
        template<typename T>
        void copy_to(T& rhs)
        {
            if(s)
                cudaMemcpy(&rhs[0], buf, s*sizeof(value_type), cudaMemcpyDeviceToHost);
        }
        void resize(size_t new_s)
        {
            value_type* new_buf;
            if(cudaMalloc(&new_buf,sizeof(value_type)*new_s) != cudaSuccess)
                throw std::bad_alloc();
            if(s)
            {
                cudaMemcpy(new_buf, buf, std::min(new_s,s)*sizeof(value_type), cudaMemcpyDeviceToDevice);
                cudaFree(buf);
            }
            if(new_s > s)
            {
                if constexpr(std::is_integral<value_type>::value || std::is_pointer<value_type>::value)
                    cudaMemset(new_buf+s,0,(new_s-s)*sizeof(value_type));
                else
                {
                    auto dp = thrust::device_pointer_cast(new_buf);
                    if constexpr(std::is_class<value_type>::value)
                        thrust::fill(dp+s,dp+new_s,value_type());
                    else
                    if constexpr(std::is_floating_point<value_type>::value)
                        thrust::fill(dp+s,dp+new_s,value_type(0));
                }
            }
            buf = new_buf;
            s = new_s;
        }
    public:
        __INLINE__ void swap(device_memory& rhs)
        {
            std::swap(buf,rhs.buf);
            std::swap(s,rhs.s);
        }
        __INLINE__ size_t size(void)    const       {return s;}
        __INLINE__ bool empty(void)     const       {return s==0;}
    public: // only in device memory
        __INLINE__ auto begin_thrust(void)                              {return thrust::device_pointer_cast(buf);}
        __INLINE__ auto begin_thrust(void)                      const   {return thrust::device_pointer_cast(buf);}
        __INLINE__ auto end_thrust(void)                                {return thrust::device_pointer_cast(buf)+s;}
        __INLINE__ auto end_thrust(void)                        const   {return thrust::device_pointer_cast(buf)+s;}
        __INLINE__ iterator get(void)                                       {return buf;}
        __INLINE__ const_iterator get(void) const                           {return buf;}
        __INLINE__ const void* begin(void)                          const   {return buf;}
        __INLINE__ const void* end(void)                            const   {return buf+s;}
        __INLINE__ void* begin(void)                                        {return buf;}
        __INLINE__ void* end(void)                                          {return buf+s;}
};

template<typename vtype>
struct device_pointer{

public:
    using value_type = vtype;
    using iterator          = value_type*;
    using const_iterator    = const value_type*;
    using reference         = value_type&;
    using const_reference   = const value_type&;
private:
    iterator buf = nullptr;
    size_t s = 0;
public:
    __INLINE__ device_pointer(void){}
    __INLINE__ device_pointer(device_memory<value_type>& rhs):buf(rhs.get()),s(rhs.size())   {}
    __INLINE__ device_pointer(const device_pointer<value_type>& rhs):buf(rhs.buf),s(rhs.s)         {}
    __INLINE__ device_pointer(iterator buf_,size_t s_):buf(buf_),s(s_)                          {}
public:
    __INLINE__ device_pointer& operator=(const device_pointer<value_type>& rhs) {buf = rhs.buf;s=rhs.s;return *this;}
public:
    __INLINE__ size_t size(void)    const       {return s;}
    __INLINE__ bool empty(void)     const       {return s==0;}
public:
    __INLINE__ auto begin_thrust(void)                      const   {return thrust::device_pointer_cast(buf);}
    __INLINE__ auto end_thrust(void)                        const   {return thrust::device_pointer_cast(buf)+s;}
public:
    __INLINE__ iterator begin(void)                                const   {return buf;}
    __INLINE__ iterator end(void)                                  const   {return buf+s;}
    template<typename index_type>
    __INLINE__ reference operator[](index_type index)              const   {return buf[index];}

    __INLINE__ iterator begin(void)                                        {return buf;}
    __INLINE__ iterator end(void)                                          {return buf+s;}
    template<typename index_type>
    __INLINE__ reference operator[](index_type index)                      {return buf[index];}
};

template<typename vtype>
struct device_const_pointer{

public:
    using value_type = vtype;
    using iterator          = const value_type*;
    using const_iterator    = const value_type*;
    using reference         = const value_type&;
    using const_reference   = const value_type&;
private:
    const_iterator buf = nullptr;
    size_t s = 0;
public:
    __INLINE__ device_const_pointer(void){}
    __INLINE__ device_const_pointer(const device_memory<value_type>& rhs):buf(rhs.get()),s(rhs.size())      {}
    __INLINE__ device_const_pointer(const device_pointer<value_type>& rhs):buf(rhs.begin()),s(rhs.size())   {}
    __INLINE__ device_const_pointer(const device_const_pointer<value_type>& rhs):buf(rhs.buf),s(rhs.s)      {}
    __INLINE__ device_const_pointer(const_iterator buf_,size_t s_):buf(buf_),s(s_)                             {}
public:
    __INLINE__ device_const_pointer& operator=(const device_memory<value_type>& rhs)        {buf = rhs.get();s=rhs.size();return *this;}
    __INLINE__ device_const_pointer& operator=(const device_pointer<value_type>& rhs)       {buf = rhs.begin();s=rhs.size();return *this;}
    __INLINE__ device_const_pointer& operator=(const device_const_pointer<value_type>& rhs) {buf = rhs.buf;s=rhs.s;return *this;}
public:
    __INLINE__ size_t size(void)    const       {return s;}
    __INLINE__ bool empty(void)     const       {return s==0;}
public:
    __INLINE__ auto begin_thrust(void)                      const   {return thrust::device_pointer_cast(buf);}
    __INLINE__ auto end_thrust(void)                        const   {return thrust::device_pointer_cast(buf)+s;}
public:
    __INLINE__ const_iterator begin(void)                          const   {return buf;}
    __INLINE__ const_iterator end(void)                            const   {return buf+s;}
    __INLINE__ iterator begin(void)                                        {return buf;}
    __INLINE__ iterator end(void)                                          {return buf+s;}

    template<typename index_type>
    __INLINE__ const_reference operator[](index_type index)              const   {return buf[index];}
    template<typename index_type>
    __INLINE__ reference operator[](index_type index)                            {return buf[index];}
};

template<typename vtype>
class host_memory{
    public:
        using value_type = vtype;
        using iterator          = value_type*;
        using const_iterator    = const value_type*;
        using reference         = value_type&;
    private:
        value_type* buf = nullptr;
        size_t s = 0;
    private:
        template<typename iter_type,typename std::enable_if<
                     std::is_same<value_type,std::iterator_traits<iter_type>::value_type>::value,bool>::type = true>
        void copy_from(iter_type from,iter_type to)
        {
            resize(to-from);
            if(s)
                cudaMemcpy(buf, &*from, s*sizeof(value_type),cudaMemcpyHostToHost);
        }
        void copy_from(const void* from,const void* to)
        {
            size_t size_in_byte = reinterpret_cast<const char*>(to)-reinterpret_cast<const char*>(from);
            resize(size_in_byte/sizeof(value_type));
            if(s)
                cudaMemcpy(buf, from, s*sizeof(value_type),cudaMemcpyDeviceToHost);
        }
    public:
        template<typename T,typename std::enable_if<std::is_class<T>::value &&
                                                    std::is_same<T::value_type,value_type>::value,bool>::type = true>
        host_memory(const T& rhs)                                    {copy_from(rhs.begin(),rhs.end());}
        host_memory(size_t new_size)                                 {resize(new_size);}
        host_memory(host_memory&& rhs)                               {swap(rhs);}
        host_memory(void){}
        template<typename iter_type>
        host_memory(iter_type from,iter_type to)                     {copy_from(from,to);}
        ~host_memory(void){clear();}
    public:
        template<typename T>
        host_memory& operator=(const T& rhs)  {copy_from(rhs);return *this;}
        void clear(void)
        {
            if(buf)
            {
                cudaFreeHost(buf);
                buf = nullptr;
                s = 0;
            }
        }
        template<typename T,typename std::enable_if<
                     std::is_class<T>::value && !std::is_same<T,device_memory<value_type> >::value,bool>::type = true>
        void copy_from(const T& rhs)
        {
            copy_from(rhs.begin(),rhs.end());
        }
        void copy_from(const device_memory<value_type>& rhs)
        {
            resize(rhs.size());
            if(s)
                cudaMemcpy(buf, rhs.begin(), s*sizeof(value_type),cudaMemcpyDeviceToHost);
        }
        void resize(size_t new_s)
        {
            iterator new_buf;
            if(cudaMallocHost(&new_buf,sizeof(value_type)*new_s) != cudaSuccess)
                throw std::bad_alloc();
            if(s)
            {
                cudaMemcpy(new_buf, buf, std::min(new_s,s)*sizeof(value_type), cudaMemcpyHostToHost);
                cudaFree(buf);
            }
            if(new_s > s)
                std::fill(new_buf+s,new_buf+new_s,0);
            buf = new_buf;
            s = new_s;
        }
    public:
        __INLINE__ void swap(host_memory& rhs)
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
};


}//namespace tipl
#endif//__CUDACC__
#endif//MEM_HPP
