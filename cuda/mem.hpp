#ifndef MEM_HPP
#define MEM_HPP
#ifdef __CUDACC__

#include <type_traits>
namespace tipl {


template<typename vtype>
class device_memory{
    public:
        using value_type = vtype;
        using iterator          = void*;
        using const_iterator    = const void*;
        using reference         = value_type&;
    private:
        value_type* buf = nullptr;
        size_t s = 0;
    public:
        template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
        device_memory(const T& rhs)                                    {copy_from(rhs);}
        device_memory(size_t new_size)                                 {resize(new_size);}
        device_memory(device_memory&& rhs)                               {swap(rhs);}
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
        device_memory& operator=(const T& rhs)  {copy_from(rhs);}
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
                cudaMemset(new_buf+s,0,(new_s-s)*sizeof(value_type));
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
        __INLINE__ value_type* get(void)                                       {return buf;}
        __INLINE__ const value_type* get(void) const                           {return buf;}
        __INLINE__ const_iterator begin(void)                          const   {return buf;}
        __INLINE__ const_iterator end(void)                            const   {return buf+s;}
        __INLINE__ iterator begin(void)                                        {return buf;}
        __INLINE__ iterator end(void)                                          {return buf+s;}
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
    public:
        template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
        host_memory(const T& rhs)                                    {copy_from(rhs);}
        host_memory(size_t new_size)                                 {resize(new_size);}
        host_memory(host_memory&& rhs)                               {swap(rhs);}
        host_memory(void){}
        template<typename iter_type,typename std::enable_if<std::is_same<value_type,std::iterator_traits<iter_type>::value_type>::value,bool>::type = true>
        host_memory(iter_type from,iter_type to)
        {
            resize(to-from);
            cudaMemcpy(buf, &*from, s*sizeof(value_type),cudaMemcpyHostToHost);
        }
    public: // from device
        host_memory(const void* from,const void* to)
        {
            size_t size_in_byte = reinterpret_cast<const char*>(to)-reinterpret_cast<const char*>(from);
            resize(size_in_byte/sizeof(value_type));
            cudaMemcpy(buf, from, s*sizeof(value_type),cudaMemcpyDeviceToHost);
        }
        ~host_memory(void){clear();}
    public:
        template<typename T>
        host_memory& operator=(const T& rhs)  {copy_from(rhs);return *this;}
        void clear(void)
        {
            if(buf)
            {
                cudaFree(buf);
                buf = nullptr;
                s = 0;
            }
        }
        template<typename T,typename std::enable_if<std::is_class<T>::value && !std::is_same<T,device_memory<value_type> >::value,bool>::type = true>
        void copy_from(const T& rhs)
        {
            resize(rhs.size());
            if(s)
                cudaMemcpy(buf, &rhs[0], s*sizeof(value_type),cudaMemcpyHostToHost);
        }
        void copy_from(const device_memory<value_type>& rhs)
        {
            resize(rhs.size());
            if(s)
                cudaMemcpy(buf, rhs.begin(), s*sizeof(value_type),cudaMemcpyDeviceToHost);
        }
        template<typename T>
        void copy_to(T& rhs)
        {
            if(s)
                cudaMemcpy(&rhs[0], buf, s*sizeof(value_type), cudaMemcpyHostToHost);
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
