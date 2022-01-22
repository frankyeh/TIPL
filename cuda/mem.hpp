#ifndef MEM_HPP
#define MEM_HPP

#ifdef __CUDACC__
template<typename vtype>
class cuda_memory{
    public:
        using value_type = vtype;
        using iterator          = value_type*;
        using const_iterator    = const value_type*;
        using reference         = value_type&;
    private:
        value_type* buf = nullptr;
        size_t s = 0;
    public:
        template<typename T>
        cuda_memory(const T& rhs)                                    {copy_from(rhs);}
        cuda_memory(cuda_memory&& rhs)                               {swap(rhs);}
        cuda_memory(void){}
        template<typename iter_type,typename std::enable_if<std::is_same<value_type,std::iterator_traits<iter_type>::value_type>::value,bool>::type = true>
        cuda_memory(iter_type from,iter_type to)
        {
            resize(to-from);
            cudaMemcpy(buf, &*from, s*sizeof(value_type), cudaMemcpyHostToDevice);
        }
        ~cuda_memory(void){clear();}
    public:
        template<typename T>
        cuda_memory& operator=(const T& rhs)  {copy_from(rhs);}
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
        void resize(size_t s_)
        {
            clear();
            if(cudaMalloc(&buf,sizeof(value_type)*s_) != cudaSuccess)
                throw std::bad_alloc();
            s = s_;
        }
    public:
        __INLINE__ void swap(cuda_memory& rhs)
        {
            std::swap(buf,rhs.buf);
            std::swap(s,rhs.s);
        }
        __INLINE__ size_t size(void)    const       {return s;}
        __INLINE__ bool empty(void)     const       {return s==0;}
        template<typename index_type>
        __INLINE__ const value_type& operator[](index_type index)   const   {return buf[index];}
        template<typename index_type>
        __INLINE__ reference operator[](index_type index)                   {return buf[index];}
        __INLINE__ iterator get(void){return buf;}
        __INLINE__ const_iterator get(void) const{return buf;}

};

#endif//__CUDACC__
#endif//MEM_HPP
