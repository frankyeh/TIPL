//---------------------------------------------------------------------------
#ifndef cuH
#define cuH

#ifdef USE_CUBLAS

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cublasXt.h"


namespace tipl{
namespace cu{


class matrix{
public:
    typedef float value_type;
    typedef float* iterator;
    typedef const float* const_iterator;
    typedef float& reference;
private:
    float* ptr = 0;
    int c = 1;
    int r = 0;
    size_t total_element = 0,buf_size = 0;
    void free_mem(void)
    {
        if(ptr)
        {
            cudaFree (ptr);
            c = 1;
            r = 0;
            total_element = 0;
            buf_size = 0;
        }
    }

public:
    matrix(void){}
    matrix(int r_){resize(r_,1);}
    matrix(int r_,int c_){resize(r_,c_);}
    matrix(const std::vector<float>& rhs)
    {
        resize(rhs.size(),1);
        *this = rhs;
    }
    matrix(const std::vector<float>& rhs,int r_,int c_)
    {
        resize(r_,c_);
        *this = rhs;
    }
    float* get(void) const{return ptr;}
    void resize(int r_,int c_)
    {
        if(r == r_ && c == c_)
            return;
        if(total_element == r_*c_)
        {
            r = r_;
            c = c_;
            return;
        }
        free_mem();
        if(r_)
        {
            if (cudaMalloc ((void**)&ptr, r_*c_*sizeof(float)) != cudaSuccess)
                throw std::exception("device memory allocation failed");
            r = r_;
            c = c_;
            total_element = r_*c_;
            buf_size = total_element*sizeof(float);
        }
    }
public:
    size_t size(void) const{return total_element;}
    bool empty(void) const{return total_element == 0;}
    ~matrix(void)
    {
        free_mem();
    }
    int col_count(void) const{return c;}
    int row_count(void) const{return r;}

public:

    const matrix& operator=(const matrix& rhs)
    {
        resize(rhs.row_count(),rhs.col_count());
        if(cudaMemcpy(ptr,rhs.ptr,buf_size,cudaMemcpyDeviceToDevice) != CUBLAS_STATUS_SUCCESS)
            throw std::exception("cudaMemcpy failed in operator=");
        return *this;
    }
    const matrix& operator=(const float* rhs)
    {
        if(cudaMemcpy(ptr,rhs,buf_size,cudaMemcpyHostToDevice) != CUBLAS_STATUS_SUCCESS)
            throw std::exception("cudaMemcpy failed in operator=");
        return *this;
    }
    const matrix& operator=(const std::vector<float>& rhs)
    {
        if(total_element != rhs.size())
            resize(rhs.size(),1);
        if(!rhs.empty())
            *this = &rhs[0];
        return *this;
    }

    void copy_to(float* rhs) const
    {
        if(cudaMemcpy(rhs,ptr,buf_size,cudaMemcpyDeviceToHost) != CUBLAS_STATUS_SUCCESS)
            throw std::exception("cudaMemcpy failed in assign");
    }

    void copy_to(std::vector<float>&lhs) const
    {
        lhs.resize(total_element);
        copy_to(&lhs[0]);
    }

};


struct context{
    cublasHandle_t handle = 0;
    context(void)
    {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            handle = 0;
    }
    ~context(void)
    {
        if(handle)
            cublasDestroy(handle);
    }
    void y_Ax(matrix& y,matrix& A, matrix& x)
    {
        if(A.col_count() != x.row_count())
            throw std::exception("Invalid y_Ax operation");
        int r = A.row_count();
        int c = x.col_count();
        int k = A.col_count();
        y.resize(r,c);
        float alpha = 1.0;
        float beta = 0.0f;
        if(cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,
                    r,c,k,&alpha,
                    A.get(),k,
                    x.get(),k,
                    &beta,
                    y.get(),r) != CUBLAS_STATUS_SUCCESS)
            throw std::exception("cublasSgemm failed in y_Ax");
    }
};




struct contextXt{
    cublasXtHandle_t handle = 0;
    std::vector<int> devices;
    contextXt(void)
    {
        if (cublasXtCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            handle = 0;
        if(handle)
        {
            int nDevices;
            cudaGetDeviceCount(&nDevices);
            devices.resize(nDevices);
            for(int i = 0;i < devices.size();++i)
                devices[i] = i;
            cublasXtDeviceSelect(handle, nDevices, &devices[0]);
        }
    }
    ~contextXt(void)
    {
        if(handle)
            cublasXtDestroy(handle);
    }
};



}
}
#endif//USE_CUBLAS


#endif//cuH
