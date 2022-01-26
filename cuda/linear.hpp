#ifndef CUDA_LINEAR_HPP
#define CUDA_LINEAR_HPP

#ifdef __CUDACC__

#include "resampling.hpp"

namespace tipl{

namespace  reg{


__global__ void mutual_information_cuda_kernel(const unsigned char* from,
                                               const unsigned char* to,
                                               int32_t* mutual_hist,
                                               unsigned int band_width,
                                               size_t size)
{
    size_t index = (uint64_t(blockIdx.x) << 8) | threadIdx.x;
    if(index < size)
        atomicAdd(mutual_hist + ((uint32_t(from[index]) << band_width) | to[index]),1);
}

__global__ void mutual_information_cuda_kernel1(const int32_t* mutual_hist,
                                                int32_t* to8_hist)
{
    for(int i = 0,pos = 0;i < blockDim.x;++i,pos += blockDim.x)
        to8_hist[threadIdx.x] += mutual_hist[pos+threadIdx.x];
}


__global__ void mutual_information_cuda_kernel2(
                                          const int32_t* from8_hist,
                                          const int32_t* to8_hist,
                                          const int32_t* mutual_hist,
                                          double* mu_log_mu)
{
    size_t index = threadIdx.x + blockDim.x*blockIdx.x;
    if (mutual_hist[index])
    {
        double mu = mutual_hist[index];
        mu_log_mu[index] = mu*std::log(mu/double(from8_hist[blockIdx.x])/to8_hist[threadIdx.x]);
    }
    else
        mu_log_mu[index] = 0.0;
}


struct mutual_information_cuda
{
    typedef double value_type;
    unsigned int band_width;
    unsigned int his_bandwidth;
    device_memory<int32_t> from8_hist;
    device_memory<unsigned char> from8;
    device_memory<unsigned char> to8;
    std::mutex init_mutex;
public:
    mutual_information_cuda(unsigned int band_width_ = 6):band_width(band_width_),his_bandwidth(1 << band_width_) {}
public:
    template<typename ImageType,typename TransformType>
    double operator()(const ImageType& from_raw,const ImageType& to_raw,const TransformType& trans)
    {
        if(from_raw.size() > to_raw.size())
        {
            auto inv_trans(trans);
            inv_trans.inverse();
            return (*this)(to_raw,from_raw,inv_trans);
        }
        {
            std::scoped_lock<std::mutex> lock(init_mutex);
            if (from8_hist.empty() || to_raw.size() != to8.size() || from_raw.size() != from8.size())
            {
                host_memory<unsigned char> host_from8(from_raw.size());
                host_memory<unsigned char> host_to8(to_raw.size());
                host_memory<int32_t> host_from8_hist;

                normalize_upper_lower(to_raw.begin(),to_raw.end(),host_to8.begin(),his_bandwidth-1);
                normalize_upper_lower(from_raw.begin(),from_raw.end(),host_from8.begin(),his_bandwidth-1);
                histogram(host_from8,host_from8_hist,0,his_bandwidth-1,his_bandwidth);

                from8_hist = host_from8_hist;
                from8 = host_from8;
                to8 = host_to8;
            }
        }
        image<ImageType::dimension,unsigned char,device_memory> to2from(from_raw.shape());
        resample_cuda(tipl::make_image(to8.get(),to_raw.shape()),to2from,trans);

        device_memory<int32_t> mutual_hist(his_bandwidth*his_bandwidth);

        mutual_information_cuda_kernel<<<from_raw.size()/256,256>>>
            (from8.get(),to2from.get(),mutual_hist.get(),band_width,from_raw.size());

        cudaDeviceSynchronize();

        device_memory<int32_t> to8_hist(his_bandwidth);
        mutual_information_cuda_kernel1<<<1,his_bandwidth>>>(mutual_hist.get(),to8_hist.get());

        cudaDeviceSynchronize();

        device_memory<double> mu_log_mu(mutual_hist.size());
        mutual_information_cuda_kernel2<<<his_bandwidth,his_bandwidth>>>
            (from8_hist.get(),to8_hist.get(),mutual_hist.get(),mu_log_mu.get());

        cudaDeviceSynchronize();

        return -accumulate(mu_log_mu,0.0);

    }
};

}//reg


}//tipl

#endif//__CUDACC__


#endif//CUDA_LINEAR_HPP
