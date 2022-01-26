#ifndef CUDA_LINEAR_HPP
#define CUDA_LINEAR_HPP

#ifdef __CUDACC__

#include "resampling.hpp"

namespace tipl{

namespace  reg{


__global__ void mutual_information_cuda_kernel(const unsigned char* from,
                                          const unsigned char* to,
                                          int32_t* mutual_hist,
                                          unsigned int band_width)
{
    auto index = tipl::pixel_index<3>::voxel2index(blockIdx.x,blockIdx.y,threadIdx.x,
                                      tipl::shape<3>(gridDim.x,gridDim.y,blockDim.x));
    unsigned int to_index = to[index];
    unsigned int from_index = uint32_t(from[index]) << band_width;
    atomicAdd(mutual_hist + from_index + to_index,1);
}

__global__ void mutual_information_cuda_kernel1(const int32_t* mutual_hist,
                                                int32_t* to_hist,
                                                unsigned int his_bandwidth)
{
    for(int i = 0,pos = 0;i < his_bandwidth;++i,pos += his_bandwidth)
        to_hist[threadIdx.x] += mutual_hist[pos+threadIdx.x];
}


__global__ void mutual_information_cuda_kernel2(
                                          const int32_t* from_hist,
                                          const int32_t* to_hist,
                                          const int32_t* mutual_hist,
                                          double* mu_log_mu)
{
    size_t index = threadIdx.x + blockDim.x*blockIdx.x;
    if (mutual_hist[index])
    {
        double mu = mutual_hist[index];
        mu_log_mu[index] = mu*std::log(mu/double(from_hist[blockIdx.x])/to_hist[threadIdx.x]);
    }
    else
        mu_log_mu[index] = 0.0;
}


struct mutual_information_cuda
{
    typedef double value_type;
    unsigned int band_width;
    unsigned int his_bandwidth;
    device_memory<int32_t> from_hist;
    device_memory<unsigned char> from;
    device_memory<unsigned char> to;
    std::mutex init_mutex;
public:
    mutual_information_cuda(unsigned int band_width_ = 6):band_width(band_width_),his_bandwidth(1 << band_width_) {}
public:
    template<typename ImageType,typename TransformType>
    double operator()(const ImageType& from_,const ImageType& to_,const TransformType& trans)
    {
        if(from_.size() > to_.size())
        {
            auto trans_(trans);
            trans_.inverse();
            return (*this)(to_,from_,trans_);
        }
        {
            std::scoped_lock<std::mutex> lock(init_mutex);
            if (from_hist.empty() || to_.size() != to.size() || from_.size() != from.size())
            {
                host_memory<unsigned char> host_from(from_.size());
                host_memory<unsigned char> host_to(to_.size());
                host_memory<int32_t> host_from_hist;
                normalize_upper_lower(to_.begin(),to_.end(),host_to.begin(),his_bandwidth-1);
                normalize_upper_lower(from_.begin(),from_.end(),host_from.begin(),his_bandwidth-1);
                histogram(host_from,host_from_hist,0,his_bandwidth-1,his_bandwidth);

                from_hist = host_from_hist;
                from = host_from;
                to = host_to;
            }
        }
        image<ImageType::dimension,unsigned char,device_memory> from2(from_.shape());
        resample_cuda(tipl::make_image(to.get(),to_.shape()),from2,trans);

        device_memory<int32_t> mutual_hist(his_bandwidth*his_bandwidth);

        mutual_information_cuda_kernel<<<dim3(from_.width(),from_.height()),from_.depth()>>>
            (from.get(),from2.get(),mutual_hist.get(),band_width);

        cudaDeviceSynchronize();

        device_memory<int32_t> to_hist(his_bandwidth);
        mutual_information_cuda_kernel1<<<1,his_bandwidth>>>(mutual_hist.get(),to_hist.get(),his_bandwidth);

        cudaDeviceSynchronize();

        device_memory<double> mu_log_mu(mutual_hist.size());
        mutual_information_cuda_kernel2<<<his_bandwidth,his_bandwidth>>>
            (from_hist.get(),to_hist.get(),mutual_hist.get(),mu_log_mu.get());

        cudaDeviceSynchronize();

        return -accumulate(mu_log_mu,0.0);

    }
};

}//reg


}//tipl

#endif//__CUDACC__


#endif//CUDA_LINEAR_HPP
