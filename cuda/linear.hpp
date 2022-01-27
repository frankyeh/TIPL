#ifndef CUDA_LINEAR_HPP
#define CUDA_LINEAR_HPP

#ifdef __CUDACC__

#include "resampling.hpp"
#include "numerical.hpp"

namespace tipl{

namespace  reg{


const int bandwidth = 6;
const int his_bandwidth = 64;

__global__ void mutual_information_cuda_kernel(const_pointer_image<3,unsigned char> from,
                                               const_pointer_image<3,unsigned char> to,
                                               shared_device_vector<int32_t> mutual_hist)
{
    size_t stride = blockDim.x*gridDim.x;
    for(size_t index = threadIdx.x + blockIdx.x*blockDim.x;
        index < from.size();index += stride)
            atomicAdd(mutual_hist.begin() + ((uint32_t(from[index]) << bandwidth) + to[index]),1);
}

__global__ void mutual_information_cuda_kernel1(const shared_device_vector<int32_t> mutual_hist,
                                                shared_device_vector<int32_t> to8_hist)
{
    for(int i = 0,pos = 0;i < blockDim.x;++i,pos += blockDim.x)
        to8_hist[threadIdx.x] += mutual_hist[pos+threadIdx.x];
}


__global__ void mutual_information_cuda_kernel2(
                                          const shared_device_vector<int32_t> from8_hist,
                                          const shared_device_vector<int32_t> to8_hist,
                                          const shared_device_vector<int32_t> mutual_hist,
                                          shared_device_vector<double> mu_log_mu)
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
    device_vector<int32_t> from8_hist;
    device_image<3,unsigned char> from8;
    device_image<3,unsigned char> to8;
    std::mutex init_mutex;
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
                host_image<3,unsigned char> host_from8(from_raw.shape());
                host_image<3,unsigned char> host_to8(to_raw.shape());
                host_vector<int32_t> host_from8_hist;

                normalize_upper_lower(to_raw.begin(),to_raw.end(),host_to8.begin(),his_bandwidth-1);
                normalize_upper_lower(from_raw.begin(),from_raw.end(),host_from8.begin(),his_bandwidth-1);
                histogram(host_from8,host_from8_hist,0,his_bandwidth-1,his_bandwidth);

                from8_hist = host_from8_hist;
                from8 = host_from8;
                to8 = host_to8;
            }
        }

        device_image<3,unsigned char> to2from(from8.shape());
        resample_cuda(to8,to2from,trans);

        device_vector<int32_t> mutual_hist(his_bandwidth*his_bandwidth);

        mutual_information_cuda_kernel<<<std::min<size_t>((from_raw.size()+255)/256,256),256>>>
                                (from8,to2from,mutual_hist);

        cudaDeviceSynchronize();

        device_vector<int32_t> to8_hist(his_bandwidth);
        mutual_information_cuda_kernel1<<<1,his_bandwidth>>>(mutual_hist,to8_hist);

        cudaDeviceSynchronize();

        device_vector<double> mu_log_mu(mutual_hist.size());
        mutual_information_cuda_kernel2<<<his_bandwidth,his_bandwidth>>>
                (from8_hist,to8_hist,mutual_hist,mu_log_mu);

        cudaDeviceSynchronize();

        return -sum_cuda(mu_log_mu,0.0);

    }
};

}//reg


}//tipl

#endif//__CUDACC__


#endif//CUDA_LINEAR_HPP
