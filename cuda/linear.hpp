#ifndef CUDA_LINEAR_HPP
#define CUDA_LINEAR_HPP

#ifdef __CUDACC__

#include "resampling.hpp"
#include "numerical.hpp"

namespace tipl{

namespace  reg{


const int bandwidth = 6;
const int his_bandwidth = 64;

template<typename T,typename V,typename U>
__global__ void mutual_information_cuda_kernel(T from,T to,V trans,U mutual_hist)
{
    TIPL_FOR(index,from.size())
    {
        tipl::pixel_index<3> pos(index,from.shape());
        tipl::vector<3> v;
        trans(pos,v);
        unsigned char to_index = 0;
        tipl::estimate<tipl::interpolation::linear>(to,v,to_index);
        atomicAdd(mutual_hist.begin() +
                  (uint32_t(from[index]) << bandwidth) +to_index,1);
    }
}

template<typename T,typename U>
__global__ void mutual_information_cuda_kernel2(T from8_hist,T mutual_hist,U mu_log_mu)
{
    int32_t to8=0;
    for(int i=0; i < his_bandwidth; ++i)
        to8 += mutual_hist[threadIdx.x + i*blockDim.x];

    size_t index = threadIdx.x + blockDim.x*blockIdx.x;

    // if mutual_hist is not zero,
    // the corresponding from8_hist and to8_hist won't be zero
    mu_log_mu[index] = mutual_hist[index] ?
        mutual_hist[index]*std::log(mutual_hist[index]/double(from8_hist[blockIdx.x])/double(to8))
            : 0.0;
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
    double operator()(const ImageType& from_raw,const ImageType& to_raw,const TransformType& trans,int thread_id = 0)
    {
        using DeviceImageType = device_image<3,typename ImageType::value_type>;
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
                to8.resize(to_raw.shape());
                normalize_upper_lower_cuda(DeviceImageType(to_raw),to8,his_bandwidth-1);

                host_image<3,unsigned char> host_from8,host_to8;
                host_vector<int32_t> host_from8_hist;

                normalize_upper_lower(from_raw,host_from8,his_bandwidth-1);
                histogram(host_from8,host_from8_hist,0,his_bandwidth-1,his_bandwidth);

                from8_hist = host_from8_hist;
                from8 = host_from8;

            }
        }

        device_vector<int32_t> mutual_hist(his_bandwidth*his_bandwidth);
        TIPL_RUN(mutual_information_cuda_kernel,from_raw.size())
                                (tipl::make_shared(from8),
                                 tipl::make_shared(to8),
                                 trans,
                                 tipl::make_shared(mutual_hist));
        if(cudaPeekAtLastError() != cudaSuccess)
            throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
        device_vector<double> mu_log_mu(his_bandwidth*his_bandwidth);
        mutual_information_cuda_kernel2<<<his_bandwidth,his_bandwidth>>>(
                        tipl::make_shared(from8_hist),
                        tipl::make_shared(mutual_hist),
                        tipl::make_shared(mu_log_mu));
        return -sum_cuda(mu_log_mu,0.0);

    }
};

}//reg


}//tipl

#endif//__CUDACC__


#endif//CUDA_LINEAR_HPP
