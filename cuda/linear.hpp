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
    size_t stride = blockDim.x*gridDim.x;
    for(size_t index = threadIdx.x + blockIdx.x*blockDim.x;
        index < from.size();index += stride)
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
    std::mutex stream_mutex;
    static constexpr int stream_count = 16;
    cudaStream_t streams[stream_count] = {0};
    size_t cur_stream_id = 0;

    std::vector<device_vector<int32_t> > mutual_hist;
    std::vector<device_vector<double> > mu_log_mu;
    auto cur_id(void)
    {
        std::lock_guard<std::mutex> lock(stream_mutex);
        ++cur_stream_id;
        if(cur_stream_id == stream_count)
            cur_stream_id = 0;
        return cur_stream_id;
    }
public:
    mutual_information_cuda(void):mutual_hist(stream_count),mu_log_mu(stream_count)
    {
        for(int i = 0;i < stream_count;++i)
        {
            mutual_hist[i].resize(his_bandwidth*his_bandwidth);
            mu_log_mu[i].resize(his_bandwidth*his_bandwidth);
            cudaStreamCreate(&streams[i]);
        }
    }
    ~mutual_information_cuda(void)
    {
        for(int i = 0;i < stream_count;++i)
            cudaStreamDestroy(streams[i]);
    }
public:
    template<typename ImageType,typename TransformType>
    double operator()(const ImageType& from_raw,const ImageType& to_raw,const TransformType& trans)
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
                host_image<3,unsigned char> host_from8,host_to8;
                host_vector<int32_t> host_from8_hist;

                normalize_upper_lower_mt(to_raw,host_to8,his_bandwidth-1);
                normalize_upper_lower_mt(from_raw,host_from8,his_bandwidth-1);
                histogram(host_from8,host_from8_hist,0,his_bandwidth-1,his_bandwidth);

                from8_hist = host_from8_hist;
                from8 = host_from8;
                to8 = host_to8;

            }
        }

        auto id = cur_id();

        thrust::fill(thrust::cuda::par.on(streams[id]),mutual_hist[id].get(),
                     mutual_hist[id].get()+mutual_hist[id].size(),0);
        mutual_information_cuda_kernel<<<std::min<size_t>((from_raw.size()+255)/256,256),256,0,streams[id]>>>
                                (tipl::make_shared(from8),
                                 tipl::make_shared(to8),
                                 trans,
                                 tipl::make_shared(mutual_hist[id]));
        if(cudaPeekAtLastError() != cudaSuccess)
            throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));

        mutual_information_cuda_kernel2<<<his_bandwidth,his_bandwidth,0,streams[id]>>>(
                        tipl::make_shared(from8_hist),
                        tipl::make_shared(mutual_hist[id]),
                        tipl::make_shared(mu_log_mu[id]));
        return -sum_cuda(mu_log_mu[id],0.0,streams[id]);

    }
};

}//reg


}//tipl

#endif//__CUDACC__


#endif//CUDA_LINEAR_HPP
