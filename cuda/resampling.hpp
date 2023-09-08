#ifndef CUDA_RESAMPLE_HPP
#define CUDA_RESAMPLE_HPP


#ifdef __CUDACC__
#include "../utility/basic_image.hpp"
#include "../numerical/interpolation.hpp"
namespace tipl{



template<tipl::interpolation itype,typename T1,typename T2>
__global__ void scale_cuda_kernel(T1 from,T2 to,double dx,double dy,double dz)
{
    TIPL_FOR(index,to.size())
    {
        tipl::pixel_index<3> pos(index,to.shape());
        tipl::vector<3> v(pos);
        v[0] *= dx;
        v[1] *= dy;
        v[2] *= dz;
        tipl::estimate<itype>(from,v,to[index]);
    }
}

template<tipl::interpolation itype = linear,typename T1,typename T2>
inline void scale_cuda(const T1& from,T2& to)
{
    double dx = double(from.width()-1)/double(to.width()-1);
    double dy = double(from.height()-1)/double(to.height()-1);
    double dz = double(from.depth()-1)/double(to.depth()-1);
    TIPL_RUN(scale_cuda_kernel<itype>,to.size())
            (tipl::make_shared(from),tipl::make_shared(to),dx,dy,dz);
}


template<typename T,typename U,typename V>
__global__ void downsample_with_padding_cuda_kernel(T in,U out,V shift)
{
    using value_type = typename T::value_type;
    TIPL_FOR(index,out.size())
    {
        downsample_with_padding_imp(pixel_index<3>(index,out.shape()),in,out,shift);
    }
}

template<typename T,typename U>
void downsample_with_padding_cuda(const T& in,U& out)
{
    if constexpr(T::dimension==3)
    {
        shape<3> out_shape((in.width()+1)/2,(in.height()+1)/2,(in.depth()+1)/2);
        if(out.size() < out_shape.size())
            out.resize(out_shape);

        std::vector<size_t> shift(8);
        shift[0] = 0;
        shift[1] = 1;
        shift[2] = in.width();
        shift[3] = 1+in.width();
        shift[4] = in.plane_size();
        shift[5] = in.plane_size()+1;
        shift[6] = in.plane_size()+in.width();
        shift[7] = in.plane_size()+1+in.width();

        device_vector<size_t> shift_(shift.begin(),shift.end());
        TIPL_RUN(downsample_with_padding_cuda_kernel,out.size())
                (tipl::make_shared(in),tipl::make_shared(out),tipl::make_shared(shift_));

        out.resize(out_shape);
    }
}


template<tipl::interpolation itype,typename T1,typename T2>
__global__ void upsample_with_padding_cuda_kernel(T1 from,T2 to)
{
    TIPL_FOR(index,to.size())
    {
        tipl::estimate<itype>(from,
            tipl::vector<3>(tipl::pixel_index<3>(index,to.shape()))*=0.5,
            to[index]);
    }
}


template<typename image_type>
void upsample_with_padding_cuda(const image_type& in,image_type& out)
{
    TIPL_RUN(upsample_with_padding_cuda_kernel<linear>,out.size())
            (tipl::make_shared(in),tipl::make_shared(out));
}

template<typename image_type>
void upsample_with_padding_cuda(image_type& in,const shape<image_type::dimension>& geo)
{
    image_type new_d(geo);
    upsample_with_padding_cuda(in,new_d);
    new_d.swap(in);
}



template<tipl::interpolation itype,typename T1,typename T2,typename U>
__global__ void resample_cuda_kernel(T1 from,T2 to,U trans)
{
    TIPL_FOR(index,to.size())
    {
        tipl::pixel_index<3> pos(index,to.shape());
        tipl::vector<3> v;
        trans(pos,v);
        tipl::estimate<itype>(from,v,to[index]);
    }
}


template<tipl::interpolation itype = linear,typename T,typename T2,typename U>
inline void resample_cuda(const T& from,T2& to,const U& trans,cudaStream_t stream = nullptr)
{
    TIPL_RUN_STREAM(resample_cuda_kernel<itype>,to.size(),stream)
            (tipl::make_shared(from),tipl::make_shared(to),trans);
    if(cudaPeekAtLastError() != cudaSuccess)
        throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
}

}

#endif//__CUDACC__

#endif//CUDA_RESAMPLE_HPP
