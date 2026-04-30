#ifndef CNN3D_DETAIL_HPP
#define CNN3D_DETAIL_HPP

#include <limits>
#include <cmath>
#include "../mt.hpp"
#include "../def.hpp"
#include "../po.hpp"

namespace tipl {
namespace ml3d {

enum class activation_type { none, relu, leaky_relu, elu };

template<typename T>
__INLINE__ T multiply_add(T a,T b,T c)
{
#ifdef __CUDA_ARCH__
    if constexpr(std::is_same_v<T,float>)
        return fmaf(a,b,c);
#endif
    return a*b+c;
}

template<activation_type Act,typename T>
__INLINE__ T apply_activation(T v,T slope)
{
    if constexpr(Act == activation_type::relu)
        return v < T(0) ? T(0) : v;
    if constexpr(Act == activation_type::leaky_relu)
        return v < T(0) ? v*slope : v;
    if constexpr(Act == activation_type::elu)
        return v < T(0) ? (T)expm1f((float)v) : v;
    return v;
}

template<activation_type Act,typename T>
inline void apply_activation_buffer(T* out,size_t count,T slope)
{
    if constexpr(Act != activation_type::none)
        for(size_t i = 0;i < count;++i)
            out[i] = apply_activation<Act>(out[i],slope);
}

template <activation_type Act, typename T>
void cpu_conv_1x1_forward(const T* in,const T* weight,const T* bias,T* out,
                          int in_c,int out_c,
                          int in_d,int in_h,int in_w)
{
    const size_t in_img_size = static_cast<size_t>(in_w)*in_h*in_d;
    const size_t out_img_size = in_img_size;
    const int plane = in_w*in_h;

    tipl::par_for(static_cast<size_t>(out_c)*in_d,[&](size_t job)
    {
        const int oc = static_cast<int>(job/in_d);
        const int z = static_cast<int>(job-static_cast<size_t>(oc)*in_d);
        T* out_slice = out+static_cast<size_t>(oc)*out_img_size+static_cast<size_t>(z)*plane;
        std::fill_n(out_slice,plane,bias[oc]);

        const T* weight_oc = weight+static_cast<size_t>(oc)*in_c;
        for(int ic = 0;ic < in_c;++ic)
        {
            const T w = weight_oc[ic];
            const T* in_slice = in+static_cast<size_t>(ic)*in_img_size+static_cast<size_t>(z)*plane;
            for(int i = 0;i < plane;++i)
                out_slice[i] = multiply_add(w,in_slice[i],out_slice[i]);
        }
        apply_activation_buffer<Act>(out_slice,plane,(T)0.01f);
    });
}

template <activation_type Act, typename T, int stride>
void cpu_conv_3x3_forward(const T* in,const T* weight,const T* bias,T* out,
                          int in_c,int out_c,
                          int in_d,int in_h,int in_w,
                          int out_d,int out_h,int out_w)
{
    const size_t in_img_size = static_cast<size_t>(in_w)*in_h*in_d;
    const size_t out_img_size = static_cast<size_t>(out_w)*out_h*out_d;
    const int in_plane = in_w*in_h;
    const int out_plane = out_w*out_h;
    constexpr int kernel_size = 3;
    constexpr int kernel_size3 = 27;
    constexpr int kernel_plane = 9;

    tipl::par_for(static_cast<size_t>(out_c)*out_d,[&](size_t job)
    {
        const int oc = static_cast<int>(job/out_d);
        const int z = static_cast<int>(job-static_cast<size_t>(oc)*out_d);
        T* out_slice = out+static_cast<size_t>(oc)*out_img_size+static_cast<size_t>(z)*out_plane;
        std::fill_n(out_slice,out_plane,bias[oc]);

        const int start_sz = z*stride-1;
        const T* weight_oc = weight+static_cast<size_t>(oc)*in_c*kernel_size3;

        for(int ic = 0;ic < in_c;++ic)
        {
            const T* in_ic = in+static_cast<size_t>(ic)*in_img_size;
            const T* weight_ic = weight_oc+ic*kernel_size3;

            for(int kz = 0;kz < kernel_size;++kz)
            {
                const int sz = start_sz+kz;
                if(static_cast<unsigned int>(sz) >= static_cast<unsigned int>(in_d))
                    continue;

                const T* in_z = in_ic+static_cast<size_t>(sz)*in_plane;
                const T* weight_z = weight_ic+kz*kernel_plane;

                for(int ky = 0;ky < kernel_size;++ky)
                {
                    const int sy0 = ky-1;
                    int y_begin = 0;
                    int y_end = out_h;

                    if(sy0 < 0)
                        y_begin = (-sy0+stride-1)/stride;
                    if(sy0+(out_h-1)*stride >= in_h)
                        y_end = (in_h-1-sy0)/stride+1;
                    if(y_end > out_h)
                        y_end = out_h;
                    if(y_begin < 0)
                        y_begin = 0;
                    if(y_begin >= y_end)
                        continue;

                    const T* weight_y = weight_z+ky*kernel_size;

                    for(int kx = 0;kx < kernel_size;++kx)
                    {
                        const T w = weight_y[kx];
                        const int sx0 = kx-1;
                        int x_begin = 0;
                        int x_end = out_w;

                        if(sx0 < 0)
                            x_begin = (-sx0+stride-1)/stride;
                        if(sx0+(out_w-1)*stride >= in_w)
                            x_end = (in_w-1-sx0)/stride+1;
                        if(x_end > out_w)
                            x_end = out_w;
                        if(x_begin < 0)
                            x_begin = 0;
                        if(x_begin >= x_end)
                            continue;

                        int sy = sy0+y_begin*stride;
                        for(int y = y_begin;y < y_end;++y,sy += stride)
                        {
                            const T* in_row = in_z+static_cast<size_t>(sy)*in_w;
                            T* out_row = out_slice+static_cast<size_t>(y)*out_w;
                            int sx = sx0+x_begin*stride;
                            for(int x = x_begin;x < x_end;++x,sx += stride)
                                out_row[x] = multiply_add(w,in_row[sx],out_row[x]);
                        }
                    }
                }
            }
        }
        apply_activation_buffer<Act>(out_slice,out_plane,(T)0.01f);
    });
}

template <activation_type Act, typename T>
void cpu_conv_3d_forward(const T* in, const T* weight, const T* bias, T* out, int in_c, int out_c, int in_d, int in_h, int in_w, int out_d, int out_h, int out_w, int kernel_size, int kernel_size3, int range, int stride)
{
    if(kernel_size == 1 && stride == 1)
        return cpu_conv_1x1_forward<Act>(in,weight,bias,out,in_c,out_c,in_d,in_h,in_w);
    if(kernel_size == 3 && stride == 1)
        return cpu_conv_3x3_forward<Act,T,1>(in,weight,bias,out,in_c,out_c,in_d,in_h,in_w,out_d,out_h,out_w);
    if(kernel_size == 3 && stride == 2)
        return cpu_conv_3x3_forward<Act,T,2>(in,weight,bias,out,in_c,out_c,in_d,in_h,in_w,out_d,out_h,out_w);
    throw std::runtime_error("cpu conv_3d supports only ks1 stride1, ks3 stride1, and ks3 stride2");
}

template <typename T>
void cpu_conv_transpose_3d_forward(const T* in, const T* weight, const T* bias, T* out, int in_c, int out_c, int in_d, int in_h, int in_w, int out_d, int out_h, int out_w, int kernel_size, int kernel_size3, int stride)
{
    if(kernel_size != 2 || stride != 2)
        throw std::runtime_error("cpu conv_transpose_3d supports only ks2 stride2");

    const size_t in_img_size = static_cast<size_t>(in_w)*in_h*in_d;
    const size_t out_img_size = static_cast<size_t>(out_w)*out_h*out_d;
    const int in_plane = in_w*in_h;
    const int out_plane = out_w*out_h;
    constexpr int kernel_size3_const = 8;
    const int weight_ic_step = out_c*kernel_size3_const;

    tipl::par_for(static_cast<size_t>(out_c)*out_d,[&](size_t job)
    {
        const int oc = static_cast<int>(job/out_d);
        const int z = static_cast<int>(job-static_cast<size_t>(oc)*out_d);
        const int in_z = z >> 1;
        const int kz = z & 1;
        T* out_slice = out+static_cast<size_t>(oc)*out_img_size+static_cast<size_t>(z)*out_plane;
        const T bias_val = bias[oc];
        const T* weight_z = weight+static_cast<size_t>(oc)*kernel_size3_const+(kz << 2);

        for(int y = 0;y < out_h;++y)
        {
            const int in_y = y >> 1;
            const int ky = y & 1;
            const T* weight_y = weight_z+(ky << 1);
            T* out_row = out_slice+static_cast<size_t>(y)*out_w;
            const T* in_y_base = in+static_cast<size_t>(in_z)*in_plane+static_cast<size_t>(in_y)*in_w;

            for(int x = 0;x < out_w;++x)
            {
                const int in_x = x >> 1;
                const int kx = x & 1;
                const T* weight_k = weight_y+kx;
                const T* in_ptr = in_y_base+in_x;
                T sum = bias_val;

                for(int ic = 0;ic < in_c;++ic,in_ptr += in_img_size,weight_k += weight_ic_step)
                    sum = multiply_add(*in_ptr,*weight_k,sum);

                out_row[x] = sum;
            }
        }
    });
}

template <activation_type Act, typename T>
void cpu_instance_norm_3d_forward(const T* in, T* out, const T* weight, const T* bias, int out_c, size_t plane_size)
{
    const double inv_plane_size = 1.0/static_cast<double>(plane_size);

    tipl::par_for(out_c,[&](size_t outc)
    {
        size_t pos = outc*plane_size;
        const T* base_ptr = in+pos;
        const T* end_ptr = base_ptr+plane_size;

        double sum = 0.0;
        double sq_sum = 0.0;

        for(const T* ptr = base_ptr;ptr < end_ptr;++ptr)
        {
            double val = *ptr;
            sum += val;
            sq_sum += val*val;
        }

        T mean = static_cast<T>(sum*inv_plane_size);
        T var = std::max(T(0),static_cast<T>(sq_sum*inv_plane_size-static_cast<double>(mean)*mean));
        T scale = weight[outc]/std::sqrt(var+T(1e-5));
        T shift = bias[outc]-mean*scale;

        const T* ptr = base_ptr;
        T* out_ptr = out+pos;
        for(;ptr < end_ptr;++ptr,++out_ptr)
            *out_ptr = apply_activation<Act>(multiply_add(*ptr,scale,shift),(T)0.01f);
    });
}

template<activation_type Act,typename T>
void cpu_batch_norm_3d_forward(const T* in,T* out,const T* weight,const T* bias,int in_c,size_t plane_size)
{
    tipl::par_for(in_c,[&](size_t c)
    {
        const T* ptr = in+c*plane_size;
        T* out_ptr = out+c*plane_size;
        const T* end_ptr = ptr+plane_size;
        T w = weight[c];
        T b = bias[c];

        for(;ptr < end_ptr;++ptr,++out_ptr)
            *out_ptr = apply_activation<Act>(multiply_add(*ptr,w,b),(T)0.01f);
    });
}

template <typename T>
void cpu_max_pool_3d_forward(const T* in, T* out, int in_c, int in_d, int in_h, int in_w, int out_d, int out_h, int out_w, int pool_size)
{
    if(pool_size != 2)
        throw std::runtime_error("cpu max_pool_3d supports only pool size 2");

    const size_t in_plane = static_cast<size_t>(in_w)*in_h;
    const size_t out_plane = static_cast<size_t>(out_w)*out_h;
    const size_t in_img_size = static_cast<size_t>(in_d)*in_plane;
    const size_t out_img_size = static_cast<size_t>(out_d)*out_plane;

    tipl::par_for(static_cast<size_t>(in_c)*out_d,[&](size_t i)
    {
        const int c = static_cast<int>(i/out_d);
        const int z = static_cast<int>(i-static_cast<size_t>(c)*out_d);
        const T* in_z0 = in+static_cast<size_t>(c)*in_img_size+static_cast<size_t>(z << 1)*in_plane;
        const T* in_z1 = in_z0+in_plane;
        T* out_ptr = out+static_cast<size_t>(c)*out_img_size+static_cast<size_t>(z)*out_plane;

        for(int y = 0,sy = 0;y < out_h;++y,sy += 2)
        {
            const T* r00 = in_z0+static_cast<size_t>(sy)*in_w;
            const T* r01 = r00+in_w;
            const T* r10 = in_z1+static_cast<size_t>(sy)*in_w;
            const T* r11 = r10+in_w;

            for(int x = 0,sx = 0;x < out_w;++x,sx += 2,++out_ptr)
            {
                T v = r00[sx];
                v = v > r00[sx+1] ? v : r00[sx+1];
                v = v > r01[sx] ? v : r01[sx];
                v = v > r01[sx+1] ? v : r01[sx+1];
                v = v > r10[sx] ? v : r10[sx];
                v = v > r10[sx+1] ? v : r10[sx+1];
                v = v > r11[sx] ? v : r11[sx];
                v = v > r11[sx+1] ? v : r11[sx+1];
                *out_ptr = v;
            }
        }
    });
}

template <typename T>
void cpu_upsample_3d_forward(const T* in, T* out, int in_c, int in_d, int in_h, int in_w, int out_d, int out_h, int out_w, int pool_size)
{
    if(pool_size != 2)
        throw std::runtime_error("cpu upsample_3d supports only pool size 2");

    const size_t in_plane = static_cast<size_t>(in_w)*in_h;
    const size_t out_plane = static_cast<size_t>(out_w)*out_h;
    const size_t in_img_size = static_cast<size_t>(in_d)*in_plane;
    const size_t out_img_size = static_cast<size_t>(out_d)*out_plane;

    tipl::par_for(static_cast<size_t>(in_c)*in_d,[&](size_t i)
    {
        const int c = static_cast<int>(i/in_d);
        const int z = static_cast<int>(i-static_cast<size_t>(c)*in_d);
        const T* in_ptr = in+static_cast<size_t>(c)*in_img_size+static_cast<size_t>(z)*in_plane;
        T* out_z0 = out+static_cast<size_t>(c)*out_img_size+static_cast<size_t>(z << 1)*out_plane;
        T* out_z1 = out_z0+out_plane;

        for(int y = 0;y < in_h;++y)
        {
            T* out_y00 = out_z0+static_cast<size_t>(y << 1)*out_w;
            T* out_y01 = out_y00+out_w;
            T* out_y10 = out_z1+static_cast<size_t>(y << 1)*out_w;
            T* out_y11 = out_y10+out_w;

            for(int x = 0;x < in_w;++x)
            {
                const T val = *in_ptr++;
                const int sx = x << 1;
                out_y00[sx] = val;
                out_y00[sx+1] = val;
                out_y01[sx] = val;
                out_y01[sx+1] = val;
                out_y10[sx] = val;
                out_y10[sx+1] = val;
                out_y11[sx] = val;
                out_y11[sx+1] = val;
            }
        }
    });
}

template <activation_type Act, typename T>
void cuda_conv_3d_forward(const T* in, const T* weight, const T* bias, T* out,
                          int in_c, int out_c,
                          int in_d, int in_h, int in_w,
                          int out_d, int out_h, int out_w,
                          int kernel_size, int kernel_size3, int range, int stride, T slope);

template <typename T>
void cuda_conv_transpose_3d_forward(const T* in, const T* weight, const T* bias, T* out,
                                    int in_c, int out_c,
                                    int in_d, int in_h, int in_w,
                                    int out_d, int out_h, int out_w,
                                    int kernel_size, int kernel_size3, int stride);

template <activation_type Act, typename T>
void cuda_instance_norm_3d_forward(const T* in, T* out, const T* weight, const T* bias,
                                   int out_c, size_t plane_size, T slope);
template<activation_type Act,typename T>
void cuda_batch_norm_3d_forward(const T* in,T* out,const T* weight,const T* bias,int in_c,size_t plane_size);

template <typename T>
void cuda_max_pool_3d_forward(const T* in, T* out,
                              int in_c, int in_d, int in_h, int in_w,
                              int out_d, int out_h, int out_w, int pool_size);
template <typename T>
void cuda_upsample_3d_forward(const T* in, T* out,
                              int in_c, int in_d, int in_h, int in_w,
                              int out_d, int out_h, int out_w, int pool_size);

template <typename T>
void cuda_copy_device_to_device(T* dest, const T* src, size_t count);

#ifdef __CUDACC__

namespace cuda_kernels
{

static constexpr int spatial_block_x = 32;
static constexpr int spatial_block_y = 4;
static constexpr int elem_block_size = 256;
static constexpr int norm_block_size = 256;

inline int div_up(size_t n,size_t d)
{
    return static_cast<int>((n+d-1)/d);
}

template<activation_type Act,typename T,int kernel_size,int stride>
__global__ __launch_bounds__(128,2)
void conv_3d_kernel(const T* __restrict__ in,const T* __restrict__ weight,const T* __restrict__ bias,T* __restrict__ out,
                    int in_c,int out_c,
                    int in_d,int in_h,int in_w,
                    int out_d,int out_h,int out_w,T slope)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y_blocks = (out_h+blockDim.y-1)/blockDim.y;
    int y_tile = blockIdx.y;
    int y0 = (y_tile%y_blocks)*blockDim.y;
    int z = y_tile/y_blocks;
    int y = y0+threadIdx.y;
    int oc = blockIdx.z;

    if(x >= out_w || y >= out_h)
        return;

    constexpr int kernel_size3 = kernel_size*kernel_size*kernel_size;
    constexpr int kernel_plane = kernel_size*kernel_size;
    constexpr int range = kernel_size/2;

    int in_plane = in_w*in_h;
    int in_img_size = in_plane*in_d;
    int out_plane = out_w*out_h;
    int out_img_size = out_plane*out_d;

    int start_sx = x*stride-range;
    int start_sy = y*stride-range;
    int start_sz = z*stride-range;

    T sum = bias[oc];
    const T* weight_oc = weight+static_cast<size_t>(oc)*in_c*kernel_size3;

    if constexpr(kernel_size == 1 && stride == 1)
    {
        size_t in_offset = static_cast<size_t>(z)*in_plane+static_cast<size_t>(y)*in_w+x;
        for(int ic = 0;ic < in_c;++ic)
            sum = multiply_add(in[static_cast<size_t>(ic)*in_img_size+in_offset],weight_oc[ic],sum);
    }
    else
    {
        bool inside = start_sx >= 0 && start_sy >= 0 && start_sz >= 0 &&
                      start_sx+kernel_size <= in_w &&
                      start_sy+kernel_size <= in_h &&
                      start_sz+kernel_size <= in_d;

        if(inside)
        {
            for(int ic = 0;ic < in_c;++ic)
            {
                const T* in_base = in+static_cast<size_t>(ic)*in_img_size+
                                   static_cast<size_t>(start_sz)*in_plane+
                                   static_cast<size_t>(start_sy)*in_w+start_sx;
                const T* weight_ic = weight_oc+ic*kernel_size3;

                #pragma unroll
                for(int kz = 0;kz < kernel_size;++kz)
                {
                    const T* in_z = in_base+kz*in_plane;
                    const T* weight_z = weight_ic+kz*kernel_plane;

                    #pragma unroll
                    for(int ky = 0;ky < kernel_size;++ky)
                    {
                        const T* in_y = in_z+ky*in_w;
                        const T* weight_y = weight_z+ky*kernel_size;

                        #pragma unroll
                        for(int kx = 0;kx < kernel_size;++kx)
                            sum = multiply_add(weight_y[kx],in_y[kx],sum);
                    }
                }
            }
        }
        else
        {
            for(int ic = 0;ic < in_c;++ic)
            {
                const T* in_ic = in+static_cast<size_t>(ic)*in_img_size;
                const T* weight_ic = weight_oc+ic*kernel_size3;

                #pragma unroll
                for(int kz = 0;kz < kernel_size;++kz)
                {
                    int sz = start_sz+kz;
                    if(static_cast<unsigned int>(sz) >= static_cast<unsigned int>(in_d))
                        continue;

                    #pragma unroll
                    for(int ky = 0;ky < kernel_size;++ky)
                    {
                        int sy = start_sy+ky;
                        if(static_cast<unsigned int>(sy) >= static_cast<unsigned int>(in_h))
                            continue;

                        const T* in_y = in_ic+static_cast<size_t>(sz)*in_plane+static_cast<size_t>(sy)*in_w;
                        const T* weight_y = weight_ic+kz*kernel_plane+ky*kernel_size;

                        #pragma unroll
                        for(int kx = 0;kx < kernel_size;++kx)
                        {
                            int sx = start_sx+kx;
                            if(static_cast<unsigned int>(sx) < static_cast<unsigned int>(in_w))
                                sum = multiply_add(weight_y[kx],in_y[sx],sum);
                        }
                    }
                }
            }
        }
    }

    out[static_cast<size_t>(oc)*out_img_size+static_cast<size_t>(z)*out_plane+static_cast<size_t>(y)*out_w+x] =
        apply_activation<Act>(sum,slope);
}

template<typename T>
__global__ __launch_bounds__(128,2)
void conv_transpose_3d_kernel(const T* __restrict__ in,const T* __restrict__ weight,const T* __restrict__ bias,T* __restrict__ out,
                              int in_c,int out_c,
                              int in_d,int in_h,int in_w,
                              int out_d,int out_h,int out_w)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y_blocks = (out_h+blockDim.y-1)/blockDim.y;
    int y_tile = blockIdx.y;
    int y0 = (y_tile%y_blocks)*blockDim.y;
    int z = y_tile/y_blocks;
    int y = y0+threadIdx.y;
    int oc = blockIdx.z;

    if(x >= out_w || y >= out_h)
        return;

    int in_x = x >> 1;
    int in_y = y >> 1;
    int in_z = z >> 1;
    int kx = x & 1;
    int ky = y & 1;
    int kz = z & 1;

    int in_plane = in_w*in_h;
    int in_img_size = in_d*in_plane;
    int out_plane = out_w*out_h;
    int out_img_size = out_d*out_plane;
    int k_offset = (kz << 2)+(ky << 1)+kx;

    T sum = bias[oc];
    const T* in_ptr = in+static_cast<size_t>(in_z)*in_plane+static_cast<size_t>(in_y)*in_w+in_x;
    const T* weight_ptr = weight+static_cast<size_t>(oc)*8+k_offset;
    int w_stride = out_c*8;

    for(int ic = 0;ic < in_c;++ic,in_ptr += in_img_size,weight_ptr += w_stride)
        sum = multiply_add(*in_ptr,*weight_ptr,sum);

    out[static_cast<size_t>(oc)*out_img_size+static_cast<size_t>(z)*out_plane+static_cast<size_t>(y)*out_w+x] = sum;
}

template<activation_type Act,typename T>
__global__ __launch_bounds__(256,2)
void batch_norm_3d_kernel(const T* __restrict__ in,T* __restrict__ out,const T* __restrict__ weight,const T* __restrict__ bias,
                          int in_c,size_t plane_size)
{
    size_t i = static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    int c = blockIdx.y;

    if(i >= plane_size)
        return;

    size_t pos = static_cast<size_t>(c)*plane_size+i;
    T val = multiply_add(in[pos],weight[c],bias[c]);

    out[pos] = apply_activation<Act>(val,(T)0.01f);
}

template<activation_type Act,typename T>
__global__ __launch_bounds__(256,1)
void instance_norm_3d_kernel(const T* __restrict__ in,T* __restrict__ out,const T* __restrict__ weight,const T* __restrict__ bias,
                             int out_c,size_t plane_size,T slope)
{
    int oc = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ T sum_buffer[norm_block_size];
    __shared__ T sq_buffer[norm_block_size];

    const T* in_ptr = in+static_cast<size_t>(oc)*plane_size;
    T* out_ptr = out+static_cast<size_t>(oc)*plane_size;

    T sum = T(0);
    T sq_sum = T(0);

    for(size_t i = tid;i < plane_size;i += blockDim.x)
    {
        T val = in_ptr[i];
        sum += val;
        sq_sum += val*val;
    }

    sum_buffer[tid] = sum;
    sq_buffer[tid] = sq_sum;
    __syncthreads();

    for(int offset = blockDim.x >> 1;offset;offset >>= 1)
    {
        if(tid < offset)
        {
            sum_buffer[tid] += sum_buffer[tid+offset];
            sq_buffer[tid] += sq_buffer[tid+offset];
        }
        __syncthreads();
    }

    T inv_plane = T(1)/static_cast<T>(plane_size);
    T mean = sum_buffer[0]*inv_plane;
    T var = sq_buffer[0]*inv_plane-mean*mean;
    if(var < T(0))
        var = T(0);

    T scale = weight[oc]/(T)sqrtf((float)var+1e-5f);
    T shift = bias[oc]-mean*scale;

    for(size_t i = tid;i < plane_size;i += blockDim.x)
    {
        T val = multiply_add(in_ptr[i],scale,shift);
        out_ptr[i] = apply_activation<Act>(val,slope);
    }
}

template<typename T>
__global__ __launch_bounds__(128,2)
void max_pool_3d_kernel(const T* __restrict__ in,T* __restrict__ out,
                        int in_c,int in_d,int in_h,int in_w,
                        int out_d,int out_h,int out_w)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y_blocks = (out_h+blockDim.y-1)/blockDim.y;
    int y_tile = blockIdx.y;
    int y0 = (y_tile%y_blocks)*blockDim.y;
    int z = y_tile/y_blocks;
    int y = y0+threadIdx.y;
    int c = blockIdx.z;

    if(x >= out_w || y >= out_h)
        return;

    int in_plane = in_w*in_h;
    int in_img_size = in_d*in_plane;
    int out_plane = out_w*out_h;
    int out_img_size = out_d*out_plane;

    int sx = x << 1;
    int sy = y << 1;
    int sz = z << 1;

    const T* p = in+static_cast<size_t>(c)*in_img_size+
                 static_cast<size_t>(sz)*in_plane+
                 static_cast<size_t>(sy)*in_w+sx;

    T max_val = p[0];
    max_val = max_val > p[1] ? max_val : p[1];
    max_val = max_val > p[in_w] ? max_val : p[in_w];
    max_val = max_val > p[in_w+1] ? max_val : p[in_w+1];

    p += in_plane;
    max_val = max_val > p[0] ? max_val : p[0];
    max_val = max_val > p[1] ? max_val : p[1];
    max_val = max_val > p[in_w] ? max_val : p[in_w];
    max_val = max_val > p[in_w+1] ? max_val : p[in_w+1];

    out[static_cast<size_t>(c)*out_img_size+static_cast<size_t>(z)*out_plane+static_cast<size_t>(y)*out_w+x] = max_val;
}

template<typename T>
__global__ __launch_bounds__(128,2)
void upsample_3d_kernel(const T* __restrict__ in,T* __restrict__ out,
                        int in_c,int in_d,int in_h,int in_w,
                        int out_d,int out_h,int out_w)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y_blocks = (in_h+blockDim.y-1)/blockDim.y;
    int y_tile = blockIdx.y;
    int y0 = (y_tile%y_blocks)*blockDim.y;
    int z = y_tile/y_blocks;
    int y = y0+threadIdx.y;
    int c = blockIdx.z;

    if(x >= in_w || y >= in_h)
        return;

    int in_plane = in_w*in_h;
    int in_img_size = in_d*in_plane;
    int out_plane = out_w*out_h;
    int out_img_size = out_d*out_plane;

    T val = in[static_cast<size_t>(c)*in_img_size+static_cast<size_t>(z)*in_plane+static_cast<size_t>(y)*in_w+x];

    int ox = x << 1;
    int oy = y << 1;
    int oz = z << 1;

    T* out_ptr = out+static_cast<size_t>(c)*out_img_size+
                 static_cast<size_t>(oz)*out_plane+
                 static_cast<size_t>(oy)*out_w+ox;

    out_ptr[0] = val;
    out_ptr[1] = val;
    out_ptr[out_w] = val;
    out_ptr[out_w+1] = val;

    out_ptr += out_plane;
    out_ptr[0] = val;
    out_ptr[1] = val;
    out_ptr[out_w] = val;
    out_ptr[out_w+1] = val;
}

} // namespace cuda_kernels

template <activation_type Act, typename T>
void cuda_conv_3d_forward(const T* in, const T* weight, const T* bias, T* out,
                          int in_c, int out_c,
                          int in_d, int in_h, int in_w,
                          int out_d, int out_h, int out_w,
                          int kernel_size, int kernel_size3, int range, int stride, T slope)
{
    dim3 block(cuda_kernels::spatial_block_x,cuda_kernels::spatial_block_y,1);
    dim3 grid(cuda_kernels::div_up(out_w,block.x),cuda_kernels::div_up(out_h,block.y)*out_d,out_c);

    if(kernel_size == 1 && stride == 1)
        cuda_kernels::conv_3d_kernel<Act,T,1,1><<<grid,block>>>(
            in,weight,bias,out,in_c,out_c,in_d,in_h,in_w,out_d,out_h,out_w,slope);
    else if(kernel_size == 3 && stride == 1)
        cuda_kernels::conv_3d_kernel<Act,T,3,1><<<grid,block>>>(
            in,weight,bias,out,in_c,out_c,in_d,in_h,in_w,out_d,out_h,out_w,slope);
    else if(kernel_size == 3 && stride == 2)
        cuda_kernels::conv_3d_kernel<Act,T,3,2><<<grid,block>>>(
            in,weight,bias,out,in_c,out_c,in_d,in_h,in_w,out_d,out_h,out_w,slope);
    else
        throw std::runtime_error("cuda conv_3d supports only ks1 stride1, ks3 stride1, and ks3 stride2");
}

template
void cuda_conv_3d_forward<activation_type::none, float>(
    const float* in, const float* weight, const float* bias, float* out,
    int in_c, int out_c,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int kernel_size3, int range, int stride, float slope);

template
void cuda_conv_3d_forward<activation_type::relu, float>(
    const float* in, const float* weight, const float* bias, float* out,
    int in_c, int out_c,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int kernel_size3, int range, int stride, float slope);

template
void cuda_conv_3d_forward<activation_type::leaky_relu, float>(
    const float* in, const float* weight, const float* bias, float* out,
    int in_c, int out_c,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int kernel_size3, int range, int stride, float slope);

template
void cuda_conv_3d_forward<activation_type::elu, float>(
    const float* in, const float* weight, const float* bias, float* out,
    int in_c, int out_c,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int kernel_size3, int range, int stride, float slope);

template <typename T>
void cuda_conv_transpose_3d_forward(const T* in, const T* weight, const T* bias, T* out,
                                    int in_c, int out_c,
                                    int in_d, int in_h, int in_w,
                                    int out_d, int out_h, int out_w,
                                    int kernel_size, int kernel_size3, int stride)
{
    if(kernel_size != 2 || stride != 2)
        throw std::runtime_error("cuda conv_transpose_3d supports only ks2 stride2");

    dim3 block(cuda_kernels::spatial_block_x,cuda_kernels::spatial_block_y,1);
    dim3 grid(cuda_kernels::div_up(out_w,block.x),cuda_kernels::div_up(out_h,block.y)*out_d,out_c);
    cuda_kernels::conv_transpose_3d_kernel<T><<<grid,block>>>(
        in,weight,bias,out,in_c,out_c,in_d,in_h,in_w,out_d,out_h,out_w);
}

template
void cuda_conv_transpose_3d_forward<float>(const float* in, const float* weight, const float* bias, float* out,
                                    int in_c, int out_c, int in_d, int in_h, int in_w, int out_d, int out_h, int out_w,
                                    int kernel_size, int kernel_size3, int stride);


template<activation_type Act,typename T>
void cuda_batch_norm_3d_forward(const T* in,T* out,const T* weight,const T* bias,
                                int in_c,size_t plane_size)
{
    dim3 block(cuda_kernels::elem_block_size);
    dim3 grid(cuda_kernels::div_up(plane_size,block.x),in_c);
    cuda_kernels::batch_norm_3d_kernel<Act,T><<<grid,block>>>(in,out,weight,bias,in_c,plane_size);
}

template void cuda_batch_norm_3d_forward<activation_type::none,float>(const float* in,float* out,const float* weight,const float* bias,int in_c,size_t plane_size);
template void cuda_batch_norm_3d_forward<activation_type::relu,float>(const float* in,float* out,const float* weight,const float* bias,int in_c,size_t plane_size);
template void cuda_batch_norm_3d_forward<activation_type::leaky_relu,float>(const float* in,float* out,const float* weight,const float* bias,int in_c,size_t plane_size);
template void cuda_batch_norm_3d_forward<activation_type::elu,float>(const float* in,float* out,const float* weight,const float* bias,int in_c,size_t plane_size);

template <activation_type Act, typename T>
void cuda_instance_norm_3d_forward(const T* in, T* out, const T* weight, const T* bias,
                                   int out_c, size_t plane_size, T slope)
{
    cuda_kernels::instance_norm_3d_kernel<Act,T><<<out_c,cuda_kernels::norm_block_size>>>(
        in,out,weight,bias,out_c,plane_size,slope);
}

template
void cuda_instance_norm_3d_forward<activation_type::none, float>(const float* in, float* out, const float* weight, const float* bias,
                                   int out_c, size_t plane_size, float slope);

template
void cuda_instance_norm_3d_forward<activation_type::relu, float>(const float* in, float* out, const float* weight, const float* bias,
                                   int out_c, size_t plane_size, float slope);

template
void cuda_instance_norm_3d_forward<activation_type::leaky_relu, float>(const float* in, float* out, const float* weight, const float* bias,
                                   int out_c, size_t plane_size, float slope);
template
void cuda_instance_norm_3d_forward<activation_type::elu, float>(const float* in, float* out, const float* weight, const float* bias,
                                   int out_c, size_t plane_size, float slope);

template <typename T>
void cuda_max_pool_3d_forward(const T* in, T* out,
                              int in_c, int in_d, int in_h, int in_w,
                              int out_d, int out_h, int out_w, int pool_size)
{
    if(pool_size != 2)
        throw std::runtime_error("cuda max_pool_3d supports only pool size 2");

    dim3 block(cuda_kernels::spatial_block_x,cuda_kernels::spatial_block_y,1);
    dim3 grid(cuda_kernels::div_up(out_w,block.x),cuda_kernels::div_up(out_h,block.y)*out_d,in_c);
    cuda_kernels::max_pool_3d_kernel<T><<<grid,block>>>(
        in,out,in_c,in_d,in_h,in_w,out_d,out_h,out_w);
}

template
void cuda_max_pool_3d_forward<float>(const float* in, float* out,
                              int in_c, int in_d, int in_h, int in_w,
                              int out_d, int out_h, int out_w, int pool_size);


template <typename T>
void cuda_upsample_3d_forward(const T* in, T* out,
                              int in_c, int in_d, int in_h, int in_w,
                              int out_d, int out_h, int out_w, int pool_size)
{
    if(pool_size != 2)
        throw std::runtime_error("cuda upsample_3d supports only pool size 2");

    dim3 block(cuda_kernels::spatial_block_x,cuda_kernels::spatial_block_y,1);
    dim3 grid(cuda_kernels::div_up(in_w,block.x),cuda_kernels::div_up(in_h,block.y)*in_d,in_c);
    cuda_kernels::upsample_3d_kernel<T><<<grid,block>>>(
        in,out,in_c,in_d,in_h,in_w,out_d,out_h,out_w);
}

template
void cuda_upsample_3d_forward<float>(const float* in, float* out,
                              int in_c, int in_d, int in_h, int in_w,
                              int out_d, int out_h, int out_w, int pool_size);


template <typename T = float>
void cuda_copy_device_to_device(T* dest, const T* src, size_t count)
{
    cudaMemcpy(dest, src, count * sizeof(T), cudaMemcpyDeviceToDevice);
}

template
void cuda_copy_device_to_device<float>(float* dest, const float* src, size_t count);

#endif // __CUDACC__




} // namespace ml3d
} // namespace tipl

#endif // CNN3D_DETAIL_HPP
