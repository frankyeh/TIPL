#ifndef CNN3D_HPP
#define CNN3D_HPP
#include <vector>
#include <memory>
#include <limits>
#include <type_traits>
#include "../utility/pixel_index.hpp"
#include "../utility/basic_image.hpp"
#include "../numerical/statistics.hpp"
#include "../mt.hpp"

namespace tipl
{
template<typename T> class device_vector;

namespace ml3d
{
namespace detail
{
template<typename T> struct is_device : std::false_type {};
template<typename T> struct is_device<tipl::device_vector<T>> : std::true_type {};
}

template<typename vector_type>
class layer
{
public:
    static constexpr bool is_gpu = detail::is_device<vector_type>::value;
    int in_channels_ = 1;
    int out_channels_ = 1;
    tipl::shape<3> dim;
public:
    layer(int channels_v):in_channels_(channels_v),out_channels_(channels_v)
    {}
    layer(int in_channels_v,int out_channels_v):in_channels_(in_channels_v),out_channels_(out_channels_v)
    {}
    virtual ~layer()
    {}
    virtual std::vector<std::pair<float*,size_t>> parameters()
    {
        return std::vector<std::pair<float*,size_t>>();
    }
    virtual void init_image(tipl::shape<3>& dim_)
    {
        dim = dim_;
    }
    virtual float* forward(float* in_ptr) = 0;
    virtual void print(std::ostream& out) const = 0;
    virtual size_t in_size() const
    {
        return dim.size()*in_channels_;
    }
    virtual size_t out_size() const
    {
        return dim.size()*out_channels_;
    }
};

template<typename vector_type>
class relu : public layer<vector_type>
{
private:
    size_t output_size_ = 0;
public:
    using layer<vector_type>::is_gpu;
    using layer<vector_type>::dim;
    using layer<vector_type>::out_channels_;

    relu(int channels_v):layer<vector_type>(channels_v)
    {}
    void init_image(tipl::shape<3>& dim_) override
    {
        dim = dim_;
        output_size_ = dim.size()*size_t(out_channels_);
    }
    float* forward(float* in_ptr) override
    {
        if constexpr(is_gpu)
        {
#ifdef __CUDACC__
            tipl::cuda_for(output_size_,[=] __device__ (size_t i)
            {
                if(in_ptr[i] < 0.0f)
                    in_ptr[i] = 0.0f;
            });
#endif
        }
        else
        {
            for(size_t i = 0;i < output_size_;++i)
                if(in_ptr[i] < 0.0f)
                    in_ptr[i] = 0.0f;
        }
        return in_ptr;
    }
    void print(std::ostream& out) const override
    {
        out << "relu " << out_channels_ << std::endl;
    }
    size_t in_size() const override
    {
        return output_size_;
    }
    size_t out_size() const override
    {
        return output_size_;
    }
};

template<typename vector_type>
class leakyrelu : public layer<vector_type>
{
private:
    size_t output_size_ = 0;
    float slope = 1e-2f;
public:
    using layer<vector_type>::is_gpu;
    using layer<vector_type>::dim;
    using layer<vector_type>::out_channels_;

    leakyrelu(int channels_v,float slope_ = 1e-2f):layer<vector_type>(channels_v),slope(slope_)
    {}
    void init_image(tipl::shape<3>& dim_) override
    {
        dim = dim_;
        output_size_ = dim.size()*size_t(out_channels_);
    }
    float* forward(float* in_ptr) override
    {
        if constexpr(is_gpu)
        {
#ifdef __CUDACC__
            float s = slope;
            tipl::cuda_for(output_size_,[=] __device__ (size_t i)
            {
                if(in_ptr[i] < 0.0f)
                    in_ptr[i] *= s;
            });
#endif
        }
        else
        {
            for(size_t i = 0;i < output_size_;++i)
                if(in_ptr[i] < 0.0f)
                     in_ptr[i] *= slope;
        }
        return in_ptr;
    }
    void print(std::ostream& out) const override
    {
        out << "leaky relu " << out_channels_ << std::endl;
    }
    size_t in_size() const override
    {
        return output_size_;
    }
    size_t out_size() const override
    {
        return output_size_;
    }
};

template<typename vector_type>
class conv_3d : public layer<vector_type>
{
private:
    int kernel_size_,kernel_size3,range;
    std::vector<int> kernel_shift;
public:
    using layer<vector_type>::is_gpu;
    using layer<vector_type>::dim;
    using layer<vector_type>::in_channels_;
    using layer<vector_type>::out_channels_;
    vector_type weight,bias,out;

    conv_3d(int in_channels_v,int out_channels_v,int kernel_size_v = 3):
        layer<vector_type>(in_channels_v,out_channels_v),kernel_size_(kernel_size_v)
    {
        kernel_size3 = kernel_size_*kernel_size_*kernel_size_;
        range = (kernel_size_-1)/2;
        weight.resize(kernel_size3*in_channels_*out_channels_);
        bias.resize(out_channels_);
    }
    std::vector<std::pair<float*,size_t>> parameters() override
    {
        std::vector<std::pair<float*,size_t>> params;
        params.push_back(std::make_pair(weight.data(),weight.size()));
        params.push_back(std::make_pair(bias.data(),bias.size()));
        return params;
    }
    void init_image(tipl::shape<3>& dim_) override
    {
        dim = dim_;
        if constexpr(!is_gpu)
        {
            kernel_shift.resize(kernel_size3);
            for(int kz = -range,index = 0;kz <= range;++kz)
                for(int ky = -range;ky <= range;++ky)
                    for(int kx = -range;kx <= range;++kx,++index)
                        kernel_shift[index] = (kz*int(dim.height())+ky)*int(dim.width())+kx;
        }
        out.resize(dim.size()*out_channels_);
    }
    float* forward(float* in) override
    {
        size_t image_size = dim.size();

        if constexpr(is_gpu)
        {
#ifdef __CUDACC__
            int out_c = out_channels_;
            int in_c = in_channels_;
            int r = range;
            int kz3 = kernel_size3;
            int dim_w = dim.width();
            int dim_h = dim.height();
            int dim_d = dim.depth();
            int p_size = dim.plane_size();
            int i_size = dim.size();
            auto w_ptr = weight.data();
            auto b_ptr = bias.data();
            auto out_ptr = out.data();

            tipl::cuda_for(out.size(),[=] __device__ (size_t i)
            {
                int c = i/i_size;
                int rem = i%i_size;
                int z = rem/p_size;
                int y = (rem%p_size)/dim_w;
                int x = rem%dim_w;

                float sum = b_ptr[c];
                const float* w = w_ptr+c*in_c*kz3;

                for(int inc = 0;inc < in_c;++inc,w += kz3)
                {
                    const float* in_c_ptr = in+inc*i_size;
                    int k_idx = 0;
                    for(int dz = -r;dz <= r;++dz)
                    {
                        int nz = z+dz;
                        if(nz < 0 || nz >= dim_d)
                        {
                            k_idx += (2*r+1)*(2*r+1);
                            continue;
                        }
                        for(int dy = -r;dy <= r;++dy)
                        {
                            int ny = y+dy;
                            if(ny < 0 || ny >= dim_h)
                            {
                                k_idx += (2*r+1);
                                continue;
                            }
                            for(int dx = -r;dx <= r;++dx,++k_idx)
                            {
                                int nx = x+dx;
                                if(nx >= 0 && nx < dim_w)
                                    sum += in_c_ptr[nz*p_size+ny*dim_w+nx]*w[k_idx];
                            }
                        }
                    }
                }
                out_ptr[i] = sum;
            });
#endif
        }
        else
        {
            size_t plane_size = dim.plane_size();
            size_t image_width = dim.width();
            size_t thread_count = tipl::max_thread_count;
            size_t thread_plane_size = dim.plane_size()*thread_count;

            tipl::par_for(thread_count,[=](size_t thread)
            {
                size_t thread_base = plane_size*thread;
                auto out_ptr = out.data()+thread_base;
                auto w_ptr = weight.data();
                for(size_t outc = 0;outc < out_channels_;++outc,out_ptr += image_size)
                {
                    {
                        auto out_plane = out_ptr;
                        for(int z = thread;z < dim[2];z += thread_count,out_plane += thread_plane_size)
                            std::fill(out_plane,out_plane+plane_size,bias[outc]);
                    }
                    auto in_ptr = in+thread_base;
                    for(int inc = 0;inc < in_channels_;++inc,in_ptr += image_size,w_ptr += kernel_size3)
                    {
                        for(int kz = -range,index = 0;kz <= range;++kz)
                            for(int ky = -range;ky <= range;++ky)
                                for(int kx = -range;kx <= range;++kx,++index)
                                {
                                    float w = w_ptr[index];
                                    auto in_plane = in_ptr+kernel_shift[index];
                                    auto out_plane = out_ptr;
                                    auto max_z = kz > 0 ? dim[2]-kz : dim[2];
                                    auto max_y = ky > 0 ? dim[1]-ky : dim[1];
                                    auto max_x = kx > 0 ? dim[0]-kx : dim[0];
                                    for(int z = thread;z < max_z;z += thread_count,in_plane += thread_plane_size,out_plane += thread_plane_size)
                                    {
                                        if(z < -kz)
                                            continue;
                                        auto in_line = in_plane;
                                        auto out_line = out_plane;
                                        for(int y = 0;y < max_y;++y,in_line += image_width,out_line += image_width)
                                        {
                                            if(y < -ky)
                                                continue;
                                            int pad = std::max<int>(0,-kx);
                                            tipl::vec::axpy(out_line+pad,out_line+max_x,w,in_line+pad);
                                        }
                                    }
                                }
                    }
                }
            },thread_count);
        }
        return out.data();
    }
    void print(std::ostream& out) const override
    {
        out << "conv3d " << in_channels_ << " " << out_channels_ << std::endl;
    }
};

template<typename vector_type>
class instance_norm_3d : public layer<vector_type>
{
public:
    using layer<vector_type>::is_gpu;
    using layer<vector_type>::dim;
    using layer<vector_type>::out_channels_;
    vector_type weight,bias;

    instance_norm_3d(int channels_v):layer<vector_type>(channels_v)
    {
        weight.resize(out_channels_);
        bias.resize(out_channels_);
    }
    std::vector<std::pair<float*,size_t>> parameters() override
    {
        std::vector<std::pair<float*,size_t>> params;
        params.push_back(std::make_pair(weight.data(),weight.size()));
        params.push_back(std::make_pair(bias.data(),bias.size()));
        return params;
    }
    float* forward(float* in) override
    {
        if constexpr(is_gpu)
        {
#ifdef __CUDACC__
            int i_size = dim.size();
            auto w_ptr = weight.data();
            auto b_ptr = bias.data();

            tipl::cuda_for(out_channels_,[=] __device__ (size_t outc)
            {
                float* in_c = in+i_size*outc;
                float m = 0.0f;
                for(size_t i = 0;i < i_size;++i)
                    m += in_c[i];
                m /= i_size;

                float var = 0.0f;
                for(size_t i = 0;i < i_size;++i)
                {
                    float diff = in_c[i]-m;
                    var += diff*diff;
                }
                var /= i_size;
                float sd = sqrt(var+1e-5f);
                float v = w_ptr[outc]/sd;
                float b = b_ptr[outc];

                for(size_t i = 0;i < i_size;++i)
                    in_c[i] = (in_c[i]-m)*v+b;
            });
#endif
        }
        else
        {
            tipl::par_for(out_channels_,[&](size_t outc)
            {
                auto in_ptr = in+dim.size()*outc;
                auto in_ptr_end = in_ptr+dim.size();
                auto m = tipl::mean(in_ptr,in_ptr_end);
                auto sd = tipl::standard_deviation(in_ptr,in_ptr_end);
                auto v = (sd == 0.0f ? 0.0f : weight[outc]/sd);
                auto b = bias[outc];
                for(size_t i = 0;i < dim.size();++i)
                    in_ptr[i] = (in_ptr[i]-m)*v+b;
            },out_channels_);
        }
        return in;
    }
    void print(std::ostream& out) const override
    {
        out << "instance_norm_3d " << out_channels_ << std::endl;
    }
};

template<typename vector_type>
class max_pool_3d : public layer<vector_type>
{
public:
    using layer<vector_type>::is_gpu;
    using layer<vector_type>::dim;
    using layer<vector_type>::out_channels_;
    tipl::shape<3> out_dim;
    vector_type out;
    int pool_size = 2;

    max_pool_3d(int channels_v):layer<vector_type>(channels_v)
    {}
    void init_image(tipl::shape<3>& dim_) override
    {
        dim = dim_;
        out_dim = tipl::shape<3>(dim[0]/pool_size,dim[1]/pool_size,dim[2]/pool_size);
        this->dim = out_dim;
        out.resize(out_dim.size()*out_channels_);
    }
    float* forward(float* in) override
    {
        if constexpr(is_gpu)
        {
#ifdef __CUDACC__
            int p_size = pool_size;
            int dim_w = dim.width();
            int dim_h = dim.height();
            int dim_d = dim.depth();
            int out_w = out_dim.width();
            int out_h = out_dim.height();
            int o_plane = out_w*out_h;
            int o_size = out_dim.size();
            int i_size = dim.size();
            auto out_ptr = out.data();

            tipl::cuda_for(out.size(),[=] __device__ (size_t i)
            {
                int c = i/o_size;
                int rem = i%o_size;
                int z = rem/o_plane;
                int y = (rem%o_plane)/out_w;
                int x = rem%out_w;

                float max_value = -1e38f;
                float* in_c = in+c*i_size;

                for(int dz = 0;dz < p_size;++dz)
                {
                    int fz = z*p_size+dz;
                    if(fz >= dim_d)
                        continue;
                    for(int dy = 0;dy < p_size;++dy)
                    {
                        int fy = y*p_size+dy;
                        if(fy >= dim_h)
                            continue;
                        for(int dx = 0;dx < p_size;++dx)
                        {
                            int fx = x*p_size+dx;
                            if(fx >= dim_w)
                                continue;
                            float val = in_c[(fz*dim_h+fy)*dim_w+fx];
                            if(val > max_value)
                                max_value = val;
                        }
                    }
                }
                out_ptr[i] = max_value;
            });
#endif
        }
        else
        {
            tipl::par_for(out_channels_,[&](size_t c)
            {
                auto in_ptr = in+c*dim.size();
                auto out_ptr = out.data()+c*out_dim.size();
                for(int z = 0,out_index = 0;z < out_dim.depth();++z)
                    for(int y = 0;y < out_dim.height();++y)
                        for(int x = 0;x < out_dim.width();++x,++out_index)
                        {
                            float max_value = std::numeric_limits<float>::lowest();
                            for(int dz = 0;dz < pool_size;++dz)
                            {
                                int from_z = z*pool_size+dz;
                                if(from_z >= dim.depth())
                                    continue;
                                for(int dy = 0;dy < pool_size;++dy)
                                {
                                    int from_y = y*pool_size+dy;
                                    if(from_y >= dim.height())
                                        continue;
                                    for(int dx = 0;dx < pool_size;++dx)
                                    {
                                        int from_x = x*pool_size+dx;
                                        if(from_x >= dim.width())
                                            continue;
                                        int in_index = (from_z*dim.height()+from_y)*dim.width()+from_x;
                                        if(in_ptr[in_index] > max_value)
                                            max_value = in_ptr[in_index];
                                    }
                                }
                            }
                            out_ptr[out_index] = max_value;
                        }
            },out_channels_);
        }
        return out.data();
    }
    void print(std::ostream& out) const override
    {
        out << "max_pool_3d " << out_channels_ << std::endl;
    }
    size_t out_size() const override
    {
        return out_dim.size()*out_channels_;
    }
};

template<typename vector_type>
class upsample_3d : public layer<vector_type>
{
public:
    using layer<vector_type>::is_gpu;
    using layer<vector_type>::dim;
    using layer<vector_type>::out_channels_;
    tipl::shape<3> out_dim;
    vector_type out;
    int pool_size = 2;

    upsample_3d(int channels_v):layer<vector_type>(channels_v)
    {}
    void init_image(tipl::shape<3>& dim_) override
    {
        dim = dim_;
        out_dim = tipl::shape<3>(dim[0]*pool_size,dim[1]*pool_size,dim[2]*pool_size);
        this->dim = out_dim;
        out.resize(out_dim.size()*out_channels_);
    }
    float* forward(float* in) override
    {
        if constexpr(is_gpu)
        {
#ifdef __CUDACC__
            int p_size = pool_size;
            int dim_w = dim.width();
            int dim_h = dim.height();
            int out_w = out_dim.width();
            int out_h = out_dim.height();
            int o_plane = out_w*out_h;
            int o_size = out_dim.size();
            int i_size = dim.size();
            auto out_ptr = out.data();

            tipl::cuda_for(out.size(),[=] __device__ (size_t i)
            {
                int c = i/o_size;
                int rem = i%o_size;
                int z = rem/o_plane;
                int y = (rem%o_plane)/out_w;
                int x = rem%out_w;

                int fz = z/p_size;
                int fy = y/p_size;
                int fx = x/p_size;

                out_ptr[i] = in[c*i_size+(fz*dim_h+fy)*dim_w+fx];
            });
#endif
        }
        else
        {
            tipl::par_for(out_channels_,[&](size_t c)
            {
                auto in_ptr = in+c*dim.size();
                auto out_ptr = out.data()+c*out_dim.size();
                for(int z = 0,out_index = 0;z < out_dim.depth();++z)
                {
                    int from_z = z/pool_size;
                    for(int y = 0;y < out_dim.height();++y)
                    {
                        int from_y = y/pool_size;
                        for(int x = 0;x < out_dim.width();++x,++out_index)
                        {
                            int from_x = x/pool_size;
                            int in_index = (from_z*dim.height()+from_y)*dim.width()+from_x;
                            out_ptr[out_index] = in_ptr[in_index];
                        }
                    }
                }
            },out_channels_);
        }
        return out.data();
    }
    void print(std::ostream& out) const override
    {
        out << "upsample_3d " << out_channels_ << std::endl;
    }
    size_t out_size() const override
    {
        return out_dim.size()*out_channels_;
    }
};

template<typename vector_type>
class network : public layer<vector_type>
{
public:
    using layer<vector_type>::in_channels_;
    using layer<vector_type>::out_channels_;
    std::vector<std::shared_ptr<layer<vector_type>>> layers;

    network():layer<vector_type>(1,1)
    {}
    network& operator<<(layer<vector_type>* l)
    {
        layers.push_back(std::shared_ptr<layer<vector_type>>(l));
        in_channels_ = layers.front()->in_channels_;
        out_channels_ = layers.back()->out_channels_;
        return *this;
    }
    network& operator<<(std::shared_ptr<layer<vector_type>> l)
    {
        layers.push_back(l);
        in_channels_ = layers.front()->in_channels_;
        out_channels_ = layers.back()->out_channels_;
        return *this;
    }
    std::vector<std::pair<float*,size_t>> parameters() override
    {
        std::vector<std::pair<float*,size_t>> param;
        for(auto& each_layer : layers)
        {
            auto new_params = each_layer->parameters();
            param.insert(param.end(),new_params.begin(),new_params.end());
        }
        return param;
    }
    void init_image(tipl::shape<3>& dim) override
    {
        for(auto& each_layer : layers)
            each_layer->init_image(dim);
    }
    float* forward(float* in) override
    {
        for(auto& each_layer : layers)
            in = each_layer->forward(in);
        return in;
    }
    void print(std::ostream& out) const override
    {
        for(auto& each_layer : layers)
            each_layer->print(out);
    }
    size_t in_size() const override
    {
        return layers.front()->in_size();
    }
    size_t out_size() const override
    {
        return layers.back()->out_size();
    }
};

}//ml3d
}//tipl

#endif//CNN3D_HPP
