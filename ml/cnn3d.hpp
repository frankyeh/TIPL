#ifndef CNN3D_HPP
#define CNN3D_HPP
#include <vector>
#include <memory>
#include <limits>
#include "../utility/pixel_index.hpp"
#include "../utility/basic_image.hpp"
#include "../numerical/statistics.hpp"
#include "../mt.hpp"

namespace tipl
{
namespace ml3d
{

enum class activation_type {
    none,
    relu,
    leaky_relu
};

class layer
{
public:
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

template <activation_type Act = activation_type::none>
class conv_3d : public layer
{
private:
    int kernel_size_,kernel_size3,range;
    std::vector<int> kernel_shift;
    float slope_;
public:
    std::vector<float> weight,bias,out;

    conv_3d(int in_channels_v,int out_channels_v,int kernel_size_v = 3, float slope_v = 1e-2f):
        layer(in_channels_v,out_channels_v),kernel_size_(kernel_size_v),slope_(slope_v)
    {
        kernel_size3 = kernel_size_*kernel_size_*kernel_size_;
        range = (kernel_size_-1)/2;
        weight.resize(kernel_size3*in_channels_*out_channels_);
        bias.resize(out_channels_);
    }
    std::vector<std::pair<float*,size_t>> parameters() override
    {
        std::vector<std::pair<float*,size_t>> params;
        params.push_back(std::make_pair(&weight[0],weight.size()));
        params.push_back(std::make_pair(&bias[0],bias.size()));
        return params;
    }
    void init_image(tipl::shape<3>& dim_) override
    {
        dim = dim_;
        kernel_shift.resize(kernel_size3);
        for(int kz = -range,index = 0;kz <= range;++kz)
            for(int ky = -range;ky <= range;++ky)
                for(int kx = -range;kx <= range;++kx,++index)
                    kernel_shift[index] = (kz*int(dim.height())+ky)*int(dim.width())+kx;
        out.resize(dim.size()*out_channels_);
    }
    float* forward(float* in) override
    {
        const size_t img_size = dim.size();
        const int in_c = in_channels_;
        const int out_c = out_channels_;
        const int w = dim.width();
        const int h = dim.height();
        const int d = dim.depth();

        tipl::par_for(out_c, [&](size_t oc) {
            float* out_ptr = out.data() + oc * img_size;
            std::fill_n(out_ptr, img_size, bias[oc]);
        });

        tipl::par_for((size_t)out_c * d, [&](size_t job) {
            const int oc = job / d;
            const int z = job % d;

            float* out_slice = out.data() + (oc * img_size) + (z * w * h);
            const float* weight_oc = weight.data() + (oc * in_c * kernel_size3);

            for (int ic = 0; ic < in_c; ++ic) {
                const float* in_channel_ptr = in + (ic * img_size);
                const float* weight_ic = weight_oc + (ic * kernel_size3);

                for (int kz = -range; kz <= range; ++kz) {
                    int sz = z + kz;
                    if (sz < 0 || sz >= d) continue;

                    const float* in_slice = in_channel_ptr + (sz * w * h);
                    const float* weight_kz = weight_ic + (kz + range) * (kernel_size_ * kernel_size_);

                    for (int ky = -range; ky <= range; ++ky) {
                        const float* weight_ky = weight_kz + (ky + range) * kernel_size_;

                        for (int kx = -range; kx <= range; ++kx) {
                            float w_val = weight_ky[kx + range];
                            if (w_val == 0.0f) continue; // Sparsity optimization

                            // Process the row with optimized boundary logic
                            for (int y = 0; y < h; ++y) {
                                int sy = y + ky;
                                if (sy < 0 || sy >= h) continue;

                                const float* in_row = in_slice + (sy * w);
                                float* out_row = out_slice + (y * w);

                                // Inner-most loop: X-axis
                                for (int x = 0; x < w; ++x) {
                                    int sx = x + kx;
                                    if (sx >= 0 && sx < w) {
                                        out_row[x] += w_val * in_row[sx];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Apply Template Activation In-Place
            if constexpr (Act != activation_type::none) {
                const int slice_size = w * h;
                for (int i = 0; i < slice_size; ++i) {
                    if (out_slice[i] < 0.0f) {
                        if constexpr (Act == activation_type::relu) {
                            out_slice[i] = 0.0f;
                        } else if constexpr (Act == activation_type::leaky_relu) {
                            out_slice[i] *= slope_;
                        }
                    }
                }
            }
        });

        return out.data();
    }
    void print(std::ostream& out_stream) const override
    {
        out_stream << "conv3d ";
        if constexpr (Act == activation_type::relu) {
            out_stream << "(relu) ";
        } else if constexpr (Act == activation_type::leaky_relu) {
            out_stream << "(leaky_relu) ";
        }
        out_stream << in_channels_ << " " << out_channels_ << std::endl;
    }
};

class instance_norm_3d : public layer
{
public:
    std::vector<float> weight,bias;
public:
    instance_norm_3d(int channels_v):layer(channels_v)
    {
        weight.resize(out_channels_);
        bias.resize(out_channels_);
    }
    std::vector<std::pair<float*,size_t>> parameters() override
    {
        std::vector<std::pair<float*,size_t>> params;
        params.push_back(std::make_pair(&weight[0],weight.size()));
        params.push_back(std::make_pair(&bias[0],bias.size()));
        return params;
    }
    float* forward(float* in) override
    {
        const size_t plane_size = dim.size();
        tipl::par_for(out_channels_, [&](size_t outc)
        {
            float* in_ptr = in + plane_size * outc;

            double sum = 0.0;
            double sq_sum = 0.0;

            for (size_t i = 0; i < plane_size; ++i) {
                float val = in_ptr[i];
                sum += val;
                sq_sum += (double)val * val;
            }

            float mean = (float)(sum / plane_size);
            float variance = (float)(sq_sum / plane_size) - (mean * mean);
            if (variance < 0) variance = 0;

            // Pre-calculate scale and shift to minimize ops in the inner loop
            float inv_std = 1.0f / std::sqrt(variance + 1e-5f);
            float scale = inv_std * weight[outc];
            float shift = bias[outc] - (mean * scale);

            for (size_t i = 0; i < plane_size; ++i) {
                in_ptr[i] = in_ptr[i] * scale + shift;
            }
        }, out_channels_);

        return in;
    }
    void print(std::ostream& out) const override
    {
        out << "norm_3d " << out_channels_ << std::endl;
    }
};

class max_pool_3d : public layer
{
public:
    tipl::shape<3> out_dim;
    std::vector<float> out;
    int pool_size = 2;
public:
    max_pool_3d(int channels_v):layer(channels_v)
    {}
    void init_image(tipl::shape<3>& dim_) override
    {
        dim = dim_;
        out_dim = tipl::shape<3>(dim[0]/pool_size,dim[1]/pool_size,dim[2]/pool_size);
        dim_ = out_dim;
        out.resize(out_dim.size()*out_channels_);
    }
    float* forward(float* in) override
    {
        const int in_w = dim.width(), in_h = dim.height(), in_d = dim.depth();
        const int out_w = out_dim.width(), out_h = out_dim.height(), out_d = out_dim.depth();
        const size_t in_plane = (size_t)in_w * in_h;
        const size_t out_plane = (size_t)out_w * out_h;

        tipl::par_for((size_t)out_channels_ * out_d, [&](size_t i)
        {
            const int c = i / out_d;
            const int z = i % out_d;

            const float* in_ptr_c = in + (c * in_d * in_plane);
            float* out_ptr_slice = out.data() + (c * out_d * out_plane) + (z * out_plane);

            const int from_z_base = z * pool_size;

            for (int y = 0; y < out_h; ++y) {
                const int from_y_base = y * pool_size;

                for (int x = 0; x < out_w; ++x) {
                    const int from_x_base = x * pool_size;
                    float max_value = -std::numeric_limits<float>::max();

                    for (int dz = 0; dz < pool_size; ++dz) {
                        int sz = from_z_base + dz;
                        if (sz >= in_d) continue;

                        const float* in_ptr_slice = in_ptr_c + (sz * in_plane);

                        for (int dy = 0; dy < pool_size; ++dy) {
                            int sy = from_y_base + dy;
                            if (sy >= in_h) continue;

                            const float* in_ptr_row = in_ptr_slice + (sy * in_w);

                            for (int dx = 0; dx < pool_size; ++dx) {
                                int sx = from_x_base + dx;
                                if (sx >= in_w) continue;

                                float val = in_ptr_row[sx];
                                if (val > max_value) {
                                    max_value = val;
                                }
                            }
                        }
                    }
                    out_ptr_slice[y * out_w + x] = max_value;
                }
            }
        });
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

class upsample_3d : public layer
{
public:
    tipl::shape<3> out_dim;
    std::vector<float> out;
    int pool_size = 2;
public:
    upsample_3d(int channels_v):layer(channels_v)
    {}
    void init_image(tipl::shape<3>& dim_) override
    {
        dim = dim_;
        out_dim = tipl::shape<3>(dim[0]*pool_size,dim[1]*pool_size,dim[2]*pool_size);
        dim_ = out_dim;
        out.resize(out_dim.size()*out_channels_);
    }
    float* forward(float* in) override
    {
        const int in_w = dim.width(), in_h = dim.height(), in_d = dim.depth();
        const int out_w = out_dim.width(), out_h = out_dim.height();
        const size_t in_plane = (size_t)in_w * in_h;
        const size_t out_plane = (size_t)out_w * out_h;

        const size_t total_planes = (size_t)out_channels_ * in_d;

        tipl::par_for(total_planes, [&](size_t i)
        {
            size_t c = i / in_d;
            size_t z = i % in_d;

            float* in_ptr_plane = in + (c * in_d + z) * in_plane;
            float* out_ptr_base = out.data() + (c * out_dim.depth() + z * pool_size) * out_plane;

            for (int y = 0; y < in_h; ++y) {
                int out_y_start = y * pool_size;
                float* in_row = in_ptr_plane + (y * in_w);

                for (int x = 0; x < in_w; ++x) {
                    float val = in_row[x];
                    int out_x_start = x * pool_size;

                    for (int dz = 0; dz < pool_size; ++dz) {
                        float* out_line = out_ptr_base + (dz * out_plane) + (out_y_start * out_w);
                        std::fill_n(out_line + out_x_start, pool_size, val);

                        for(int dy = 1; dy < pool_size; ++dy) {
                            std::copy_n(out_line + out_x_start, pool_size,
                                       out_line + out_x_start + (dy * out_w));
                        }
                    }
                }
            }
        });
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

class network : public layer
{
public:
    network():layer(1,1)
    {}
    std::vector<std::shared_ptr<layer>> layers;

    network& operator<<(layer* l)
    {
        layers.push_back(std::shared_ptr<layer>(l));
        in_channels_ = layers.front()->in_channels_;
        out_channels_ = layers.back()->out_channels_;
        return *this;
    }
    network& operator<<(std::shared_ptr<layer> l)
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
    void init_image(tipl::shape<3>& dim_) override
    {
        dim = dim_;
        for(auto& each_layer : layers)
            each_layer->init_image(dim_);
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

template<typename prog_type = void>
class unet3d : public network {
    std::deque<std::shared_ptr<network> > encoding,decoding,up;
    std::shared_ptr<layer> output;
private:
    std::shared_ptr<layer> add_layer(layer* l)
    {
        std::shared_ptr<layer> new_layer(l);
        layers.push_back(new_layer);
        return new_layer;
    }
public:
    prog_type* prog = nullptr;
    unet3d(const std::vector<std::vector<int> >& features_down,
           const std::vector<std::vector<int> >& features_up,
           const std::vector<int>& kernel_size,
            int in_channels_v,int out_channels_v)
    {
        in_channels_ = in_channels_v;
        for(int level=0; level< features_down.size(); level++)
        {
            std::shared_ptr<network> n_en(new network);
            if(level)
                *n_en.get() << add_layer(new max_pool_3d(features_down[level][0]));
            add_conv_block(*n_en.get(),features_down[level],kernel_size[level]);
            encoding.push_back(n_en);
        }
        for(int level=features_down.size()-2; level>=0; level--)
        {
            std::shared_ptr<network> n_up(new network),n_de(new network);
            *n_up.get() << add_layer(new upsample_3d(features_up[level+1].back()));
            add_conv_block(*n_up.get(),{features_up[level+1].back(),features_down[level].back()},kernel_size[level]);
            add_conv_block(*n_de.get(),features_up[level],kernel_size[level]);

            up.push_front(n_up);
            decoding.push_front(n_de);
        }
        output = add_layer(new conv_3d<activation_type::none>(features_up[0].back(), out_channels_v, 1));
        out_channels_ = out_channels_v;
    }
    void add_conv_block(network& n,const std::vector<int>& rhs,size_t ks)
    {
        int count = 0;
        for(auto next_count : rhs)
        {
            if(count)
            {
                n << add_layer(new conv_3d<activation_type::leaky_relu>(count, next_count,ks));
                n << add_layer(new instance_norm_3d(next_count));
            }
            count = next_count;
        }
    }
    virtual void init_image(tipl::shape<3>& dim_)
    {
        dim = dim_;
        for(int level=0; level< encoding.size(); level++)
            encoding[level]->init_image(dim_);

        for(int level=encoding.size()-2; level>=0; level--)
        {
            up[level]->init_image(dim_);
            // create space for concatenation
            {
                auto conv = dynamic_cast<conv_3d<activation_type::leaky_relu>*>(encoding[level]->layers[encoding[level]->layers.size()-2].get());
                conv->out.resize(up[level]->out_size()+encoding[level]->out_size());
            }
            decoding[level]->init_image(dim_);
        }
        output->init_image(dim_);
    }
    virtual float* forward(float* in)
    {
        std::vector<float*> buf;
        for(int level=0; level < encoding.size(); level++)
        {
            if constexpr (!std::is_void_v<prog_type>) {
                if(prog && !(*prog)(int(level),int(encoding.size()*2)))
                    return nullptr;
            }
            buf.push_back(in = encoding[level]->forward(in));
        }
        for(int level=encoding.size()-2; level>=0; level--)
        {
            if constexpr (!std::is_void_v<prog_type>) {
                if(prog && !(*prog)(int(encoding.size()*2)-level,int(encoding.size()*2+1)))
                    return nullptr;
            }
            buf.pop_back();
            auto in2 = up[level]->forward(in);
            std::copy_n(in2,up[level]->out_size(),buf.back()+encoding[level]->out_size());
            in=decoding[level]->forward(buf.back());
        }
        return output->forward(in);
    }
};

}//ml
}//tipl

#endif//CNN3D_HPP
