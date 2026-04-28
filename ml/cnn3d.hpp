#ifndef CNN3D_HPP
#define CNN3D_HPP

#include <vector>
#include <memory>
#include <limits>
#include <cmath>
#include <deque>
#include <algorithm>
#include <functional>
#include <iostream>
#include "../utility/pixel_index.hpp"
#include "../utility/basic_image.hpp"
#include "../numerical/statistics.hpp"
#include "../mt.hpp"
#include "../def.hpp"
#include "../po.hpp"

namespace tipl {
namespace ml3d {

enum class activation_type { none, relu, leaky_relu };

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

class layer {
public:
    int in_channels_ = 1, out_channels_ = 1;
    size_t out_size = 0 , out_buffer_size = 0;
    tipl::shape<3> dim;
    float* out = nullptr;
    bool is_gpu = false;

    layer(int channels) : in_channels_(channels), out_channels_(channels) {}
    layer(int in_c, int out_c) : in_channels_(in_c), out_channels_(out_c) {}
    virtual ~layer() = default;

    virtual std::vector<std::pair<float*, size_t>> parameters() { return {}; }
    virtual void init_image(tipl::shape<3>& dim_)
    {
        dim = dim_;
        out_buffer_size = out_size = dim.size() * out_channels_;
    }
    virtual float* forward(float* in_ptr) = 0;
    virtual void print(std::ostream& out) const = 0;
    virtual void allocate(float*& ptr, bool is_gpu_mem)
    {
        is_gpu = is_gpu_mem;
        out = ptr; ptr += out_buffer_size;
    }
    virtual bool change_dim(void) const {return false;}
};

static constexpr const char* kernel_size_keyword = "ks";
static constexpr const char* stride_keyword = "stride";
static constexpr const char* leaky_relu_keyword = "leaky_relu";
static constexpr const char* relu_keyword = "relu";

template <activation_type Act = activation_type::none>
class conv_3d : public layer {
    int kernel_size_, kernel_size3, range, stride_;
public:
    static constexpr const char* keyword = "conv";
    float* weight = nullptr;
    float* bias = nullptr;
    size_t weight_size = 0, bias_size = 0;
    tipl::shape<3> out_dim;

    conv_3d(int in_c, int out_c, int ks = 3, int stride = 1)
        : layer(in_c, out_c), kernel_size_(ks), kernel_size3(ks * ks * ks), range((ks - 1) / 2),
          stride_(stride){
        weight_size = kernel_size3 * in_channels_ * out_channels_;
        bias_size = out_channels_;
    }

    std::vector<std::pair<float*, size_t>> parameters() override {
        return {{weight, weight_size}, {bias, bias_size}};

    }

    void init_image(tipl::shape<3>& dim_) override {
        dim = dim_;
        out_dim = dim_ = tipl::s(dim_[0]/stride_,dim_[1]/stride_,dim_[2]/stride_);
        out_buffer_size = out_size = out_dim.size() * out_channels_;
    }

    void allocate(float*& ptr, bool is_gpu_mem) override {
        weight = ptr; ptr += weight_size;
        bias = ptr; ptr += bias_size;
        layer::allocate(ptr,is_gpu_mem);
    }


    float* forward(float* in) override
    {
        if constexpr (tipl::use_cuda)
        {
            if (this->is_gpu) {
                cuda_conv_3d_forward<Act>(in, weight, bias, out, in_channels_, out_channels_,
                                          dim.depth(), dim.height(), dim.width(),
                                          out_dim.depth(), out_dim.height(), out_dim.width(),
                                          kernel_size_, kernel_size3, range, stride_, 0.01f);
                return out;
            }
        }

        const size_t in_img_size = dim.size(), out_img_size = out_dim.size();
        const int in_c = in_channels_, out_c = out_channels_;
        const int in_w = dim.width(), in_h = dim.height(), in_d = dim.depth();
        const int out_w = out_dim.width(), out_h = out_dim.height(), out_d = out_dim.depth();

        const int in_plane = in_w * in_h;
        const int out_plane = out_w * out_h;
        const int kernel_plane = kernel_size_ * kernel_size_;
        const int stride_in_w = stride_ * in_w;
        const int start_base_sy_in_w = -range * in_w; // Replaces ky * in_w

        tipl::par_for(static_cast<size_t>(out_c) * out_d, [&](size_t job) {
            const int oc = job / out_d, z = job % out_d;

            float* out_slice = out + (oc * out_img_size) + (z * out_plane);
            const float* weight_oc = weight + (oc * in_c * kernel_size3);

            float bias_val = bias[oc];
            std::fill_n(out_slice, out_plane, bias_val);

            const int start_sz = z * stride_ - range;
            const int start_slice_offset = start_sz * in_plane;

            const float* in_channel_ptr = in;
            const float* weight_ic = weight_oc;

            for (int ic = 0; ic < in_c; ++ic, in_channel_ptr += in_img_size, weight_ic += kernel_size3) {

                const float* weight_kz = weight_ic;
                int sz = start_sz;
                int slice_offset = start_slice_offset;

                for (int kz = -range; kz <= range; ++kz, weight_kz += kernel_plane, ++sz, slice_offset += in_plane) {

                    // Fast bounds check using unsigned cast
                    if (static_cast<unsigned int>(sz) >= static_cast<unsigned int>(in_d)) continue;

                    // Valid pointer addition using the safely maintained integer offset
                    const float* in_slice = in_channel_ptr + slice_offset;

                    const float* weight_ky = weight_kz;
                    int base_sy_in_w = start_base_sy_in_w;

                    // Replaced ky * in_w with additive base_sy_in_w += in_w
                    for (int ky = -range; ky <= range; ++ky, weight_ky += kernel_size_, base_sy_in_w += in_w) {

                        const float* weight_kx = weight_ky;

                        for (int kx = -range; kx <= range; ++kx, ++weight_kx) {
                            float w_val = *weight_kx;
                            if (w_val == 0.0f) continue;

                            int sy_in_w = base_sy_in_w;
                            int y_out_w = 0;

                            for (int y = 0, sy = ky; y < out_h; ++y, sy += stride_, sy_in_w += stride_in_w, y_out_w += out_w) {
                                if (static_cast<unsigned int>(sy) >= static_cast<unsigned int>(in_h)) continue;

                                const float* in_row = in_slice + sy_in_w;
                                float* out_row = out_slice + y_out_w;

                                for (int x = 0, sx = kx; x < out_w; ++x, sx += stride_) {
                                    if (static_cast<unsigned int>(sx) < static_cast<unsigned int>(in_w)) {
                                        out_row[x] += w_val * in_row[sx];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if constexpr (Act != activation_type::none) {
                for (int i = 0; i < out_plane; ++i) {
                    if (out_slice[i] < 0.0f) {
                        if constexpr (Act == activation_type::relu) out_slice[i] = 0.0f;
                        else if constexpr (Act == activation_type::leaky_relu) out_slice[i] *= 0.01f;
                    }
                }
            }
        });
        return out;
    }
    void print(std::ostream& os) const override
    {
        os << keyword << out_channels_ << "," << kernel_size_keyword << kernel_size_ << "," << stride_keyword << stride_;
        if(Act == activation_type::relu)
            os << "," << relu_keyword;
        if(Act == activation_type::leaky_relu)
            os << "," << leaky_relu_keyword;
    }
    bool change_dim(void) const override{return stride_ != 1;}
};

class conv_transpose_3d : public layer {
    int kernel_size_, kernel_size3, stride_;
public:
    static constexpr const char* keyword = "conv_trans";
    float* weight = nullptr;
    float* bias = nullptr;
    size_t weight_size = 0, bias_size = 0;
    tipl::shape<3> out_dim;

    conv_transpose_3d(int in_c, int out_c, int ks = 2, int stride = 2)
        : layer(in_c, out_c), kernel_size_(ks), kernel_size3(ks * ks * ks), stride_(stride){
        weight_size = kernel_size3 * in_channels_ * out_channels_;
        bias_size = out_channels_;
    }

    std::vector<std::pair<float*, size_t>> parameters() override {
        return {{weight, weight_size}, {bias, bias_size}};
    }

    void init_image(tipl::shape<3>& dim_) override {
        dim = dim_;
        out_dim = dim_ = tipl::s(dim_[0]*stride_,dim_[1]*stride_,dim_[2]*stride_);
        out_buffer_size = out_size = out_dim.size() * out_channels_;
    }

    void allocate(float*& ptr, bool is_gpu_mem) override {
        weight = ptr; ptr += weight_size;
        bias = ptr; ptr += bias_size;
        layer::allocate(ptr,is_gpu_mem);
    }

    float* forward(float* in) override
    {
        if constexpr (tipl::use_cuda)
            if (this->is_gpu)
            {
                cuda_conv_transpose_3d_forward(in, weight, bias, out, in_channels_, out_channels_,
                                               dim.depth(), dim.height(), dim.width(),
                                               out_dim.depth(), out_dim.height(), out_dim.width(),
                                               kernel_size_, kernel_size3, stride_);
                return out;
            }

        const size_t in_img_size = dim.size();
        const size_t out_img_size = out_dim.size();
        const int in_c = in_channels_, out_c = out_channels_;
        const int in_w = dim.width(), in_h = dim.height(), in_d = dim.depth();
        const int out_w = out_dim.width(), out_h = out_dim.height(), out_d = out_dim.depth();

        // --- PRECOMPUTE PLANES & STRIDES ONCE ---
        const int in_plane = in_w * in_h;
        const int out_plane = out_w * out_h;
        const int kernel_plane = kernel_size_ * kernel_size_;
        const int weight_ic_step = out_c * kernel_size3;

        tipl::par_for(static_cast<size_t>(out_c) * out_d, [&](size_t job)
        {
            const int oc = job / out_d;
            const int z = job % out_d;

            float* out_ptr = this->out + (oc * out_img_size) + (z * out_plane);

            int in_z = z / stride_;
            int kz = z % stride_;

            float bias_val = bias[oc];

            const float* weight_base = weight + (oc * kernel_size3) + (kz * kernel_plane);
            const float* in_base = in + (in_z * in_plane);

            const float* weight_ky_base = weight_base;
            const float* in_y_base = in_base;
            int ky = 0;

            for (int y = 0; y < out_h; ++y)
            {
                int kx = 0;
                const float* in_x_ptr = in_y_base;
                const float* weight_kx_ptr = weight_ky_base;

                for (int x = 0; x < out_w; ++x, ++out_ptr)
                {
                    float sum = bias_val;
                    const float* w_ic_ptr = weight_kx_ptr;
                    const float* in_ic_ptr = in_x_ptr;

                    // Removed unnecessary brackets for single-line loop
                    for (int ic = 0; ic < in_c; ++ic, w_ic_ptr += weight_ic_step, in_ic_ptr += in_img_size)
                        sum += (*w_ic_ptr) * (*in_ic_ptr);

                    *out_ptr = sum;

                    if (++kx == stride_)
                    {
                        kx = 0;
                        in_x_ptr++;
                        weight_kx_ptr = weight_ky_base;
                    }
                    else // Removed brackets for single statement
                        weight_kx_ptr++;
                }

                if (++ky == stride_)
                {
                    ky = 0;
                    in_y_base += in_w;
                    weight_ky_base = weight_base;
                }
                else // Removed brackets for single statement
                    weight_ky_base += kernel_size_;
            }
        });
        return out;
    }
    void print(std::ostream& os) const override { os << keyword << out_channels_ << "," << kernel_size_keyword << kernel_size_ << "," << stride_keyword << stride_; }
    bool change_dim(void) const override{return stride_ != 1;}
};

template <activation_type Act = activation_type::none>
class instance_norm_3d : public layer {
public:
    static constexpr const char* keyword = "norm";
    float* weight = nullptr;
    float* bias = nullptr;
    size_t weight_size = 0, bias_size = 0;

    instance_norm_3d(int c)
        : layer(c) {
        weight_size = c;
        bias_size = c;
    }

    std::vector<std::pair<float*, size_t>> parameters() override {
        return {{weight, weight_size}, {bias, bias_size}};
    }
    void init_image(tipl::shape<3>& dim_) override
    {
        layer::init_image(dim_);
        out_buffer_size = 0; // in-place
    }
    void allocate(float*& ptr, bool is_gpu_mem) override
    {
        weight = ptr; ptr += weight_size;
        bias = ptr; ptr += bias_size;
        layer::allocate(ptr,is_gpu_mem);
    }

    float* forward(float* in) override
    {
        if constexpr (tipl::use_cuda)
            if (this->is_gpu)
            {
                cuda_instance_norm_3d_forward<Act>(in, in, weight, bias, out_channels_, dim.size(), 0.01f);
                return in;
            }

        const size_t plane_size = dim.size();

        // --- PRECOMPUTE INVERSE DIVISION ---
        const double inv_plane_size = 1.0 / static_cast<double>(plane_size);

        tipl::par_for(out_channels_, [&](size_t outc)
        {
            size_t pos = outc * plane_size;
            float* const base_ptr = in + pos;
            float* const end_ptr = base_ptr + plane_size;

            double sum = 0.0;
            double sq_sum = 0.0;

            // Pass 1: Pointer sweeping (Zero multiplication/addition for indexing)
            for (const float* ptr = base_ptr; ptr < end_ptr; ++ptr)
            {
                double val = *ptr;
                sum += val;
                sq_sum += val * val;
            }

            // Replaced slow divisions with fast multiplications
            float mean = static_cast<float>(sum * inv_plane_size);
            float var = std::max(0.0f, static_cast<float>(sq_sum * inv_plane_size - static_cast<double>(mean) * mean));

            float scale = weight[outc] / std::sqrt(var + 1e-5f);
            float shift = bias[outc] - (mean * scale);

            // Pass 2: Pointer sweeping for apply phase
            for (float* ptr = base_ptr, *out_ptr = out + pos; ptr < end_ptr; ++ptr, ++out_ptr)
            {
                float val = (*ptr) * scale + shift;

                if constexpr (Act != activation_type::none)
                    if (val < 0.0f)
                    {
                        if constexpr (Act == activation_type::relu)
                            val = 0.0f;
                        else if constexpr (Act == activation_type::leaky_relu)
                            val *= 0.01f;
                    }

                *out_ptr = val;
            }
        });

        return in;
    }
    void print(std::ostream& os) const override {
        os << keyword;
        if(Act == activation_type::relu)
            os << "," << relu_keyword;
        if(Act == activation_type::leaky_relu)
            os << "," << leaky_relu_keyword;
    }
};

class max_pool_3d : public layer {
public:
    static constexpr const char* keyword = "max_pool";
    tipl::shape<3> out_dim;
    int pool_size = 2;

    max_pool_3d(int c) : layer(c) {}

    void init_image(tipl::shape<3>& dim_) override {
        dim = dim_;
        out_dim = dim_ = {dim[0] / pool_size, dim[1] / pool_size, dim[2] / pool_size};
        out_buffer_size = out_size = out_dim.size()*out_channels_;
    }

    float* forward(float* in) override
    {
        if constexpr (tipl::use_cuda)
            if (this->is_gpu)
            {
                cuda_max_pool_3d_forward(in, out, out_channels_, dim.depth(), dim.height(), dim.width(),
                                         out_dim.depth(), out_dim.height(), out_dim.width(), pool_size);
                return out;
            }

        const int in_w = dim.width(), in_h = dim.height(), in_d = dim.depth();
        const int out_w = out_dim.width(), out_h = out_dim.height(), out_d = out_dim.depth();

        // --- PRECOMPUTE PLANES ---
        const size_t in_plane = static_cast<size_t>(in_w) * in_h;
        const size_t out_plane = static_cast<size_t>(out_w) * out_h;

        tipl::par_for(static_cast<size_t>(out_channels_) * out_d, [&](size_t i)
        {
            const int c = i / out_d;
            const int z = i % out_d;

            // Sequential pointer for output writing
            float* out_ptr = this->out + (c * out_d * out_plane) + (z * out_plane);
            const float* in_ptr_c = in + (c * in_d * in_plane);

            // --- Z-DIMENSION HOISTING ---
            const int start_sz = z * pool_size;
            const int max_dz = std::min(pool_size, in_d - start_sz); // Hoisted bounds check

            if (max_dz <= 0) return;

            const float* in_ptr_slice_base = in_ptr_c + (start_sz * in_plane);

            // Replaced y * pool_size with additive sy_base
            for (int y = 0, sy_base = 0; y < out_h; ++y, sy_base += pool_size)
            {
                // --- Y-DIMENSION HOISTING ---
                const int max_dy = std::min(pool_size, in_h - sy_base);
                if (max_dy <= 0) break;

                const float* in_ptr_row_base = in_ptr_slice_base + (sy_base * in_w);

                // Replaced x * pool_size with additive sx_base. out_ptr steps automatically.
                for (int x = 0, sx_base = 0; x < out_w; ++x, sx_base += pool_size, ++out_ptr)
                {
                    // --- X-DIMENSION HOISTING ---
                    const int max_dx = std::min(pool_size, in_w - sx_base);
                    if (max_dx <= 0) break;

                    float max_val = -std::numeric_limits<float>::max();

                    // Base pointer for this specific pooling window
                    const float* z_ptr = in_ptr_row_base + sx_base;

                    // --- INNER LOOPS: ZERO Math, ZERO Multiplications, ZERO If-Checks ---
                    for (int dz = 0; dz < max_dz; ++dz, z_ptr += in_plane)
                    {
                        const float* y_ptr = z_ptr;
                        for (int dy = 0; dy < max_dy; ++dy, y_ptr += in_w)
                        {
                            const float* x_ptr = y_ptr;
                            for (int dx = 0; dx < max_dx; ++dx, ++x_ptr)
                                if (*x_ptr > max_val)
                                    max_val = *x_ptr;
                        }
                    }

                    *out_ptr = max_val;
                }
            }
        });
        return out;
    }

    void print(std::ostream& os) const override { os << keyword; }
    bool change_dim(void) const override{return true;}
};

class upsample_3d : public layer {
public:
    static constexpr const char* keyword = "upsample";
    tipl::shape<3> out_dim;
    int pool_size = 2;

    upsample_3d(int c) : layer(c) {}

    void init_image(tipl::shape<3>& dim_) override {
        dim = dim_;
        out_dim = dim_ = {dim[0] * pool_size, dim[1] * pool_size, dim[2] * pool_size};
        out_buffer_size = out_size = out_dim.size()*out_channels_;
    }

    float* forward(float* in) override
    {
        if constexpr (tipl::use_cuda)
            if (this->is_gpu)
            {
                cuda_upsample_3d_forward(in, out, out_channels_, dim.depth(), dim.height(), dim.width(),
                                         out_dim.depth(), out_dim.height(), out_dim.width(), pool_size);
                return out;
            }

        const int in_w = dim.width(), in_h = dim.height(), in_d = dim.depth();
        const int out_w = out_dim.width();

        // --- PRECOMPUTE PLANES & STEPS ---
        const size_t in_plane = static_cast<size_t>(in_w) * in_h;
        const size_t out_plane = static_cast<size_t>(out_w) * out_dim.height();
        const size_t y_step = static_cast<size_t>(pool_size) * out_w;

        tipl::par_for(static_cast<size_t>(out_channels_) * in_d, [&](size_t i)
        {
            const size_t c = i / in_d;
            const size_t z = i % in_d;

            // Base input pointer for this specific channel and depth
            const float* in_ptr = in + (c * in_d + z) * in_plane;

            // Base output pointer for this specific channel and depth
            float* out_y_base = this->out + (c * out_dim.depth() + z * pool_size) * out_plane;

            // Y Loop: Replaced out_y_start = y * pool_size and in_row stepping
            for (int y = 0; y < in_h; ++y, out_y_base += y_step)
            {
                float* out_x_base = out_y_base;

                // X Loop: Replaced out_x_start = x * pool_size
                for (int x = 0; x < in_w; ++x, out_x_base += pool_size)
                {
                    // Perfectly sequential memory read! Replaces in_row[x] array lookup.
                    const float val = *in_ptr++;

                    float* out_z_base = out_x_base;

                    // DZ Loop: Replaced dz * out_plane
                    for (int dz = 0; dz < pool_size; ++dz, out_z_base += out_plane)
                    {
                        float* out_line = out_z_base;

                        // DY Loop: Replaced dy * out_w and eradicated std::copy_n
                        for(int dy = 0; dy < pool_size; ++dy, out_line += out_w)
                            std::fill_n(out_line, pool_size, val);
                    }
                }
            }
        });
        return out;
    }

    void print(std::ostream& os) const override { os << keyword; }
    bool change_dim(void) const override{return true;}
};

class network : public layer
{
public:
    std::function<bool(void)> prog = nullptr;
    std::vector<std::shared_ptr<layer>> layers;

    network() : layer(1,1) {}
    network(int in_c,int out_c) : layer(in_c,out_c) {}

    std::vector<std::pair<float*,size_t>> parameters() override
    {
        std::vector<std::pair<float*,size_t>> param;
        for(auto& l : layers)
        {
            auto p = l->parameters();
            param.insert(param.end(),p.begin(),p.end());
        }
        return param;
    }

    void init_image(tipl::shape<3>& dim_) override
    {
        dim = dim_;
        for(auto& l : layers)
            l->init_image(dim_);
        out_size = layers.back()->out_size;
        out_buffer_size = 0;
    }

    template<typename Container>
    void allocate_memory(Container& memory)
    {
        size_t total_size = 0;
        auto p = parameters();
        size_t p_size = p.size();
        for(size_t i = 0; i < p_size; ++i)
            total_size += p[i].second;

        size_t l_size = layers.size();
        for(size_t i = 0; i < l_size; ++i)
            total_size += layers[i]->out_buffer_size;

        memory.resize(total_size);

        float* ptr = total_size > 0 ? (float*)memory.data() : nullptr;
        bool is_gpu_mem = (memory_location<Container>::at == CUDA);
        allocate(ptr,is_gpu_mem);
    }

    void allocate(float*& ptr,bool is_gpu_mem) override
    {
        this->is_gpu = is_gpu_mem;
        for(auto& l : layers)
            l->allocate(ptr,is_gpu_mem);
    }

    float* forward(float* in) override
    {
        for(auto& l : layers)
        {
            if(prog && !prog())
                return nullptr;
            in = l->forward(in);
        }
        return in;
    }

    void print(std::ostream& os) const override
    {
        bool first = true;
        for(auto& l : layers)
        {
            if(!first)
                os << (l->change_dim() ? "\n" : "+");
            l->print(os);
            first = false;
        }
    }

    std::shared_ptr<layer> create_layer(const std::string& def,int in_c)
    {
        std::unordered_map<std::string,std::string> params;
        for(const auto& arg : tipl::split(def,','))
        {
            size_t pos = arg.find_first_of("0123456789");
            if(pos != std::string::npos)
                params[arg.substr(0,pos)] = arg.substr(pos);
            else
                params[arg] = "1";
        }
        std::shared_ptr<layer> l;

        if(params.count(max_pool_3d::keyword))
            l.reset(new max_pool_3d(in_c));
        else if(params.count(upsample_3d::keyword))
            l.reset(new upsample_3d(in_c));
        else if(params.count(conv_transpose_3d::keyword))
        {
            int ks = params.count(kernel_size_keyword) ? std::stoi(params[kernel_size_keyword]) : 2;
            int stride = params.count(stride_keyword) ? std::stoi(params[stride_keyword]) : 2;
            l.reset(new conv_transpose_3d(in_c,std::stoi(params[conv_transpose_3d::keyword]),ks,stride));
        }
        else if(params.count(conv_3d<>::keyword))
        {
            int out_ch = std::stoi(params[conv_3d<>::keyword]);
            int ks = params.count(kernel_size_keyword) ? std::stoi(params[kernel_size_keyword]) : 3;
            int stride = params.count(stride_keyword) ? std::stoi(params[stride_keyword]) : 1;

            if(params.count(leaky_relu_keyword))
                l.reset(new conv_3d<activation_type::leaky_relu>(in_c,out_ch,ks,stride));
            else
                l.reset(new conv_3d<activation_type::none>(in_c,out_ch,ks,stride));
        }
        else if(params.count(instance_norm_3d<>::keyword))
        {
            if(params.count(leaky_relu_keyword))
                l.reset(new instance_norm_3d<activation_type::leaky_relu>(in_c));
            else
                l.reset(new instance_norm_3d<activation_type::none>(in_c));
        }
        else
            throw std::runtime_error("unknown layer");
        layers.push_back(l);
        return l;
    }
};

class unet3d : public network
{
    std::vector<std::vector<std::shared_ptr<layer>>> encoding, decoding, up;
    std::vector<std::shared_ptr<layer>> en_tail,up_tail;
public:
    unet3d(const std::string& structure,int in_c,int out_c) : network(in_c,out_c)
    {
        std::vector<std::vector<std::string> > enc_tokens,dec_tokens;

        {
            std::vector<std::string> all_lines(tipl::split_in_lines(structure));
            if(all_lines.empty())
                throw std::runtime_error("invalid u-net structure");
            size_t enc_count = all_lines.size() / 2 + 1;
            for(size_t i = 0;i < enc_count;++i)
                enc_tokens.push_back(tipl::split(all_lines[i],'+'));
            for(size_t i = enc_count;i < all_lines.size();++i)
                dec_tokens.push_back(tipl::split(all_lines[i],'+'));
        }

        std::vector<size_t> skip_count;
        encoding.resize(enc_tokens.size());
        for(int level = 0;level < enc_tokens.size();++level)
        {
            for(const auto& token : enc_tokens[level])
                encoding[level].push_back(create_layer(token));
            skip_count.push_back(layers.back()->out_channels_);
        }

        for(int level = 1;level < dec_tokens.size()-1;++level)
            if(dec_tokens[0].back() != dec_tokens[level].back())
                throw std::runtime_error("invalid u-net structure: the tail layers of encoding blocks are different");

        for(const auto& tokens : dec_tokens)
        {
            up.insert(up.begin(),std::vector<std::shared_ptr<layer>>());
            decoding.insert(decoding.begin(),std::vector<std::shared_ptr<layer>>());

            skip_count.pop_back();

            for(size_t t = 0; t < tokens.size() - 1; ++t)
            {
                up[0].push_back(create_layer(tokens[t]));
                if(layers.back()->out_channels_ == skip_count.back() && !tipl::begins_with(tokens[t+1],"norm"))
                    break;
            }

            // skip connection add additional channels
            decoding[0].push_back(create_layer(tokens[up[0].size()],skip_count.back()));

            for(size_t t = up[0].size()+1; t < tokens.size(); ++t)
            {
                auto l = create_layer(tokens[t]);
                if(is_output_layer(l) && t == tokens.size() - 1)
                    break;
                decoding[0].push_back(l);
            }
        }
    }
    std::shared_ptr<layer> create_layer(const std::string& def,int in_c = 0)
    {
        if(layers.empty())
            in_c = in_channels_;
        else
        {
            if(is_output_layer(layers.back())) // the deep supervision outputs
                in_c += layers.back()->in_channels_; // reverse to its input channel
            else
                in_c += layers.back()->out_channels_; // most common condition, follow previous layer's out channels
        }
        return network::create_layer(def,in_c);
    }
    bool is_output_layer(const std::shared_ptr<layer> l) const
    {
        return l->out_channels_ == out_channels_ && l->in_channels_ != l->out_channels_;
    }

    void init_image(tipl::shape<3>& dim_) override
    {
        network::init_image(dim_);

        // find all tail layers basic on out_buffer_size
        auto tail_layer = [](std::vector<std::shared_ptr<layer>>& block)
        {
            for(auto it = block.rbegin(); it != block.rend(); ++it)
                if((*it)->out_buffer_size > 0)
                    return *it;
            throw std::runtime_error("invalid u-net structure: cannot find tail layer");
        };

        en_tail.resize(encoding.size());
        up_tail.resize(up.size());
        for(size_t level = 0;level < encoding.size();++level)
            en_tail[level] = tail_layer(encoding[level]);
        for(size_t level = 0;level < up.size();++level)
            up_tail[level] = tail_layer(up[level]);

        for(size_t i = 0; i < up_tail.size(); ++i)
            {
                en_tail[i]->out_buffer_size += up_tail[i]->out_size;
                up_tail[i]->out_buffer_size = 0; // Prevent the up layer from requesting its own independent memory block
            }
    }
    void allocate(float*& ptr, bool is_gpu_mem) override
    {
        network::allocate(ptr, is_gpu_mem);
        for(size_t i = 0; i < up_tail.size(); ++i)
            up_tail[i]->out = en_tail[i]->out + en_tail[i]->out_size;
    }
    float* forward(float* in) override
    {
        auto forward_block = [&](const std::vector<std::shared_ptr<layer>>& block, float* in_ptr)
        {
            for(auto& l : block)
                in_ptr = l->forward(in_ptr);
            return in_ptr;
        };

        int n_levels = static_cast<int>(encoding.size());
        for(int i = 0; i < n_levels; ++i)
        {
            if(prog && !prog())
                return nullptr;
            in = forward_block(encoding[i], in);
        }
        for(int i = n_levels - 2; i >= 0; --i)
        {
            if(prog && !prog())
                return nullptr;
            forward_block(up[i], in); // This writes directly into the pre-allocated tail location of the encoder buffer
            in = forward_block(decoding[i], en_tail[i]->out);
        }
        return layers.back()->forward(in);
    }
};

#ifdef __CUDACC__

namespace cuda_kernels
{

template<activation_type Act,typename T>
__global__ void conv_3d_kernel(const T* in,const T* weight,const T* bias,T* out,
                               int in_c,int out_c,
                               int in_d,int in_h,int in_w,
                               int out_d,int out_h,int out_w,
                               int kernel_size,int kernel_size3,int range,int stride,T slope)
{
    size_t out_plane = out_w*out_h;
    size_t out_img_size = out_plane*out_d;
    size_t total_out_size = static_cast<size_t>(out_c)*out_img_size;
    size_t idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx >= total_out_size)
        return;

    int x = idx%out_w;
    int y = (idx/out_w)%out_h;
    int z = (idx/out_plane)%out_d;
    int oc = idx/out_img_size;

    int in_plane = in_w*in_h;
    int in_img_size = in_plane*in_d;
    int kernel_plane = kernel_size*kernel_size;

    const T* weight_oc = weight+(oc*in_c*kernel_size3);
    T sum = bias[oc];

    int base_z = z*stride;
    int base_y = y*stride;
    int base_x = x*stride;

    for(int ic = 0;ic < in_c;++ic)
    {
        const T* in_channel_ptr = in+(ic*in_img_size);
        const T* weight_ic = weight_oc+(ic*kernel_size3);

        for(int kz = -range;kz <= range;++kz)
        {
            int sz = base_z+kz;
            if(sz < 0 || sz >= in_d)
                continue;

            int sz_offset = sz*in_plane;
            const T* weight_kz = weight_ic+(kz+range)*kernel_plane;

            for(int ky = -range;ky <= range;++ky)
            {
                int sy = base_y+ky;
                if(sy < 0 || sy >= in_h)
                    continue;

                int sy_offset = sz_offset+sy*in_w;
                const T* weight_ky = weight_kz+(ky+range)*kernel_size;

                for(int kx = -range;kx <= range;++kx)
                {
                    int sx = base_x+kx;
                    if(sx < 0 || sx >= in_w)
                        continue;

                    sum += weight_ky[kx+range]*in_channel_ptr[sy_offset+sx];
                }
            }
        }
    }

    if constexpr(Act == activation_type::relu)
        if(sum < T(0))
            sum = T(0);
    if constexpr(Act == activation_type::leaky_relu)
        if(sum < T(0))
            sum *= slope;

    out[idx] = sum;
}

template<typename T>
__global__ void conv_transpose_3d_kernel(const T* in,const T* weight,const T* bias,T* out,
                                         int in_c,int out_c,
                                         int in_d,int in_h,int in_w,
                                         int out_d,int out_h,int out_w,
                                         int kernel_size,int kernel_size3,int stride)
{
    size_t out_plane = out_w*out_h;
    size_t out_img_size = out_plane*out_d;
    size_t total_out = static_cast<size_t>(out_c)*out_img_size;
    size_t idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx >= total_out)
        return;

    int x = idx%out_w;
    int y = (idx/out_w)%out_h;
    int z = (idx/out_plane)%out_d;
    int oc = idx/out_img_size;

    int in_x = x/stride;
    int kx = x%stride;
    int in_y = y/stride;
    int ky = y%stride;
    int in_z = z/stride;
    int kz = z%stride;

    T sum = bias[oc];

    int k_offset = oc*kernel_size3+kz*(kernel_size*kernel_size)+ky*kernel_size+kx;
    int in_plane = in_w*in_h;
    int in_img_size = in_d*in_plane;
    int in_offset_base = in_z*in_plane+in_y*in_w+in_x;
    int w_stride = out_c*kernel_size3;

    const T* in_ptr = in+in_offset_base;
    const T* w_ptr = weight+k_offset;

    for(int ic = 0;ic < in_c;++ic,in_ptr += in_img_size,w_ptr += w_stride)
        sum += (*in_ptr)*(*w_ptr);

    out[idx] = sum;
}

template<activation_type Act,typename T>
__global__ void instance_norm_3d_kernel(const T* in,T* out,const T* weight,const T* bias,
                                        int out_c,size_t plane_size,T slope)
{
    int oc = blockIdx.x*blockDim.x+threadIdx.x;
    if(oc >= out_c)
        return;

    const T* in_ptr = in+plane_size*oc;
    T* out_ptr = out+plane_size*oc;
    double sum = 0.0,sq_sum = 0.0;

    for(size_t i = 0;i < plane_size;++i)
    {
        T val = in_ptr[i];
        sum += val;
        sq_sum += (double)val*val;
    }

    double inv_plane = 1.0/plane_size;
    T mean = (T)(sum*inv_plane);
    T var = (T)fmaxf(0.0f,(float)(sq_sum*inv_plane-mean*mean));

    T scale = weight[oc]/(T)sqrtf((float)var+1e-5f);
    T shift = bias[oc]-(mean*scale);

    for(size_t i = 0;i < plane_size;++i)
    {
        T val = in_ptr[i]*scale+shift;
        if constexpr(Act == activation_type::relu)
            if(val < T(0))
                val = T(0);
        if constexpr(Act == activation_type::leaky_relu)
            if(val < T(0))
                val *= slope;
        out_ptr[i] = val;
    }
}

template<typename T>
__global__ void max_pool_3d_kernel(const T* in,T* out,
                                   int in_c,int in_d,int in_h,int in_w,
                                   int out_d,int out_h,int out_w,int pool_size)
{
    size_t out_plane = out_w*out_h;
    size_t out_img_size = out_plane*out_d;
    size_t total_out = static_cast<size_t>(in_c)*out_img_size;
    size_t idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx >= total_out)
        return;

    int x = idx%out_w;
    int y = (idx/out_w)%out_h;
    int z = (idx/out_plane)%out_d;
    int c = idx/out_img_size;

    int from_z_base = z*pool_size;
    int from_y_base = y*pool_size;
    int from_x_base = x*pool_size;

    int in_plane = in_w*in_h;
    int in_img_size = in_d*in_plane;
    const T* in_ptr = in+(c*in_img_size);

    T max_val = (T)-1e38f;

    for(int dz = 0;dz < pool_size;++dz)
    {
        int sz = from_z_base+dz;
        if(sz >= in_d)
            continue;

        int sz_offset = sz*in_plane;

        for(int dy = 0;dy < pool_size;++dy)
        {
            int sy = from_y_base+dy;
            if(sy >= in_h)
                continue;

            int sy_offset = sz_offset+sy*in_w;

            for(int dx = 0;dx < pool_size;++dx)
            {
                int sx = from_x_base+dx;
                if(sx >= in_w)
                    continue;

                T val = in_ptr[sy_offset+sx];
                if(val > max_val)
                    max_val = val;
            }
        }
    }

    out[idx] = max_val;
}

template<typename T>
__global__ void upsample_3d_kernel(const T* in,T* out,
                                   int in_c,int in_d,int in_h,int in_w,
                                   int out_d,int out_h,int out_w,int pool_size)
{
    size_t out_plane = out_w*out_h;
    size_t out_img_size = out_plane*out_d;
    size_t total_out = static_cast<size_t>(in_c)*out_img_size;
    size_t idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx >= total_out)
        return;

    int x = idx%out_w;
    int y = (idx/out_w)%out_h;
    int z = (idx/out_plane)%out_d;
    int c = idx/out_img_size;

    int in_x = x/pool_size;
    int in_y = y/pool_size;
    int in_z = z/pool_size;

    int in_plane = in_w*in_h;
    int in_img_size = in_d*in_plane;

    out[idx] = in[c*in_img_size+in_z*in_plane+in_y*in_w+in_x];
}

} // namespace cuda_kernels

template <activation_type Act, typename T>
void cuda_conv_3d_forward(const T* in, const T* weight, const T* bias, T* out,
                          int in_c, int out_c,
                          int in_d, int in_h, int in_w,
                          int out_d, int out_h, int out_w,
                          int kernel_size, int kernel_size3, int range, int stride, T slope)
{
    size_t total_out_size = static_cast<size_t>(out_c) * out_d * out_h * out_w;
    int block_size = 256;
    int grid_size = (total_out_size + block_size - 1) / block_size;
    cuda_kernels::conv_3d_kernel<Act, T><<<grid_size, block_size>>>(
        in, weight, bias, out, in_c, out_c, in_d, in_h, in_w, out_d, out_h, out_w, kernel_size, kernel_size3, range, stride, slope);
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

template <typename T>
void cuda_conv_transpose_3d_forward(const T* in, const T* weight, const T* bias, T* out,
                                    int in_c, int out_c,
                                    int in_d, int in_h, int in_w,
                                    int out_d, int out_h, int out_w,
                                    int kernel_size, int kernel_size3, int stride) {
    size_t total_out = static_cast<size_t>(out_c) * out_d * out_h * out_w;
    int block_size = 256;
    int grid_size = (total_out + block_size - 1) / block_size;
    cuda_kernels::conv_transpose_3d_kernel<T><<<grid_size, block_size>>>(
        in, weight, bias, out, in_c, out_c, in_d, in_h, in_w, out_d, out_h, out_w, kernel_size, kernel_size3, stride);
}

template
void cuda_conv_transpose_3d_forward<float>(const float* in, const float* weight, const float* bias, float* out,
                                    int in_c, int out_c, int in_d, int in_h, int in_w, int out_d, int out_h, int out_w,
                                    int kernel_size, int kernel_size3, int stride);

template <activation_type Act, typename T = float>
void cuda_instance_norm_3d_forward(const T* in, T* out, const T* weight, const T* bias,
                                   int out_c, size_t plane_size, T slope) {
    int block_size = 256;
    int grid_size = (out_c + block_size - 1) / block_size;
    cuda_kernels::instance_norm_3d_kernel<Act, T><<<grid_size, block_size>>>(
        in, out, weight, bias, out_c, plane_size, slope);
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


template <typename T>
void cuda_max_pool_3d_forward(const T* in, T* out,
                              int in_c, int in_d, int in_h, int in_w,
                              int out_d, int out_h, int out_w, int pool_size) {
    size_t total_out = static_cast<size_t>(out_d) * out_h * out_w * in_c;
    int block_size = 256;
    int grid_size = (total_out + block_size - 1) / block_size;
    cuda_kernels::max_pool_3d_kernel<T><<<grid_size, block_size>>>(
        in, out, in_c, in_d, in_h, in_w, out_d, out_h, out_w, pool_size);
}

template
void cuda_max_pool_3d_forward<float>(const float* in, float* out,
                              int in_c, int in_d, int in_h, int in_w,
                              int out_d, int out_h, int out_w, int pool_size);


template <typename T>
void cuda_upsample_3d_forward(const T* in, T* out,
                              int in_c, int in_d, int in_h, int in_w,
                              int out_d, int out_h, int out_w, int pool_size) {
    size_t total_out = static_cast<size_t>(out_d) * out_h * out_w * in_c;
    int block_size = 256;
    int grid_size = (total_out + block_size - 1) / block_size;
    cuda_kernels::upsample_3d_kernel<T><<<grid_size, block_size>>>(
        in, out, in_c, in_d, in_h, in_w, out_d, out_h, out_w, pool_size);
}

template
void cuda_upsample_3d_forward<float>(const float* in, float* out,
                              int in_c, int in_d, int in_h, int in_w,
                              int out_d, int out_h, int out_w, int pool_size);


template <typename T = float>
void cuda_copy_device_to_device(T* dest, const T* src, size_t count) {
    cudaMemcpy(dest, src, count * sizeof(T), cudaMemcpyDeviceToDevice);
}

// Explicit instantiation for cuda_copy_device_to_device
template
void cuda_copy_device_to_device<float>(float* dest, const float* src, size_t count);

#endif // __CUDACC__

} // namespace ml3d
} // namespace tipl

#endif // CNN3D_HPP
