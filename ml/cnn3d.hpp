#ifndef CNN3D_HPP
#define CNN3D_HPP

#include <vector>
#include <memory>
#include <limits>
#include <cmath>
#include <deque>
#include <algorithm>
#include "../utility/pixel_index.hpp"
#include "../utility/basic_image.hpp"
#include "../numerical/statistics.hpp"
#include "../mt.hpp"

namespace tipl {
namespace ml3d {

enum class activation_type { none, relu, leaky_relu };

class layer {
public:
    int in_channels_ = 1, out_channels_ = 1;
    size_t out_size = 0;
    tipl::shape<3> dim;

    layer(int channels) : in_channels_(channels), out_channels_(channels) {}
    layer(int in_c, int out_c) : in_channels_(in_c), out_channels_(out_c) {}
    virtual ~layer() = default;

    virtual std::vector<std::pair<float*, size_t>> parameters() { return {}; }
    virtual void init_image(tipl::shape<3>& dim_)
    {
        dim = dim_;
        out_size = dim.size() * out_channels_;
    }
    virtual float* forward(float* in_ptr) = 0;
    virtual void print(std::ostream& out) const = 0;
    virtual size_t alloc_buffer_size() const { return out_size; }
    virtual void allocate(float*& ptr) {}
};

template <activation_type Act = activation_type::none>
class conv_3d : public layer {
    int kernel_size_, kernel_size3, range;
    float slope_;
public:
    float* weight = nullptr;
    float* bias = nullptr;
    float* out = nullptr;
    size_t weight_size = 0, bias_size = 0;

    conv_3d(int in_c, int out_c, int ks = 3, float slope = 1e-2f)
        : layer(in_c, out_c), kernel_size_(ks), kernel_size3(ks * ks * ks), range((ks - 1) / 2), slope_(slope) {
        weight_size = kernel_size3 * in_channels_ * out_channels_;
        bias_size = out_channels_;
    }

    std::vector<std::pair<float*, size_t>> parameters() override {
        return {{weight, weight_size}, {bias, bias_size}};
    }

    void allocate(float*& ptr) override {
        weight = ptr; ptr += weight_size;
        bias = ptr; ptr += bias_size;
        out = ptr; ptr += out_size;
    }

    float* forward(float* in) override {
        const size_t img_size = dim.size();
        const int in_c = in_channels_, out_c = out_channels_;
        const int w = dim.width(), h = dim.height(), d = dim.depth();

        tipl::par_for(out_c, [&](size_t oc) {
            std::fill_n(out + oc * img_size, img_size, bias[oc]);
        });

        tipl::par_for(static_cast<size_t>(out_c) * d, [&](size_t job) {
            const int oc = job / d, z = job % d;
            float* out_slice = out + (oc * img_size) + (z * w * h);
            const float* weight_oc = weight + (oc * in_c * kernel_size3);

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
                            if (w_val == 0.0f) continue;

                            for (int y = 0; y < h; ++y) {
                                int sy = y + ky;
                                if (sy < 0 || sy >= h) continue;
                                const float* in_row = in_slice + (sy * w);
                                float* out_row = out_slice + (y * w);

                                for (int x = 0; x < w; ++x) {
                                    int sx = x + kx;
                                    if (sx >= 0 && sx < w) out_row[x] += w_val * in_row[sx];
                                }
                            }
                        }
                    }
                }
            }

            if constexpr (Act != activation_type::none) {
                const int slice_size = w * h;
                for (int i = 0; i < slice_size; ++i) {
                    if (out_slice[i] < 0.0f) {
                        if constexpr (Act == activation_type::relu) out_slice[i] = 0.0f;
                        else if constexpr (Act == activation_type::leaky_relu) out_slice[i] *= slope_;
                    }
                }
            }
        });
        return out;
    }
    void print(std::ostream& os) const override {
        os << "conv3d " << (Act == activation_type::relu ? "(relu) " : Act == activation_type::leaky_relu ? "(leaky_relu) " : "")
           << in_channels_ << " " << out_channels_ << '\n';
    }
};

template <activation_type Act = activation_type::none>
class instance_norm_3d : public layer {
    float slope_;
public:
    float* weight = nullptr;
    float* bias = nullptr;
    size_t weight_size = 0, bias_size = 0;

    instance_norm_3d(int c, float slope = 1e-2f)
        : layer(c), slope_(slope) {
        weight_size = c;
        bias_size = c;
    }

    std::vector<std::pair<float*, size_t>> parameters() override {
        return {{weight, weight_size}, {bias, bias_size}};
    }

    void allocate(float*& ptr) override {
        weight = ptr; ptr += weight_size;
        bias = ptr; ptr += bias_size;
    }

    float* forward(float* in) override {
        const size_t plane_size = dim.size();

        tipl::par_for(out_channels_, [&](size_t outc) {
            float* in_ptr = in + plane_size * outc;
            double sum = 0.0, sq_sum = 0.0;

            for (size_t i = 0; i < plane_size; ++i) {
                float val = in_ptr[i];
                sum += val;
                sq_sum += static_cast<double>(val) * val;
            }

            float mean = static_cast<float>(sum / plane_size);
            float var = std::max(0.0f, static_cast<float>(sq_sum / plane_size - mean * mean));

            float scale = weight[outc] / std::sqrt(var + 1e-5f);
            float shift = bias[outc] - (mean * scale);

            for (size_t i = 0; i < plane_size; ++i) {
                float val = in_ptr[i] * scale + shift;

                if constexpr (Act != activation_type::none) {
                    if (val < 0.0f) {
                        if constexpr (Act == activation_type::relu) {
                            val = 0.0f;
                        } else if constexpr (Act == activation_type::leaky_relu) {
                            val *= slope_;
                        }
                    }
                }
                in_ptr[i] = val;
            }
        }, out_channels_);

        return in;
    }
    size_t alloc_buffer_size() const override { return 0; }
    void print(std::ostream& os) const override {
        os << "norm_3d "
           << (Act == activation_type::relu ? "(relu) " : Act == activation_type::leaky_relu ? "(leaky_relu) " : "")
           << out_channels_ << '\n';
    }
};

class max_pool_3d : public layer {
public:
    tipl::shape<3> out_dim;
    float* out = nullptr;
    int pool_size = 2;

    max_pool_3d(int c) : layer(c) {}

    void init_image(tipl::shape<3>& dim_) override {
        dim = dim_;
        out_dim = dim_ = {dim[0] / pool_size, dim[1] / pool_size, dim[2] / pool_size};
        out_size = out_dim.size()*out_channels_;
    }

    void allocate(float*& ptr) override {
        out = ptr; ptr += out_size;
    }

    float* forward(float* in) override {
        const int in_w = dim.width(), in_h = dim.height(), in_d = dim.depth();
        const int out_w = out_dim.width(), out_h = out_dim.height(), out_d = out_dim.depth();
        const size_t in_plane = static_cast<size_t>(in_w) * in_h, out_plane = static_cast<size_t>(out_w) * out_h;

        tipl::par_for(static_cast<size_t>(out_channels_) * out_d, [&](size_t i) {
            const int c = i / out_d, z = i % out_d;
            const float* in_ptr_c = in + (c * in_d * in_plane);
            float* out_ptr_slice = out + (c * out_d * out_plane) + (z * out_plane);
            const int from_z_base = z * pool_size;

            for (int y = 0; y < out_h; ++y) {
                const int from_y_base = y * pool_size;
                for (int x = 0; x < out_w; ++x) {
                    const int from_x_base = x * pool_size;
                    float max_val = -std::numeric_limits<float>::max();

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
                                if (sx < in_w) max_val = std::max(max_val, in_ptr_row[sx]);
                            }
                        }
                    }
                    out_ptr_slice[y * out_w + x] = max_val;
                }
            }
        });
        return out;
    }

    void print(std::ostream& os) const override { os << "max_pool_3d " << out_channels_ << '\n'; }
};

class upsample_3d : public layer {
public:
    tipl::shape<3> out_dim;
    float* out = nullptr;
    int pool_size = 2;

    upsample_3d(int c) : layer(c) {}

    void init_image(tipl::shape<3>& dim_) override {
        dim = dim_;
        out_dim = dim_ = {dim[0] * pool_size, dim[1] * pool_size, dim[2] * pool_size};
        out_size = out_dim.size()*out_channels_;
    }


    void allocate(float*& ptr) override {
        out = ptr; ptr += out_size;
    }

    float* forward(float* in) override {
        const int in_w = dim.width(), in_h = dim.height(), in_d = dim.depth();
        const int out_w = out_dim.width(), out_h = out_dim.height();
        const size_t in_plane = static_cast<size_t>(in_w) * in_h, out_plane = static_cast<size_t>(out_w) * out_h;

        tipl::par_for(static_cast<size_t>(out_channels_) * in_d, [&](size_t i) {
            size_t c = i / in_d, z = i % in_d;
            float* in_ptr_plane = in + (c * in_d + z) * in_plane;
            float* out_ptr_base = out + (c * out_dim.depth() + z * pool_size) * out_plane;

            for (int y = 0; y < in_h; ++y) {
                int out_y_start = y * pool_size;
                float* in_row = in_ptr_plane + (y * in_w);

                for (int x = 0; x < in_w; ++x) {
                    int out_x_start = x * pool_size;
                    for (int dz = 0; dz < pool_size; ++dz) {
                        float* out_line = out_ptr_base + (dz * out_plane) + (out_y_start * out_w);
                        std::fill_n(out_line + out_x_start, pool_size, in_row[x]);
                        for(int dy = 1; dy < pool_size; ++dy) {
                            std::copy_n(out_line + out_x_start, pool_size, out_line + out_x_start + (dy * out_w));
                        }
                    }
                }
            }
        });
        return out;
    }

    void print(std::ostream& os) const override { os << "upsample_3d " << out_channels_ << '\n'; }
};

class network : public layer {
public:
    std::vector<std::shared_ptr<layer>> layers;
    network() : layer(1, 1) {}
    std::vector<std::pair<float*, size_t>> parameters() override {
        std::vector<std::pair<float*, size_t>> param;
        for (auto& l : layers) {
            auto p = l->parameters();
            param.insert(param.end(), p.begin(), p.end());
        }
        return param;
    }

    void init_image(tipl::shape<3>& dim_) override {
        dim = dim_;
        for (auto& l : layers) l->init_image(dim_);
        out_size = layers.back()->out_size;
    }

    void allocate_memory(std::vector<float>& memory)
    {
        size_t total_size = 0;
        for (const auto& p : parameters()) total_size += p.second;
        for (auto& l : layers) total_size += l->alloc_buffer_size();
        memory.resize(total_size);
        float* ptr = memory.data();
        allocate(ptr);
    }

    void allocate(float*& ptr) override {
        for (auto& l : layers)
            l->allocate(ptr);
    }

    float* forward(float* in) override {
        for (auto& l : layers) in = l->forward(in);
        return in;
    }

    void print(std::ostream& os) const override { for (auto& l : layers) l->print(os); }
};


enum class unet_version { standard, deep_supervision };
template<typename prog_type = void, unet_version Version = unet_version::standard>
class unet3d : public network {
    std::deque<std::vector<std::shared_ptr<layer>>> encoding, decoding, up;
    std::shared_ptr<layer> output;

    std::shared_ptr<layer> add_layer(layer* l) {
        layers.emplace_back(l);
        return layers.back();
    }

    void add_conv_block(std::vector<std::shared_ptr<layer>>& block, const std::vector<int>& rhs, size_t ks) {
        int count = 0;
        for (int next_c : rhs) {
            if (count) {
                constexpr auto conv_act = (Version == unet_version::standard) ? activation_type::leaky_relu : activation_type::none;
                constexpr auto norm_act = (Version == unet_version::standard) ? activation_type::none : activation_type::leaky_relu;
                block.push_back(add_layer(new conv_3d<conv_act>(count, next_c, ks)));
                block.push_back(add_layer(new instance_norm_3d<norm_act>(next_c)));
            }
            count = next_c;
        }
    }

    float* forward_block(const std::vector<std::shared_ptr<layer>>& block, float* in_ptr) {
        for (auto& l : block) in_ptr = l->forward(in_ptr);
        return in_ptr;
    }

public:
    prog_type* prog = nullptr;

    unet3d(const std::vector<std::vector<int>>& f_down, const std::vector<std::vector<int>>& f_up,
           const std::vector<int>& ks, int in_c, int out_c) {
        in_channels_ = in_c;
        out_channels_ = out_c;

        for (size_t i = 0; i < f_down.size(); ++i) {
            std::vector<std::shared_ptr<layer>> en_block;
            if (i > 0) en_block.push_back(add_layer(new max_pool_3d(f_down[i][0])));
            add_conv_block(en_block, f_down[i], ks[i]);
            encoding.push_back(std::move(en_block));
        }

        for (int i = static_cast<int>(f_down.size()) - 2; i >= 0; --i) {
            std::vector<std::shared_ptr<layer>> up_block, de_block;
            up_block.push_back(add_layer(new upsample_3d(f_up[i+1].back())));
            add_conv_block(up_block, {f_up[i+1].back(), f_down[i].back()}, ks[i]);
            add_conv_block(de_block, f_up[i], ks[i]);

            up.push_front(std::move(up_block));
            decoding.push_front(std::move(de_block));

            if constexpr (Version == unet_version::deep_supervision) {
                auto ds_head = add_layer(new conv_3d<activation_type::none>(f_up[i].back(), out_c, 1));
                if (i == 0) output = ds_head;
            }
        }

        if constexpr (Version == unet_version::standard) {
            output = add_layer(new conv_3d<activation_type::none>(f_up[0].back(), out_c, 1));
        }
    }

    void init_image(tipl::shape<3>& dim_) override {
        network::init_image(dim_);
        for (int i = static_cast<int>(encoding.size()) - 2; i >= 0; --i)
            encoding[i][encoding[i].size() - 2]->out_size += up[i].back()->out_size;
    }

    float* forward(float* in) override {
        std::vector<float*> buf;
        int n_levels = static_cast<int>(encoding.size());

        for (int i = 0; i < n_levels; ++i) {
            if constexpr (!std::is_void_v<prog_type>) {
                if (prog && !(*prog)(i, n_levels * 2)) return nullptr;
            }
            buf.push_back(in = forward_block(encoding[i], in));
        }

        for (int i = n_levels - 2; i >= 0; --i) {
            if constexpr (!std::is_void_v<prog_type>) {
                if (prog && !(*prog)(n_levels * 2 - i, n_levels * 2 + 1)) return nullptr;
            }
            buf.pop_back();
            float* encoder_skip = buf.back();
            float* decoder_up = forward_block(up[i], in);

            size_t copy_size = up[i].back()->out_size;
            size_t skip_offset = encoding[i].back()->out_size;

            std::copy_n(decoder_up, copy_size, encoder_skip + skip_offset);

            in = forward_block(decoding[i], encoder_skip);
        }
        return output->forward(in);
    }
};

} // namespace ml3d
} // namespace tipl

#endif // CNN3D_HPP
