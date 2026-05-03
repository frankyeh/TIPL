#ifndef CNN3D_HPP
#define CNN3D_HPP

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <deque>
#include <algorithm>
#include <stdexcept>
#include "../utility/shape.hpp"
#include "../def.hpp"
#include "../cu.hpp"
#include "cnn3d_detail.hpp"

namespace tipl {
namespace ml3d {

template <activation_type Act,typename T>
void cuda_conv_3d_forward(const T* in,const T* weight,const T* bias,T* out,int in_c,int out_c,int in_d,int in_h,int in_w,int out_d,int out_h,int out_w,int kernel_size,int kernel_size3,int range,int stride,T slope);

template <typename T>
void cuda_conv_transpose_3d_forward(const T* in,const T* weight,const T* bias,T* out,int in_c,int out_c,int in_d,int in_h,int in_w,int out_d,int out_h,int out_w,int kernel_size,int kernel_size3,int stride);

template <activation_type Act,typename T>
void cuda_batch_norm_3d_forward(const T* in,T* out,const T* weight,const T* bias,int in_c,size_t plane_size);

template <activation_type Act,typename T>
void cuda_instance_norm_3d_forward(const T* in,T* out,const T* weight,const T* bias,int out_c,size_t plane_size,T slope);

template <typename T>
void cuda_max_pool_3d_forward(const T* in,T* out,int in_c,int in_d,int in_h,int in_w,int out_d,int out_h,int out_w,int pool_size);

template <typename T>
void cuda_upsample_3d_forward(const T* in,T* out,int in_c,int in_d,int in_h,int in_w,int out_d,int out_h,int out_w,int pool_size);


static constexpr const char* kernel_size_keyword = "ks";
static constexpr const char* stride_keyword = "stride";
static constexpr const char* leaky_relu_keyword = "leaky_relu";
static constexpr const char* elu_keyword = "elu";
static constexpr const char* relu_keyword = "relu";


class layer
{
public:
    int in_channels_ = 1, out_channels_ = 1;
    size_t out_size = 0, out_buffer_size = 0;
    tipl::shape<3> dim;
    float* out = nullptr;
    bool is_gpu = false;

    layer(int channels) : in_channels_(channels), out_channels_(channels) {}
    layer(int in_c,int out_c) : in_channels_(in_c), out_channels_(out_c) {}
    virtual ~layer() = default;

    virtual std::vector<std::pair<float*,size_t>> parameters() { return {}; }
    virtual size_t param_size(void)       {return 0;}
    virtual const tipl::shape<3>& init_image(const tipl::shape<3>& dim_)
    {
        dim = dim_;
        out_buffer_size = out_size = dim.size() * out_channels_;
        return dim;
    }
    virtual void forward(const float* in_ptr,float* out_ptr) = 0;
    virtual void print(std::ostream& out) const = 0;
    virtual float* allocate_param(float* ptr,bool is_gpu_mem)
    {
        is_gpu = is_gpu_mem;
        return ptr;
    }
    virtual bool change_dim(void) const { return false; }

protected:
    template<activation_type Act>
    void print_activation(std::ostream& os) const
    {
        if constexpr(Act == activation_type::relu) os << "," << relu_keyword;
        if constexpr(Act == activation_type::leaky_relu) os << "," << leaky_relu_keyword;
        if constexpr(Act == activation_type::elu) os << "," << elu_keyword;
    }
};

class weight_bias_layer : public layer
{
public:
    float* weight = nullptr;
    float* bias = nullptr;
    size_t weight_size = 0, bias_size = 0;

    weight_bias_layer(int channels) : layer(channels), bias_size(channels) {}
    weight_bias_layer(int in_c,int out_c) : layer(in_c,out_c), bias_size(out_c) {}

    std::vector<std::pair<float*,size_t>> parameters() override   { return {{weight,weight_size},{bias,bias_size}}; }
    size_t param_size(void) override                              { return weight_size + bias_size;}
    float* allocate_param(float* ptr,bool is_gpu_) override
    {
        weight = ptr; ptr += weight_size;
        bias = ptr; ptr += bias_size;
        is_gpu = is_gpu_;
        return ptr;
    }
};

template <activation_type Act = activation_type::none>
class conv_3d : public weight_bias_layer
{
    int kernel_size_, kernel_size3, range, stride_;
public:
    static constexpr const char* keyword = "conv";
    tipl::shape<3> out_dim;

    conv_3d(int in_c,int out_c,int ks = 3,int stride = 1)
        : weight_bias_layer(in_c,out_c), kernel_size_(ks), kernel_size3(ks * ks * ks), range((ks - 1) / 2), stride_(stride) { weight_size = kernel_size3 * in_channels_ * out_channels_; }

    const tipl::shape<3>& init_image(const tipl::shape<3>& dim_) override
    {
        dim = dim_;
        out_dim = tipl::s(dim_[0] / stride_,dim_[1] / stride_,dim_[2] / stride_);
        out_buffer_size = out_size = out_dim.size() * out_channels_;
        return out_dim;
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_conv_3d_forward<Act,float>(in,weight,bias,out_ptr,in_channels_,out_channels_,dim.depth(),dim.height(),dim.width(),out_dim.depth(),out_dim.height(),out_dim.width(),kernel_size_,kernel_size3,range,stride_,0.01f),void();
        cpu_conv_3d_forward<Act>(in,weight,bias,out_ptr,in_channels_,out_channels_,dim.depth(),dim.height(),dim.width(),out_dim.depth(),out_dim.height(),out_dim.width(),kernel_size_,kernel_size3,range,stride_);
    }

    void print(std::ostream& os) const override
    {
        os << keyword << out_channels_ << "," << kernel_size_keyword << kernel_size_ << "," << stride_keyword << stride_;
        this->print_activation<Act>(os);
    }
    bool change_dim(void) const override { return stride_ != 1; }
};

class conv_transpose_3d : public weight_bias_layer
{
    int kernel_size_, kernel_size3, stride_;
public:
    static constexpr const char* keyword = "conv_trans";
    tipl::shape<3> out_dim;

    conv_transpose_3d(int in_c,int out_c,int ks = 2,int stride = 2)
        : weight_bias_layer(in_c,out_c), kernel_size_(ks), kernel_size3(ks * ks * ks), stride_(stride) { weight_size = kernel_size3 * in_channels_ * out_channels_; }

    const tipl::shape<3>& init_image(const tipl::shape<3>& dim_) override
    {
        dim = dim_;
        out_dim = tipl::s(dim_[0] * stride_,dim_[1] * stride_,dim_[2] * stride_);
        out_buffer_size = out_size = out_dim.size() * out_channels_;
        return out_dim;
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_conv_transpose_3d_forward<float>(in,weight,bias,out_ptr,in_channels_,out_channels_,dim.depth(),dim.height(),dim.width(),out_dim.depth(),out_dim.height(),out_dim.width(),kernel_size_,kernel_size3,stride_),void();
        cpu_conv_transpose_3d_forward(in,weight,bias,out_ptr,in_channels_,out_channels_,dim.depth(),dim.height(),dim.width(),out_dim.depth(),out_dim.height(),out_dim.width(),kernel_size_,kernel_size3,stride_);
    }

    void print(std::ostream& os) const override { os << keyword << out_channels_ << "," << kernel_size_keyword << kernel_size_ << "," << stride_keyword << stride_; }
    bool change_dim(void) const override { return stride_ != 1; }
};

template<activation_type Act = activation_type::none>
class batch_norm_3d : public weight_bias_layer
{
public:
    static constexpr const char* keyword = "bnorm";

    batch_norm_3d(int c) : weight_bias_layer(c) { weight_size = c; }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_batch_norm_3d_forward<Act,float>(in,out_ptr,weight,bias,out_channels_,dim.size()),void();
        cpu_batch_norm_3d_forward<Act>(in,out_ptr,weight,bias,out_channels_,dim.size());
    }

    void print(std::ostream& os) const override
    {
        os << keyword;
        this->print_activation<Act>(os);
    }
};

template <activation_type Act = activation_type::none>
class instance_norm_3d : public weight_bias_layer
{
public:
    static constexpr const char* keyword = "norm";

    instance_norm_3d(int c) : weight_bias_layer(c) { weight_size = c; }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_instance_norm_3d_forward<Act,float>(in,out_ptr,weight,bias,out_channels_,dim.size(),0.01f),void();
        cpu_instance_norm_3d_forward<Act>(in,out_ptr,weight,bias,out_channels_,dim.size());
    }

    void print(std::ostream& os) const override
    {
        os << keyword;
        this->print_activation<Act>(os);
    }
};

class max_pool_3d : public layer
{
public:
    static constexpr const char* keyword = "max_pool";
    tipl::shape<3> out_dim;
    int pool_size = 2;

    max_pool_3d(int c) : layer(c) {}

    const tipl::shape<3>& init_image(const tipl::shape<3>& dim_) override
    {
        dim = dim_;
        out_dim = {dim[0] / pool_size,dim[1] / pool_size,dim[2] / pool_size};
        out_buffer_size = out_size = out_dim.size() * out_channels_;
        return out_dim;
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_max_pool_3d_forward<float>(in,out_ptr,out_channels_,dim.depth(),dim.height(),dim.width(),out_dim.depth(),out_dim.height(),out_dim.width(),pool_size),void();
        cpu_max_pool_3d_forward(in,out_ptr,out_channels_,dim.depth(),dim.height(),dim.width(),out_dim.depth(),out_dim.height(),out_dim.width(),pool_size);
    }

    void print(std::ostream& os) const override { os << keyword; }
    bool change_dim(void) const override { return true; }
};

class upsample_3d : public layer
{
public:
    static constexpr const char* keyword = "upsample";
    tipl::shape<3> out_dim;
    int pool_size = 2;

    upsample_3d(int c) : layer(c) {}

    const tipl::shape<3>& init_image(const tipl::shape<3>& dim_) override
    {
        dim = dim_;
        out_dim = {dim[0] * pool_size,dim[1] * pool_size,dim[2] * pool_size};
        out_buffer_size = out_size = out_dim.size() * out_channels_;
        return out_dim;
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_upsample_3d_forward<float>(in,out_ptr,out_channels_,dim.depth(),dim.height(),dim.width(),out_dim.depth(),out_dim.height(),out_dim.width(),pool_size),void();
        cpu_upsample_3d_forward(in,out_ptr,out_channels_,dim.depth(),dim.height(),dim.width(),out_dim.depth(),out_dim.height(),out_dim.width(),pool_size);
    }

    void print(std::ostream& os) const override { os << keyword; }
    bool change_dim(void) const override { return true; }
};



class network : public layer
{
protected:
    template<template<activation_type> class LayerType,typename... Args>
    std::shared_ptr<layer> make_act_layer(const std::unordered_map<std::string,std::string>& params,Args... args)
    {
        if(params.count(elu_keyword)) return std::make_shared<LayerType<activation_type::elu>>(args...);
        if(params.count(leaky_relu_keyword)) return std::make_shared<LayerType<activation_type::leaky_relu>>(args...);
        if(params.count(relu_keyword)) return std::make_shared<LayerType<activation_type::relu>>(args...);
        return std::make_shared<LayerType<activation_type::none>>(args...);
    }
    float* get_ptr(std::vector<float>& mem,tipl::device_vector<float>& gpu_mem,size_t total_size)
    {
        float* ptr = nullptr;
        if constexpr(tipl::use_cuda)
            if(is_gpu)
            {
                gpu_mem.resize(total_size);
                ptr = gpu_mem.data();
            }
        if(!ptr)
        {
            mem.resize(total_size);
            ptr = mem.data();
        }
        return ptr;
    }
protected:
    tipl::device_vector<float> gpu_memory;
    std::vector<float> memory;
protected:
    tipl::device_vector<float> gpu_buf_memory;
    std::vector<float> buf_memory;

public:
    std::function<bool(int,int)> prog = nullptr;
    std::vector<std::shared_ptr<layer>> layers;

    network() : layer(1,1) {}
    network(int in_c,int out_c) : layer(in_c,out_c) {}

    std::vector<std::pair<float*,size_t>> parameters() override
    {
        size_t size = param_size();
        if(!size)
            return {};
        if(memory.empty())
            allocate_param((memory = std::vector<float>(size)).data(),is_gpu = false);
        std::vector<std::pair<float*,size_t>> param;
        for(auto& l : layers)
        {
            auto p = l->parameters();
            param.insert(param.end(),p.begin(),p.end());
        }
        return param;
    }
    size_t param_size(void) override
    {
        size_t total_size = 0;
        for(auto& l : layers)
            total_size += l->param_size();
        return total_size;
    }

    const tipl::shape<3>& init_image(const tipl::shape<3>& dim_) override
    {
        auto out_dim = dim = dim_;
        for(auto& l : layers)
            out_dim = l->init_image(out_dim);
        out_size = layers.back()->out_size;
        out_buffer_size = 0;
        out = nullptr;
        return dim;
    }
    void to_gpu(void)
    {
        if constexpr(tipl::use_cuda)
            allocate_param((gpu_memory = memory).data(),is_gpu = true);
        memory = std::vector<float>();
    }
    float* allocate_param(float* ptr,bool is_gpu_mem) override
    {
        is_gpu = is_gpu_mem;
        for(auto& l : layers)
            ptr = l->allocate_param(ptr,is_gpu);
        return ptr;
    }
    virtual void allocate_buffer(void)
    {
        size_t max_mem = 0, in_beg = std::numeric_limits<size_t>::max(), in_end = in_beg;
        std::vector<size_t> out_loc(layers.size());
        for(size_t i = 0; i < layers.size(); ++i)
        {
            size_t buf_size = layers[i]->out_buffer_size;
            if(buf_size < in_beg)
                out_loc[i] = in_beg = 0;
            else
                out_loc[i] = in_beg = in_end;
            in_end = in_beg + buf_size;
            if(in_end > max_mem)
                max_mem = in_end;
        }

        float* ptr = get_ptr(buf_memory,gpu_buf_memory,max_mem);
        for(size_t i = 0; i < layers.size(); ++i)
            layers[i]->out = ptr + out_loc[i];
        out = layers.back()->out;
    }
    void forward(const float* in_ptr,float*) override
    {
        if(!out)
            allocate_buffer();
        for(size_t i = 0;i < layers.size();++i)
        {
            if(prog && !prog(i,layers.size())) return;
            layers[i]->forward(in_ptr,layers[i]->out);
            in_ptr = layers[i]->out;
        }
    }

    void push_back(std::shared_ptr<layer> l) { layers.push_back(l); }
    size_t size(void) const { return layers.size(); }
    auto back(void) { return layers.back(); }

    void print(std::ostream& os) const override
    {
        bool first = true;
        for(auto& l : layers)
        {
            if(!first) os << (l->change_dim() ? "\n" : "+");
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
            if(pos != std::string::npos) params[arg.substr(0,pos)] = arg.substr(pos);
            else params[arg] = "1";
        }
        std::shared_ptr<layer> l;

        if(params.count(max_pool_3d::keyword)) l.reset(new max_pool_3d(in_c));
        else if(params.count(upsample_3d::keyword)) l.reset(new upsample_3d(in_c));
        else if(params.count(conv_transpose_3d::keyword))
        {
            int ks = params.count(kernel_size_keyword) ? std::stoi(params[kernel_size_keyword]) : 2;
            int stride = params.count(stride_keyword) ? std::stoi(params[stride_keyword]) : 2;
            if(ks != 2 || stride != 2) throw std::runtime_error("conv_trans supports only ks2 stride2");
            l.reset(new conv_transpose_3d(in_c,std::stoi(params[conv_transpose_3d::keyword]),ks,stride));
        }
        else if(params.count(conv_3d<>::keyword))
        {
            int out_ch = std::stoi(params[conv_3d<>::keyword]);
            int ks = params.count(kernel_size_keyword) ? std::stoi(params[kernel_size_keyword]) : 3;
            int stride = params.count(stride_keyword) ? std::stoi(params[stride_keyword]) : 1;

            if(!((ks == 1 && stride == 1) || (ks == 3 && (stride == 1 || stride == 2))))
                throw std::runtime_error("conv supports only ks1 stride1, ks3 stride1, and ks3 stride2");

            l = make_act_layer<conv_3d>(params,in_c,out_ch,ks,stride);
        }
        else if(params.count(instance_norm_3d<>::keyword)) l = make_act_layer<instance_norm_3d>(params,in_c);
        else if(params.count(batch_norm_3d<>::keyword)) l = make_act_layer<batch_norm_3d>(params,in_c);
        else throw std::runtime_error("unknown layer:"+params[0]);

        layers.push_back(l);
        return l;
    }
};

class unet3d : public network
{
protected:
    std::vector<std::vector<std::shared_ptr<layer>>> encoding, decoding;
    std::vector<std::shared_ptr<layer>> decoding_head;
protected:
    tipl::device_vector<float> gpu_skip_memory;
    std::vector<float> skip_memory;
public:
    unet3d(const std::string& structure,int in_c,int out_c) : network(in_c,out_c)
    {
        std::vector<std::vector<std::string>> enc_tokens, dec_tokens;
        {
            std::vector<std::string> all_lines(tipl::split_in_lines(structure));
            if(all_lines.size() < 3) throw std::runtime_error("invalid u-net structure");
            size_t enc_count = all_lines.size() / 2 + 1;
            for(size_t i = 0;i < all_lines.size();++i)
                (i < enc_count ? enc_tokens : dec_tokens).push_back(tipl::split(all_lines[i],'+'));
        }

        encoding.resize(enc_tokens.size());
        for(int level = 0;level < enc_tokens.size();++level)
            for(const auto& token : enc_tokens[level])
                encoding[level].push_back(create_layer(token));

        for(int level = dec_tokens.size() - 1; level >= 0; --level)
        {
            const auto& tokens = dec_tokens[dec_tokens.size() - 1 - level];
            decoding.insert(decoding.begin(),std::vector<std::shared_ptr<layer>>());
            decoding_head.insert(decoding_head.begin(),layers.back());

            decoding[0].push_back(create_layer(tokens[0],encoding[level].back()->out_channels_));
            for(size_t t = 1; t < tokens.size(); ++t)
            {
                auto l = create_layer(tokens[t]);
                if(tokens[t] == dec_tokens.back().back()) break;
                decoding[0].push_back(l);
            }
        }
    }
    std::shared_ptr<layer> create_layer(const std::string& def,int in_c = 0)
    {
        if(layers.empty()) in_c = in_channels_;
        else in_c += layers.back()->out_channels_;
        return network::create_layer(def,in_c);
    }

    void allocate_buffer(void) override
    {
        size_t total_skip_memory = 0;
        for(size_t i = 0; i < decoding.size(); ++i)
        {
            total_skip_memory += encoding[i].back()->out_size + decoding_head[i]->out_size;
            encoding[i].back()->out_buffer_size = 0;
            decoding_head[i]->out_buffer_size = 0;
        }
        network::allocate_buffer();
        auto skip_ptr = get_ptr(skip_memory,gpu_skip_memory,total_skip_memory);
        for(size_t i = 0; i < decoding_head.size(); ++i)
        {
            encoding[i].back()->out = skip_ptr;
            decoding_head[i]->out = skip_ptr + encoding[i].back()->out_size;
            skip_ptr += encoding[i].back()->out_size + decoding_head[i]->out_size;
        }
    }

    void forward(const float* in_ptr,float*) override
    {
        if(!out)
            allocate_buffer();

        auto forward_block = [&](const auto& block, const float* in_p) -> const float*
        {
            for(auto& l : block)
            {
                l->forward(in_p,l->out);
                in_p = l->out;
            }
            return in_p;
        };
        for(size_t i = 0; i < encoding.size(); ++i)
        {
            if(prog && !prog(i,encoding.size()*2))
                return;
            in_ptr = forward_block(encoding[i], in_ptr);
        }
        for(int i = int(decoding_head.size()) - 1; i >= 0; --i)
        {
            if(prog && !prog(encoding.size()*2-i-1,encoding.size()*2))
                return;
            in_ptr = forward_block(decoding[i], encoding[i].back()->out);
        }
        layers.back()->forward(in_ptr,layers.back()->out);
    }
    void forward(tipl::image<3>& in)
    {
        if(dim != in.shape())
            init_image(in.shape());
        if constexpr(tipl::use_cuda)
            if(is_gpu) return forward(tipl::device_image<3,float>(in).data(),nullptr),void();
        forward(in.data(),nullptr);
    }
};

} // namespace ml3d
} // namespace tipl

#endif // CNN3D_HPP
