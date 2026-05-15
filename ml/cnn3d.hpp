#ifndef CNN3D_HPP
#define CNN3D_HPP

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <deque>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <functional>
#include <type_traits>
#include "../utility/shape.hpp"
#include "../def.hpp"
#include "../cu.hpp"
#include "../po.hpp"
#include "cnn3d_detail.hpp"

namespace tipl {
namespace ml3d {

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
};

template <activation_type Act = activation_type::none>
class conv_xy_3d : public weight_bias_layer
{
public:
    static constexpr const char* keyword = "conv_xy";
    tipl::shape<3> out_dim;

    conv_xy_3d(int in_c,int out_c) : weight_bias_layer(in_c,out_c)
    {
        weight_size = 27*in_channels_*out_channels_;
    }

    const tipl::shape<3>& init_image(const tipl::shape<3>& dim_) override
    {
        dim = dim_;
        out_dim = tipl::s(dim_[0] >> 1,dim_[1] >> 1,dim_[2]);
        out_buffer_size = out_size = out_dim.size()*out_channels_;
        return out_dim;
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_conv_xy_3d_forward<Act,float>(
                           in,weight,bias,out_ptr,
                           in_channels_,out_channels_,
                           dim.depth(),dim.height(),dim.width(),
                           out_dim.depth(),out_dim.height(),out_dim.width(),
                           0.01f),void();

        cpu_conv_xy_3d_forward<Act>(
            in,weight,bias,out_ptr,
            in_channels_,out_channels_,
            dim.depth(),dim.height(),dim.width(),
            out_dim.depth(),out_dim.height(),out_dim.width());
    }

    void print(std::ostream& os) const override
    {
        os << keyword << out_channels_ << "," << kernel_size_keyword << "3," << stride_keyword << "2";
        this->print_activation<Act>(os);
    }
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
};

class conv_transpose_xy_3d : public weight_bias_layer
{
public:
    static constexpr const char* keyword = "conv_trans_xy";
    tipl::shape<3> out_dim;

    conv_transpose_xy_3d(int in_c,int out_c) : weight_bias_layer(in_c,out_c)
    {
        weight_size = 4*in_channels_*out_channels_; // 2x2x1
    }

    const tipl::shape<3>& init_image(const tipl::shape<3>& dim_) override
    {
        dim = dim_;
        out_dim = tipl::s(dim_[0]*2,dim_[1]*2,dim_[2]);
        out_buffer_size = out_size = out_dim.size()*out_channels_;
        return out_dim;
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_conv_transpose_xy_3d_forward<float>(
                           in,weight,bias,out_ptr,
                           in_channels_,out_channels_,
                           dim.depth(),dim.height(),dim.width(),
                           out_dim.depth(),out_dim.height(),out_dim.width()),void();

        cpu_conv_transpose_xy_3d_forward(
            in,weight,bias,out_ptr,
            in_channels_,out_channels_,
            dim.depth(),dim.height(),dim.width(),
            out_dim.depth(),out_dim.height(),out_dim.width());
    }

    void print(std::ostream& os) const override
    {
        os << keyword << out_channels_ << "," << kernel_size_keyword << "2," << stride_keyword << "2";
    }
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

    max_pool_3d(int c) : layer(c) {}

    const tipl::shape<3>& init_image(const tipl::shape<3>& dim_) override
    {
        dim = dim_;
        out_dim = {dim[0] / 2,dim[1] / 2,dim[2] / 2};
        out_buffer_size = out_size = out_dim.size() * out_channels_;
        return out_dim;
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_max_pool_3d_forward<float>(in,out_ptr,out_channels_,dim.depth(),dim.height(),dim.width(),out_dim.depth(),out_dim.height(),out_dim.width()),void();
        cpu_max_pool_3d_forward(in,out_ptr,out_channels_,dim.depth(),dim.height(),dim.width(),out_dim.depth(),out_dim.height(),out_dim.width());
    }

    void print(std::ostream& os) const override { os << keyword; }
};

class avg_pool_3d : public layer
{
public:
    static constexpr const char* keyword = "avg_pool";
    tipl::shape<3> out_dim;

    avg_pool_3d(int c) : layer(c) {}

    const tipl::shape<3>& init_image(const tipl::shape<3>& dim_) override
    {
        dim = dim_;
        out_dim = {dim[0] / 2,dim[1] / 2,dim[2] / 2};
        out_buffer_size = out_size = out_dim.size() * out_channels_;
        return out_dim;
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_avg_pool_3d_forward<float>(in,out_ptr,out_channels_,dim.depth(),dim.height(),dim.width(),out_dim.depth(),out_dim.height(),out_dim.width()),void();
        cpu_avg_pool_3d_forward(in,out_ptr,out_channels_,dim.depth(),dim.height(),dim.width(),out_dim.depth(),out_dim.height(),out_dim.width());
    }

    void print(std::ostream& os) const override { os << keyword; }
};

class avg_pooling_xy_3d : public layer
{
public:
    static constexpr const char* keyword = "avg_pooling_xy";
    tipl::shape<3> out_dim;

    avg_pooling_xy_3d(int c) : layer(c) {}

    const tipl::shape<3>& init_image(const tipl::shape<3>& dim_) override
    {
        dim = dim_;
        out_dim = {dim[0]/2,dim[1]/2,dim[2]};
        out_buffer_size = out_size = out_dim.size()*out_channels_;
        return out_dim;
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_avg_pooling_xy_3d_forward<float>(
                           in,out_ptr,out_channels_,
                           dim.depth(),dim.height(),dim.width(),
                           out_dim.depth(),out_dim.height(),out_dim.width()),void();

        cpu_avg_pooling_xy_3d_forward(
            in,out_ptr,out_channels_,
            dim.depth(),dim.height(),dim.width(),
            out_dim.depth(),out_dim.height(),out_dim.width());
    }

    void print(std::ostream& os) const override     {os << keyword;}
};

class upsample_3d : public layer
{
public:
    static constexpr const char* keyword = "upsample";
    tipl::shape<3> out_dim;

    upsample_3d(int c) : layer(c) {}

    const tipl::shape<3>& init_image(const tipl::shape<3>& dim_) override
    {
        dim = dim_;
        out_dim = {dim[0] * 2,dim[1] * 2,dim[2] * 2};
        out_buffer_size = out_size = out_dim.size() * out_channels_;
        return out_dim;
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_upsample_3d_forward<float>(in,out_ptr,out_channels_,dim.depth(),dim.height(),dim.width(),out_dim.depth(),out_dim.height(),out_dim.width()),void();
        cpu_upsample_3d_forward(in,out_ptr,out_channels_,dim.depth(),dim.height(),dim.width(),out_dim.depth(),out_dim.height(),out_dim.width());
    }

    void print(std::ostream& os) const override { os << keyword; }
};

class layer_sequence
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
public:
    std::vector<std::shared_ptr<layer>> layers;

    std::vector<std::pair<float*,size_t>> parameters(void)
    {
        std::vector<std::pair<float*,size_t>> param;
        for(auto& l : layers)
        {
            auto p = l->parameters();
            param.insert(param.end(),p.begin(),p.end());
        }
        return param;
    }

    size_t param_size(void) const
    {
        size_t total_size = 0;
        for(auto& l : layers)
            total_size += l->param_size();
        return total_size;
    }

    float* allocate_param(float* ptr,bool is_gpu)
    {
        for(auto& l : layers)
            ptr = l->allocate_param(ptr,is_gpu);
        return ptr;
    }

    tipl::shape<3> init_image(const tipl::shape<3>& dim)
    {
        auto out_dim = dim;
        for(auto& l : layers)
            out_dim = l->init_image(out_dim);
        return out_dim;
    }

    size_t arrange_buffer(float* ptr = nullptr)
    {
        size_t max_mem = 0;
        size_t in_beg = std::numeric_limits<size_t>::max();
        size_t in_end = in_beg;

        for(auto& l : layers)
        {
            size_t buf_size = l->out_buffer_size;
            if(buf_size < in_beg)
                in_beg = 0;
            else
                in_beg = in_end;

            in_end = in_beg + buf_size;

            if(ptr)
                l->out = ptr + in_beg;
            if(in_end > max_mem)
                max_mem = in_end;
        }
        return max_mem;
    }

    const float* forward(const float* in_ptr,const std::function<bool(size_t,size_t)>& prog = nullptr)
    {
        for(size_t i = 0;i < layers.size();++i)
        {
            if(prog && !prog(i,layers.size()))
                return in_ptr;
            layers[i]->forward(in_ptr,layers[i]->out);
            in_ptr = layers[i]->out;
        }
        return in_ptr;
    }

    void print(std::ostream& os) const
    {
        bool first = true;
        for(auto& l : layers)
        {
            if(!first)
                os << "+";
            l->print(os);
            first = false;
        }
    }

    std::shared_ptr<layer> make_layer(const std::string& def,int in_c);
    std::shared_ptr<layer> append_layer(const std::string& def,int in_c)
    {
        auto l = make_layer(def,in_c);
        layers.push_back(l);
        return l;
    }

    void push_back(std::shared_ptr<layer> l) { layers.push_back(l); }
    size_t size(void) const { return layers.size(); }
    bool empty(void) const { return layers.empty(); }
    auto back(void) { return layers.back(); }
    auto back(void) const { return layers.back(); }
};

template<activation_type Act = activation_type::none>
class split_merge_3d : public layer
{
public:
    static constexpr const char* keyword = "split_merge";

    layer_sequence main_branch;
    layer_sequence skip_branch;
    bool skip_is_identity = false;

    tipl::shape<3> out_dim;
    size_t main_branch_mem_size = 0;
    size_t skip_branch_mem_size = 0;

public:
    split_merge_3d(int in_c,bool skip_identity_ = false) :
        layer(in_c,in_c), skip_is_identity(skip_identity_) {}

public:
    layer_sequence& main_layers(void) { return main_branch; }
    layer_sequence& skip_layers(void) { return skip_branch; }
    const layer_sequence& main_layers(void) const { return main_branch; }
    const layer_sequence& skip_layers(void) const { return skip_branch; }

    void set_skip_identity(bool value = true) { skip_is_identity = value; }

    std::vector<std::pair<float*,size_t>> parameters() override
    {
        auto p1 = main_branch.parameters();
        auto p2 = skip_branch.parameters();
        p1.insert(p1.end(),p2.begin(),p2.end());
        return p1;
    }

    size_t param_size(void) override
    {
        return main_branch.param_size()+skip_branch.param_size();
    }

    float* allocate_param(float* ptr,bool is_gpu_) override
    {
        is_gpu = is_gpu_;
        ptr = main_branch.allocate_param(ptr,is_gpu);
        ptr = skip_branch.allocate_param(ptr,is_gpu);
        return ptr;
    }

    const tipl::shape<3>& init_image(const tipl::shape<3>& dim_) override
    {
        dim = dim_;

        out_dim = dim_;
        if(!main_branch.empty())
            out_dim = main_branch.init_image(dim_);
        else
            throw std::runtime_error("split_merge requires a non-empty main branch");

        tipl::shape<3> skip_dim = dim_;
        int main_out_channels = main_branch.back()->out_channels_;
        int skip_out_channels = in_channels_;

        if(skip_is_identity)
        {
            skip_dim = dim_;
            skip_out_channels = in_channels_;
        }
        else
        {
            if(skip_branch.empty())
                throw std::runtime_error("split_merge skip branch is empty and not identity");
            skip_dim = skip_branch.init_image(dim_);
            skip_out_channels = skip_branch.back()->out_channels_;
        }

        if(skip_dim != out_dim)
            throw std::runtime_error("split_merge main/skip branch output dimensions mismatch");
        if(skip_out_channels != main_out_channels)
            throw std::runtime_error("split_merge main/skip branch channel count mismatch");

        out_channels_ = main_out_channels;
        out_size = out_dim.size()*out_channels_;

        main_branch_mem_size = main_branch.arrange_buffer();
        skip_branch_mem_size = skip_is_identity ? 0 : skip_branch.arrange_buffer();

        out_buffer_size = out_size + main_branch_mem_size + skip_branch_mem_size;
        return out_dim;
    }

    void forward(const float* in,float* out_ptr) override
    {
        float* main_ptr = out_ptr + out_size;
        float* skip_ptr = main_ptr + main_branch_mem_size;

        main_branch.arrange_buffer(main_ptr);
        const float* main_out = main_branch.forward(in);

        const float* skip_out = in;
        if(!skip_is_identity)
        {
            skip_branch.arrange_buffer(skip_ptr);
            skip_out = skip_branch.forward(in);
        }

        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
            {
                cuda_add_3d_forward<Act,float>(main_out,skip_out,out_ptr,out_size,0.01f);
                return;
            }

        cpu_add_3d_forward<Act>(main_out,skip_out,out_ptr,out_size,(float)0.01f);
    }

    void print(std::ostream& os) const override
    {
        auto print_inline_branch = [&](const layer_sequence& seq)
        {
            bool first = true;
            for(auto& l : seq.layers)
            {
                if(!first)
                    os << "+";
                l->print(os);
                first = false;
            }
        };

        os << keyword << "(";
        print_inline_branch(main_branch);
        os << "|";
        if(skip_is_identity)
            os << "identity";
        else
            print_inline_branch(skip_branch);
        os << ")";
        this->print_activation<Act>(os);
    }

};


inline std::shared_ptr<layer> layer_sequence::make_layer(const std::string& def_,int in_c)
{
    auto build_layer_sequence = [&](layer_sequence& seq,const std::string& branch,int in_c)
    {
        int cur_c = in_c;
        for(const auto& token : tipl::split(branch,'+','(',')'))
        {
            if(token.empty())
                continue;
            auto l = seq.append_layer(token,cur_c);
            cur_c = l->out_channels_;
        }
    };

    std::string def = tipl::trim_space(def_);

    if(tipl::begins_with(def,std::string(split_merge_3d<>::keyword)+"("))
    {
        const size_t open = def.find('(');
        int depth = 0;
        size_t close = std::string::npos;

        for(size_t i = open;i < def.size();++i)
        {
            if(def[i] == '(')
                ++depth;
            else if(def[i] == ')' && --depth == 0)
            {
                close = i;
                break;
            }
        }
        if(close == std::string::npos)
            throw std::runtime_error("invalid split_merge syntax: " + def);

        const std::string inside = def.substr(open+1,close-open-1);
        auto branches = tipl::split(inside,'|','(',')');
        if(branches.size() != 2)
            throw std::runtime_error("split_merge requires two branches: " + def);

        const std::string main_branch = tipl::trim_space(branches[0]);
        const std::string skip_branch = tipl::trim_space(branches[1]);

        auto tail = tipl::trim_space(def.substr(close+1));
        if(!tail.empty() && tail[0] == ',')
            tail.erase(tail.begin());

        auto options = tipl::split(tail,',','(',')');
        std::unordered_map<std::string,std::string> params;
        for(auto& opt : options)
            if(!tipl::trim_space(opt).empty())
                params[tipl::trim_space(opt)] = "1";

        auto make_split = [&](auto act_tag)
        {
            using split_type = split_merge_3d<decltype(act_tag)::value>;
            bool identity = skip_branch == "identity";
            auto l = std::make_shared<split_type>(in_c,identity);

            build_layer_sequence(l->main_branch,main_branch,in_c);
            if(!identity)
                build_layer_sequence(l->skip_branch,skip_branch,in_c);

            if(l->main_branch.empty())
                throw std::runtime_error("split_merge main branch is empty: " + def);

            // Critical: out_channels_ must be known during parsing,
            // before init_image(), because the next layer uses it.
            l->out_channels_ = l->main_branch.back()->out_channels_;
            return std::static_pointer_cast<layer>(l);
        };

        if(params.count(elu_keyword))
            return make_split(std::integral_constant<activation_type,activation_type::elu>{});
        if(params.count(leaky_relu_keyword))
            return make_split(std::integral_constant<activation_type,activation_type::leaky_relu>{});
        if(params.count(relu_keyword))
            return make_split(std::integral_constant<activation_type,activation_type::relu>{});
        return make_split(std::integral_constant<activation_type,activation_type::none>{});
    }

    std::unordered_map<std::string,std::string> params;
    for(const auto& arg : tipl::split(def,',','(',')'))
    {
        auto a = tipl::trim_space(arg);
        size_t pos = a.find_first_of("0123456789");
        if(pos != std::string::npos)
            params[a.substr(0,pos)] = a.substr(pos);
        else
            params[a] = "1";
    }

    std::shared_ptr<layer> l;

    if(params.count(max_pool_3d::keyword))
        l.reset(new max_pool_3d(in_c));
    else if(params.count(avg_pool_3d::keyword))
        l.reset(new avg_pool_3d(in_c));
    else if(params.count(avg_pooling_xy_3d::keyword))
        l.reset(new avg_pooling_xy_3d(in_c));
    else if(params.count(upsample_3d::keyword))
        l.reset(new upsample_3d(in_c));
    else if(params.count(conv_transpose_3d::keyword))
    {
        int ks = params.count(kernel_size_keyword) ? std::stoi(params[kernel_size_keyword]) : 2;
        int stride = params.count(stride_keyword) ? std::stoi(params[stride_keyword]) : 2;
        if(ks != 2 || stride != 2)
            throw std::runtime_error("conv_trans supports only ks2 stride2");
        l.reset(new conv_transpose_3d(in_c,std::stoi(params[conv_transpose_3d::keyword]),ks,stride));
    }
    else if(params.count(conv_transpose_xy_3d::keyword))
    {
        int ks = params.count(kernel_size_keyword) ? std::stoi(params[kernel_size_keyword]) : 2;
        int stride = params.count(stride_keyword) ? std::stoi(params[stride_keyword]) : 2;
        if(ks != 2 || stride != 2)
            throw std::runtime_error("conv_trans_xy supports only ks2 stride2, meaning ks2x2x1 stride2x2x1");
        l.reset(new conv_transpose_xy_3d(in_c,std::stoi(params[conv_transpose_xy_3d::keyword])));
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
    else if(params.count(conv_xy_3d<>::keyword))
    {
        int out_ch = std::stoi(params[conv_xy_3d<>::keyword]);
        int ks = params.count(kernel_size_keyword) ? std::stoi(params[kernel_size_keyword]) : 3;
        int stride = params.count(stride_keyword) ? std::stoi(params[stride_keyword]) : 2;
        if(ks != 3 || stride != 2)
            throw std::runtime_error("conv_xy supports only ks3 stride2, meaning stride2x2x1");

        l = make_act_layer<conv_xy_3d>(params,in_c,out_ch);
    }
    else if(params.count(instance_norm_3d<>::keyword))
        l = make_act_layer<instance_norm_3d>(params,in_c);
    else if(params.count(batch_norm_3d<>::keyword))
        l = make_act_layer<batch_norm_3d>(params,in_c);
    else
        throw std::runtime_error("unknown layer:" + def);

    return l;
}

class network : public layer, public layer_sequence
{
protected:
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

    tipl::device_vector<float> gpu_buf_memory;
    std::vector<float> buf_memory;

public:
    std::function<bool(int,int)> prog = nullptr;

public:
    network() : layer(1,1) {}
    network(int in_c,int out_c) : layer(in_c,out_c) {}

    std::vector<std::pair<float*,size_t>> parameters() override
    {
        size_t size = param_size();
        if(!size)
            return {};
        if(memory.empty())
            allocate_param((memory = std::vector<float>(size)).data(),is_gpu = false);
        return layer_sequence::parameters();
    }

    size_t param_size(void) override    {return layer_sequence::param_size();}
    const tipl::shape<3>& init_image(const tipl::shape<3>& dim_) override
    {
        dim = dim_;
        layer_sequence::init_image(dim_);
        out_size = layers.empty() ? 0 : layers.back()->out_size;
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

    float* allocate_param(float* ptr,bool is_gpu_mem) override  {return layer_sequence::allocate_param(ptr,is_gpu = is_gpu_mem);}
    virtual void allocate_buffer(void)
    {
        size_t max_mem = arrange_buffer();
        float* ptr = get_ptr(buf_memory,gpu_buf_memory,max_mem);
        arrange_buffer(ptr);
        out = layers.empty() ? nullptr : layers.back()->out;
    }

    void forward(const float* in_ptr,float*) override
    {
        if(!out)
            allocate_buffer();
        layer_sequence::forward(in_ptr,[&](size_t i,size_t n){return !prog || prog(int(i),int(n));});
    }

    void print(std::ostream& os) const override     {layer_sequence::print(os);}
    std::shared_ptr<layer> create_layer(const std::string& def,int in_c)     {return append_layer(def,in_c);}
};

class unet3d : public network
{
protected:
    std::vector<std::vector<std::shared_ptr<layer>>> encoding, decoding;
    std::vector<std::shared_ptr<layer>> decoding_head;
protected:
    tipl::device_vector<float> gpu_skip_memory;
    std::vector<float> skip_memory;
    std::string arch;

public:
    int round_up_multiple[3] = {1,1,1};

public:
    unet3d(const std::string& structure,int in_c,int out_c) : network(in_c,out_c), arch(structure)
    {
        std::vector<std::vector<std::string>> enc_tokens, dec_tokens;
        {
            std::vector<std::string> all_lines(tipl::split_by_line_breaks(structure));
            if(all_lines.size() < 3) throw std::runtime_error("invalid u-net structure");
            size_t enc_count = all_lines.size() / 2 + 1;
            for(size_t i = 0;i < all_lines.size();++i)
                (i < enc_count ? enc_tokens : dec_tokens).push_back(tipl::split(all_lines[i],'+','(',')'));
        }

        for(size_t i = 1;i < enc_tokens.size();++i)
        {
            bool xy = false;
            for(const auto& token : enc_tokens[i])
                if(token.find("conv_xy") != std::string::npos)
                    xy = true;

            round_up_multiple[0] <<= 1;
            round_up_multiple[1] <<= 1;
            if(!xy)
                round_up_multiple[2] <<= 1;
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
        size_t persistent_size = 0;
        size_t max_memory_reached = 0;

        std::vector<size_t> enc_offset(decoding.size());
        std::vector<size_t> dec_offset(decoding.size());

        for(size_t i = 0;i < decoding.size();++i)
        {
            auto enc = encoding[i].back();
            auto dec = decoding_head[i];

            enc_offset[i] = persistent_size;
            max_memory_reached = std::max(max_memory_reached,
                                          persistent_size + enc->out_buffer_size);
            persistent_size += enc->out_size;

            dec_offset[i] = persistent_size;
            max_memory_reached = std::max(max_memory_reached,
                                          persistent_size + dec->out_buffer_size);
            persistent_size += dec->out_size;

            enc->out_buffer_size = 0;
            dec->out_buffer_size = 0;
        }

        network::allocate_buffer();

        auto skip_ptr = get_ptr(skip_memory,gpu_skip_memory,
                                std::max(persistent_size,max_memory_reached));

        for(size_t i = 0;i < decoding_head.size();++i)
        {
            encoding[i].back()->out = skip_ptr + enc_offset[i];
            decoding_head[i]->out = skip_ptr + dec_offset[i];
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
    void print(std::ostream& os) const override     {os << arch;}
};

} // namespace ml3d
} // namespace tipl

#endif // CNN3D_HPP
