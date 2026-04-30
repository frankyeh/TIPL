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

static constexpr const char* kernel_size_keyword = "ks";
static constexpr const char* stride_keyword = "stride";
static constexpr const char* leaky_relu_keyword = "leaky_relu";
static constexpr const char* elu_keyword = "elu";
static constexpr const char* relu_keyword = "relu";


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
    virtual void forward(const float* in_ptr,float* out_ptr) = 0;
    virtual void print(std::ostream& out) const = 0;
    virtual void allocate(float*& ptr, bool is_gpu_mem)
    {
        is_gpu = is_gpu_mem;
        out = ptr; ptr += out_buffer_size;
    }
    virtual bool change_dim(void) const {return false;}
};

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
        out_dim = dim_ = tipl::s(dim_[0] / stride_, dim_[1] / stride_, dim_[2] / stride_);
        out_buffer_size = out_size = out_dim.size() * out_channels_;
    }

    void allocate(float*& ptr, bool is_gpu_mem) override {
        weight = ptr; ptr += weight_size;
        bias = ptr; ptr += bias_size;
        layer::allocate(ptr, is_gpu_mem);
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_conv_3d_forward<Act>(in, weight, bias, out_ptr, in_channels_, out_channels_, dim.depth(), dim.height(), dim.width(), out_dim.depth(), out_dim.height(), out_dim.width(), kernel_size_, kernel_size3, range, stride_, 0.01f),void();
        cpu_conv_3d_forward<Act>(in, weight, bias, out_ptr, in_channels_, out_channels_, dim.depth(), dim.height(), dim.width(), out_dim.depth(), out_dim.height(), out_dim.width(), kernel_size_, kernel_size3, range, stride_);
    }

    void print(std::ostream& os) const override
    {
        os << keyword << out_channels_ << "," << kernel_size_keyword << kernel_size_ << "," << stride_keyword << stride_;
        if constexpr(Act == activation_type::relu)
            os << "," << relu_keyword;
        if constexpr(Act == activation_type::leaky_relu)
            os << "," << leaky_relu_keyword;
        if constexpr(Act == activation_type::elu)
            os << "," << elu_keyword;
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
        out_dim = dim_ = tipl::s(dim_[0] * stride_, dim_[1] * stride_, dim_[2] * stride_);
        out_buffer_size = out_size = out_dim.size() * out_channels_;
    }

    void allocate(float*& ptr, bool is_gpu_mem) override {
        weight = ptr; ptr += weight_size;
        bias = ptr; ptr += bias_size;
        layer::allocate(ptr, is_gpu_mem);
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_conv_transpose_3d_forward(in, weight, bias, out_ptr, in_channels_, out_channels_, dim.depth(), dim.height(), dim.width(), out_dim.depth(), out_dim.height(), out_dim.width(), kernel_size_, kernel_size3, stride_), void();
        cpu_conv_transpose_3d_forward(in, weight, bias, out_ptr, in_channels_, out_channels_, dim.depth(), dim.height(), dim.width(), out_dim.depth(), out_dim.height(), out_dim.width(), kernel_size_, kernel_size3, stride_);
    }

    void print(std::ostream& os) const override { os << keyword << out_channels_ << "," << kernel_size_keyword << kernel_size_ << "," << stride_keyword << stride_; }
    bool change_dim(void) const override{return stride_ != 1;}
};

template<activation_type Act = activation_type::none>
class batch_norm_3d : public layer
{
public:
    static constexpr const char* keyword = "bnorm";
    float* weight = nullptr;
    float* bias = nullptr;
    size_t weight_size = 0, bias_size = 0;

    batch_norm_3d(int c) : layer(c)
    {
        weight_size = c;
        bias_size = c;
    }
    std::vector<std::pair<float*,size_t>> parameters() override
    {
        return {{weight,weight_size},{bias,bias_size}};
    }
    void allocate(float*& ptr,bool is_gpu_mem) override
    {
        weight = ptr; ptr += weight_size;
        bias = ptr; ptr += bias_size;
        layer::allocate(ptr,is_gpu_mem);
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_batch_norm_3d_forward<Act>(in,out_ptr,weight,bias,out_channels_,dim.size()),void();
        cpu_batch_norm_3d_forward<Act>(in,out_ptr,weight,bias,out_channels_,dim.size());
    }

    void print(std::ostream& os) const override
    {
        os << keyword;
        if constexpr(Act == activation_type::relu)
            os << "," << relu_keyword;
        if constexpr(Act == activation_type::leaky_relu)
            os << "," << leaky_relu_keyword;
        if constexpr(Act == activation_type::elu)
            os << "," << elu_keyword;
    }
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
    void allocate(float*& ptr, bool is_gpu_mem) override
    {
        weight = ptr; ptr += weight_size;
        bias = ptr; ptr += bias_size;
        layer::allocate(ptr, is_gpu_mem);
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_instance_norm_3d_forward<Act>(in, out_ptr, weight, bias, out_channels_, dim.size(), 0.01f), void();
        cpu_instance_norm_3d_forward<Act>(in, out_ptr, weight, bias, out_channels_, dim.size()), in;
    }

    void print(std::ostream& os) const override {
        os << keyword;
        if constexpr(Act == activation_type::relu)
            os << "," << relu_keyword;
        if constexpr(Act == activation_type::leaky_relu)
            os << "," << leaky_relu_keyword;
        if constexpr(Act == activation_type::elu)
            os << "," << elu_keyword;
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
        out_buffer_size = out_size = out_dim.size() * out_channels_;
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_max_pool_3d_forward(in, out_ptr, out_channels_, dim.depth(), dim.height(), dim.width(), out_dim.depth(), out_dim.height(), out_dim.width(), pool_size), void();
        cpu_max_pool_3d_forward(in, out_ptr, out_channels_, dim.depth(), dim.height(), dim.width(), out_dim.depth(), out_dim.height(), out_dim.width(), pool_size);
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
        out_buffer_size = out_size = out_dim.size() * out_channels_;
    }

    void forward(const float* in,float* out_ptr) override
    {
        if constexpr(tipl::use_cuda)
            if(this->is_gpu)
                return cuda_upsample_3d_forward(in, out_ptr, out_channels_, dim.depth(), dim.height(), dim.width(), out_dim.depth(), out_dim.height(), out_dim.width(), pool_size), void();
        cpu_upsample_3d_forward(in, out_ptr, out_channels_, dim.depth(), dim.height(), dim.width(), out_dim.depth(), out_dim.height(), out_dim.width(), pool_size);
    }

    void print(std::ostream& os) const override { os << keyword; }
    bool change_dim(void) const override{return true;}
};

class network : public layer
{
protected:
    tipl::device_vector<float> gpu_memory;
    std::vector<float> memory;
public:
    std::function<bool(int,int)> prog = nullptr;
    std::vector<std::shared_ptr<layer>> layers;

    network() : layer(1,1) {}
    network(int in_c,int out_c) : layer(in_c,out_c) {}

    std::vector<std::pair<float*,size_t>> parameters() override
    {
        if(memory.empty())
        {
            size_t total_size = 0;
            for(auto& each_layer : layers)
            {
                for(auto& each_param : each_layer->parameters())
                    total_size += each_param.second;
                total_size += each_layer->out_buffer_size;
            }
            if(!total_size)
                throw std::runtime_error("no memory to allocate for network");
            memory.resize(total_size);
            auto ptr = memory.data();
            allocate(ptr,is_gpu = false);
        }
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
    void to_gpu(void)
    {
        gpu_memory = memory;
        memory = std::vector<float>();
        auto ptr = gpu_memory.data();
        allocate(ptr,is_gpu = true);
    }
    void allocate(float*& ptr, bool is_gpu_mem) override
    {
        is_gpu = is_gpu_mem;
        for(auto& l : layers)
            l->allocate(ptr,is_gpu);
        out = layers.back()->out;
    }
    void forward(const float* in_ptr,float*) override
    {
        for(size_t i = 0;i < layers.size();++i)
        {
            if(prog && !prog(i,layers.size()))
                return;
            layers[i]->forward(in_ptr,layers[i]->out);
            in_ptr = layers[i]->out;
        }
    }

    void push_back(std::shared_ptr<layer> l)    {layers.push_back(l);}
    size_t size(void) const                     {return layers.size();}
    auto back(void)                             {return layers.back();}

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
            if(ks != 2 || stride != 2)
                throw std::runtime_error("conv_trans supports only ks2 stride2");
            l.reset(new conv_transpose_3d(in_c,std::stoi(params[conv_transpose_3d::keyword]),ks,stride));
        }
        else if(params.count(conv_3d<>::keyword))
        {
            int out_ch = std::stoi(params[conv_3d<>::keyword]);
            int ks = params.count(kernel_size_keyword) ? std::stoi(params[kernel_size_keyword]) : 3;
            int stride = params.count(stride_keyword) ? std::stoi(params[stride_keyword]) : 1;

            if(!((ks == 1 && stride == 1) || (ks == 3 && (stride == 1 || stride == 2))))
                throw std::runtime_error("conv supports only ks1 stride1, ks3 stride1, and ks3 stride2");

            if(params.count(elu_keyword))
                l.reset(new conv_3d<activation_type::elu>(in_c,out_ch,ks,stride));
            else if(params.count(leaky_relu_keyword))
                l.reset(new conv_3d<activation_type::leaky_relu>(in_c,out_ch,ks,stride));
            else if(params.count(relu_keyword))
                l.reset(new conv_3d<activation_type::relu>(in_c,out_ch,ks,stride));
            else
                l.reset(new conv_3d<activation_type::none>(in_c,out_ch,ks,stride));
        }
        else if(params.count(instance_norm_3d<>::keyword))
        {
            if(params.count(elu_keyword))
                l.reset(new instance_norm_3d<activation_type::elu>(in_c));
            else if(params.count(leaky_relu_keyword))
                l.reset(new instance_norm_3d<activation_type::leaky_relu>(in_c));
            else if(params.count(relu_keyword))
                l.reset(new instance_norm_3d<activation_type::relu>(in_c));
            else
                l.reset(new instance_norm_3d<activation_type::none>(in_c));
        }
        else if(params.count(batch_norm_3d<>::keyword))
        {
            if(params.count(elu_keyword))
                l.reset(new batch_norm_3d<activation_type::elu>(in_c));
            else if(params.count(leaky_relu_keyword))
                l.reset(new batch_norm_3d<activation_type::leaky_relu>(in_c));
            else if(params.count(relu_keyword))
                l.reset(new batch_norm_3d<activation_type::relu>(in_c));
            else
                l.reset(new batch_norm_3d<activation_type::none>(in_c));
        }
        else
            throw std::runtime_error("unknown layer:"+params[0]);
        layers.push_back(l);
        return l;
    }
};

class unet3d : public network
{
    std::vector<network> encoding, decoding, up;
public:
    unet3d(const std::string& structure,int in_c,int out_c) : network(in_c,out_c)
    {
        std::vector<std::vector<std::string> > enc_tokens,dec_tokens;
        {
            std::vector<std::string> all_lines(tipl::split_in_lines(structure));
            if(all_lines.size() < 3)
                throw std::runtime_error("invalid u-net structure");
            size_t enc_count = all_lines.size() / 2 + 1;
            for(size_t i = 0;i < all_lines.size();++i)
                (i < enc_count ? enc_tokens : dec_tokens).push_back(tipl::split(all_lines[i],'+'));
        }

        encoding.resize(enc_tokens.size());
        for(int level = 0;level < enc_tokens.size();++level)
            for(const auto& token : enc_tokens[level])
                encoding[level].push_back(create_layer(token));


        auto channel_count = layers.back()->out_channels_;
        for(int level = dec_tokens.size() - 1; level >= 0; --level)
        {
            const auto& tokens = dec_tokens[dec_tokens.size() - 1 - level];
            up.insert(up.begin(),network());
            decoding.insert(decoding.begin(),network());

            size_t skip_conn_loc = std::find(tokens.begin(),tokens.end(),"skip_conn")-tokens.begin();
            if(skip_conn_loc + 1 >= tokens.size())
                throw std::runtime_error("invalid u-net structure: cannot find skip connection location");

            for(size_t t = 0; t < skip_conn_loc; ++t)
                up[0].push_back(network::create_layer(tokens[t],t ? layers.back()->out_channels_ : channel_count));

            // skip connection add additional channels
            decoding[0].push_back(create_layer(tokens[skip_conn_loc + 1],encoding[level].back()->out_channels_));

            for(size_t t = skip_conn_loc + 2; t < tokens.size(); ++t)
            {
                auto l = create_layer(tokens[t]);
                if(tokens[t] == dec_tokens.back().back()) // final or deep supervision outputs
                    break;
                decoding[0].push_back(l);
            }
            channel_count = decoding[0].back()->out_channels_;
        }
    }
    std::shared_ptr<layer> create_layer(const std::string& def,int in_c = 0)
    {
        if(layers.empty())
            in_c = in_channels_;
        else
            in_c += layers.back()->out_channels_; // most common condition, follow previous layer's out channels
        return network::create_layer(def,in_c);
    }

    void init_image(tipl::shape<3>& dim_) override
    {
        network::init_image(dim_);
        for(size_t i = 0; i < up.size(); ++i)
            encoding[i].back()->out_buffer_size += up[i].back()->out_size;
    }
    void allocate(float*& ptr, bool is_gpu_) override
    {
        network::allocate(ptr,is_gpu = is_gpu_);
        for(size_t i = 0; i < up.size(); ++i)
            up[i].back()->out = encoding[i].back()->out + encoding[i].back()->out_size;
    }
    void forward(const float* in_ptr,float*) override
    {
        int n_levels = static_cast<int>(encoding.size());
        for(int i = 0; i < n_levels; ++i)
        {
            if(prog && !prog(i,n_levels+n_levels))
                return;
            encoding[i].forward(in_ptr,nullptr);
            in_ptr = encoding[i].back()->out;
        }
        for(int i = n_levels - 2; i >= 0; --i)
        {
            if(prog && !prog(n_levels+n_levels-i-1,n_levels+n_levels))
                return;
            up[i].forward(in_ptr,nullptr);
            decoding[i].forward(encoding[i].back()->out,nullptr);
            in_ptr = decoding[i].back()->out;
        }
        layers.back()->forward(in_ptr,layers.back()->out);
    }
    void forward(tipl::image<3>& in)
    {
        if constexpr(tipl::use_cuda)
            if(is_gpu)
            {
                forward(tipl::device_image<3,float>(in).data(),nullptr);
                return;
            }
        forward(in.data(),nullptr);
    }
};


} // namespace ml3d
} // namespace tipl

#endif // CNN3D_HPP
