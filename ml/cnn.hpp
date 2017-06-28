#ifndef CNN_HPP
#define CNN_HPP
#include <algorithm>
#include <regex>
#include <exception>
#include <set>
#include <deque>
#include <sstream>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <numeric>
#include <iomanip>
#include <memory>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

#include "image/numerical/matrix.hpp"
#include "image/numerical/numerical.hpp"
#include "image/numerical/basic_op.hpp"
#include "image/numerical/resampling.hpp"
#include "image/utility/geometry.hpp"
#include "image/utility/basic_image.hpp"
#include "image/utility/multi_thread.hpp"


namespace image
{
namespace ml
{

const float bias_cap = 10.0f;
const float weight_cap = 100.0f;

template<class value_type>
inline float tanh_f(value_type v)
{
    if(v < -bias_cap)
        return -1.0;
    if(v > bias_cap)
        return 1.0f;
    const float ep = expf(v + v);
    return (ep - float(1)) / (ep + float(1));
}
template<class value_type>
inline float tanh_df(value_type y)
{
    return float(1) - y * y;
}


template<class value_type>
inline float sigmoid_f(value_type v)
{
    if(v < -bias_cap)
        return 0.0f;
    if(v > bias_cap)
        return 1.0f;
    return float(1) / (float(1) + expf(-v));
}
template<class value_type>
inline float sigmoid_df(value_type y)
{
    return y * (float(1) - y);
}

template<class value_type>
inline float relu_f(value_type v)
{
    if(v > bias_cap)
        return bias_cap;
    return v > value_type(0) ? float(v) : float(0);
}
template<class value_type>
inline float relu_df(value_type y)
{
    return y > value_type(0) ? float(1) : float(0);
}

enum activation_type { tanh, sigmoid, relu, identity};
enum status_type { training,testing};

class basic_layer
{

public:
    activation_type af;
    status_type status;
    int input_size;
    int output_size;
    float weight_base;
    std::vector<float> weight,bias;
public:

    virtual ~basic_layer() {}
    basic_layer(activation_type af_ = activation_type::tanh):af(af_),status(testing),weight_base(1){}
    void init( int input_size_, int output_size_, int weight_dim, int bias_dim)
    {
        input_size = input_size_;
        output_size = output_size_;
        weight.resize(weight_dim);
        bias.resize(bias_dim);
    }
    void initialize_weight(image::uniform_dist<float>& gen)
    {
        for(int i = 0; i < weight.size(); ++i)
            weight[i] = gen()*weight_base;
        std::fill(bias.begin(), bias.end(), 0.0f);
    }

    virtual bool init(const image::geometry<3>& in_dim,const image::geometry<3>& out_dim) = 0;
    virtual void forward_propagation(const float* data,float* out) = 0;
    void forward_af(float* data)
    {
        if(af == activation_type::tanh)
            for(int i = 0; i < output_size; ++i)
                data[i] = tanh_f(data[i]);
        if(af == activation_type::sigmoid)
            for(int i = 0; i < output_size; ++i)
                data[i] = sigmoid_f(data[i]);
        if(af == activation_type::relu)
            for(int i = 0; i < output_size; ++i)
                data[i] = relu_f(data[i]);
    }
    virtual void to_image(basic_image<float,2>& I)
    {
        I.clear();
    }
    virtual void calculate_dwdb(const float*,
                                  const float*,
                                  std::vector<float>&,
                                  std::vector<float>&){}
    virtual void back_propagation(float* in_dE_da,
                                  float* out_dE_da,
                                  const float*) = 0;
    void back_af(float* dE_da,const float* prev_out)
    {
        if(af == activation_type::tanh)
            for(int i = 0; i < output_size; ++i)
                dE_da[i] *= tanh_df(prev_out[i]);
        if(af == activation_type::sigmoid)
            for(int i = 0; i < output_size; ++i)
                dE_da[i] *= sigmoid_df(prev_out[i]);
        if(af == activation_type::relu)
            for(int i = 0; i < output_size; ++i)
                if(prev_out[i] <= 0)
                    dE_da[i] = 0;
    }
    virtual unsigned int computation_cost(void) const
    {
        return unsigned int(weight.size());
    }
};



class fully_connected_layer : public basic_layer
{
    image::geometry<3> in_dim;
public:
    fully_connected_layer(activation_type af_):basic_layer(af_){}
    bool init(const image::geometry<3>& in_dim_,const image::geometry<3>& out_dim) override
    {
        in_dim = in_dim_;
        basic_layer::init(in_dim.size(), out_dim.size(),in_dim.size() * out_dim.size(), out_dim.size());
        weight_base = std::sqrtf(6.0f / (float)(input_size+output_size));
        return true;
    }
    void forward_propagation(const float* data,float* out) override
    {
        for(int i = 0,i_pos = 0;i < output_size;++i,i_pos += input_size)
            out[i] = bias[i] + image::vec::dot(&weight[i_pos],&weight[i_pos]+input_size,&data[0]);
    }
    void calculate_dwdb(const float* in_dE_da,
                        const float* prev_out,
                        std::vector<float>& dweight,
                        std::vector<float>& dbias) override
    {
        image::add(&dbias[0],&dbias[0]+output_size,in_dE_da);
        for(int i = 0,i_pos = 0; i < output_size; i++,i_pos += input_size)
            if(in_dE_da[i] != float(0))
                image::vec::axpy(&dweight[i_pos],&dweight[i_pos]+input_size,in_dE_da[i],prev_out);
    }
    void to_image(basic_image<float,2>& I)
    {
        std::vector<float> w(weight),b(bias);
        image::normalize_abs(w);
        image::normalize_abs(b);
        if(in_dim[0] == 1)
        {
            I.resize(geometry<2>(int(bias.size()),int(weight.size()/bias.size()+3)));
            std::copy(w.begin(),w.end(),I.begin()+I.width());
            std::copy(b.begin(),b.end(),I.end()-I.width()*2);
        }
        else
        {
            int n = int(weight.size()/in_dim.plane_size());
            int col = in_dim[2];
            int row = n/col;
            I.resize(geometry<2>(col* (in_dim.width()+1)+1,row * (in_dim.height() +1) + 3));
            for(int y = 0,index = 0;y < row;++y)
                for(int x = 0;x < col;++x,++index)
                {
                    image::draw(image::make_image(&w[0] + index*in_dim.plane_size(),image::geometry<2>(in_dim[0],in_dim[1])),
                                I,image::geometry<2>(x*(in_dim.width()+1),y*(in_dim.height()+1)+1));
                }
            std::copy(b.begin(),b.end(),I.end()-b.size());
        }
    }
    void back_propagation(float* in_dE_da,// output_size
                          float* out_dE_da,// input_size
                          const float*) override
    {
        image::mat::left_vector_product(&weight[0],in_dE_da,out_dE_da,image::dyndim(output_size,input_size));
    }
};


class partial_connected_layer : public basic_layer
{
protected:
    std::vector<std::vector<int> > w2o_1,w2o_2,o2w_1,o2w_2,i2w_1,i2w_2,b2o;
    std::vector<int> o2b;
public:
    partial_connected_layer(activation_type af_): basic_layer(af_){}
    void init(int in_dim, int out_dim, int weight_dim, int bias_dim)
    {
        basic_layer::init(in_dim, out_dim, weight_dim, bias_dim);
        w2o_1.resize(weight_dim);
        w2o_2.resize(weight_dim);
        o2w_1.resize(out_dim);
        o2w_2.resize(out_dim);
        i2w_1.resize(in_dim);
        i2w_2.resize(in_dim);
        b2o.resize(bias_dim);
        o2b.resize(out_dim);
        weight_base = std::sqrtf(6.0f / (float)(max_size(o2w_1) + max_size(i2w_1)));
    }

    template <class Container>
    static size_t max_size(const Container& c)
    {
        typedef typename Container::value_type value_t;
        return std::max_element(c.begin(), c.end(), [](const value_t& left, const value_t& right)
        {
            return left.size() < right.size();
        })->size();
    }

    void connect_weight(int input_index, int outputindex, int weight_index)
    {
        w2o_1[weight_index].push_back(input_index);
        w2o_2[weight_index].push_back(outputindex);
        o2w_1[outputindex].push_back(weight_index);
        o2w_2[outputindex].push_back(input_index);
        i2w_1[input_index].push_back(weight_index);
        i2w_2[input_index].push_back(outputindex);
    }


    void forward_propagation(const float* data,float* out) override
    {
        for(int i = 0; i < output_size; ++i)
        {
            const std::vector<int>& o2w_1i = o2w_1[i];
            const std::vector<int>& o2w_2i = o2w_2[i];
            float sum(0);
            for(int j = 0;j < o2w_1i.size();++j)
                sum += weight[o2w_1i[j]] * data[o2w_2i[j]];
            out[i] = sum + bias[o2b[i]];
        }
    }
    void calculate_dwdb(const float* in_dE_da,
                        const float* prev_out,
                        std::vector<float>& dweight,
                        std::vector<float>& dbias) override
    {
        for(int i = 0; i < w2o_1.size(); i++)
        {
            const std::vector<int>& w2o_1i = w2o_1[i];
            const std::vector<int>& w2o_2i = w2o_2[i];
            float sum(0);
            for(int j = 0;j < w2o_1i.size();++j)
                sum += prev_out[w2o_1i[j]] * in_dE_da[w2o_2i[j]];
            dweight[i] += sum;
        }

        for(int i = 0; i < b2o.size(); i++)
        {
            const std::vector<int>& outs = b2o[i];
            float sum(0);
            for(int j = 0;j < outs.size();++j)
                sum += in_dE_da[outs[j]];
            dbias[i] += sum;
        }
    }
    void back_propagation(float* in_dE_da,// output_size
                          float* out_dE_da,// input_size
                          const float*) override
    {
        for(int i = 0; i != input_size; i++)
        {
            const std::vector<int>& i2w_1i = i2w_1[i];
            const std::vector<int>& i2w_2i = i2w_2[i];
            float sum(0);
            for(int j = 0;j < i2w_1i.size();++j)
                sum += weight[i2w_1i[j]] * in_dE_da[i2w_2i[j]];
            out_dE_da[i] = sum;
        }
    }
};


class average_pooling_layer : public partial_connected_layer
{
public:
    int pool_size;
    average_pooling_layer(activation_type af_,int pool_size_)
        : partial_connected_layer(af_),pool_size(pool_size_){}
    bool init(const image::geometry<3>& in_dim,const image::geometry<3>& out_dim) override
    {
        partial_connected_layer::init(in_dim.size(),in_dim.size()/pool_size/pool_size,in_dim.depth(), in_dim.depth());
        if(out_dim != image::geometry<3>(in_dim.width()/ pool_size, in_dim.height() / pool_size, in_dim.depth()) ||
                in_dim.depth() != out_dim.depth())
            return false;
        for(int c = 0; c < in_dim.depth(); ++c)
            for(int y = 0; y < in_dim.height(); y += pool_size)
                for(int x = 0; x < in_dim.width(); x += pool_size)
                {
                    int dymax = std::min(pool_size, in_dim.height() - y);
                    int dxmax = std::min(pool_size, in_dim.width() - x);
                    int dstx = x / pool_size;
                    int dsty = y / pool_size;

                    for(int dy = 0; dy < dymax; ++dy)
                        for(int dx = 0; dx < dxmax; ++dx)
                            connect_weight((in_dim.height()*c + y + dy)*in_dim.width() + dx + x,
                                           (out_dim.height()*c + dsty)*out_dim.width() + dstx, c);

                }
        for(int c = 0, index = 0; c < in_dim.depth(); ++c)
            for(int y = 0; y < out_dim.height(); ++y)
                for(int x = 0; x < out_dim.width(); ++x, ++index)
                {
                    o2b[index] = c;
                    b2o[c].push_back(index);
                }
        weight_base = std::sqrtf(6.0f / (float)(max_size(o2w_1) + max_size(i2w_1)));
        return true;
    }
    void to_image(basic_image<float,2>& I)
    {
        I.resize(geometry<2>(int(weight.size()),5));
        std::vector<float> w(weight),b(bias);
        image::normalize_abs(w);
        image::normalize_abs(b);
        std::copy(w.begin(),w.end(),I.begin()+I.width());
        std::copy(b.begin(),b.end(),I.end()-I.width()*2);
    }
    virtual unsigned int computation_cost(void) const
    {
        return input_size*pool_size*pool_size;
    }
};

class max_pooling_layer : public basic_layer
{
    std::vector<std::vector<int> > o2i;
    std::vector<int> i2o;
    geometry<3> in_dim;
    geometry<3> out_dim;

public:
    int pool_size;
public:
    max_pooling_layer(activation_type af_,int pool_size_)
        : basic_layer(af_),pool_size(pool_size_){}
    bool init(const image::geometry<3>& in_dim_,const image::geometry<3>& out_dim_) override
    {
        in_dim = in_dim_;
        out_dim = out_dim_;
        basic_layer::init(in_dim.size(),out_dim.size(),0,0);
        if(out_dim != image::geometry<3>(in_dim.width()/ pool_size, in_dim.height() / pool_size, in_dim.depth()))
            return false;
        init_connection();
        weight_base = std::sqrtf(6.0f / (float)(o2i[0].size()+1));
        return true;
    }
    void forward_propagation(const float* data,float* out) override
    {
        for(int i = 0; i < o2i.size(); i++)
        {
            float max_value = std::numeric_limits<float>::lowest();
            for(auto j : o2i[i])
            {
                if(data[j] > max_value)
                    max_value = data[j];
            }
            out[i] = max_value;
        }
    }
    void back_propagation(float* in_dE_da,// output_size
                          float* out_dE_da,// input_size
                          const float* prev_out) override
    {
        std::vector<int> max_idx(out_dim.size());

        for(int i = 0; i < o2i.size(); i++)
        {
            float max_value = std::numeric_limits<float>::lowest();
            for(auto j : o2i[i])
            {
                if(prev_out[j] > max_value)
                {
                    max_value = prev_out[j];
                    max_idx[i] = j;
                }
            }
        }
        for(int i = 0; i < i2o.size(); i++)
        {
            int outi = i2o[i];
            out_dE_da[i] = (max_idx[outi] == i) ? in_dE_da[outi] : float(0);
        }
    }
    virtual unsigned int computation_cost(void) const
    {
        return unsigned int(out_dim.size()*pool_size*pool_size/10.0f);
    }
private:
    void init_connection(void)
    {
        i2o.resize(in_dim.size());
        o2i.resize(out_dim.size());
        for(int c = 0,out_index = 0; c < in_dim.depth(); ++c)
            for(int y = 0; y < out_dim.height(); ++y)
                for(int x = 0; x < out_dim.width(); ++x,++out_index)
                {
                    int from_x = x * pool_size;
                    int from_y = y * pool_size;
                    for(int dy = 0; dy < pool_size; dy++)
                        for(int dx = 0; dx < pool_size; dx++)
                        if(from_x + dx < in_dim.width() &&
                           from_y + dy < in_dim.height())
                        {
                            int in_index = (in_dim.height() * c + from_y + dy) * in_dim.width() + from_x + dx;
                            i2o[in_index] = out_index;
                            o2i[out_index].push_back(in_index);
                        }
                }
    }

};


class convolutional_layer : public basic_layer
{
    geometry<3> in_dim,out_dim;
public:
    int kernel_size,kernel_size2;
    // check if any kernel is zero and re-initialize it

public:
    convolutional_layer(activation_type af_,int kernel_size_)
        : basic_layer(af_),
          kernel_size(kernel_size_),kernel_size2(kernel_size_*kernel_size_)
    {
    }
    bool init(const image::geometry<3>& in_dim_,const image::geometry<3>& out_dim_) override
    {
        in_dim = in_dim_;
        out_dim = out_dim_;
        if(in_dim.width()-out_dim.width()+1 != kernel_size ||
           in_dim.height()-out_dim.height()+1 != kernel_size)
            return false;
        //weight_dim = image::geometry<3>(kernel_size,kernel_size,in_dim.depth() * out_dim.depth()),
        basic_layer::init(in_dim_.size(), out_dim_.size(),kernel_size2* in_dim.depth() * out_dim.depth(), out_dim.depth());
        weight_base = std::sqrtf(6.0f / (float)(kernel_size2 * in_dim.depth() + kernel_size2 * out_dim.depth()));
        return true;
    }
    void to_image(basic_image<float,2>& I)
    {
        std::vector<float> w(weight),b(bias);
        image::normalize_abs(w);
        image::normalize_abs(b);
        I.resize(geometry<2>(out_dim.depth()* (kernel_size+1)+1,in_dim.depth() * (kernel_size+1) + 3));
        for(int x = 0,index = 0;x < out_dim.depth();++x)
            for(int y = 0;y < in_dim.depth();++y,++index)
            {
                image::draw(image::make_image(&w[0] + index*kernel_size2,image::geometry<2>(kernel_size,kernel_size)),
                            I,image::geometry<2>(x*(kernel_size+1),y*(kernel_size+1)+1));
            }

        std::copy(b.begin(),b.end(),I.end()-I.width()*2);
    }
    void forward_propagation(const float* data,float* out) override
    {
        for(int o = 0, o_index = 0,o_index2 = 0; o < out_dim.depth(); ++o, o_index += out_dim.plane_size())
        {
            std::fill(out+o_index,out+o_index+out_dim.plane_size(),bias[o]);
            for(int inc = 0, inc_index = 0; inc < in_dim.depth(); inc++, inc_index += in_dim.plane_size(),o_index2 += kernel_size2)
                {
                    for(int y = 0, y_index = 0, index = 0; y < out_dim.height(); y++, y_index += in_dim.width())
                    {
                        for(int x = 0; x < out_dim.width(); x++, ++index)
                        {
                            const float * w = &weight[o_index2];
                            const float * p = &data[inc_index] + y_index + x;
                            float sum(0);
                            for(int wy = 0; wy < kernel_size; wy++)
                            {
                                sum += image::vec::dot(w,w+kernel_size,p);
                                w += kernel_size;
                                p += in_dim.width();
                            }
                            out[o_index+index] += sum;
                        }
                    }
                }
        }
    }
    void calculate_dwdb(const float* in_dE_da,
                        const float* prev_out,
                        std::vector<float>& dweight,
                        std::vector<float>& dbias) override
    {
        // accumulate dw
        for(int outc = 0, outc_pos = 0, w_index = 0; outc < out_dim.depth(); outc++, outc_pos += out_dim.plane_size())
        {
            for(int inc = 0;inc < in_dim.depth();++inc,w_index += kernel_size2)
            {
                for(int wy = 0, index = w_index; wy < kernel_size; wy++)
                {
                    for(int wx = 0; wx < kernel_size; wx++, ++index)
                    {
                        const float * prevo = prev_out + (in_dim.height() * inc + wy) * in_dim.width() + wx;
                        const float * delta = &in_dE_da[outc_pos];
                        float sum(0);
                        for(int y = 0; y < out_dim.height(); y++, prevo += in_dim.width(), delta += out_dim.width())
                            sum += vec::dot(prevo, prevo + out_dim.width(), delta);
                        dweight[index] += sum;
                    }
                }
            }
        }
        {
            for(int outc = 0, outc_pos = 0; outc < out_dim.depth(); outc++, outc_pos += out_dim.plane_size())
            {
                const float *delta = &in_dE_da[outc_pos];
                dbias[outc] += std::accumulate(delta, delta + out_dim.plane_size(), float(0));
            }
        }
    }
    void back_propagation(float* in_dE_da,// output_size
                          float* out_dE_da,// input_size
                          const float*) override
    {
        // propagate delta to previous layer
        for(int outc = 0, outc_pos = 0,w_index = 0; outc < out_dim.depth(); ++outc, outc_pos += out_dim.plane_size())
        {
            for(int inc = 0, inc_pos = 0; inc < in_dim.depth(); ++inc, inc_pos += in_dim.plane_size(),w_index += kernel_size2)
            {
                const float *pdelta_src = in_dE_da + outc_pos;
                float *pdelta_dst = out_dE_da + inc_pos;
                for(int y = 0, y_pos = 0, index = 0; y < out_dim.height(); y++, y_pos += in_dim.width())
                    for(int x = 0; x < out_dim.width(); x++, ++index)
                    {
                        const float * ppw = &weight[w_index];
                        const float ppdelta_src = pdelta_src[index];
                        float *p = pdelta_dst + y_pos + x;
                        for(int wy = 0; wy < kernel_size; wy++,ppw += kernel_size,p += in_dim.width())
                            image::vec::axpy(p,p+kernel_size,ppdelta_src,ppw);
                    }

            }
        }
    }
    virtual unsigned int computation_cost(void) const
    {
        return out_dim.size()*in_dim.depth()*kernel_size*kernel_size;
    }
};


class dropout_layer : public basic_layer
{
private:
    unsigned int dim;
    image::bernoulli bgen;
public:
    dropout_layer(float dropout_rate)
        : basic_layer(activation_type::identity),
          bgen(dropout_rate)
    {
    }
    bool init(const image::geometry<3>& in_dim_,const image::geometry<3>& out_dim_) override
    {
        if(in_dim_.size() != out_dim_.size())
            return false;
        dim =in_dim_.size();
        basic_layer::init(dim,dim,0,0);
        return true;
    }

    void back_propagation(float* in_dE_da,// output_size
                          float* out_dE_da,// input_size
                          const float* pre_out) override
    {
        for(unsigned int i = 0; i < dim; i++)
            out_dE_da[i] = (pre_out[i] == 0.0f) ? 0: in_dE_da[i];
    }
    void forward_propagation(const float* data,float* out) override
    {
        if(status == testing)
        {
            std::copy(data,data+dim,out);
            return;
        }
        for(unsigned int i = 0; i < dim; i++)
            out[i] = bgen() ? 0.0f: data[i];
    }
};

class soft_max_layer : public basic_layer{
public:
    soft_max_layer(void)
        : basic_layer(activation_type::identity)
    {
    }
    bool init(const image::geometry<3>& in_dim,const image::geometry<3>& out_dim) override
    {
        if(in_dim.size() != out_dim.size())
            return false;
        basic_layer::init(in_dim.size(),in_dim.size(),0,0);
        return true;
    }
    void forward_propagation(const float* data,float* out) override
    {
        float m = *std::max_element(data,data+input_size);
        for(int i = 0;i < input_size;++i)
            out[i] = expf(data[i]-m);
        float sum = std::accumulate(out,out+output_size,float(0));
        if(sum != 0)
            image::divide_constant(out,out+output_size,sum);
    }
    void back_propagation(float* in_dE_da,// output_size
                          float* out_dE_da,// input_size
                          const float* prev_out) override
    {
        std::copy(in_dE_da,in_dE_da+input_size,out_dE_da);
        image::minus_constant(out_dE_da,out_dE_da+output_size,image::vec::dot(in_dE_da,in_dE_da+input_size,prev_out));
        image::multiply(out_dE_da,out_dE_da+output_size,prev_out);
    }
};

template<typename value_type,typename label_type>
class network_data
{
public:
    image::geometry<3> input,output;
    std::vector<std::vector<value_type> > data;
    std::vector<label_type> data_label;


    void clear(void)
    {
        data.clear();
        data_label.clear();
    }
    bool is_empty(void) const
    {
        return data.empty();
    }

    bool load_from_file(const char* file_name)
    {
        std::ifstream in(file_name,std::ios::binary);
        if(!in)
            return false;
        unsigned int i,j;
        in.read((char*)&input[0],sizeof(input[0])*3);
        in.read((char*)&output[0],sizeof(output[0])*3);
        in.read((char*)&i,4);
        in.read((char*)&j,4);
        data_label.resize(i);
        in.read((char*)&data_label[0],sizeof(label_type)*i);
        data.resize(i);
        for(unsigned int k = 0;k < i;++k)
        {
            data[k].resize(j);
            in.read((char*)&data[k][0],sizeof(value_type)*j);
        }
        return !!in;
    }
    bool save_to_file(const char* file_name) const
    {
        std::ofstream out(file_name,std::ios::binary);
        if(!out)
            return false;
        unsigned int data_size = data.size();
        unsigned int data_dim = data[0].size();
        out.write((const char*)&input[0],sizeof(input[0])*3);
        out.write((const char*)&output[0],sizeof(output[0])*3);
        out.write((const char*)&data_size,sizeof(unsigned int));
        out.write((const char*)&data_dim,sizeof(unsigned int));
        out.write((const char*)&data_label[0],data_label.size());
        for(unsigned int i = 0;i < data.size();++i)
            out.write((const char*)&data[i][0],data[i].size()*sizeof(float));
        return true;
    }
    void get_label_pile(std::vector<std::vector<unsigned int> >& label_pile) const
    {
        label_pile.clear();
        label_pile.resize(output.size());
        for(unsigned int i = 0;i < data_label.size();++i)
            if(data_label[i] < label_pile.size())
                label_pile[data_label[i]].push_back(i);
    }

    void sample_test_from(network_data& rhs,float sample_ratio = 0.01)
    {
        input = rhs.input;
        output = rhs.output;

        std::vector<std::vector<unsigned int> > label_pile;
        get_label_pile(label_pile);
        std::vector<int> list_to_remove(rhs.data.size());
        for(int i = 0;i < output.size();++i)
        {
            if(label_pile[i].empty())
                continue;
            std::random_shuffle(label_pile[i].begin(),label_pile[i].end());
            int sample_count = std::max<int>(1,label_pile[i].size()*sample_ratio);
            for(int j = 0;j < sample_count;++j)
            {
                int index = label_pile[i][j];
                while(list_to_remove[index])
                    index = label_pile[i][j];
                data.push_back(rhs.data[index]);
                data_label.push_back(rhs.data_label[index]);
                list_to_remove[index] = 1;
            }
        }
        for(int i = rhs.data.size()-1;i >= 0;--i)
        if(list_to_remove[i])
        {
            rhs.data[i] = rhs.data.back();
            rhs.data_label[i] = rhs.data_label.back();
            rhs.data.pop_back();
            rhs.data_label.pop_back();
        }
    }
};

class network
{
    std::vector<std::shared_ptr<basic_layer> > layers;
    std::vector<image::geometry<3> > geo;
    unsigned int data_size;
private:// for training
    std::vector<std::vector<std::vector<float> > > dweight,dbias;
    std::vector<std::vector<float> > in_out,back_df;
    std::vector<float*> in_out_ptr,back_df_ptr;
    float target_value_min,target_value_max;
    unsigned int training_count = 0;
    unsigned int training_error_count = 0;
    unsigned int output_size = 0;
public:
    float learning_rate = 0.01f;
    float w_decay_rate = 0.0001f;
    float b_decay_rate = 0.05f;
    float momentum = 0.9f;
    int batch_size = 64;
    int epoch= 20;
    std::string error_msg;
private:
    float rate_decay = 1.0f;
public:
    network():data_size(0){}
    void reset(void)
    {
        layers.clear();
        geo.clear();
        dweight.clear();
        dbias.clear();
        training_count = 0;
        training_error_count = 0;
        data_size = 0;
    }

    unsigned int get_output_size(void) const{return output_size;}
    bool add(const image::geometry<3>& dim)
    {
        if(!layers.empty())
        {
            if(!layers.back()->init(geo.back(),dim))
                return false;
        }
        geo.push_back(dim);
        data_size += dim.size();
        output_size = dim.size();
        return true;
    }
    unsigned int computation_cost(void) const
    {
        unsigned int cost = 0;
        for(auto& layer : layers)
            cost += layer->computation_cost();
        return cost;
    }
    void get_min_max(std::vector<float>& wmin,
                     std::vector<float>& wmax,
                     std::vector<float>& bmin,
                     std::vector<float>& bmax)
    {
        for(auto& layer : layers)
        if(!layer->weight.empty())
        {
            wmin.push_back(*std::min_element(layer->weight.begin(),layer->weight.end()));
            wmax.push_back(*std::max_element(layer->weight.begin(),layer->weight.end()));
            bmin.push_back(*std::min_element(layer->bias.begin(),layer->bias.end()));
            bmax.push_back(*std::max_element(layer->bias.begin(),layer->bias.end()));
        }
        else
        {
            wmin.push_back(0);
            wmax.push_back(0);
            bmin.push_back(0);
            bmax.push_back(0);
        }
    }

    void to_image(color_image& I,std::vector<float> in,int label,int layer_height = 20,int max_width = 0)
    {
        std::vector<image::color_image> Is(layers.size());
        Is.resize(layers.size());

        int total_height = 0;
        image::uniform_dist<float> gen(0.0f,3.14159265358979323846f*2.0f);
        std::vector<image::basic_image<float,2> > layer_images(layers.size());
        image::par_for(layers.size(),[&](int i)
        {
            layers[i]->to_image(layer_images[i]);
        });

        for(int i = 0;i < layers.size();++i)
            max_width = std::max<int>(max_width,layer_images[i].width());

        for(int i = 0;i < layers.size();++i)
        {
            image::rgb_color b;
            b.from_hsl(gen(),0.5,0.85);
            if(layer_images[i].empty())
            {
                total_height += layer_height;
                continue;
            }
            for(int j = 0;j < 2 && layer_images[i].width() < max_width*0.5f;++j)
                image::upsampling_nearest(layer_images[i]);

            total_height += std::max<int>(layer_images[i].height(),layer_height);
            {
                Is[i].resize(layer_images[i].geometry());
                std::fill(Is[i].begin(),Is[i].end(),b);
                for(int j = 0;j < layer_images[i].size();++j)
                {
                    if(layer_images[i][j] == 0)
                    {
                        Is[i][j] = b;
                        continue;
                    }
                    unsigned char s(std::min<int>(255,512.0*std::fabs(layer_images[i][j])));
                    if(layer_images[i][j] < 0) // red
                        Is[i][j] = image::rgb_color(s,0,0);
                    if(layer_images[i][j] > 0) // blue
                        Is[i][j] = image::rgb_color(0,0,s);
                }
            }
        }

        std::vector<image::color_image> values(geo.size());
        {
            std::vector<float> out(data_size),back(data_size);
            float* out_buf = &out[0];
            float* back_buf = &back[0];
            float* in_buf = &in[0];
            forward_propagation(in_buf,out_buf);
            back_propagation(in_buf,label,out_buf,back_buf);
            for(int i = 0;i < geo.size();++i)
            {
                int col = std::max<int>(1,(max_width-1)/(geo[i].width()+1));
                int row = std::max<int>(1,geo[i][2]/col+1);
                if(i == 0)
                    values[i].resize(image::geometry<2>(col*(geo[i].width()+1)+1,row*(geo[i].height()+1)+1));
                else
                    values[i].resize(image::geometry<2>(col*(geo[i].width()+1)+1,int(2.0f*row*(geo[i].height()+1)+2)));
                std::fill(values[i].begin(),values[i].end(),image::rgb_color(255,255,255));
                int draw_width = 0;
                for(int y = 0,j = 0;y < row;++y)
                    for(int x = 0;j < geo[i][2] && x < col;++x,++j)
                    {
                        auto v1 = image::make_image((i == 0 ? in_buf : out_buf)+geo[i].plane_size()*j,image::geometry<2>(geo[i][0],geo[i][1]));
                        auto v2 = image::make_image((i == 0 ? in_buf : back_buf)+geo[i].plane_size()*j,image::geometry<2>(geo[i][0],geo[i][1]));
                        image::normalize_abs(v1);
                        image::normalize_abs(v2);
                        image::color_image Iv1(v1.geometry()),Iv2(v2.geometry());
                        for(int j = 0;j < Iv1.size();++j)
                        {
                            unsigned char s1(std::min<int>(255,int(255.0f*std::fabs(v1[j]))));
                            if(v1[j] < 0) // red
                                Iv1[j] = image::rgb_color(s1,0,0);
                            if(v1[j] >= 0) // blue
                                Iv1[j] = image::rgb_color(0,0,s1);
                            unsigned char s2(std::min<int>(255,int(255.0f*std::fabs(v2[j]))));
                            if(v2[j] < 0) // red
                                Iv2[j] = image::rgb_color(s2,0,0);
                            if(v2[j] >= 0) // blue
                                Iv2[j] = image::rgb_color(0,0,s2);
                        }
                        image::draw(Iv1,values[i],image::geometry<2>(x*(geo[i].width()+1)+1,y*(geo[i].height()+1)+1));
                        if(i)
                            image::draw(Iv2,values[i],image::geometry<2>(x*(geo[i].width()+1)+1,row*(geo[i].height()+1)+1+y*(geo[i].height()+1)+1));
                        draw_width = std::max<int>(draw_width,Iv1.width() + x*(geo[i].width()+1)+1);
                    }
                while((draw_width << 1) < max_width && values[i].height() < 50)
                {
                    image::upsampling_nearest(values[i]);
                    draw_width <<= 1;
                }
                total_height += values[i].height();
                back_buf += geo[i].size();
                out_buf += geo[i].size();
            }
        }

        I.resize(image::geometry<2>(max_width,total_height));
        std::fill(I.begin(),I.end(),image::rgb_color(255,255,255));        
        int cur_height = 0;
        for(int i = 0;i < geo.size();++i)
        {
            // input image
            image::draw(values[i],I,image::geometry<2>(0,cur_height));
            cur_height += values[i].height();

            // network wieghts
            if(i < layers.size())
            {
                image::rgb_color b;
                if(Is[i].empty())
                    b.from_hsl(gen(),0.5,0.85);
                else
                    b = Is[i][0];
                image::fill_rect(I,image::geometry<2>(0,cur_height),
                                   image::geometry<2>(max_width,cur_height+std::max<int>(Is[i].height(),layer_height)),b);
                image::draw(Is[i],I,image::geometry<2>(1,cur_height +
                                                       (Is[i].height() < layer_height ? (layer_height- Is[i].height())/2: 0)));
                cur_height += std::max<int>(Is[i].height(),layer_height);
            }
        }
    }
    bool add(const std::vector<std::string>& list)
    {
        for(auto& str: list)
            if(!add(str))
            {
                error_msg = str;
                return false;
            }
        return true;
    }
    bool add(const std::string& text)
    {
        // parse by |
        {
            std::regex reg("[|]");
            std::sregex_token_iterator first{text.begin(), text.end(),reg, -1},last;
            std::vector<std::string> list = {first, last};
            if(list.size() > 1)
                return add(list);
        }
        // parse by ","
        std::regex reg(",");
        std::sregex_token_iterator first{text.begin(), text.end(),reg, -1},last;
        std::vector<std::string> list = {first, last};

        {
            std::regex integer("(\\+|-)?[[:digit:]]+");
            if(list.size() == 3 &&
               std::regex_match(list[0],integer) &&
               std::regex_match(list[1],integer) &&
               std::regex_match(list[2],integer))
            {
                int x,y,z;
                std::istringstream(list[0]) >> x;
                std::istringstream(list[1]) >> y;
                std::istringstream(list[2]) >> z;
                return add(image::geometry<3>(x,y,z));
            }
        }

        if(list.empty())
            return false;
        if(list[0] == "soft_max")
        {
            layers.push_back(std::make_shared<soft_max_layer>());
            return true;
        }
        if(list[0] == "dropout")
        {
            float param = 0.9f;
            std::istringstream(list[1]) >> param;
            layers.push_back(std::make_shared<dropout_layer>(param));
            return true;
        }
        if(list.size() < 2)
            return false;
        activation_type af;
        if(list[1] == "tanh")
            af = activation_type::tanh;
        else
            if(list[1] == "sigmoid")
                af = activation_type::sigmoid;
            else
                if(list[1] == "relu")
                    af = activation_type::relu;
                else
                    if(list[1] == "identity")
                        af = activation_type::identity;
                    else
                        return false;

        if(list[0] == "full")
        {
            layers.push_back(std::make_shared<fully_connected_layer>(af));
            return true;
        }

        if(list.size() < 3)
            return false;
        int param;
        std::istringstream(list[2]) >> param;
        if(list[0] == "avg_pooling")
        {
            layers.push_back(std::make_shared<average_pooling_layer>(af,param));
            return true;
        }
        if(list[0] == "max_pooling")
        {
            layers.push_back(std::make_shared<max_pooling_layer>(af,param));
            return true;
        }
        if(list[0] == "conv")
        {
            layers.push_back(std::make_shared<convolutional_layer>(af,param));
            return true;
        }
        return false;
    }
    void save_to_file(const char* file_name)
    {
        std::ostringstream out;
        for(int i = 0;i < geo.size();++i)
        {
            if(i)
                out << "|";
            out << geo[i][0] << "," << geo[i][1] << "," << geo[i][2];
            if(i < layers.size())
            {
                out << "|";
                std::string af_type;
                if(layers[i]->af == activation_type::tanh)
                    af_type = "tanh";
                if(layers[i]->af == activation_type::sigmoid)
                    af_type = "sigmoid";
                if(layers[i]->af == activation_type::relu)
                    af_type = "relu";
                if(layers[i]->af == activation_type::identity)
                    af_type = "identity";

                if(dynamic_cast<convolutional_layer*>(layers[i].get()))
                    out << "conv," << af_type << "," << dynamic_cast<convolutional_layer*>(layers[i].get())->kernel_size;
                if(dynamic_cast<max_pooling_layer*>(layers[i].get()))
                    out << "max_pooling," << af_type << "," << dynamic_cast<max_pooling_layer*>(layers[i].get())->pool_size;
                if(dynamic_cast<average_pooling_layer*>(layers[i].get()))
                    out << "avg_pooling," << af_type << "," << dynamic_cast<average_pooling_layer*>(layers[i].get())->pool_size;
                if(dynamic_cast<fully_connected_layer*>(layers[i].get()))
                    out << "full," << af_type;
                if(dynamic_cast<soft_max_layer*>(layers[i].get()))
                    out << "soft_max";
            }
        }
        std::string nn_text = out.str();
        std::ofstream file(file_name,std::ios::binary);
        size_t nn_text_length = nn_text.length();
        file.write((const char*)&nn_text_length,sizeof(nn_text_length));
        file.write((const char*)&*nn_text.begin(),nn_text_length);
        for(auto& layer : layers)
            if(!layer->weight.empty())
            {
                file.write((const char*)&*layer->weight.begin(),layer->weight.size()*4);
                file.write((const char*)&*layer->bias.begin(),layer->bias.size()*4);
            }
    }
    bool load_from_file(const char* file_name)
    {
        std::ifstream in(file_name,std::ios::binary);
        if(!in)
            return false;
        unsigned int nn_text_length = 0;
        in.read((char*)&nn_text_length,4);
        std::string nn_text;
        nn_text.resize(nn_text_length);
        in.read((char*)&*nn_text.begin(),nn_text_length);
        if(!in)
            return false;
        layers.clear();
        add(nn_text);
        for(auto& layer : layers)
            if(!layer->weight.empty())
            {
                in.read((char*)&*layer->weight.begin(),layer->weight.size()*4);
                in.read((char*)&*layer->bias.begin(),layer->bias.size()*4);
            }
        return !!in;
    }




    void forward_propagation(const float* input,float* out_ptr)
    {
        for(int k = 0;k < layers.size();++k)
        {
            float* next_ptr = out_ptr + layers[k]->input_size;
            layers[k]->forward_propagation(k == 0 ? input : out_ptr,next_ptr);
            layers[k]->forward_af(next_ptr);
            out_ptr = next_ptr;
        }
    }
    void back_propagation(const float* data_entry,unsigned int label,const float* input,float* out_ptr)
    {
        const float* out_ptr2 = input + data_size - output_size;
        float* df_ptr = out_ptr + data_size - output_size;
        // calculate difference
        image::copy_ptr(out_ptr2,df_ptr,output_size);
        for(unsigned int i = 0;i < output_size;++i)
            df_ptr[i] -= ((label == i) ? target_value_max : target_value_min);
        for(int k = (int)layers.size()-1;k >= 0;--k)
        {
            layers[k]->back_af(df_ptr,out_ptr2);
            const float* next_out_ptr = (k == 0 ? data_entry : out_ptr2 - layers[k]->input_size);
            float* next_df_ptr = df_ptr - layers[k]->input_size;
            layers[k]->back_propagation(df_ptr,next_df_ptr,next_out_ptr);
            out_ptr2 = next_out_ptr;
            df_ptr = next_df_ptr;
        }
    }
    void calculate_dwdb(const float* data_entry,const float* out_ptr,float* df_ptr,
                                                std::vector<std::vector<float> >& dweight,
                                                std::vector<std::vector<float> >& dbias)
    {
        for(int k = 0;k < layers.size();++k)
        {
            df_ptr += layers[k]->input_size;
            if(!layers[k]->weight.empty())
                layers[k]->calculate_dwdb(df_ptr,k == 0 ? data_entry : out_ptr,
                                          dweight[k],dbias[k]);
            out_ptr += layers[k]->input_size;
        }
    }

    void initialize_training(void)
    {
        image::uniform_dist<float> gen(-1.0,1.0);
        for(auto layer : layers)
            layer->initialize_weight(gen);
        int thread_count = std::thread::hardware_concurrency();
        dweight.resize(thread_count);
        dbias.resize(thread_count);
        in_out.resize(thread_count);
        back_df.resize(thread_count);
        in_out_ptr.resize(thread_count);
        back_df_ptr.resize(thread_count);
        for(int i = 0;i < thread_count;++i)
        {
            dweight[i].resize(layers.size());
            dbias[i].resize(layers.size());
            for(int j = 0;j < layers.size();++j)
            {
                dweight[i][j].resize(layers[j]->weight.size());
                dbias[i][j].resize(layers[j]->bias.size());
            }
            in_out[i].resize(data_size);
            back_df[i].resize(data_size);
            in_out_ptr[i] = &in_out[i][0];
            back_df_ptr[i] = &back_df[i][0];
        }

        if(layers.back()->af == activation_type::tanh)
        {
            target_value_min = -0.8f;
            target_value_max = 0.8f;
        }
        else
        {
            target_value_min = 0.1f;
            target_value_max = 0.9f;
        }
        rate_decay = 1.0f;
    }
    float get_training_error(void) const
    {
        return 100.0f*training_error_count/training_count;
    }

    template <class network_data_type,class train_seq_type>
    void train_batch(const network_data_type& network_data,const train_seq_type& train_seq,
                     bool &terminated)
    {
        for(auto layer : layers)
            layer->status = training;
        if(dweight.empty())
            initialize_training();
        const auto& data = network_data.data;
        const auto& label_id = network_data.data_label;
        int size = batch_size;
        training_count = 0;
        training_error_count = 0;
        for(int i = 0;i < train_seq.size();i += size)
        {
            int size = std::min<int>(batch_size,train_seq.size()-i);
            // train a batch
            par_for2(size, [&](int m, int thread_id)
            {
                ++training_count;
                int data_index = train_seq[i+m];
                if(terminated)
                    return;
                forward_propagation(&data[data_index][0],in_out_ptr[thread_id]);

                auto ptr = in_out_ptr[thread_id] + data_size - output_size;
                if(label_id[data_index] != std::max_element(ptr,ptr+output_size)-ptr)
                    ++training_error_count;

                back_propagation(&data[data_index][0],label_id[data_index],
                                    in_out_ptr[thread_id],back_df_ptr[thread_id]);
                calculate_dwdb(&data[data_index][0],in_out_ptr[thread_id],back_df_ptr[thread_id],
                                    dweight[thread_id],dbias[thread_id]);
            });

            // update_weights
            par_for(layers.size(),[this,size](int j)
            {
                if(layers[j]->weight.empty())
                    return;
                std::vector<float> dw(layers[j]->weight.size());
                std::vector<float> db(layers[j]->bias.size());
                for(int k = 0;k < dweight.size();++k)
                {
                    image::add_mt(dw,dweight[k][j]);
                    image::add_mt(db,dbias[k][j]);
                    image::multiply_constant_mt(dweight[k][j],momentum);
                    image::multiply_constant_mt(dweight[k][j],momentum);
                }

                {
                    image::multiply_constant_mt(layers[j]->weight,1.0f-w_decay_rate*rate_decay);
                    image::multiply_constant_mt(layers[j]->bias,1.0f-b_decay_rate*rate_decay);
                }
                {
                    image::vec::axpy(&layers[j]->weight[0],&layers[j]->weight[0] + layers[j]->weight.size(),-learning_rate*rate_decay/float(size),&dw[0]);
                    image::vec::axpy(&layers[j]->bias[0],&layers[j]->bias[0] + layers[j]->bias.size(),-learning_rate*rate_decay/float(size),&db[0]);
                }

                image::upper_lower_threshold(layers[j]->bias,-bias_cap,bias_cap);
                image::upper_lower_threshold(layers[j]->weight,-weight_cap,weight_cap);
            });
        }
    }
    template <class data_type>
    void normalize_data(data_type& data)
    {
        image::par_for(data.size(),[&](unsigned int i){
           image::normalize_abs(data[i]);
        });
    }

    template <class network_data_type,class iter_type>
    void train(const network_data_type& data,bool &terminated,iter_type iter_fun)
    {
        std::vector<std::vector<unsigned int> > label_pile;
        unsigned int sample_count = 0;
        data.get_label_pile(label_pile);
        for(int i = 0;i < output_size;++i)
            if(label_pile.empty())
                return;
            else
                sample_count = std::max<unsigned int>(sample_count,label_pile[i].size());
        // rearrange training data
        for(int iter = 0; iter < epoch && !terminated;iter++ ,iter_fun())
        {
            rate_decay = std::pow(0.8,iter);
            for(int i = 0;i < output_size;++i)
                std::random_shuffle(label_pile[i].begin(),label_pile[i].end());

            std::vector<unsigned int> training_sequence;
            std::vector<unsigned int> label_pile_index(output_size);
            for(unsigned int j = 0;j < sample_count;++j)
                for(int i = 0;i < output_size;++i)
                {
                    training_sequence.push_back(label_pile[i][label_pile_index[i]++]);
                    if(label_pile_index[i] >= label_pile[i].size())
                        label_pile_index[i] = 0;
                }
            train_batch(data,training_sequence,terminated);
        }
    }

    void predict(std::vector<float>& in)
    {
        std::vector<float> out(data_size);
        forward_propagation(&in[0],&out[0]);
        in.resize(output_size);
        std::copy(out.end()-in.size(),out.end(),in.begin());
    }

    size_t predict_label(const std::vector<float>& in)
    {
        std::vector<float> result(in);
        predict(result);
        return std::max_element(result.begin(),result.end())-result.begin();
    }

    template<class input_type>
    size_t predict_label(const input_type& in)
    {
        std::vector<float> result(in.begin(),in.end());
        predict(result);
        return std::max_element(result.begin(),result.end())-result.begin();
    }

    void test(const std::vector<std::vector<float>>& data,
              std::vector<std::vector<float> >& test_result)
    {
        for(auto layer : layers)
            layer->status = testing;
        test_result.resize(data.size());
        par_for((int)data.size(), [&](int i)
        {
            test_result[i] = data[i];
            predict(test_result[i]);
        });
    }
    void test(const std::vector<std::vector<float> >& data,
              std::vector<int>& test_result)
    {
        for(auto layer : layers)
            layer->status = testing;
        test_result.resize(data.size());
        par_for((int)data.size(), [&](int i)
        {
            test_result[i] = int(predict_label(data[i]));
        });
    }
    template<typename data_type,typename label_type>
    float test_error(const data_type& data,
                     const label_type& test_result)
    {
        std::vector<int> result;
        test(data,result);
        int num_error = 0,num_total = 0;
        for (size_t i = 0; i < result.size(); i++)
        {
            if (result[i] != test_result[i])
                num_error++;
            num_total++;
        }
        return (float)num_error * 100.0f / (float)num_total;
    }
};

inline bool operator << (network& n, const image::geometry<3>& dim)
{
    return n.add(dim);
}
inline bool operator << (network& n, const std::string& text)
{
    return n.add(text);
}


template<class geo_type>
void iterate_cnn(geo_type in_dim,
             const geo_type& out_dim,
             std::vector<std::string>& list,
             int reduce_size = 2,
             int max_cost = 10000)
{
    const int max_kernel = 5;
    unsigned int layer_cost = 0;
    std::multimap<int, std::tuple<image::geometry<3>,std::string,char> > candidates;
    std::multimap<int, std::string> sorted_list;


    if(reduce_size)
    {
        std::string in_str;
        {
            std::ostringstream sout;
            sout << in_dim[0] << "," << in_dim[1] << "," << in_dim[2];
            in_str = sout.str();
            in_str += "|";
        }
        in_dim[0] /= reduce_size;
        in_dim[1] /= reduce_size;
        int pool_size = reduce_size;
        candidates.insert(std::make_pair(0,std::make_tuple(in_dim,in_str + "max_pooling,identity,"+std::to_string(pool_size)+"|",char(1))));
    }
    else
    {
        candidates.insert(std::make_pair(0,std::make_tuple(in_dim,std::string(),char(1))));
    }

    while(!candidates.empty())
    {
        int cur_cost = candidates.begin()->first;
        geo_type cur_dim = std::get<0>(candidates.begin()->second);
        std::string cur_string = std::get<1>(candidates.begin()->second);
        char tag = std::get<2>(candidates.begin()->second);
        candidates.erase(candidates.begin());
        {
            std::ostringstream sout;
            sout << cur_dim[0] << "," << cur_dim[1] << "," << cur_dim[2];
            cur_string += sout.str();
            cur_string += "|";
        }
        // add max pooling
        if(!tag && cur_dim.width() > out_dim.width())
        {
            int pool_size = 2;
            if(pool_size < cur_dim.width()/2 && pool_size < cur_dim.height()/2)
            {
                geo_type in_dim2(cur_dim[0]/pool_size,cur_dim[1]/pool_size,cur_dim[2]);
                int cost = in_dim2.size()*pool_size*pool_size+in_dim2.size()+layer_cost;
                if(cur_cost+cost > max_cost || in_dim2.size() < out_dim.size())
                    ;
                else
                    candidates.insert(std::make_pair(cur_cost+cost,std::make_tuple(in_dim2,cur_string+"max_pooling,identity,"+std::to_string(pool_size)+"|",char(1))));
            }
        }
        // add convolutional layer
        for(int kernel = 3;kernel < cur_dim.width() && kernel < cur_dim.height() && kernel <= max_kernel;++kernel)
        {
            // feature layer
            for(int feature = 4;feature <= 64;feature *= 2)
            {
                geo_type in_dim2(cur_dim[0]-kernel+1,cur_dim[1]-kernel+1,feature);
                if(in_dim2.size() < out_dim.size())
                    continue;
                int cost = in_dim2.size()*cur_dim.depth()*kernel*kernel+cur_dim.size()+layer_cost;
                if(cur_cost+cost > max_cost)
                    break;
                candidates.insert(std::make_pair(cur_cost+cost,std::make_tuple(in_dim2,cur_string+"conv,relu,"+std::to_string(kernel)+"|",char(0))));
            }
        }
        // add fully connected
        if(cur_cost)
        {
            for(int i = 2;i <= 256;i *= 2)
            {
                if(i <= out_dim.size() || i >= cur_dim.size())
                    continue;
                geo_type in_dim2(1,1,i);
                int cost = cur_dim.size()*in_dim2.size()+i+layer_cost;
                if(cur_cost+cost > max_cost)
                    break;
                candidates.insert(std::make_pair(cur_cost+cost,std::make_tuple(in_dim2,cur_string+"full,relu|",char(1))));
            }
        }
        // end
        if(cur_cost)
        {
            std::ostringstream sout;
            sout << out_dim[0] << "," << out_dim[1] << "," << out_dim[2];
            std::string s = cur_string + std::string("full,relu|")+sout.str();
            int cost = cur_dim.size()*out_dim.size()+out_dim.size();
            if(cost+cur_cost < max_cost)
                sorted_list.insert(std::make_pair(cost+cur_cost,s));
        }
    }
    for(auto& p:sorted_list)
        list.push_back(p.second);
}





}//ml
}//image

#endif//CNN_HPP
