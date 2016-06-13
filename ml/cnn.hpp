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
#include "image/utility/geometry.hpp"
#include "image/utility/basic_image.hpp"
#include "image/utility/multi_thread.hpp"

namespace image
{
namespace ml
{

const float bias_cap = 10.0f;
const float weight_cap = 100.0f;
enum activation_type { tanh, sigmoid, relu, identity};

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


class basic_layer
{

public:
    activation_type af;
    int input_size;
    int output_size;
    float weight_base;
    std::vector<float> weight,bias;
public:

    virtual ~basic_layer() {}
    basic_layer(activation_type af_ = activation_type::tanh):af(af_),weight_base(1){}
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
        std::fill(bias.begin(), bias.end(), 0);
    }
    virtual void update(const std::vector<float>& dweight,
                        const std::vector<float>& dbias,float learning_rate)
    {
        if(weight.empty())
            return;
        image::vec::axpy(&weight[0],&weight[0] + weight.size(),-learning_rate,&dweight[0]);
        image::vec::axpy(&bias[0],&bias[0] + bias.size(),-learning_rate,&dbias[0]);
        image::multiply_constant(&bias[0],&bias[0] + bias.size(),0.95);
    }
    virtual void init(const image::geometry<3>& in_dim,const image::geometry<3>& out_dim) = 0;
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
    virtual void calculate_dwdb(const float* dE_da,
                                  const float* prev_out,
                                  std::vector<float>& dweight,
                                  std::vector<float>& dbias){}
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
        return weight.size();
    }
};



class fully_connected_layer : public basic_layer
{
    image::geometry<3> in_dim;
public:
    fully_connected_layer(activation_type af_):basic_layer(af_){}
    void init(const image::geometry<3>& in_dim_,const image::geometry<3>& out_dim) override
    {
        in_dim = in_dim_;
        basic_layer::init(in_dim.size(), out_dim.size(),in_dim.size() * out_dim.size(), out_dim.size());
        weight_base = std::sqrt(6.0 / (float)(input_size+output_size));
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
            I.resize(geometry<2>(bias.size(),weight.size()/bias.size()+3));
            while(I.height() > I.width())
                I.resize(geometry<2>(I.width()*2,I.height()/2+1));
            std::copy(w.begin(),w.end(),I.begin()+I.width());
            std::copy(b.begin(),b.end(),I.end()-I.width()*2);
        }
        else
        {
            int n = weight.size()/in_dim.plane_size();
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
        weight_base = std::sqrt(6.0 / (float)(max_size(o2w_1) + max_size(i2w_1)));
    }

    template <class Container>
    static int max_size(const Container& c)
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
    void init(const image::geometry<3>& in_dim,const image::geometry<3>& out_dim) override
    {
        partial_connected_layer::init(in_dim.size(),in_dim.size()/pool_size/pool_size,in_dim.depth(), in_dim.depth());
        if(out_dim != image::geometry<3>(in_dim.width()/ pool_size, in_dim.height() / pool_size, in_dim.depth()) ||
                in_dim.depth() != out_dim.depth())
            throw std::runtime_error("invalid size in the average pooling layer");
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
        weight_base = std::sqrt(6.0 / (float)(max_size(o2w_1) + max_size(i2w_1)));
    }
    void to_image(basic_image<float,2>& I)
    {
        I.resize(geometry<2>(weight.size(),5));
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
    void init(const image::geometry<3>& in_dim_,const image::geometry<3>& out_dim_) override
    {
        in_dim = in_dim_;
        out_dim = out_dim_;
        basic_layer::init(in_dim.size(),out_dim.size(),0,0);
        if(out_dim != image::geometry<3>(in_dim.width()/ pool_size, in_dim.height() / pool_size, in_dim.depth()))
            throw std::runtime_error("invalid size in the max pooling layer");
        init_connection();
        weight_base = std::sqrt(6.0 / (float)(o2i[0].size()+1));
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
        return out_dim.size()*pool_size*pool_size/10.0;
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



struct connection_table
{
    basic_image<unsigned char,2> c;
    connection_table(){}
    connection_table(const unsigned char *ar, int rows, int cols) : c(ar,geometry<2>(cols,rows))
    {
    }
    connection_table(int rows, int cols):c(geometry<2>(cols,rows))
    {
        std::vector<unsigned char> p(c.width());
        std::fill(p.begin(),p.begin()+p.size()/2,1);
        for(int y = 0;y < c.height();++y)
        {
            std::random_shuffle(p.begin(),p.end());
            std::copy(p.begin(),p.end(),c.begin()+y*c.width());
        }
    }
    bool is_connected(int x, int y) const
    {
        return is_empty() ? true : c[y * c.width() + x];
    }

    bool is_empty() const
    {
        return c.size() == 0;
    }

};

class convolutional_layer : public basic_layer
{
    connection_table connection;
    geometry<3> in_dim,out_dim,weight_dim;
public:
    int kernel_size;
public:
    convolutional_layer(activation_type af_,int kernel_size_,const connection_table& connection_ = connection_table())
        : basic_layer(af_),
          kernel_size(kernel_size_),
          connection(connection_)
    {
    }
    void init(const image::geometry<3>& in_dim_,const image::geometry<3>& out_dim_) override
    {
        in_dim = in_dim_;
        out_dim = out_dim_;
        if(in_dim.width()-out_dim.width()+1 != kernel_size ||
           in_dim.height()-out_dim.height()+1 != kernel_size)
            throw std::runtime_error("invalid layer dimension at the convolutional layer");
        weight_dim = image::geometry<3>(kernel_size,kernel_size,in_dim.depth() * out_dim.depth()),
        basic_layer::init(in_dim_.size(), out_dim_.size(),weight_dim.size(), out_dim.depth());
        weight_base = std::sqrt(6.0 / (float)(weight_dim.plane_size() * in_dim.depth() + weight_dim.plane_size() * out_dim.depth()));
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
                image::draw(image::make_image(&w[0] + index*kernel_size*kernel_size,image::geometry<2>(kernel_size,kernel_size)),
                            I,image::geometry<2>(x*(kernel_size+1),y*(kernel_size+1)+1));
            }

        std::copy(b.begin(),b.end(),I.end()-I.width()*2);
    }
    void forward_propagation(const float* data,float* out) override
    {
        for(int o = 0, o_index = 0,o_index2 = 0; o < out_dim.depth(); ++o, o_index += out_dim.plane_size())
        {
            std::fill(out+o_index,out+o_index+out_dim.plane_size(),bias[o]);
            for(int inc = 0, inc_index = 0; inc < in_dim.depth(); inc++, inc_index += in_dim.plane_size(),o_index2 += weight_dim.plane_size())
                if(connection.is_connected(o, inc))
                {
                    for(int y = 0, y_index = 0, index = 0; y < out_dim.height(); y++, y_index += in_dim.width())
                    {
                        for(int x = 0; x < out_dim.width(); x++, ++index)
                        {
                            const float * w = &weight[o_index2];
                            const float * p = &data[inc_index] + y_index + x;
                            float sum(0);
                            for(int wy = 0; wy < weight_dim.height(); wy++)
                            {
                                sum += image::vec::dot(w,w+weight_dim.width(),p);
                                w += weight_dim.width();
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
            for(int inc = 0;inc < in_dim.depth();++inc,w_index += weight_dim.plane_size())
            {
                if(connection.is_connected(outc, inc))
                for(int wy = 0, index = w_index; wy < weight_dim.height(); wy++)
                {
                    for(int wx = 0; wx < weight_dim.width(); wx++, ++index)
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
            for(int inc = 0, inc_pos = 0; inc < in_dim.depth(); ++inc, inc_pos += in_dim.plane_size(),w_index += weight_dim.plane_size())
            if(connection.is_connected(outc, inc))
            {
                const float *pdelta_src = in_dE_da + outc_pos;
                float *pdelta_dst = out_dE_da + inc_pos;
                for(int y = 0, y_pos = 0, index = 0; y < out_dim.height(); y++, y_pos += in_dim.width())
                    for(int x = 0; x < out_dim.width(); x++, ++index)
                    {
                        const float * ppw = &weight[w_index];
                        const float ppdelta_src = pdelta_src[index];
                        float *p = pdelta_dst + y_pos + x;
                        for(int wy = 0; wy < weight_dim.height(); wy++,ppw += weight_dim.width(),p += in_dim.width())
                            image::vec::axpy(p,p+weight_dim.width(),ppdelta_src,ppw);
                    }

            }
        }
    }
    virtual unsigned int computation_cost(void) const
    {
        return out_dim.size()*in_dim.depth()*kernel_size*kernel_size;
    }
};

/*
class dropout_layer : public basic_layer
{
private:
    unsigned int dim;
    std::vector<bool> drop;
    image::bernoulli bgen;
public:
    dropout_layer(float dropout_rate)
        : basic_layer(activation_type::identity),
          bgen(dropout_rate),
          drop(0)
    {
    }
    void init(const image::geometry<3>& in_dim_,const image::geometry<3>& out_dim_) override
    {
        if(in_dim_.size() != out_dim_.size())
            throw std::runtime_error("invalid layer dimension in the dropout layer");
        dim =in_dim_.size();
        basic_layer::init(dim,dim,0,0);
    }

    void back_propagation(float* in_dE_da,// output_size
                          float* out_dE_da,// input_size
                          const float*) override
    {
        if(drop.empty())
        {
            drop.resize(dim);
            for(int i = 0;i < dim;++i)
                drop[i] = bgen();
            return;
        }
        for(int i = 0; i < drop.size(); i++)
            out_dE_da[i] = drop[i] ? 0: in_dE_da[i];
    }
    void forward_propagation(const float* data,float* out) override
    {
        if(drop.empty())
            return;
        for(int i = 0; i < drop.size(); i++)
            if(drop[i])
                data[i] = 0;
    }
};
*/

class soft_max_layer : public basic_layer{
public:
    soft_max_layer(void)
        : basic_layer(activation_type::identity)
    {
    }
    void init(const image::geometry<3>& in_dim,const image::geometry<3>& out_dim) override
    {
        if(in_dim.size() != out_dim.size())
            throw std::runtime_error("invalid layer dimension at soft_max_layer");
        basic_layer::init(in_dim.size(),in_dim.size(),0,0);
    }
    void forward_propagation(const float* data,float* out) override
    {
        for(int i = 0;i < input_size;++i)
            out[i] = expf(data[i]);
        float sum = std::accumulate(out,out+output_size,float(0));
        if(sum != 0)
            image::divide_constant(out,out+output_size,sum);
    }
    void back_propagation(float* in_dE_da,// output_size
                          float* out_dE_da,// input_size
                          const float* prev_out) override
    {
        for(int i = 0;i < output_size;++i)
        {
            float sum = float(0);
            for(int j = 0;j < output_size;++j)
                sum += (i == j) ?  in_dE_da[j]*(float(1)-prev_out[i]) : -in_dE_da[j]*prev_out[i];
            out_dE_da[i] = sum;
        }
    }
};

template<typename value_type,typename label_type>
struct network_data
{
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
    network():data_size(0){}
    void reset(void)
    {
        layers.clear();
        geo.clear();
        data_size = 0;
    }

    unsigned int get_output_size(void) const{return output_size;}
    void add(basic_layer* new_layer)
    {
        layers.push_back(std::shared_ptr<basic_layer>(new_layer));
    }
    void add(const image::geometry<3>& dim)
    {
        if(!layers.empty())
            layers.back()->init(geo.back(),dim);
        geo.push_back(dim);
        data_size += dim.size();
        output_size = dim.size();
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

    void to_image(color_image& I,std::vector<float> in,int layer_height = 20,int max_width = 0)
    {
        std::vector<image::color_image> Is(layers.size());
        Is.resize(layers.size());

        int total_height = 0;
        image::uniform_dist<float> gen(0,3.14159265358979323846*2);
        for(int i = 0;i < layers.size();++i)
        {
            image::rgb_color b;
            b.from_hsl(gen(),0.5,0.85);
            image::basic_image<float,2> I2;
            layers[i]->to_image(I2);
            if(I2.width() > max_width)
                max_width = I2.width();
            total_height += std::max<int>(I2.height(),layer_height);
            if(!I2.empty())
            {
                Is[i].resize(I2.geometry());
                std::fill(Is[i].begin(),Is[i].end(),b);
                for(int j = 0;j < I2.size();++j)
                {
                    if(I2[j] == 0)
                    {
                        Is[i][j] = b;
                        continue;
                    }
                    unsigned char s(std::min<int>(255,300.0*std::fabs(I2[j])));
                    if(I2[j] < 0) // red
                        Is[i][j] = image::rgb_color(s,0,0);
                    if(I2[j] > 0) // blue
                        Is[i][j] = image::rgb_color(0,0,s);
                }
            }
        }

        std::vector<image::color_image> values(geo.size());
        {
            std::vector<float> out(data_size);
            float* out_buf = &out[0];
            float* in_buf = &in[0];
            forward_propagation(in_buf,out_buf);
            for(int i = 0;i < geo.size();++i)
            {
                int col = std::max<int>(1,(max_width-1)/(geo[i].width()+1));
                int row = std::max<int>(1,geo[i][2]/col+1);
                values[i].resize(image::geometry<2>(col*(geo[i].width()+1)+1,row*(geo[i].height()+1)+1));
                std::fill(values[i].begin(),values[i].end(),image::rgb_color(255,255,255));
                for(int y = 0,j = 0;y < row;++y)
                    for(int x = 0;j < geo[i][2] && x < col;++x,++j)
                    {
                        auto v = image::make_image((i == 0 ? in_buf : out_buf)+geo[i].plane_size()*j,image::geometry<2>(geo[i][0],geo[i][1]));
                        image::normalize_abs(v);
                        image::color_image Iv(v.geometry());
                        for(int j = 0;j < Iv.size();++j)
                        {
                            unsigned char s(std::min<int>(255,300.0*std::fabs(v[j])));
                            if(v[j] < 0) // red
                                Iv[j] = image::rgb_color(s,0,0);
                            if(v[j] >= 0) // blue
                                Iv[j] = image::rgb_color(0,0,s);
                        }
                        image::draw(Iv,values[i],image::geometry<2>(x*(geo[i].width()+1)+1,y*(geo[i].height()+1)+1));
                    }
                total_height += values[i].height();
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
                image::draw(Is[i],I,image::geometry<2>(1,cur_height + (Is[i].height() < layer_height ? (layer_height- Is[i].height())/2: 0)));
                cur_height += std::max<int>(Is[i].height(),layer_height);
            }
        }
    }
    void add(const std::vector<std::string>& list)
    {
        for(auto& str: list)
            add(str);
    }
    void add(const std::string& text)
    {
        // parse by |
        {
            std::regex reg("[|]");
            std::sregex_token_iterator first{text.begin(), text.end(),reg, -1},last;
            std::vector<std::string> list = {first, last};
            if(list.size() > 1)
            {
                add(list);
                return;
            }
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
                add(image::geometry<3>(x,y,z));
                return;
            }
        }

        if(list.empty())
            throw std::runtime_error(std::string("Invalid network construction text:") + text);
        if(list[0] == "soft_max")
        {
            add(new soft_max_layer());
            return;
        }
        if(list.size() < 2)
            throw std::runtime_error(std::string("Invalid network construction text:") + text);
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
                        throw std::runtime_error(std::string("Invalid activation function type:") + text);

        if(list[0] == "full")
        {
            add(new fully_connected_layer(af));
            return;
        }

        if(list.size() < 3)
            throw std::runtime_error(std::string("Invalid network construction text:") + text);
        int param;
        std::istringstream(list[2]) >> param;
        if(list[0] == "avg_pooling")
        {
            add(new average_pooling_layer(af,param));
            return;
        }
        if(list[0] == "max_pooling")
        {
            add(new max_pooling_layer(af,param));
            return;
        }
        if(list[0] == "conv")
        {
            add(new convolutional_layer(af,param));
            return;
        }
        throw std::runtime_error(std::string("Invalid network layer text:") + text);
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
        unsigned int nn_text_length = nn_text.length();
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
            target_value_min = -0.8;
            target_value_max = 0.8;
        }
        else
        {
            target_value_min = 0.1;
            target_value_max = 0.9;
        }

        training_count = 0;
        training_error_count = 0;
    }
    double get_training_error(void) const
    {
        return 100.0*training_error_count/training_count;
    }

    void update_weights(float learning_rate)
    {
        par_for(layers.size(),[this,learning_rate](int j)
        {
            if(layers[j]->weight.empty())
                return;
            std::vector<float> dw(layers[j]->weight.size());
            std::vector<float> db(layers[j]->bias.size());
            for(int k = 0;k < dweight.size();++k)
            {
                image::add(dw,dweight[k][j]);
                image::add(db,dbias[k][j]);
                std::fill(dweight[k][j].begin(),dweight[k][j].end(),float(0));
                std::fill(dweight[k][j].begin(),dweight[k][j].end(),float(0));
            }
            layers[j]->update(dw,db,learning_rate);
            image::upper_lower_threshold(layers[j]->bias,-bias_cap,bias_cap);
            image::upper_lower_threshold(layers[j]->weight,-weight_cap,weight_cap);
        });
    }

    template <class data_type,class label_id_type>
    void train_a_batch(const data_type& data,const label_id_type& label_id,int i,int size,bool &terminated)
    {
        par_for2(size, [this,&label_id,&data,i,&terminated](int m, int thread_id)
        {
            ++training_count;
            int data_index = i+m;
            if(terminated)
                return;
            forward_propagation(&data[data_index][0],in_out_ptr[thread_id]);

            const float* out_ptr2 = in_out_ptr[thread_id] + data_size - output_size;
            if(label_id[data_index] != std::max_element(out_ptr2,out_ptr2+output_size)-out_ptr2)
                ++training_error_count;
            float* df_ptr = back_df_ptr[thread_id] + data_size - output_size;

            image::copy_ptr(out_ptr2,df_ptr,output_size);
            for(int i = 0;i < output_size;++i)
                df_ptr[i] -= ((label_id[data_index] == i) ? target_value_max : target_value_min);
            for(int k = layers.size()-1;k >= 0;--k)
            {
                layers[k]->back_af(df_ptr,out_ptr2);
                const float* next_out_ptr = (k == 0 ? &data[data_index][0] : out_ptr2 - layers[k]->input_size);
                float* next_df_ptr = df_ptr - layers[k]->input_size;
                if(!layers[k]->weight.empty())
                    layers[k]->calculate_dwdb(df_ptr,next_out_ptr,dweight[thread_id][k],dbias[thread_id][k]);
                layers[k]->back_propagation(df_ptr,next_df_ptr,next_out_ptr);
                out_ptr2 = next_out_ptr;
                df_ptr = next_df_ptr;
            }

        },std::thread::hardware_concurrency());
        if(training_count > 1000)
        {
            training_count *= 0.9;
            training_error_count *= 0.9;
        }
    }

    template <class data_type,class label_type>
    void train_batch(const data_type& data,const label_type& label_id,
                       unsigned int batch_count,bool &terminated,float learning_rate = 0.0001)
    {
        if(dweight.empty())
            initialize_training();
        int batch_size = output_size*2;
        for(int i = 0,j = 0;i < data.size() && j < batch_count;i += batch_size,++j)
        {
            int size = std::min<int>(batch_size,data.size()-i);
            train_a_batch(data,label_id,i,size,terminated);
            update_weights(learning_rate/batch_size);
        }
    }

    template <class data_type,class label_type,class iter_type>
    void train(const data_type& data,
               const label_type& label_id,int iteration_count,bool &terminated,
               iter_type iter_fun = []{},float learning_rate = 0.0001)
    {
        for(int iter = 0; iter < iteration_count && !terminated;iter++ ,learning_rate *= 0.85,iter_fun())
            train_batch(data,label_id,data.size()*2/output_size+1,terminated,learning_rate);
    }

    void predict(std::vector<float>& in)
    {
        std::vector<float> out(data_size);
        forward_propagation(&in[0],&out[0]);
        in.resize(output_size);
        std::copy(out.end()-in.size(),out.end(),in.begin());
    }

    int predict_label(const std::vector<float>& in)
    {
        std::vector<float> result(in);
        predict(result);
        return std::max_element(result.begin(),result.end())-result.begin();
    }

    template<class input_type>
    int predict_label(const input_type& in)
    {
        std::vector<float> result(in.begin(),in.end());
        predict(result);
        return std::max_element(result.begin(),result.end())-result.begin();
    }

    void test(const std::vector<std::vector<float>>& data,
              std::vector<std::vector<float> >& test_result)
    {
        test_result.resize(data.size());
        par_for((int)data.size(), [&](int i)
        {
            test_result[i] = data[i];
            predict(test_result[i]);
        });
    }
    void test(const std::vector<std::vector<float>>& data,
              std::vector<int>& test_result)
    {
        test_result.resize(data.size());
        par_for((int)data.size(), [&](int i)
        {
            test_result[i] = predict_label(data[i]);
        });
    }



};

inline network& operator << (network& n, basic_layer* layer)
{
    n.add(layer);
    return n;
}
inline network& operator << (network& n, const image::geometry<3>& dim)
{
    n.add(dim);
    return n;
}
inline network& operator << (network& n, const std::string& text)
{
    n.add(text);
    return n;
}


template<class geo_type>
void iterate_cnn(const geo_type& in_dim,
             const geo_type& out_dim,
             std::vector<std::string>& list,
             int max_size = 10000)
{
    const int base_cost = 2000;
    const int max_kernel = 7;
    int max_cost = std::numeric_limits<int>::max();
    std::multimap<int, std::tuple<image::geometry<3>,std::string,char> > candidates;
    candidates.insert(std::make_pair(0,std::make_tuple(in_dim,std::string(),char(1))));
    while(list.size() < max_size && !candidates.empty())
    {
        int cur_cost = candidates.begin()->first;
        geo_type cur_dim = std::get<0>(candidates.begin()->second);
        std::string cur_string = std::get<1>(candidates.begin()->second);
        char tag = std::get<2>(candidates.begin()->second);
        while(candidates.size() > max_size)
        {
            max_cost = std::min<int>(max_cost,(--candidates.end())->first);
            candidates.erase(--candidates.end());
        }
        candidates.erase(candidates.begin());
        {
            std::ostringstream sout;
            sout << cur_dim[0] << "," << cur_dim[1] << "," << cur_dim[2];
            cur_string += sout.str();
            cur_string += "|";
        }

        // add convolutional layer
        for(int kernel = 3;kernel < cur_dim.width() && kernel < cur_dim.height() && kernel <= max_kernel;++kernel)
        {
            // feature layer
            for(int feature = 4;feature <= 64;feature *= 2)
            {
                geo_type in_dim2(cur_dim[0]-kernel+1,cur_dim[1]-kernel+1,feature);
                int cost = in_dim2.size()*cur_dim.depth()*kernel*kernel+base_cost;
                if(cur_cost+cost > max_cost || in_dim2.size() < out_dim.size())
                    break;
                candidates.insert(std::make_pair(cur_cost+cost,std::make_tuple(in_dim2,cur_string+"conv,relu,"+std::to_string(kernel)+"|",char(0))));
                candidates.insert(std::make_pair(cur_cost+cost,std::make_tuple(in_dim2,cur_string+"conv,tanh,"+std::to_string(kernel)+"|",char(0))));
            }
        }
        // add max pooling
        if(!tag && cur_dim.width() > out_dim.width())
        {
            for(int pool_size = 2;pool_size < 4 && pool_size < cur_dim.width()/2 && pool_size < cur_dim.height()/2;++pool_size)
            {
                geo_type in_dim2(cur_dim[0]/pool_size,cur_dim[1]/pool_size,cur_dim[2]);
                int cost = in_dim2.size()*pool_size*pool_size+base_cost;
                if(cur_cost+cost > max_cost || in_dim2.size() < out_dim.size())
                    break;
                candidates.insert(std::make_pair(cur_cost+cost,std::make_tuple(in_dim2,cur_string+"max_pooling,identity,"+std::to_string(pool_size)+"|",char(1))));
            }
        }
        // add fully connected
        if(cur_cost)
        {
            for(int i = 2;i <= 256;i *= 2)
            {
                if(i < out_dim.size())
                    continue;
                geo_type in_dim2(1,1,i);
                int cost = cur_dim.size()*in_dim2.size()+base_cost;
                if(in_dim2.size() > cur_dim.size() || cur_cost+cost > max_cost)
                    break;
                candidates.insert(std::make_pair(cur_cost+cost,std::make_tuple(in_dim2,cur_string+"full,tanh|",char(1))));
                candidates.insert(std::make_pair(cur_cost+cost,std::make_tuple(in_dim2,cur_string+"full,relu|",char(1))));
            }
        }
        // end
        if(cur_cost)
        {
            std::ostringstream sout;
            sout << out_dim[0] << "," << out_dim[1] << "," << out_dim[2];
            std::string str = sout.str();
            std::string str1 = std::string("full,tanh|")+str;
            std::string str2 = std::string("full,relu|")+str;
            list.push_back(cur_string + str1);
            list.push_back(cur_string + str2);
        }
    }
}





}//ml
}//image

#endif//CNN_HPP
