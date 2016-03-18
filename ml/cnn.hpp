#ifndef CNN_HPP
#define CNN_HPP
#include <algorithm>
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
#include <random>
#include "image/numerical/matrix.hpp"
#include "image/numerical/numerical.hpp"
#include "image/utility/geometry.hpp"
#include "image/utility/multi_thread.hpp"

namespace image
{
namespace ml
{


template<typename T>
T uniform(T min, T max)
{
    static std::mt19937 gen(0);
    std::uniform_real_distribution<float> dst(min, max);
    return dst(gen);
}
//---------------------------------------------------------------------------
template<typename T>
bool bernoulli(T p)
{
    return uniform(float(0), float(1)) <= p;
}

const float bias_cap = 50.0;
enum activation_type { tanh, sigmoid, relu, identity};

template<typename value_type>
inline float tanh_f(value_type v)
{
    if(v < -10.0)
        return -1.0;
    if(v > 10.0)
        return 1.0;
    const float ep = std::exp(v + v);
    return (ep - float(1)) / (ep + float(1));
}
template<typename value_type>
inline float tanh_df(value_type y)
{
    return float(1) - y * y;
}


template<typename value_type>
inline float sigmoid_f(value_type v)
{
    if(v < -10.0)
        return 0.0;
    if(v > 10.0)
        return 1.0;
    return float(1) / (float(1) + std::exp(-v));
}
template<typename value_type>
inline float sigmoid_df(value_type y)
{
    return y * (float(1) - y);
}

template<typename value_type>
inline float relu_f(value_type v)
{
    return std::max<value_type>(0, v);
}
template<typename value_type>
inline float relu_df(value_type y)
{
    return y > value_type(0) ? value_type(1) : value_type(0);
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
    void reset(void)
    {
        for(int i = 0; i < weight.size(); ++i)
            weight[i] = uniform(-weight_base, weight_base);
        std::fill(bias.begin(), bias.end(), 0);
    }
    virtual void update(const std::vector<float>& dweight,
                        const std::vector<float>& dbias,float learning_rate)
    {
        if(weight.empty())
            return;
        image::vec::axpy(&weight[0],&weight[0] + weight.size(),-learning_rate,&dweight[0]);
        image::vec::axpy(&bias[0],&bias[0] + bias.size(),-learning_rate,&dbias[0]);
    }
    virtual void init(const image::geometry<3>& in_dim,const image::geometry<3>& out_dim) = 0;
    virtual void forward_propagation(std::vector<float>& data) = 0;
    void forward_af(std::vector<float>& data)
    {
        if(af == activation_type::tanh)
            for(int i = 0; i < data.size(); ++i)
                data[i] = tanh_f(data[i]);
        if(af == activation_type::sigmoid)
            for(int i = 0; i < data.size(); ++i)
                data[i] = sigmoid_f(data[i]);
        if(af == activation_type::relu)
            for(int i = 0; i < data.size(); ++i)
                data[i] = relu_f(data[i]);
    }

    virtual void back_propagation(std::vector<float>& dE_da,
                                  const std::vector<float>& prev_out,
                                  std::vector<float>& dweight,
                                  std::vector<float>& dbias) = 0;
    void back_af(std::vector<float>& dE_da,const std::vector<float>& prev_out)
    {
        if(af == activation_type::tanh)
            for(int i = 0; i < dE_da.size(); ++i)
                dE_da[i] *= tanh_df(prev_out[i]);
        if(af == activation_type::sigmoid)
            for(int i = 0; i < dE_da.size(); ++i)
                dE_da[i] *= sigmoid_df(prev_out[i]);
        if(af == activation_type::relu)
            for(int i = 0; i < dE_da.size(); ++i)
                dE_da[i] *= relu_df(prev_out[i]);
    }
};



class fully_connected_layer : public basic_layer
{
public:
    fully_connected_layer(activation_type af_):basic_layer(af_){}
    void init(const image::geometry<3>& in_dim,const image::geometry<3>& out_dim) override
    {
        basic_layer::init(in_dim.size(), out_dim.size(),in_dim.size() * out_dim.size(), out_dim.size());
        weight_base = std::sqrt(6.0 / (float)(input_size+output_size));
    }

    void forward_propagation(std::vector<float>& data) override
    {
        std::vector<float> wx(bias);
        for(int i = 0,i_pos = 0;i < output_size;++i,i_pos += input_size)
            wx[i] += image::vec::dot(&weight[i_pos],&weight[i_pos]+input_size,&data[0]);
        data.swap(wx);
    }

    void back_propagation(std::vector<float>& in_dE_da,
                          const std::vector<float>& prev_out,
                          std::vector<float>& dweight,
                          std::vector<float>& dbias) override
    {
        image::add(dbias,in_dE_da);
        for(int i = 0,i_pos = 0; i < output_size; i++,i_pos += input_size)
            if(in_dE_da[i] != float(0))
                image::vec::axpy(&dweight[i_pos],&dweight[i_pos]+input_size,in_dE_da[i],&prev_out[0]);
        std::vector<float> dE_da(input_size);
        image::mat::left_vector_product(&weight[0],&in_dE_da[0],&dE_da[0],image::dyndim(in_dE_da.size(),dE_da.size()));
        dE_da.swap(in_dE_da);
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

    template <typename Container>
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


    void forward_propagation(std::vector<float>& data) override
    {
        std::vector<float> wx(output_size);
        for(int i = 0; i < output_size; ++i)
        {
            const std::vector<int>& o2w_1i = o2w_1[i];
            const std::vector<int>& o2w_2i = o2w_2[i];
            float sum(0);
            for(int j = 0;j < o2w_1i.size();++j)
                sum += weight[o2w_1i[j]] * data[o2w_2i[j]];
            wx[i] = sum + bias[o2b[i]];
        }
        data.swap(wx);
    }

    void back_propagation(std::vector<float>& in_dE_da,
                          const std::vector<float>& prev_out,
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

        std::vector<float> dE_da(input_size);
        for(int i = 0; i != input_size; i++)
        {
            const std::vector<int>& i2w_1i = i2w_1[i];
            const std::vector<int>& i2w_2i = i2w_2[i];

            float sum(0);
            for(int j = 0;j < i2w_1i.size();++j)
                sum += weight[i2w_1i[j]] * in_dE_da[i2w_2i[j]];
            dE_da[i] = sum;
        }
        dE_da.swap(in_dE_da);
    }

};


class average_pooling_layer : public partial_connected_layer
{
    int pool_size;
public:
    average_pooling_layer(activation_type af_,int pool_size_)
        : partial_connected_layer(af_),pool_size(pool_size_){}
    void init(const image::geometry<3>& in_dim,const image::geometry<3>& out_dim) override
    {
        partial_connected_layer::init(in_dim.size(),in_dim.size()/pool_size/pool_size,in_dim.depth(), in_dim.depth());
        if(out_dim != image::geometry<3>(in_dim.width()/ pool_size, in_dim.height() / pool_size, in_dim.depth()) ||
                in_dim.depth() != out_dim.depth())
            throw std::runtime_error("invalid size in the max pooling layer");
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

};

class max_pooling_layer : public basic_layer
{
    int pool_size;
    std::vector<std::vector<int> > o2i;
    std::vector<int> i2o;
    geometry<3> in_dim;
    geometry<3> out_dim;

public:
    max_pooling_layer(activation_type af_,int pool_size_)
        : basic_layer(af_),pool_size(pool_size_){}
    void init(const image::geometry<3>& in_dim_,const image::geometry<3>& out_dim_) override
    {
        basic_layer::init(in_dim_.size(),in_dim_.size()/pool_size/pool_size,0,0);
        in_dim = in_dim_;
        out_dim = out_dim_;
        if(out_dim != image::geometry<3>(in_dim_.width()/ pool_size, in_dim_.height() / pool_size, in_dim_.depth()))
            throw std::runtime_error("invalid size in the max pooling layer");
        init_connection(pool_size);
        weight_base = std::sqrt(6.0 / (float)(o2i[0].size()+1));
    }
    void forward_propagation(std::vector<float>& data) override
    {
        std::vector<float> wx(basic_layer::output_size);

        for(int i = 0; i < basic_layer::output_size; i++)
        {
            float max_value = std::numeric_limits<float>::lowest();
            for(auto j : o2i[i])
            {
                if(data[j] > max_value)
                    max_value = data[j];
            }
            wx[i] = max_value;
        }
        data.swap(wx);
    }

    void back_propagation(std::vector<float>& in_dE_da,
                          const std::vector<float>& prev_out,
                          std::vector<float>& dweight,
                          std::vector<float>& dbias) override
    {
        std::vector<int> max_idx(out_dim.size());

        for(int i = 0; i < basic_layer::output_size; i++)
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
        std::vector<float> dE_da(input_size);
        for(int i = 0; i < input_size; i++)
        {
            int outi = i2o[i];
            dE_da[i] = (max_idx[outi] == i) ? in_dE_da[outi] : float(0);
        }
        dE_da.swap(in_dE_da);
    }
private:
    void init_connection(int pool_size)
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
            throw std::runtime_error("invalid layer dimension");
        weight_dim = image::geometry<3>(kernel_size,kernel_size,in_dim_.depth() * out_dim_.depth()),
        basic_layer::init(in_dim_.size(), out_dim_.size(),weight_dim.size(), out_dim_.depth());
        weight_base = std::sqrt(6.0 / (float)(weight_dim.plane_size() * in_dim.depth() + weight_dim.plane_size() * out_dim.depth()));
    }

    void forward_propagation(std::vector<float>& data) override
    {
        std::vector<float> wx(output_size);
        for(int o = 0, o_index = 0,o_index2 = 0; o < out_dim.depth(); ++o, o_index += out_dim.plane_size())
        {
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
                            wx[o_index+index] += sum;
                        }
                    }
                }
            image::add_constant(&wx[o_index],&wx[o_index]+out_dim.plane_size(),bias[o]);
        }

        data.swap(wx);
    }

    void back_propagation(std::vector<float>& in_dE_da,
                          const std::vector<float>& prev_out,
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
                        const float * prevo = &prev_out[(in_dim.height() * inc + wy) * in_dim.width() + wx];
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
        std::vector<float> dE_da(input_size);
        // propagate delta to previous layer
        for(int outc = 0, outc_pos = 0,w_index = 0; outc < out_dim.depth(); ++outc, outc_pos += out_dim.plane_size())
        {
            for(int inc = 0, inc_pos = 0; inc < in_dim.depth(); ++inc, inc_pos += in_dim.plane_size(),w_index += weight_dim.plane_size())
            if(connection.is_connected(outc, inc))
            {
                const float *pdelta_src = &in_dE_da[outc_pos];
                float *pdelta_dst = &dE_da[inc_pos];
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
        dE_da.swap(in_dE_da);
    }

};

class dropout_layer : public basic_layer
{
private:
    float dropout_rate;
    unsigned int dim;
    std::vector<bool> drop;
public:
    dropout_layer(float dropout_rate_)
        : basic_layer(activation_type::identity),
          dropout_rate(dropout_rate_),
          drop(0)
    {
    }
    void init(const image::geometry<3>& in_dim_,const image::geometry<3>& out_dim_) override
    {
        if(in_dim_.size() != out_dim_.size())
            throw std::runtime_error("invalid layer dimension");
        dim =in_dim_.size();
        basic_layer::init(dim,dim,0,0);
    }

    void back_propagation(std::vector<float>& in_dE_da,
                          const std::vector<float>& prev_out,
                          std::vector<float>& dweight,
                          std::vector<float>& dbias) override
    {
        if(drop.empty())
        {
            drop.resize(dim);
            for(int i = 0;i < dim;++i)
                drop[i] = bernoulli(dropout_rate);
            return;
        }
        for(int i = 0; i < drop.size(); i++)
            if(drop[i])
                in_dE_da[i] = 0;
    }
    void forward_propagation(std::vector<float>& data) override
    {
        if(drop.empty())
            return;
        for(int i = 0; i < drop.size(); i++)
            if(drop[i])
                data[i] = 0;
    }
};

class soft_max_layer : public basic_layer{
public:
    soft_max_layer(void)
        : basic_layer(activation_type::identity)
    {
    }
    void init(const image::geometry<3>& in_dim,const image::geometry<3>& out_dim) override
    {
        if(in_dim.size() != out_dim.size())
            throw std::runtime_error("invalid layer dimension");
        basic_layer::init(in_dim.size(),in_dim.size(),0,0);
    }
    void forward_propagation(std::vector<float>& data) override
    {
        image::exp(data);
        float sum = std::accumulate(data.begin(),data.end(),float(0));
        if(sum != 0)
            image::divide_constant(data.begin(),data.end(),sum);
    }
    void back_propagation(std::vector<float>& in_dE_da,
                          const std::vector<float>& prev_out,
                          std::vector<float>& dweight,
                          std::vector<float>& dbias) override
    {
        std::vector<float> dE_da(in_dE_da.size());
        for(int i = 0;i < in_dE_da.size();++i)
        {
            float sum = float(0);
            for(int j = 0;j < in_dE_da.size();++j)
                sum += (i == j) ?  in_dE_da[j]*(float(1)-prev_out[i]) : -in_dE_da[j]*prev_out[i];
            dE_da[i] = sum;
        }
        dE_da.swap(in_dE_da);
    }
};

class network
{
    std::vector<std::shared_ptr<basic_layer> > layers;
    image::geometry<3> cur_dim;
public:
    network(){}
    void reset(void)
    {
        layers.clear();
    }

    void add(basic_layer* new_layer)
    {
        layers.push_back(std::shared_ptr<basic_layer>(new_layer));
    }
    void add(const image::geometry<3>& dim)
    {
        if(!layers.empty())
            layers.back()->init(cur_dim,dim);
        cur_dim = dim;
    }

    void predict(std::vector<float>& in)
    {
        for(auto layer : layers)
        {
            layer->forward_propagation(in);
            layer->forward_af(in);
        }
    }
    int predict_label(const std::vector<float>& in)
    {
        std::vector<float> result(in);
        predict(result);
        return std::max_element(result.begin(),result.end())-result.begin();
    }



    template <typename data_type,typename label_type,typename iter_type>
    void train(const data_type& data,const label_type& label,int iteration_count,bool &termminated,
               iter_type iter_fun = [&]{},bool reset_weights = true)
    {
        if(reset_weights)
        {
            for(auto layer : layers)
                layer->reset();
        }
        int thread_count = std::thread::hardware_concurrency();
        std::vector<std::vector<std::vector<float> > > dweight(thread_count),dbias(thread_count);
        for(int i = 0;i < thread_count;++i)
        {
            dweight[i].resize(layers.size());
            dbias[i].resize(layers.size());
            for(int j = 0;j < layers.size();++j)
            {
                dweight[i][j].resize(layers[j]->weight.size());
                dbias[i][j].resize(layers[j]->bias.size());
            }
        }

        float learning_rate = 0.02;
        int batch_size = 20;
        for(int iter = 0; iter < iteration_count; iter++ && !termminated,learning_rate *= 0.85,iter_fun())
        {
            for(int i = 0;i < data.size() && !termminated;i += batch_size)
            {
                int size = std::min<int>(batch_size,data.size()-i);
                par_for2(size, [&](int m, int thread_id)
                {
                    if(termminated)
                        return;
                    std::vector<std::vector<float> > out(layers.size()+1);
                    out[0] = data[i + m];
                    for(int k = 0;k < layers.size();++k)
                    {
                        out[k+1] = out[k];
                        layers[k]->forward_propagation(out[k+1]);
                        layers[k]->forward_af(out[k+1]);
                    }
                    std::vector<float> output = out.back();
                    image::minus(output,label[i + m]);// diff of mse
                    for(int k = layers.size()-1;k >= 0;--k)
                    {
                        layers[k]->back_af(output,out[k+1]);
                        layers[k]->back_propagation(output,out[k],dweight[thread_id][k],dbias[thread_id][k]);
                    }

                },thread_count);
                par_for(layers.size(),[&](int j)
                {
                    if(layers[j]->weight.empty())
                        return;
                    std::vector<float> dw(layers[j]->weight.size());
                    std::vector<float> db(layers[j]->bias.size());
                    for(int k = 0;k < thread_count;++k)
                    {
                        image::add(dw,dweight[k][j]);
                        image::add(db,dbias[k][j]);
                        std::fill(dweight[k][j].begin(),dweight[k][j].end(),float(0));
                        std::fill(dweight[k][j].begin(),dweight[k][j].end(),float(0));
                    }
                    layers[j]->update(dw,db,learning_rate/batch_size);
                    image::upper_lower_threshold(layers[j]->bias,-bias_cap,bias_cap);
                });
            }
        }
    }

    void test(const std::vector<std::vector<float>>& data,std::vector<std::vector<float> >& test_result)
    {
        test_result.resize(data.size());
        par_for((int)data.size(), [&](int i)
        {
            test_result[i] = data[i];
            predict(test_result[i]);
        });
    }
    void test(const std::vector<std::vector<float>>& data,std::vector<int>& test_result)
    {
        test_result.resize(data.size());
        par_for((int)data.size(), [&](int i)
        {
            test_result[i] = predict_label(data[i]);
        });
    }
    float target_value_min() const
    {
        if(layers.back()->af == activation_type::tanh)
            return -0.8;
        if(layers.back()->af == activation_type::sigmoid)
            return 0.1;
        if(layers.back()->af == activation_type::relu)
            return 0.1;
        if(layers.back()->af == activation_type::identity)
            return 0.1;
        return 0;
    }
    float target_value_max() const
    {
        if(layers.back()->af == activation_type::tanh)
            return 0.8;
        if(layers.back()->af == activation_type::sigmoid)
            return 0.9;
        if(layers.back()->af == activation_type::relu)
            return 0.9;
        if(layers.back()->af == activation_type::identity)
            return 0.9;
        return 1;
    }
};

template<typename type>
network& operator << (network& n, type* layer)
{
    n.add(layer);
    return n;
}
template<typename type>
network& operator << (network& n, const type& dim)
{
    n.add(dim);
    return n;
}

}//ml
}//image

#endif//CNN_HPP
