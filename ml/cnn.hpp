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

#include "image/numerical/matrix.hpp"
#include "image/numerical/numerical.hpp"
#include "image/utility/geometry.hpp"
#include "image/utility/multi_thread.hpp"

namespace image
{
namespace ml
{
namespace cnn
{

enum activation_type { tanh, sigmoid, relu, identity};

template<typename value_type>
inline float tanh_f(value_type v)
{
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
    activation_type af,previous_af;
    int input_size;
    int output_size;
    float weight_base;
    std::vector<float> weight,bias,dweight,dbias;
public:

    virtual ~basic_layer() {}
    basic_layer(activation_type af_ = activation_type::tanh):af(af_),previous_af(activation_type::identity),weight_base(1){}
    void init( int input_size_, int output_size_, int weight_dim, int bias_dim)
    {
        input_size = input_size_;
        output_size = output_size_;
        weight.resize(weight_dim);
        bias.resize(bias_dim);
        dweight.resize(weight_dim);
        dbias.resize(bias_dim);
    }
    void reset(void)
    {
        for(int i = 0; i < weight.size(); ++i)
            weight[i] = uniform(-weight_base, weight_base);
        std::fill(bias.begin(), bias.end(), 0);
    }
    virtual void update(float learning_rate)
    {
        if(weight.empty())
            return;
        image::vec::axpy(&weight[0],&weight[0] + weight.size(),-learning_rate,&dweight[0]);
        image::vec::axpy(&bias[0],&bias[0] + bias.size(),-learning_rate,&dbias[0]);
        std::fill(&dweight[0],&dweight[0]+ dweight.size(),0);
        std::fill(&dbias[0],&dbias[0]+ dbias.size(),0);
    }

    virtual std::string name() const = 0;
    void save(std::ostream& os) const
    {
        for(auto w : weight) os << w << " ";
        for(auto b : bias) os << b << " ";
    }

    void load(std::istream& is)
    {
        for(auto& w : weight) is >> w;
        for(auto& b : bias) is >> b;
    }
    virtual void init(const image::geometry<3>& in_dim,const image::geometry<3>& out_dim) = 0;
    virtual void forward_propagation(std::vector<float>& in) = 0;
    virtual void back_propagation(std::vector<float>& in_dE_da,const std::vector<float>& prev_out) = 0;
};

template <typename Char, typename CharTraits>
std::basic_ostream<Char, CharTraits>& operator << (std::basic_ostream<Char, CharTraits>& os, const basic_layer& v)
{
    v.save(os);
    return os;
}

template <typename Char, typename CharTraits>
std::basic_istream<Char, CharTraits>& operator >> (std::basic_istream<Char, CharTraits>& os, basic_layer& v)
{
    v.load(os);
    return os;
}



class fully_connected_layer : public basic_layer
{
public:
    fully_connected_layer(activation_type act):basic_layer(af){}
    void init(const image::geometry<3>& in_dim,const image::geometry<3>& out_dim) override
    {
        basic_layer::init(in_dim.size(), out_dim.size(),in_dim.size() * out_dim.size(), out_dim.size());
        weight_base = std::sqrt(6.0 / (float)(input_size+output_size));
    }

    void forward_propagation(std::vector<float>& in) override
    {
        std::vector<float> wx(output_size);
        for(int i = 0;i < output_size;++i)
        {
            float sum(0);
            for(int c = 0, c_index = i; c < input_size; c++, c_index += output_size)
                sum += weight[c_index] * in[c];
            sum += bias[i];
            wx[i] = sum;
        }

        if(af == activation_type::tanh)
            for(int i = 0; i < output_size; ++i)
                wx[i] = tanh_f(wx[i]);
        if(af == activation_type::sigmoid)
            for(int i = 0; i < output_size; ++i)
                wx[i] = sigmoid_f(wx[i]);
        if(af == activation_type::relu)
            for(int i = 0; i < output_size; ++i)
                wx[i] = relu_f(wx[i]);
        in.swap(wx);
    }

    void back_propagation(std::vector<float>& in_dE_da,const std::vector<float>& prev_out) override
    {
        std::vector<float> dE_da(input_size);
        for(int c = 0,c_index = 0; c < input_size; c++,c_index += output_size)
            dE_da[c] = vec::dot(&in_dE_da[0], &in_dE_da[0] + output_size, &weight[c_index]);


        if(previous_af == activation_type::tanh)
            for(int c = 0; c < input_size; c++)
                dE_da[c] *= tanh_df(prev_out[c]);
        if(previous_af == activation_type::sigmoid)
            for(int c = 0; c < input_size; c++)
                dE_da[c] *= sigmoid_df(prev_out[c]);
        if(previous_af == activation_type::relu)
            for(int c = 0; c < input_size; c++)
                dE_da[c] *= relu_df(prev_out[c]);

        float* dw_ptr = &dweight[0];
        for(int c = 0; c < input_size; c++,dw_ptr += output_size)
            if(prev_out[c] != float(0))
                image::vec::axpy(dw_ptr,dw_ptr+output_size,prev_out[c],&in_dE_da[0]);
        image::add(dbias,in_dE_da);
        dE_da.swap(in_dE_da);
    }
    std::string name() const override
    {
        return "fully-connected";
    }

};


class partial_connected_layer : public basic_layer
{
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


    void forward_propagation(std::vector<float>& in) override
    {
        std::vector<float> wx(output_size);
        for(int i = 0; i < output_size; ++i)
        {
            const std::vector<int>& o2w_1i = o2w_1[i];
            const std::vector<int>& o2w_2i = o2w_2[i];
            float sum(0);
            for(int j = 0;j < o2w_1i.size();++j)
                sum += weight[o2w_1i[j]] * in[o2w_2i[j]];
            wx[i] = sum + bias[o2b[i]];
        }

        if(af == activation_type::tanh)
            for(int i = 0; i < output_size; ++i)
                wx[i] = tanh_f(wx[i]);
        if(af == activation_type::sigmoid)
            for(int i = 0; i < output_size; ++i)
                wx[i] = sigmoid_f(wx[i]);
        if(af == activation_type::relu)
            for(int i = 0; i < output_size; ++i)
                wx[i] = relu_f(wx[i]);
        in.swap(wx);
    }

    void back_propagation(std::vector<float>& in_dE_da,const std::vector<float>& prev_out) override
    {
        std::vector<float> dE_da(input_size);

        for(int i = 0; i != input_size; i++)
        {
            const std::vector<int>& i2w_1i = i2w_1[i];
            const std::vector<int>& i2w_2i = i2w_2[i];

            float sum(0);
            for(int j = 0;j < i2w_1i.size();++j)
                sum += weight[i2w_1i[j]] * in_dE_da[i2w_2i[j]];

            if(previous_af == activation_type::tanh)
                dE_da[i] = sum * tanh_df(prev_out[i]);
            if(previous_af == activation_type::sigmoid)
                dE_da[i] = sum * sigmoid_df(prev_out[i]);
            if(previous_af == activation_type::relu)
                dE_da[i] = sum * relu_df(prev_out[i]);

        }

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
        dE_da.swap(in_dE_da);
    }
protected:
    std::vector<std::vector<int> > w2o_1,w2o_2,o2w_1,o2w_2,i2w_1,i2w_2,b2o;
    std::vector<int> o2b;
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
            throw std::exception("invalid size in the max pooling layer");
        init_connection(pool_size);
        weight_base = std::sqrt(6.0 / (float)(o2i[0].size()+1));
    }

    void forward_propagation(std::vector<float>& in) override
    {
        std::vector<float> wx(basic_layer::output_size);

        for(int i = 0; i < basic_layer::output_size; i++)
        {
            float max_value = std::numeric_limits<float>::lowest();
            for(auto j : o2i[i])
            {
                if(in[j] > max_value)
                    max_value = in[j];
            }
            wx[i] = max_value;
        }
        if(af == activation_type::tanh)
            for(int i = 0; i < basic_layer::output_size; ++i)
                wx[i] = tanh_f(wx[i]);
        if(af == activation_type::sigmoid)
            for(int i = 0; i < basic_layer::output_size; ++i)
                wx[i] = sigmoid_f(wx[i]);
        if(af == activation_type::relu)
            for(int i = 0; i < basic_layer::output_size; ++i)
                wx[i] = relu_f(wx[i]);
        in.swap(wx);
    }

    void back_propagation(std::vector<float>& in_dE_da,const std::vector<float>& prev_out) override
    {
        std::vector<float> dE_da(input_size);
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
        if(previous_af == activation_type::tanh)
            for(int i = 0; i < input_size; i++)
            {
                int outi = i2o[i];
                dE_da[i] = (max_idx[outi] == i) ? in_dE_da[outi] * tanh_df(prev_out[i]) : float(0);
            }
        if(previous_af == activation_type::sigmoid)
            for(int i = 0; i < input_size; i++)
            {
                int outi = i2o[i];
                dE_da[i] = (max_idx[outi] == i) ? in_dE_da[outi] * sigmoid_df(prev_out[i]) : float(0);
            }
        if(previous_af == activation_type::relu)
            for(int i = 0; i < input_size; i++)
            {
                int outi = i2o[i];
                dE_da[i] = (max_idx[outi] == i) ? in_dE_da[outi] * relu_df(prev_out[i]) : float(0);
            }
        if(previous_af == activation_type::identity)
            for(int i = 0; i < input_size; i++)
            {
                int outi = i2o[i];
                dE_da[i] = (max_idx[outi] == i) ? in_dE_da[outi] : float(0);
            }
        dE_da.swap(in_dE_da);
    }
    std::string name() const override
    {
        return "max-pool";
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


class average_pooling_layer : public partial_connected_layer
{
    geometry<3> in_dim;
    geometry<3> out_dim;
    int pool_size;
public:
    typedef partial_connected_layer basic_layer;
    average_pooling_layer(activation_type af_,int pool_size_)
        : basic_layer(af_),pool_size(pool_size_){}
    void init(const image::geometry<3>& in_dim_,const image::geometry<3>& out_dim_) override
    {
        basic_layer::init(in_dim_.size(),in_dim_.size()/pool_size/pool_size,in_dim_.depth(), in_dim_.depth());
        in_dim = in_dim_;
        out_dim = out_dim_;
        if(out_dim != image::geometry<3>(in_dim_.width()/ pool_size, in_dim_.height() / pool_size, in_dim_.depth()))
            throw std::exception("invalid size in the max pooling layer");
        init_connection(pool_size);
        weight_base = std::sqrt(6.0 / (float)(max_size(o2w_1) + max_size(i2w_1)));
    }
    std::string name() const override
    {
        return "ave-pool";
    }

private:
    void init_connection(int pool_size)
    {
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
            throw std::exception("invalid layer dimension");
        weight_dim = image::geometry<3>(kernel_size,kernel_size,in_dim_.depth() * out_dim_.depth()),
        basic_layer::init(in_dim_.size(), out_dim_.size(),weight_dim.size(), out_dim_.depth());
        weight_base = std::sqrt(6.0 / (float)(weight_dim.plane_size() * in_dim.depth() + weight_dim.plane_size() * out_dim.depth()));
    }

    void forward_propagation(std::vector<float>& in) override
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
                            const float * p = &in[inc_index] + y_index + x;
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

            if(!bias.empty())
                image::add_constant(&wx[o_index],&wx[o_index]+out_dim.plane_size(),bias[o]);
        }

        if(af == activation_type::tanh)
            for(int i = 0; i < output_size; ++i)
                wx[i] = tanh_f(wx[i]);
        if(af == activation_type::sigmoid)
            for(int i = 0; i < output_size; ++i)
                wx[i] = sigmoid_f(wx[i]);
        if(af == activation_type::relu)
            for(int i = 0; i < output_size; ++i)
                wx[i] = relu_f(wx[i]);
        in.swap(wx);
    }

    void back_propagation(std::vector<float>& in_dE_da,const std::vector<float>& prev_out) override
    {
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

        if(previous_af == activation_type::tanh)
            for(int i = 0; i < in_dim.size(); ++i)
                dE_da[i] *= tanh_df(prev_out[i]);
        if(previous_af == activation_type::sigmoid)
            for(int i = 0; i < in_dim.size(); ++i)
                dE_da[i] *= sigmoid_df(prev_out[i]);
        if(previous_af == activation_type::relu)
            for(int i = 0; i < in_dim.size(); ++i)
                dE_da[i] *= relu_df(prev_out[i]);

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
        dE_da.swap(in_dE_da);
    }

    std::string name() const override
    {
        return "conv";
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
            throw std::exception("invalid layer dimension");
        dim =in_dim_.size();
        basic_layer::init(dim,dim,0,0);
    }

    void back_propagation(std::vector<float>& in_dE_da,const std::vector<float>& prev_out) override
    {
        if(drop.empty())
            return;
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
    void update(float learning_rate) override
    {
        drop.resize(dim);
        for(int i = 0;i < dim;++i)
            drop[i] = bernoulli(dropout_rate);
    }

    std::string name() const override
    {
        return "dropout";
    }
};


class network
{
    std::vector<std::shared_ptr<basic_layer> > layers;
    image::geometry<3> cur_dim;
public:
    network(){}

    void add(basic_layer* new_layer)
    {
        if(!layers.empty())
            new_layer->previous_af = layers.back()->af;
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
            layer->forward_propagation(in);
    }
    int predict_label(const std::vector<float>& in)
    {
        std::vector<float> result(in);
        predict(result);
        return std::max_element(result.begin(),result.end())-result.begin();
    }



    template <typename data_type,typename label_type,typename iter_type>
    void train(const data_type& data,const label_type& label,int iteration_count,iter_type iter_fun = [&]{},bool reset_weights = true)
    {
        if(reset_weights)
        {
            for(auto layer : layers)
                layer->reset();
        }
        float learning_rate = 0.02;
        int batch_size = 20;
        for(int iter = 0; iter < iteration_count; iter++,learning_rate *= 0.85,iter_fun())
        {
            for(int i = 0;i < data.size();i += batch_size)
            {
                int size = std::min<int>(batch_size,data.size()-i);
                par_for(size, [&](int m)
                {
                    std::vector<std::vector<float> > out(layers.size());
                    for(int k = 0;k < layers.size();++k)
                    {
                        out[k] = (k == 0 ? data[i + m] : out[k-1]);
                        layers[k]->forward_propagation(out[k]);
                    }
                    std::vector<float> output = out.back();
                    image::minus(output,label[i + m]);// diff of mse
                    if(layers.back()->af == activation_type::tanh)
                        for(int i = 0; i < out.size(); i++)
                            output[i] *= tanh_df(output[i]);
                    if(layers.back()->af == activation_type::sigmoid)
                        for(int i = 0; i < out.size(); i++)
                            output[i] *= sigmoid_df(output[i]);
                    if(layers.back()->af == activation_type::relu)
                        for(int i = 0; i < out.size(); i++)
                            output[i] *= relu_df(output[i]);

                    for(int k = layers.size()-1;k >= 0;--k)
                        layers[k]->back_propagation(output,k == 0? data[i + m] : out[k-1]);
                });
                for(auto& layer : layers)
                    layer->update(learning_rate/batch_size);
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
    void save(std::ostream& os) const
    {
        os.precision(std::numeric_limits<float>::digits10);
        for(auto layer : layers)
            layer->save(os);
    }

    void load(std::istream& is)
    {
        is.precision(std::numeric_limits<float>::digits10);
        for(auto layer : layers)
            layer->load(is);
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

network& operator << (network& n, basic_layer* layer)
{
    n.add(layer);
    return n;
}
network& operator << (network& n, const image::geometry<3>& dim)
{
    n.add(dim);
    return n;
}

template <typename Char, typename CharTraits>
std::basic_ostream<Char, CharTraits>& operator << (std::basic_ostream<Char, CharTraits>& os, const network& n)
{
    n.save(os);
    return os;
}

template <typename Char, typename CharTraits>
std::basic_istream<Char, CharTraits>& operator >> (std::basic_istream<Char, CharTraits>& os, network& n)
{
    n.load(os);
    return os;
}




}//cnn
}//ml
}//image
