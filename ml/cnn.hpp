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
#include <random>

#include "tipl/numerical/matrix.hpp"
#include "tipl/numerical/numerical.hpp"
#include "tipl/numerical/basic_op.hpp"
#include "tipl/numerical/resampling.hpp"
#include "tipl/utility/geometry.hpp"
#include "tipl/utility/basic_image.hpp"
#include "tipl/utility/multi_thread.hpp"


namespace tipl
{
namespace ml
{


template<class value_type>
inline float relu_f(value_type v)
{
    return v > value_type(0) ? float(v) : float(0);
}
template<class value_type>
inline float relu_df(value_type y)
{
    return y > value_type(0) ? float(1) : float(0);
}

enum activation_type { relu, identity};
enum status_type { training,testing};

class basic_layer
{

public:
    activation_type af;
    status_type status;
    int input_size;
    int output_size;
    float weight_base;
    float wlearning_base_rate = 1.0f;
    float blearning_base_rate = 1.0f;
    std::vector<float> weight,bias;
public:

    virtual ~basic_layer() {}
    basic_layer(activation_type af_ = activation_type::relu):af(af_),status(testing),weight_base(1){}
    void init( int input_size_, int output_size_, int weight_dim, int bias_dim)
    {
        input_size = input_size_;
        output_size = output_size_;
        weight.resize(weight_dim);
        bias.resize(bias_dim);
    }
    virtual void initialize_weight(tipl::uniform_dist<float>& gen)
    {
        for(int i = 0; i < weight.size(); ++i)
            weight[i] = gen()*weight_base;
        if(!bias.empty())
            std::fill(bias.begin(), bias.end(), 0.0f);
    }

    virtual bool init(const tipl::geometry<3>& in_dim,const tipl::geometry<3>& out_dim) = 0;
    virtual void forward_propagation(const float* data,float* out) = 0;
    void forward_af(float* data)
    {
        if(af == activation_type::relu)
            for(int i = 0; i < output_size; ++i)
                data[i] = relu_f(data[i]);
    }
    virtual void to_image(image<float,2>& I)
    {
        I.clear();
    }
    virtual void to_image(color_image& Is,int max_width)
    {
        Is.clear();
        image<float,2> I;
        to_image(I);
        if(I.empty())
            return;
        while(max_width && I.width() > max_width)
            tipl::downsampling(I);
        for(int j = 0;j < 2 && I.width() < max_width*0.5f;++j)
            tipl::upsampling_nearest(I);
        Is.resize(I.geometry());
        std::fill(Is.begin(),Is.end(),tipl::rgb(0xFFFFFFFF));
        for(int j = 0;j < I.size();++j)
        {
            if(I[j] == 0)
                continue;
            unsigned char s(std::min<int>(255,((int)std::fabs(I[j]*1024.0f))));
            if(I[j] < 0) // red
                Is[j] = tipl::rgb(s,0,0);
            if(I[j] > 0) // blue
                Is[j] = tipl::rgb(0,0,s);
        }
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
        if(af == activation_type::relu)
            for(int i = 0; i < output_size; ++i)
                if(prev_out[i] <= 0)
                    dE_da[i] = 0;
    }
    virtual unsigned int computation_cost(void) const
    {
        return (unsigned int)(weight.size());
    }
    virtual void update(float rw,const std::vector<float>& dw,
                        float rb,const std::vector<float>& db)
    {
        tipl::vec::axpy(&weight[0],&weight[0] + weight.size(),rw,&dw[0]);
        tipl::vec::axpy(&bias[0],&bias[0] + bias.size(),rb,&db[0]);
    }
};


class fully_connected_layer : public basic_layer
{
    tipl::geometry<3> in_dim;
public:
    fully_connected_layer(activation_type af_):basic_layer(af_){}
    bool init(const tipl::geometry<3>& in_dim_,const tipl::geometry<3>& out_dim) override
    {
        in_dim = in_dim_;
        basic_layer::init(in_dim.size(), out_dim.size(),in_dim.size() * out_dim.size(), out_dim.size());
        weight_base = (float)std::sqrt(6.0f / (float)(input_size+output_size));
        return true;
    }
    void initialize_weight(tipl::uniform_dist<float>& gen)
    {
        //basic_layer::initialize_weight(gen);
        basic_layer::initialize_weight(gen);

        if(input_size > output_size)
        {
            std::vector<float> u(output_size*output_size),s(output_size);
            tipl::mat::svd(&weight[0],&u[0],&s[0],tipl::dyndim(output_size,input_size));
        }
    }

    void forward_propagation(const float* data,float* out) override
    {
        for(int i = 0,i_pos = 0;i < output_size;++i,i_pos += input_size)
            out[i] = bias[i] + tipl::vec::dot(&weight[i_pos],&weight[i_pos]+input_size,&data[0]);
    }
    void calculate_dwdb(const float* in_dE_da,
                        const float* prev_out,
                        std::vector<float>& dweight,
                        std::vector<float>& dbias) override
    {
        tipl::add(&dbias[0],&dbias[0]+output_size,in_dE_da);
        for(int i = 0,i_pos = 0; i < output_size; i++,i_pos += input_size)
            if(in_dE_da[i] != float(0))
                tipl::vec::axpy(&dweight[i_pos],&dweight[i_pos]+input_size,in_dE_da[i],prev_out);
    }
    void to_image(image<float,2>& I)
    {
        std::vector<float> w(weight),b(bias);
        tipl::normalize_abs(w);
        tipl::normalize_abs(b);
        if(in_dim[1] == 1 || in_dim[0] == 1)
        {
            int width = in_dim.size()+3;
            I.resize(geometry<2>(width,b.size()));
            for(int row = 0,row_pos = 0,w_pos = 0;row < b.size();++row,row_pos += width,w_pos += in_dim.size())
            {
                std::copy(&w[w_pos],&w[w_pos]+in_dim.size(),&I[row_pos]);
                I[row_pos+width-2] = b[row];
            }
        }
        else
        {
            int n = int(weight.size()/in_dim.plane_size());
            int col = in_dim[2];
            int pad = 3;
            while(col < std::sqrt(n))
            {
                col += in_dim[2];
                pad += 3;
            }
            int row = n/col;
            I.resize(geometry<2>(col* (in_dim.width()+1)+pad,row * (in_dim.height() +1) + 1));
            int b_pos = 0;
            for(int y = 0,index = 0;y < row;++y)
                for(int x = 0;x < col;++x,++index)
                {
                    tipl::draw(tipl::make_image(&w[0] + index*in_dim.plane_size(),tipl::geometry<2>(in_dim[0],in_dim[1])),
                                I,tipl::geometry<2>(x*(in_dim.width()+1)+x/in_dim[2],y*(in_dim.height()+1)+1));
                    if((x+1)%in_dim[2] == 0)
                        I.at(x*(in_dim.width()+1)+x/in_dim[2]+in_dim.width(),y*(in_dim.height()+1)+1 + in_dim.height()/2) = b[b_pos++];
                }
            //std::copy(b.begin(),b.end(),I.end()-b.size());
        }
    }
    void back_propagation(float* in_dE_da,// output_size
                          float* out_dE_da,// input_size
                          const float*) override
    {
        tipl::mat::left_vector_product(&weight[0],in_dE_da,out_dE_da,tipl::dyndim(output_size,input_size));
    }
};


class partially_connected_layer : public fully_connected_layer
{
public:
    std::vector<std::vector<int> > mapping;
public:
    partially_connected_layer(activation_type af_):fully_connected_layer(af_){}
    bool init(const tipl::geometry<3>& in_dim_,const tipl::geometry<3>& out_dim) override
    {
        fully_connected_layer::init(in_dim_,out_dim);
        mapping.resize(output_size);
        return true;
    }
    void initialize_weight(tipl::uniform_dist<float>& gen)
    {
        size_t w_sum = 0;
        for(int i = 0;i < mapping.size();++i)
            w_sum += (mapping[i].empty() ? input_size : mapping[i].size());
        weight.resize(w_sum);
        basic_layer::initialize_weight(gen);
    }
    void forward_propagation(const float* data,float* out) override
    {
        for(int i = 0,i_pos = 0;i < output_size;++i)
            if(!mapping[i].empty())
            {
                const auto& cur_mapping = mapping[i];
                size_t new_size = cur_mapping.size();
                float sum = bias[i];
                for(int j = 0;j < new_size;++j)
                    sum += weight[i_pos++]*data[cur_mapping[j]];
                out[i] = sum;
            }
            else
            {
                out[i] = bias[i] + tipl::vec::dot(&weight[i_pos],&weight[i_pos]+input_size,&data[0]);
                i_pos += input_size;
            }
        //for(int i = 0,i_pos = 0;i < output_size;++i,i_pos += input_size)
        //    out[i] = bias[i] + tipl::vec::dot(&weight[i_pos],&weight[i_pos]+input_size,&data[0]);
    }
    void to_image(image<float,2>& I)
    {
        I.clear();
    }

    void calculate_dwdb(const float* in_dE_da,
                        const float* prev_out,
                        std::vector<float>& dweight,
                        std::vector<float>& dbias) override
    {
        tipl::add(&dbias[0],&dbias[0]+output_size,in_dE_da);
        for(int i = 0,i_pos = 0; i < output_size; i++)
        {
            if(in_dE_da[i] != float(0))
            {
                const auto& cur_mapping = mapping[i];
                size_t new_size = cur_mapping.size();
                float w = in_dE_da[i];
                if(new_size)
                {
                    for(int j = 0;j < new_size;++j)
                        dweight[i_pos++] += w*prev_out[cur_mapping[j]];
                }
                else
                {
                    tipl::vec::axpy(&dweight[i_pos],&dweight[i_pos]+input_size,in_dE_da[i],prev_out);
                    i_pos += input_size;
                }
            }
            else
                i_pos += (mapping[i].empty() ? input_size : mapping[i].size());
        }
        /*
        tipl::add(&dbias[0],&dbias[0]+output_size,in_dE_da);
        for(int i = 0,i_pos = 0; i < output_size; i++,i_pos += input_size)
            if(in_dE_da[i] != float(0))
                tipl::vec::axpy(&dweight[i_pos],&dweight[i_pos]+input_size,in_dE_da[i],prev_out);
        */

    }
    void back_propagation(float* ,// output_size
                          float* ,// input_size
                          const float*) override
    {
        /*
        for(int i = 0,i_pos = 0;i < output_size;++i)
        {
            const auto& cur_mapping = mapping[i];
            size_t new_size = cur_mapping.size();
            float w = in_dE_da[i];
            for(int j = 0;j < new_size;++j)
                out_dE_da[cur_mapping[j]] += weight[i_pos++]*w;
        }
        */
        //    tipl::mat::left_vector_product(&weight[0],in_dE_da,out_dE_da,tipl::dyndim(output_size,input_size));
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
    bool init(const tipl::geometry<3>& in_dim_,const tipl::geometry<3>& out_dim_) override
    {
        in_dim = in_dim_;
        out_dim = out_dim_;
        basic_layer::init(in_dim.size(),out_dim.size(),0,0);
        if(out_dim != tipl::geometry<3>(in_dim.width()/ pool_size, in_dim.height() / pool_size, in_dim.depth()))
            return false;
        init_connection();
        weight_base = (float)std::sqrt(6.0f / (float)(o2i[0].size()+1));
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
        return (unsigned int)(out_dim.size()*pool_size*pool_size/10.0f);
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
    bool init(const tipl::geometry<3>& in_dim_,const tipl::geometry<3>& out_dim_) override
    {
        in_dim = in_dim_;
        out_dim = out_dim_;
        if(in_dim.width()-out_dim.width()+1 != kernel_size ||
           in_dim.height()-out_dim.height()+1 != kernel_size)
            return false;
        //weight_dim = tipl::geometry<3>(kernel_size,kernel_size,in_dim.depth() * out_dim.depth()),
        basic_layer::init(in_dim_.size(), out_dim_.size(),kernel_size2* in_dim.depth() * out_dim.depth(), out_dim.depth());
        weight_base = (float)std::sqrt(6.0f / (float)(kernel_size2 * in_dim.depth() + kernel_size2 * out_dim.depth()));
        return true;
    }
    void to_image(image<float,2>& I)
    {
        std::vector<float> w(weight),b(bias);
        tipl::normalize_abs(w);
        tipl::normalize_abs(b);
        I.resize(geometry<2>(out_dim.depth()* (kernel_size+1)+1,in_dim.depth() * (kernel_size+1) + 3));
        for(int x = 0,index = 0;x < out_dim.depth();++x)
            for(int y = 0;y < in_dim.depth();++y,++index)
            {
                tipl::draw(tipl::make_image(&w[0] + index*kernel_size2,tipl::geometry<2>(kernel_size,kernel_size)),
                            I,tipl::geometry<2>(x*(kernel_size+1),y*(kernel_size+1)+1));
            }

        for(int i = 0,pos = I.size()-I.width()*2+kernel_size/2;i < b.size();++i,pos += kernel_size+1)
            I[pos] = b[i];
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
                                sum += tipl::vec::dot(w,w+kernel_size,p);
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
                dbias[outc] += std::accumulate(delta, delta + out_dim.plane_size(),0.0f);
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
                            tipl::vec::axpy(p,p+kernel_size,ppdelta_src,ppw);
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
    tipl::geometry<3> dim;
    tipl::bernoulli bgen;
public:
    float dropout_rate;
    dropout_layer(float dropout_rate_)
        : basic_layer(activation_type::identity),dropout_rate(dropout_rate_),
          bgen(dropout_rate_)
    {
    }
    bool init(const tipl::geometry<3>& in_dim_,const tipl::geometry<3>& out_dim_) override
    {
        if(in_dim_.size() != out_dim_.size())
            return false;
        dim =in_dim_;
        basic_layer::init(dim.size(),dim.size(),0,0);
        return true;
    }

    void back_propagation(float* in_dE_da,// output_size
                          float* out_dE_da,// input_size
                          const float* pre_out) override
    {
        pre_out += dim.size();
        if(dim.plane_size() == 1)
            for(unsigned int i = 0; i < dim.size(); i++)
                out_dE_da[i] = (pre_out[i] == 0.0f ? 0.0f: in_dE_da[i]);
        else
        {
            for(unsigned int i = 0; i < dim.depth(); i++,
                pre_out += dim.plane_size(),out_dE_da += dim.plane_size(),in_dE_da += dim.plane_size())
            if(pre_out[0] == 0.0f)
                std::fill(out_dE_da,out_dE_da+dim.plane_size(),0.0f);
            else
                std::copy(in_dE_da,in_dE_da+dim.plane_size(),out_dE_da);
        }
    }
    void forward_propagation(const float* data,float* out) override
    {
        if(status == testing)
        {
            std::copy(data,data+dim.size(),out);
            return;
        }
        if(dim.plane_size() == 1)
            for(unsigned int i = 0; i < dim.size(); i++)
                out[i] = bgen() ? 0.0f: (data[i] == 0.0f ? std::numeric_limits<float>::min() : data[i]);
        else
        for(unsigned int i = 0; i < dim.depth(); i++,data += dim.plane_size(),out += dim.plane_size())
            if(bgen())
                std::fill(out,out+dim.plane_size(),0.0f);
            else
            {
                std::copy(data,data+dim.plane_size(),out);
                if(*out == 0.0f)
                    *out = std::numeric_limits<float>::min();
            }

    }
};

class soft_max_layer : public basic_layer{
public:
    soft_max_layer(void)
        : basic_layer(activation_type::identity)
    {
    }
    bool init(const tipl::geometry<3>& in_dim,const tipl::geometry<3>& out_dim) override
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
            tipl::divide_constant(out,out+output_size,sum);
    }
    void back_propagation(float* in_dE_da,// output_size
                          float* out_dE_da,// input_size
                          const float* prev_out) override
    {
        std::copy(in_dE_da,in_dE_da+input_size,out_dE_da);
        tipl::minus_constant(out_dE_da,out_dE_da+output_size,tipl::vec::dot(in_dE_da,in_dE_da+input_size,prev_out));
        tipl::multiply(out_dE_da,out_dE_da+output_size,prev_out);
    }
};


template<typename value_type,typename label_type>
class network_data
{
public:
    tipl::geometry<3> input,output;
    std::vector<std::vector<value_type> > data;
    std::vector<label_type> data_label;

    size_t size(void) const{return data_label.size();}
    void clear(void)
    {
        data.clear();
        data_label.clear();
    }
    bool empty(void) const
    {
        return data.empty();
    }
    template <typename io_type>
    bool load_from_file(const char* file_name)
    {
        io_type in;
        if(!in.open(file_name))
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
    template <typename io_type>
    bool save_to_file(const char* file_name) const
    {
        io_type out;
        if(!out.open(file_name))
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

template<typename value_type,typename label_type>
class network_data_proxy{
public:
    const network_data<value_type,label_type>* source = 0;
    std::vector<unsigned int> pos;
public:
    network_data_proxy(void){}
    network_data_proxy(const network_data<value_type,label_type>& rhs)
    {
        (*this) = rhs;
    }
    network_data_proxy& operator=(const network_data<value_type,label_type>& rhs)
    {
        source = &rhs;
        pos.resize(rhs.size());
        for(int i = 0;i < pos.size();++i)
            pos[i] = i;
        return *this;
    }
    const value_type* get_data(unsigned int index) const{return &source->data[pos[index]][0];}
    label_type get_label(unsigned int index) const{return source->data_label[pos[index]];}
    size_t size(void)const{return pos.size();}
    bool empty(void)const{return pos.empty();}
    template <typename seed_type>
    void shuffle(seed_type& seed){std::shuffle(pos.begin(),pos.end(),seed);}
    void homogenize(void)
    {
        if(source->output.size() == 1)
            return;
        std::vector<std::vector<unsigned int> > pile(source->output.size());
        for(int i = 0;i < pos.size();++i)
            if(source->data_label[pos[i]] < pile.size())
                pile[source->data_label[pos[i]]].push_back(pos[i]);
        int max_size = 0;
        for(int i = 0;i < pile.size();++i)
            max_size = std::max<int>(max_size,pile[i].size());
        for(int i = 0;i < pile.size();++i)
            if(!pile[i].empty())
            {
                int dup = max_size/pile[i].size()-1;
                for(int j = 0;j < dup;++j)
                    pos.insert(pos.end(),pile[i].begin(),pile[i].end());
            }
    }
};

template<typename value_type,typename label_type>
void data_fold_for_cv(const network_data<value_type,label_type>& rhs,
                      std::vector<network_data_proxy<value_type,label_type> > & training_data,
                      std::vector<network_data_proxy<value_type,label_type> > & testing_data,
                      int total_fold = 10)
{
    training_data = std::vector<network_data_proxy<value_type,label_type> >(total_fold);
    testing_data = std::vector<network_data_proxy<value_type,label_type> >(total_fold);
    for(int i = 0;i < total_fold;++i)
    {
        training_data[i].source = &rhs;
        testing_data[i].source = &rhs;
    }
    auto idx = tipl::arg_sort(rhs.data_label,std::less<float>());
    for(int i = 0,bin = 0;i < rhs.size();++i,++bin)
    {
        if(bin >= total_fold)
            bin = 0;
        for(int j = 0;j < total_fold;++j)
            if(bin == j)
                testing_data[j].pos.push_back(idx[i]);
            else
                training_data[j].pos.push_back(idx[i]);
    }
}
/*

    void rotate_permute(void)
    {
        tipl::uniform_dist<int> gen(2);
        for(int j = 0;j < size();++j)
        {
            auto I = tipl::make_image(&data[j][0],input);
            if(gen())
                tipl::flip_x(I);
            if(gen())
                tipl::flip_y(I);
            if(gen())
                tipl::swap_xy(I);
        }
    }
 */
class network
{
public:
    std::vector<std::shared_ptr<basic_layer> > layers;
    std::vector<tipl::geometry<3> > geo;
    unsigned int data_size = 0;
    unsigned int output_size = 0;
    bool initialized = false;
    std::string error_msg;
public:
    network(){}
    void reset(void)
    {
        layers.clear();
        geo.clear();
        data_size = 0;
        initialized = false;
    }
    const network& operator=(const network& rhs)
    {
        reset();
        add(rhs.get_layer_text());
        for(int i = 0;i < rhs.layers.size();++i)
            if(!layers[i]->weight.empty())
            {
                layers[i]->weight = rhs.layers[i]->weight;
                layers[i]->bias = rhs.layers[i]->bias;
            }
        data_size = rhs.data_size;
        output_size = rhs.output_size;
        initialized = rhs.initialized;
        return *this;
    }
    const network& operator=(network&& rhs)
    {
        layers.swap(rhs.layers);
        geo.swap(rhs.geo);
        data_size = rhs.data_size;
        output_size = rhs.output_size;
        initialized = rhs.initialized;
        return *this;
    }

    void init_weights(unsigned int seed = 0)
    {
        tipl::uniform_dist<float> gen(-1.0,1.0,seed);
        for(auto layer : layers)
            layer->initialize_weight(gen);
        initialized = true;
    }
    void sort_fully_layer(void)
    {
        for(int i = 0;i+1 < layers.size();++i)
        {
            fully_connected_layer* l1;
            fully_connected_layer* l2;
            if((l1 = dynamic_cast<fully_connected_layer*>(layers[i].get())) &&
               (l2 = dynamic_cast<fully_connected_layer*>(layers[i+1].get())))
            {
                auto& w1 = l1->weight;
                auto& b1 = l1->bias;
                auto& w2 = l2->weight;
                int n = l1->input_size;
                int m = l1->output_size;
                std::vector<float> vector_length(m);
                tipl::par_for(m,[&](int i)
                {
                    int pos = i*n;
                    vector_length[i] = tipl::vec::norm2(w1.begin()+pos,w1.begin()+pos+n);
                });
                auto idx = tipl::arg_sort(vector_length,std::greater<float>());
                std::vector<float> nw1(w1),nb1(b1),nw2(w2);
                for(int i = 0,pos = 0;i < m;++i,pos += n)
                    if(idx[i] != i)
                    {
                        tipl::copy_ptr(nw1.begin()+idx[i]*n,w1.begin()+pos,n);
                        b1[i] = nb1[idx[i]];
                        for(int j = i,k = idx[i];j < nw2.size();j += m,k += m)
                            w2[j] = nw2[k];
                    }
            }
        }
    }

    bool empty(void) const{return layers.empty();}
    unsigned int get_output_size(void) const{return output_size;}
    unsigned int get_input_size(void) const{return geo.empty() ? 0: geo[0].size();}
    tipl::geometry<3> get_input_dim(void) const{return geo.empty() ? tipl::geometry<3>(): geo.front();}
    tipl::geometry<3> get_output_dim(void) const{return geo.empty() ? tipl::geometry<3>(): geo.back();}
    bool add(const tipl::geometry<3>& dim)
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
        for(auto layer : layers)
            cost += layer->computation_cost();
        return cost;
    }
    void get_min_max(std::vector<float>& wmin,
                     std::vector<float>& wmax,
                     std::vector<float>& bmin,
                     std::vector<float>& bmax)
    {
        for(auto layer : layers)
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
    void get_layer_images(std::vector<tipl::color_image>& Is,int max_width = 600)
    {
        Is.clear();
        Is.resize(layers.size());
        tipl::par_for(layers.size(),[&](int i)
        {
            layers[i]->to_image(Is[i],max_width);
        });
    }

    void to_image(color_image& I,std::vector<float> in,int label,int layer_height = 20,int max_width = 600)
    {
        std::vector<tipl::color_image> Is;
        get_layer_images(Is);
        int total_height = 0;
        for(int i = 0;i < Is.size();++i)
            total_height += std::max<int>(Is[i].height(),layer_height);

        std::vector<tipl::color_image> values(geo.size());
        {
            std::vector<float> out(data_size),back(data_size);
            float* out_buf = &out[0];
            float* back_buf = &back[0];
            float* in_buf = &in[0];
            forward_propagation(in_buf,out_buf);
            back_propagation(label,out_buf,back_buf);
            for(int i = 0;i < geo.size();++i)
            {
                int col = std::max<int>(1,(max_width-1)/(geo[i].width()+1));
                int row = std::max<int>(1,geo[i][2]/col+1);
                if(i == 0)
                    values[i].resize(tipl::geometry<2>(col*(geo[i].width()+1)+1,row*(geo[i].height()+1)+1));
                else
                    values[i].resize(tipl::geometry<2>(col*(geo[i].width()+1)+1,int(2.0f*row*(geo[i].height()+1)+2)));
                std::fill(values[i].begin(),values[i].end(),tipl::rgb(255,255,255));
                int draw_width = 0;
                for(int y = 0,j = 0;y < row;++y)
                    for(int x = 0;j < geo[i][2] && x < col;++x,++j)
                    {
                        auto v1 = tipl::make_image((i == 0 ? in_buf : out_buf)+geo[i].plane_size()*j,tipl::geometry<2>(geo[i][0],geo[i][1]));
                        auto v2 = tipl::make_image((i == 0 ? in_buf : back_buf)+geo[i].plane_size()*j,tipl::geometry<2>(geo[i][0],geo[i][1]));
                        tipl::normalize_abs(v1);
                        tipl::normalize_abs(v2);
                        tipl::color_image Iv1(v1.geometry()),Iv2(v2.geometry());
                        for(int j = 0;j < Iv1.size();++j)
                        {
                            unsigned char s1(std::min<int>(255,int(255.0f*std::fabs(v1[j]))));
                            if(v1[j] < 0) // red
                                Iv1[j] = tipl::rgb(s1,0,0);
                            if(v1[j] >= 0) // blue
                                Iv1[j] = tipl::rgb(0,0,s1);
                            unsigned char s2(std::min<int>(255,int(255.0f*std::fabs(v2[j]))));
                            if(v2[j] < 0) // red
                                Iv2[j] = tipl::rgb(s2,0,0);
                            if(v2[j] >= 0) // blue
                                Iv2[j] = tipl::rgb(0,0,s2);
                        }
                        tipl::draw(Iv1,values[i],tipl::geometry<2>(x*(geo[i].width()+1)+1,y*(geo[i].height()+1)+1));
                        if(i)
                            tipl::draw(Iv2,values[i],tipl::geometry<2>(x*(geo[i].width()+1)+1,row*(geo[i].height()+1)+1+y*(geo[i].height()+1)+1));
                        draw_width = std::max<int>(draw_width,Iv1.width() + x*(geo[i].width()+1)+1);
                    }
                while((draw_width << 1) < max_width && values[i].height() < 50)
                {
                    tipl::upsampling_nearest(values[i]);
                    draw_width <<= 1;
                }
                while(draw_width > max_width)
                {
                    tipl::downsampling(values[i]);
                    draw_width >>= 1;
                }
                total_height += values[i].height();
                back_buf += geo[i].size();
                out_buf += geo[i].size();
            }
        }

        I.resize(tipl::geometry<2>(max_width,total_height));
        std::fill(I.begin(),I.end(),tipl::rgb(255,255,255));
        int cur_height = 0;
        for(int i = 0;i < geo.size();++i)
        {
            // input image
            tipl::draw(values[i],I,tipl::geometry<2>(0,cur_height));
            cur_height += values[i].height();

            // network wieghts
            if(i < layers.size())
            {
                tipl::rgb b;
                if(Is[i].empty())
                    b.from_hsl(0.5,0.5,0.85);
                else
                    b = Is[i][0];
                tipl::fill_rect(I,tipl::geometry<2>(0,cur_height),
                                   tipl::geometry<2>(max_width,cur_height+std::max<int>(Is[i].height(),layer_height)),b);
                tipl::draw(Is[i],I,tipl::geometry<2>(1,cur_height +
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
        initialized = false;
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
                return add(tipl::geometry<3>(x,y,z));
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
        if(list[0] == "partially")
        {
            layers.push_back(std::make_shared<partially_connected_layer>(af));
            return true;
        }
        if(list.size() < 3)
            return false;
        int param;
        std::istringstream(list[2]) >> param;
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
    std::shared_ptr<basic_layer> get_layer(int i) const{return layers[i];}
    const tipl::geometry<3>& get_geo(int i) const{return geo[i];}
    std::string get_layer_text(void) const
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
                if(layers[i]->af == activation_type::relu)
                    af_type = "relu";
                if(layers[i]->af == activation_type::identity)
                    af_type = "identity";

                if(dynamic_cast<convolutional_layer*>(layers[i].get()))
                    out << "conv," << af_type << "," << dynamic_cast<convolutional_layer*>(layers[i].get())->kernel_size;
                if(dynamic_cast<max_pooling_layer*>(layers[i].get()))
                    out << "max_pooling," << af_type << "," << dynamic_cast<max_pooling_layer*>(layers[i].get())->pool_size;
                if(dynamic_cast<partially_connected_layer*>(layers[i].get()))
                    out << "partially," << af_type;
                else
                    if(dynamic_cast<fully_connected_layer*>(layers[i].get()))
                        out << "full," << af_type;
                if(dynamic_cast<soft_max_layer*>(layers[i].get()))
                    out << "soft_max";
                if(dynamic_cast<dropout_layer*>(layers[i].get()))
                    out << "dropout," << dynamic_cast<dropout_layer*>(layers[i].get())->dropout_rate;

            }
        }
        return out.str();
    }
    template<typename io_type>
    bool save_to_file(const char* file_name)
    {
        io_type file;
        if(!file.open(file_name))
            return false;
        std::string nn_text = get_layer_text();
        unsigned int nn_text_length = nn_text.length();
        file.write((const char*)&nn_text_length,sizeof(nn_text_length));
        file.write((const char*)&*nn_text.begin(),nn_text_length);
        for(auto layer : layers)
            if(!layer->weight.empty())
            {
                file.write((const char*)&*layer->weight.begin(),layer->weight.size()*4);
                file.write((const char*)&*layer->bias.begin(),layer->bias.size()*4);
            }
        return true;
    }
    template<typename io_type>
    bool load_from_file(const char* file_name)
    {
        io_type in;
        if(!in.open(file_name))
            return false;
        size_t nn_text_length = 0;
        in.read((char*)&nn_text_length,4);
        std::string nn_text;
        nn_text.resize(nn_text_length);
        in.read((char*)&*nn_text.begin(),nn_text_length);
        if(!in)
            return false;
        reset();
        add(nn_text);
        for(auto layer : layers)
            if(!layer->weight.empty())
            {
                in.read((char*)&*layer->weight.begin(),layer->weight.size()*4);
                in.read((char*)&*layer->bias.begin(),layer->bias.size()*4);
            }
        initialized = true;
        set_test_mode(true);
        return !!in;
    }




    void forward_propagation(const float* input,float* out_ptr) const
    {
        for(int k = 0;k < layers.size();++k)
        {
            if(k == 0)
                std::copy(input,input+layers[0]->input_size,out_ptr);
            float* next_ptr = out_ptr + layers[k]->input_size;
            layers[k]->forward_propagation(out_ptr,next_ptr);
            layers[k]->forward_af(next_ptr);
            out_ptr = next_ptr;
        }
    }
    void forward_propagation(std::vector<float>& in) const
    {
        std::vector<float> out(data_size);
        forward_propagation(&in[0],&out[0]);
        in.resize(output_size);
        std::copy(out.end()-in.size(),out.end(),in.begin());
    }
    template<typename label_type>
    float back_propagation(label_type label,const float* input,float* out_ptr) const
    {
        const float target_value_min = 0.2f;
        const float target_value_max = 0.8f;
        const float* out_ptr2 = input + data_size - output_size;
        float* df_ptr = out_ptr + data_size - output_size;
        // calculate difference
        tipl::copy_ptr(out_ptr2,df_ptr,output_size);
        float error = 0.0f;
        if(output_size == 1) // regression
            error = std::fabs(df_ptr[0] -= label);
        else
        for(unsigned int i = 0;i < output_size;++i)
        {
            if(label == i)
            {
                df_ptr[i] -= target_value_max;
                if(df_ptr[i] > 0.0f)
                    df_ptr[i] = 0.0f;
            }
            else
            {
                df_ptr[i] -= target_value_min;
                if(df_ptr[i] < 0.0f)
                    df_ptr[i] = 0.0f;
            }
            error += std::abs(df_ptr[i]);
        }
        for(int k = (int)layers.size()-1;k >= 0;--k)
        {
            layers[k]->back_af(df_ptr,out_ptr2);
            const float* next_out_ptr = out_ptr2 - layers[k]->input_size;
            float* next_df_ptr = df_ptr - layers[k]->input_size;
            layers[k]->back_propagation(df_ptr,next_df_ptr,next_out_ptr);
            out_ptr2 = next_out_ptr;
            df_ptr = next_df_ptr;
        }
        return error;
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

    void set_test_mode(bool test)
    {
        for(auto layer : layers)
            layer->status = (test ? testing:training);
    }

    template<typename output_type>
    void predict(std::vector<float>& in,output_type& output)const
    {
        predict(&in[0],output);
    }

    template<typename value_type,typename output_type>
    void predict(const value_type* in,output_type& output)const
    {
        std::vector<value_type> result(data_size);
        forward_propagation(in,&result[0]);
        if(output_size == 1)
            output = result.back();
        else
            output = std::max_element(result.end()-output_size,result.end())-result.end()+output_size;
    }

    template<typename value_type,typename label_type,typename result_type>
    void predict(const network_data_proxy<value_type,label_type>& data,result_type& test_result) const
    {
        test_result.resize(data.size());
        par_for((int)data.size(), [&](int i)
        {
            predict(data.get_data(i),test_result[i]);
        });
    }
};

inline bool operator << (network& n, const tipl::geometry<3>& dim)
{
    return n.add(dim);
}
inline bool operator << (network& n, const std::string& text)
{
    return n.add(text);
}



//template<typename optimizer>
class trainer{
private:
    std::mt19937 rd_gen;
private:// for training
    std::vector<std::vector<std::vector<float> > > dweight,dbias;
    std::vector<std::vector<float> > in_out,back_df;
    std::vector<float*> in_out_ptr,back_df_ptr;
public:
    float w_decay_rate = 0.005f;
    float learning_rate = 0.01f;
    float rate_decay = 1.0f;
    float momentum = 0.9f;
    float bias_cap = 10.0f;
    float weight_cap = 100.0f;
    int batch_size = 64;
    int epoch= 20;
    int repeat = 1;
public:
    std::vector<unsigned int> error_table;
    unsigned int training_count = 0;
    unsigned int training_error_count = 0;
    float training_error_value = 0.0;
    std::vector<float> train_result;
public:
    void reset(void)
    {
        dweight.clear();
        dbias.clear();
        training_count = 0;
        training_error_count = 0;
        training_error_value = 0.0f;
    }

    float get_training_error(void) const
    {
        return 100.0f*training_error_count/training_count;
    }
    float get_training_error_value(void) const
    {
        return training_error_value/training_count;
    }
    void initialize_training(const network& nn)
    {
        int thread_count = std::thread::hardware_concurrency();
        dweight.resize(thread_count);
        dbias.resize(thread_count);
        in_out.resize(thread_count);
        back_df.resize(thread_count);
        in_out_ptr.resize(thread_count);
        back_df_ptr.resize(thread_count);
        for(int i = 0;i < thread_count;++i)
        {
            dweight[i].resize(nn.layers.size());
            dbias[i].resize(nn.layers.size());
            for(int j = 0;j < nn.layers.size();++j)
            {
                dweight[i][j].resize(nn.layers[j]->weight.size());
                dbias[i][j].resize(nn.layers[j]->bias.size());
            }
            in_out[i].resize(nn.data_size);
            back_df[i].resize(nn.data_size);
            in_out_ptr[i] = &in_out[i][0];
            back_df_ptr[i] = &back_df[i][0];
        }
    }
    template <class network_data_type>
    void train_batch(network& nn,network_data_type& data,bool &terminated)
    {
        nn.set_test_mode(false);
        int size = batch_size;
        training_count = 0;
        training_error_count = 0;
        training_error_value = 0.0f;
        if(!error_table.empty())
            std::fill(error_table.begin(),error_table.end(),0);
        for(int i = 0;i < data.size() && !terminated;i += size)
        {
            // train a batch
            par_for2(std::min<int>(batch_size,data.size()-i),[&](int m, int thread_id)
            {
                ++training_count;
                int data_index = i+m;
                if(terminated)
                    return;
                nn.forward_propagation(data.get_data(data_index),in_out_ptr[thread_id]);
                auto label = data.get_label(data_index);
                auto ptr = in_out_ptr[thread_id] + nn.data_size - nn.output_size;
                if(nn.output_size != 1) // regression
                {
                    size_t predicted_label = std::max_element(ptr,ptr+nn.output_size)-ptr;
                    if(data_index < train_result.size())
                        train_result[data_index] = predicted_label;
                    if(label != predicted_label)
                        ++training_error_count;
                    if(!error_table.empty())
                    {
                        size_t pos = nn.output_size*label + predicted_label;
                        if(pos < error_table.size())
                            ++error_table[pos];
                    }
                }
                else
                {
                    if(data_index < train_result.size())
                        train_result[data_index] = *ptr;
                }
                training_error_value += nn.back_propagation(label,in_out_ptr[thread_id],back_df_ptr[thread_id]);
                nn.calculate_dwdb(data.get_data(data_index),in_out_ptr[thread_id],back_df_ptr[thread_id],
                                    dweight[thread_id],dbias[thread_id]);
            });
            // update_weights
            par_for(nn.layers.size(),[this,&nn](int j)
            {
                if(nn.layers[j]->weight.empty())
                    return;
                std::vector<float> dw(nn.layers[j]->weight.size());
                std::vector<float> db(nn.layers[j]->bias.size());
                par_for(dweight.size(),[this,&dw,&db,j](int k)
                {
                    tipl::add(dw,dweight[k][j]);
                    tipl::add(db,dbias[k][j]);
                    tipl::multiply_constant(dweight[k][j],momentum);
                    tipl::multiply_constant(dbias[k][j],momentum);
                });

                if(nn.layers[j]->wlearning_base_rate == 1 && nn.layers[j]->blearning_base_rate == 1)
                {
                    nn.layers[j]->wlearning_base_rate = learning_rate*0.01f/(tipl::max_abs_value(dw)+1.0f);
                    nn.layers[j]->blearning_base_rate = learning_rate*0.01f/(tipl::max_abs_value(db)+1.0f);
                }

                nn.layers[j]->update(-nn.layers[j]->wlearning_base_rate*rate_decay,dw,
                                  -nn.layers[j]->blearning_base_rate*rate_decay,db);

                tipl::upper_lower_threshold(nn.layers[j]->bias,-bias_cap,bias_cap);
                tipl::upper_lower_threshold(nn.layers[j]->weight,-weight_cap,weight_cap);
            });
        }
    }

    template <typename value_type,typename label_type,typename iter_type>
    void train(network& nn,
               network_data_proxy<value_type,label_type>& data,
               bool &terminated,iter_type iter_fun,int seed = 0)
    {
        reset();
        initialize_training(nn);
        //int w_decay_count = 0;
        std::vector<std::vector<float> > w_decay;
        if(!nn.initialized)
        {
            nn.init_weights(seed);
            for(int i = 0;i < nn.layers.size();++i)
            {
                nn.layers[i]->wlearning_base_rate = 1.0f;
                nn.layers[i]->blearning_base_rate = 1.0f;
            }
        }
        /*
        if(w_decay_rate != 0.0f)
        {
            w_decay.resize(nn.layers.size());
            tipl::par_for(nn.layers.size(),[&](int i)
            {
                w_decay[i] = nn.layers[i]->weight;
                tipl::multiply_constant(w_decay[i].begin(),w_decay[i].end(),w_decay_rate);
            });
            w_decay_count = 1.0f/w_decay_rate;
        }*/
        for(int run = 0;run < repeat && !terminated;++run)
        {
            rate_decay = 1.0f;
            for(int iter = 0; iter < epoch && !terminated;iter++ ,iter_fun())
            {
                data.shuffle(rd_gen);
                train_batch(nn,data,terminated);
                rate_decay *= 0.998f;
                /*
                if(w_decay_count)
                {
                    tipl::par_for(nn.layers.size(),[&](int i)
                    {
                        if(!nn.layers[i]->weight.empty())
                            tipl::minus(nn.layers[i]->weight,w_decay[i]);
                    });
                    --w_decay_count;
                }*/
            }
        }

    }

};


struct iterate_cnn_data{
    tipl::geometry<3> dim;
    std::string str;
    int num_conv = 0;
    int depth = 0;
    enum {root = 1, conv = 2, max_pooling = 4, fully = 8, fully_dropout = 16} previous_layer;

};

template<typename str_list_type>
void iterate_cnn(
             const tipl::geometry<3>& in_dim,
             const tipl::geometry<3>& out_dim,
             str_list_type& list,
             int max_conv = 4,
             int max_depth = 12,
             int max_list = 10000)
{
    unsigned int layer_cost = 0;
    std::multimap<int, std::string> sorted_list;

    for(int width = 20; width <= 80; width *= 2)
    {
        iterate_cnn_data new_layer;
        new_layer.dim = in_dim;
        new_layer.str = std::string();
        new_layer.previous_layer = iterate_cnn_data::root;

        std::multimap<int, iterate_cnn_data> candidates;
        candidates.insert(std::make_pair(0,new_layer));
        for(int list_size = 0;list_size < max_list && !candidates.empty();++list_size)
        {
            int max_cost = std::numeric_limits<int>::max();
            if(candidates.size() > max_list)
                max_cost = (--candidates.end())->first;
            int cur_cost = candidates.begin()->first;
            iterate_cnn_data cur_layer = candidates.begin()->second;
            cur_layer.depth++;
            candidates.erase(candidates.begin());
            // output dimension
            {
                std::ostringstream sout;
                sout << cur_layer.dim[0] << "," << cur_layer.dim[1] << "," << cur_layer.dim[2];
                cur_layer.str += sout.str();
                cur_layer.str += "|";
            }
            if(cur_layer.depth <= max_depth)
            {
                // add max pooling
                if(cur_layer.previous_layer == iterate_cnn_data::conv && cur_layer.dim.width() > out_dim.width())
                {
                    new_layer = cur_layer;
                    new_layer.dim[0] /= 2;
                    new_layer.dim[1] /= 2;
                    new_layer.str += std::string("max_pooling,identity,2|");
                    new_layer.previous_layer = iterate_cnn_data::max_pooling;
                    int cost = cur_layer.dim.size()+layer_cost;
                    if(cur_cost+cost < max_cost && new_layer.dim.size() > out_dim.size())
                        candidates.insert(std::make_pair(cur_cost+cost,new_layer));
                }
                // add convolutional layer
                if((cur_layer.previous_layer == iterate_cnn_data::root ||
                   cur_layer.previous_layer == iterate_cnn_data::conv ||
                   cur_layer.previous_layer == iterate_cnn_data::max_pooling) &&
                        cur_layer.num_conv < max_conv)
                for(int kernel = 3;kernel <= 5;kernel += 2)
                {
                    if(cur_layer.dim[0] < kernel+1 || cur_layer.dim[1] < kernel+1)
                        break;
                    new_layer = cur_layer;
                    ++new_layer.num_conv;
                    new_layer.dim[0] -= kernel-1;
                    new_layer.dim[1] -= kernel-1;
                    new_layer.dim[2] = width/2;
                    new_layer.str += std::string("conv,relu,")+std::to_string(kernel)+"|";
                    new_layer.previous_layer = iterate_cnn_data::conv;

                    int cost = new_layer.dim.size()*cur_layer.dim.depth()*kernel*kernel+layer_cost;
                    if(cur_cost+cost < max_cost)
                        candidates.insert(std::make_pair(cur_cost+cost,new_layer));
                }
                // add fully connected
                if(cur_layer.previous_layer != iterate_cnn_data::root)
                {
                    new_layer = cur_layer;
                    new_layer.dim[0] = 1;
                    new_layer.dim[1] = 1;
                    new_layer.dim[2] = width;
                    new_layer.str += std::string("full,relu|1,1,")+std::to_string(width)+"|dropout,0.1|";
                    new_layer.previous_layer = iterate_cnn_data::fully;
                    int cost = cur_layer.dim.size()*new_layer.dim.size()+layer_cost;
                    if(cur_cost+cost < max_cost)
                        candidates.insert(std::make_pair(cur_cost+cost,new_layer));
                }
            }

            // end
            if(cur_layer.depth > 2)
            {
                std::ostringstream sout;
                sout << out_dim[0] << "," << out_dim[1] << "," << out_dim[2];
                std::string s = cur_layer.str + std::string("full,relu|")+sout.str();
                int cost = cur_layer.dim.size()*out_dim.size()+out_dim.size();
                if(cost+cur_cost < max_cost)
                    sorted_list.insert(std::make_pair(cost+cur_cost,s));
            }
        }
    }
    for(auto& p:sorted_list)
        list.push_back(p.second);
}





}//ml
}//image

#endif//CNN_HPP
