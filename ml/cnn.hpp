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

#include "../numerical/matrix.hpp"
#include "../numerical/numerical.hpp"
#include "../numerical/basic_op.hpp"
#include "../numerical/resampling.hpp"
#include "../utility/shape.hpp"
#include "../utility/basic_image.hpp"
#include "../mt.hpp"


namespace tipl
{
namespace ml
{


enum status_type { training,testing};

class basic_activation_layer{
protected:
    int output_size = 0;
public:
    virtual void init(const tipl::shape<3>& out_dim){output_size = out_dim.size();}
    virtual void forward_af(float*){;}
    virtual void back_af(float*,const float*){;}
};


class relu_layer : public basic_activation_layer{
public:
    void forward_af(float* y)
    {

        for(int i = 0; i < output_size; ++i)
            if(y[i] < 0.0f)
                y[i] = 0.0f;

    }
    void back_af(float* dOut,const float* x)
    {
        for(int i = 0; i < output_size; ++i)
             if(x[i] < 0)
                 dOut[i] = 0;
    }
};

class leaky_relu_layer : public basic_activation_layer{
public:
    void forward_af(float* y)
    {

        for(int i = 0; i < output_size; ++i)
            if(y[i] < 0.0f)
                y[i] *= 0.01f;

    }
    void back_af(float* dOut,const float* x)
    {
        for(int i = 0; i < output_size; ++i)
             if(x[i] < 0)
                 dOut[i] *= 0.01f;
    }
};

class pull_layer : public basic_activation_layer{
public:
    void forward_af(float* y)
    {
        tipl::add(y,y+output_size,y-output_size-output_size);
    }
};

class push_layer : public basic_activation_layer{
public:
    void back_af(float* dOut,const float*)
    {

        tipl::add(dOut,dOut+output_size,dOut+output_size+output_size);
    }
};


class basic_layer
{

public:
    std::vector<std::shared_ptr<basic_activation_layer> > af;
public:
    status_type status;
    int input_size;
    int output_size;
    float weight_base;
    std::vector<float> weight,bias;
public:

    virtual ~basic_layer() {}
    basic_layer(void):status(testing),weight_base(1){}
    virtual bool init(const tipl::shape<3>& in_dim_,const tipl::shape<3>& out_dim)
    {
        input_size = in_dim_.size();
        output_size = out_dim.size();
        for(auto f : af)
            f->init(out_dim);
        return true;
    }
    virtual void initialize_weight(tipl::uniform_dist<float>& gen)
    {
        for(int i = 0; i < weight.size(); ++i)
            weight[i] = gen()*weight_base;
        if(!bias.empty())
            std::fill(bias.begin(), bias.end(), 0.0f);
    }
    virtual void forward_propagation(const float* x,float* y) = 0;

    template<typename io_stream>
    void read(io_stream& in)
    {
        if(!weight.empty())
        {
            in.read((char*)&weight[0],weight.size()*4);
            in.read((char*)&bias[0],bias.size()*4);
        }
    }
    template<typename io_stream>
    void write(io_stream& out)
    {
        if(!weight.empty())
        {
            out.write((const char*)&weight[0],weight.size()*4);
            out.write((const char*)&bias[0],bias.size()*4);
        }
    }

    virtual void calculate_dwdb(const float*,
                                  const float*,
                                  std::vector<float>&,
                                  std::vector<float>&){}
    virtual void back_propagation(float* dOut,// output_size
                                  float* dX,// input_size
                                  const float*) = 0;

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
public:
    float bn_ratio = 0.0f;
private:
    float weight_scale = 0.0f;
    unsigned int count = 0;
public:
    tipl::shape<3> in_dim;
    fully_connected_layer(void){}
    bool init(const tipl::shape<3>& in_dim_,const tipl::shape<3>& out_dim) override
    {
        in_dim = in_dim_;
        basic_layer::init(in_dim_,out_dim);
        weight.resize(in_dim.size() * out_dim.size());
        bias.resize(out_dim.size());
        weight_base = (float)std::sqrt(6.0f / (float)(input_size+output_size));
        count = 0;
        weight_scale = 0.0f;
        return true;
    }
    void initialize_weight(tipl::uniform_dist<float>& gen) override
    {
        basic_layer::initialize_weight(gen);
    }

    void forward_propagation(const float* x,float* y) override
    {
        for(int i = 0,i_pos = 0;i < output_size;++i,i_pos += input_size)
            y[i] = bias[i] + tipl::vec::dot(&weight[i_pos],&weight[i_pos]+input_size,&x[0]);
        if(!af.empty() && bn_ratio != 0.0f)
        {
            ++count;
            float mean = tipl::mean(y,y+output_size);
            weight_scale += tipl::variance(y,y+output_size,mean);
        }
    }
    //dW += dOut * x
    //db += dOut
    void calculate_dwdb(const float* dOut,
                        const float* x,
                        std::vector<float>& dweight,
                        std::vector<float>& dbias) override
    {
        tipl::add(&dbias[0],&dbias[0]+output_size,dOut);
        for(int i = 0,i_pos = 0; i < output_size; i++,i_pos += input_size)
            if(dOut[i] != float(0))
                tipl::vec::axpy(&dweight[i_pos],&dweight[i_pos]+input_size,dOut[i],x);
    }
    // dX = dOut * W
    void back_propagation(float* dOut,// output_size
                          float* dX,// input_size
                          const float*) override
    {
        tipl::mat::left_vector_product(&weight[0],dOut,dX,tipl::shape<2>(output_size,input_size));
    }

    void update(float rw,const std::vector<float>& dw,
                float rb,const std::vector<float>& db) override
    {
        if(!af.empty() && bn_ratio != 0.0f)
        {
            weight_scale = std::sqrt(weight_scale/count);
            tipl::multiply_constant(weight,1.0f-(weight_scale-1.0f)*bn_ratio);
            count = 0;
            weight_scale = 0.0f;
        }
        basic_layer::update(rw,dw,rb,db);
    }
};


class max_pooling_layer : public basic_layer
{
    std::vector<std::vector<int> > o2i;
    std::vector<int> i2o;
    shape<3> in_dim;
    shape<3> out_dim;

public:
    int pool_size;
public:
    max_pooling_layer(int pool_size_):pool_size(pool_size_){}
    bool init(const tipl::shape<3>& in_dim_,const tipl::shape<3>& out_dim_) override
    {
        basic_layer::init(in_dim_,out_dim_);
        in_dim = in_dim_;
        out_dim = out_dim_;
        if(out_dim != tipl::shape<3>(in_dim.width()/ pool_size, in_dim.height() / pool_size, in_dim.depth()))
            return false;
        init_connection();
        weight_base = (float)std::sqrt(6.0f / (float)(o2i[0].size()+1));
        return true;
    }

    void forward_propagation(const float* x,float* y) override
    {
        for(int i = 0; i < o2i.size(); i++)
        {
            float max_value = std::numeric_limits<float>::lowest();
            for(auto j : o2i[i])
            {
                if(x[j] > max_value)
                    max_value = x[j];
            }
            y[i] = max_value;
        }
    }
    void back_propagation(float* dOut,// output_size
                          float* dX,// input_size
                          const float* x) override
    {
        std::vector<int> max_idx(out_dim.size());

        for(int i = 0; i < o2i.size(); i++)
        {
            float max_value = std::numeric_limits<float>::lowest();
            for(auto j : o2i[i])
            {
                if(x[j] > max_value)
                {
                    max_value = x[j];
                    max_idx[i] = j;
                }
            }
        }
        for(int i = 0; i < i2o.size(); i++)
        {
            int outi = i2o[i];
            dX[i] = (max_idx[outi] == i) ? dOut[outi] : float(0);
        }
    }
    virtual unsigned int computation_cost(void) const override
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
private:
    int strip,strip_2,shift,max_x,max_y;
    std::vector<int> kernel_index;
public:
    shape<3> in_dim,out_dim;
    int kernel_size,kernel_size2;
    // check if any kernel is zero and re-initialize it

public:
    convolutional_layer(int kernel_size_) : kernel_size(kernel_size_),kernel_size2(kernel_size_*kernel_size_)
    {
    }
    bool init(const tipl::shape<3>& in_dim_,const tipl::shape<3>& out_dim_) override
    {
        basic_layer::init(in_dim_, out_dim_);
        in_dim = in_dim_;
        out_dim = out_dim_;
        if(in_dim.width() != out_dim.width() ||
           in_dim.height() != out_dim.height())
            return false;
        strip = (kernel_size-1);
        strip_2 = (kernel_size-1)/2;
        shift = strip_2*out_dim.width()+strip_2;
        max_y = out_dim.height()-strip;
        max_x = out_dim.width()-strip;
        kernel_index.resize(kernel_size2);
        for(int y = 0,index = 0;y < kernel_size;++y)
            for(int x = 0;x < kernel_size;++x,++index)
                kernel_index[index] = x + y*in_dim.width();

        weight.resize(kernel_size2* in_dim.depth() * out_dim.depth());
        bias.resize(out_dim.depth());
        weight_base = (float)std::sqrt(6.0f / (float)(kernel_size2 * in_dim.depth() + kernel_size2 * out_dim.depth()));
        return true;
    }

    void forward_propagation(const float* in,float* out) override
    {

        float* out_ptr = out;
        const float* w_ptr = &weight[0];
        // for each output channel
        for(int outc = 0; outc < out_dim.depth(); ++outc, out_ptr += out_dim.plane_size())
        {
            std::fill(out_ptr,out_ptr+out_dim.plane_size(),bias[outc]);
            const float* in_ptr = in;
            // for each input channel
            for(int inc = 0; inc < in_dim.depth(); inc++, in_ptr += in_dim.plane_size(),w_ptr += kernel_size2)
                {
                    const float* in_line = in_ptr;
                    float* out_line = out_ptr + shift;

                    for(int y = 0; y < max_y; y++,in_line += in_dim.width(),out_line += out_dim.width())
                    {
                        for(int x = 0; x < max_x; x++)
                        {
                            const float * w = w_ptr;
                            const float * p = in_line + x;
                            float sum(0);
                            for(int i = 0; i < kernel_size2; i++)
                                sum += w[i]*p[kernel_index[i]];
                            out_line[x] += sum;
                        }
                    }
                }
        }
    }
    void calculate_dwdb(const float* dOut,
                        const float* x,
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
                        const float * prevo = x + (in_dim.height() * inc + wy) * in_dim.width() + wx;
                        const float * delta = &dOut[outc_pos]+shift;
                        float sum(0);
                        for(int y = 0; y < max_y; y++, prevo += in_dim.width(), delta += out_dim.width())
                            sum += vec::dot(prevo, prevo + max_x, delta);
                        dweight[index] += sum;
                    }
                }
            }
        }
        {
            for(int outc = 0, outc_pos = 0; outc < out_dim.depth(); outc++, outc_pos += out_dim.plane_size())
            {
                const float *delta = &dOut[outc_pos];
                dbias[outc] += std::accumulate(delta, delta + out_dim.plane_size(),0.0f);
            }
        }
    }
    void back_propagation(float* dOut,// output_size
                          float* dX,// input_size
                          const float*) override
    {
        // propagate delta to previous layer
        for(int outc = 0, outc_pos = 0,w_index = 0; outc < out_dim.depth(); ++outc, outc_pos += out_dim.plane_size())
        {
            for(int inc = 0, inc_pos = 0; inc < in_dim.depth(); ++inc, inc_pos += in_dim.plane_size(),w_index += kernel_size2)
            {
                float *dX_pos = dX + inc_pos;
                const float *dOut_pos = dOut + outc_pos + shift;
                for(int y = 0, y_pos = 0; y < max_y; y++, y_pos += in_dim.width(),dOut_pos += out_dim.width())
                {
                    for(int x = 0; x < max_x; x++)
                    {
                        const float * ppw = &weight[w_index];
                        const float v = dOut_pos[x];
                        float *p = dX_pos + y_pos + x;
                        for(int i = 0; i < kernel_size2; i++)
                            p[kernel_index[i]] += v*ppw[i];
                    }
                }

            }
        }
    }
    virtual unsigned int computation_cost(void) const override
    {
        return out_dim.size()*in_dim.depth()*kernel_size*kernel_size;
    }
};


class soft_max_layer : public basic_layer{
public:
    soft_max_layer(void){}
    bool init(const tipl::shape<3>& in_dim,const tipl::shape<3>& out_dim) override
    {
        if(in_dim.size() != out_dim.size())
            return false;
        basic_layer::init(in_dim,in_dim);
        return true;
    }
    void forward_propagation(const float* x,float* y) override
    {
        float m = *std::max_element(x,x+input_size);
        for(int i = 0;i < input_size;++i)
            y[i] = expf(x[i]-m);
        float sum = std::accumulate(y,y+output_size,float(0));
        if(sum != 0)
            tipl::divide_constant(y,y+output_size,sum);
    }
    void back_propagation(float* dOut,// output_size
                          float* dX,// input_size
                          const float* x) override
    {
        std::copy(dOut,dOut+input_size,dX);
        tipl::minus_constant(dX,dX+output_size,tipl::vec::dot(dOut,dOut+input_size,x));
        tipl::multiply(dX,dX+output_size,x);
    }
};

class max_layer : public basic_layer{
public:
    max_layer(void){}
    bool init(const tipl::shape<3>& in_dim,const tipl::shape<3>& out_dim) override
    {
        if(in_dim.size() != out_dim.size())
            return false;
        basic_layer::init(in_dim,in_dim);
        return true;
    }
    void forward_propagation(const float* x,float* y) override
    {
        int pos = std::max_element(x,x+input_size)-x;
        std::fill(y,y+input_size,0.0f);
        y[pos] = 1.0f;
    }
    void back_propagation(float* dOut,// output_size
                          float* dX,// input_size
                          const float*) override
    {
        std::copy(dOut,dOut+input_size,dX);
    }
};


template<typename label_type>
class network_data
{
public:
    tipl::shape<3> input,output;
    std::vector<std::vector<float> > data;
    std::vector<label_type> data_label;
public:
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
            in.read((char*)&data[k][0],4*j);
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

template<typename label_type>
class data_normalization;

template<>
class data_normalization<float>{
public:
    float mean;
    float r;
    void operator()(network_data<float>& data)
    {
        mean = tipl::mean(data.data_label);
        float sd = tipl::standard_deviation(data.data_label.begin(),data.data_label.end(),mean);
        if(sd != 0.0)
            r = 0.2f/sd;
        else
            r = 0.0f;
        tipl::minus_constant(data.data_label,mean);
        tipl::multiply_constant(data.data_label,r);
        tipl::add_constant(data.data_label,0.5f);
    }
    float get_value(float value)
    {
        value -= 0.5f;
        if(r != 0.0f)
            value /= r;
        value += mean;
        return value;
    }
};


template<typename label_type>
class network_data_proxy{
public:
    const network_data<label_type>* source = 0;
    std::vector<unsigned int> pos;
public:
    network_data_proxy(void){}
    network_data_proxy(const network_data<label_type>& rhs)
    {
        (*this) = rhs;
    }
    network_data_proxy& operator=(const network_data<label_type>& rhs)
    {
        source = &rhs;
        pos.resize(rhs.size());
        for(int i = 0;i < pos.size();++i)
            pos[i] = i;
        return *this;
    }
    const float* get_data(unsigned int index) const{return &source->data[pos[index]][0];}
    const label_type& get_label(unsigned int index) const{return source->data_label[pos[index]];}
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

    float calculate_mae(const std::vector<float>& result) const
    {
        float sum_error = 0.0f;
        for(int i = 0;i < size();++i)
            sum_error += std::fabs(result[i]-get_label(i));
        return sum_error/size();
    }
    float calculate_r(const std::vector<float>& result) const
    {
        std::vector<float> d(size());
        for(int i = 0;i < d.size();++i)
            d[i] = get_label(i);
        return tipl::correlation(d.begin(),d.end(),result.begin());
    }
    float calculate_mae(const std::vector<std::vector<float> >& result,size_t index) const
    {
        float sum_error = 0.0f;
        for(int i = 0;i < size();++i)
            sum_error += std::fabs(result[i][index]-get_label(i)[index]);
        return sum_error/size();
    }

    float calculate_r(const std::vector<std::vector<float> >& result,size_t index) const
    {
        std::vector<float> d(size()),d2(size());
        for(int i = 0;i < d.size();++i)
        {
            d[i] = get_label(i)[index];
            d2[i] = result[i][index];
        }
        return tipl::correlation(d.begin(),d.end(),d2.begin());
    }

    int calculate_miss(const std::vector<label_type>& result) const
    {
        int mis_count = 0;
        for(int i = 0;i < size();++i)
            if(result[i] != get_label(i))
                ++mis_count;
        return mis_count;
    }
};

template<typename label_type>
void data_fold_for_cv(const network_data<label_type>& rhs,
                      std::vector<network_data_proxy<label_type> > & training_data,
                      std::vector<network_data_proxy<label_type> > & testing_data,
                      int total_fold = 10,bool stratified = true)
{
    training_data = std::vector<network_data_proxy<label_type> >(total_fold);
    testing_data = std::vector<network_data_proxy<label_type> >(total_fold);
    for(int i = 0;i < total_fold;++i)
    {
        training_data[i].source = &rhs;
        testing_data[i].source = &rhs;
    }
    if(stratified)
    {
        auto order = arg_sort(rhs.data_label,std::less<label_type>());
        for(int i = 0,bin = 0;i < rhs.size();++i,++bin)
        {
            if(bin >= total_fold)
                bin = 0;
            for(int j = 0;j < total_fold;++j)
                if(bin == j)
                    testing_data[j].pos.push_back(order[i]);
                else
                    training_data[j].pos.push_back(order[i]);
        }
    }
    else
    {
        for(int j = 0;j < total_fold;++j)
        for(int i = 0;i < rhs.size();++i)
        {
            if(j == total_fold*i/rhs.size())
                testing_data[j].pos.push_back(i);
            else
                training_data[j].pos.push_back(i);
        }
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
    std::vector<tipl::shape<3> > geo;
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
                        auto p = nw1.begin()+idx[i]*n;
                        std::copy(p,p+n,w1.begin()+pos);
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
    tipl::shape<3> get_input_dim(void) const{return geo.empty() ? tipl::shape<3>(): geo.front();}
    tipl::shape<3> get_output_dim(void) const{return geo.empty() ? tipl::shape<3>(): geo.back();}
    bool add(const tipl::shape<3>& dim)
    {
        if(!layers.empty())
        {
            if(!layers.back()->init(geo.back(),dim))
                return false;
            data_size += dim.size();
        }
        geo.push_back(dim);
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
                return add(tipl::shape<3>(x,y,z));
            }
        }

        if(list.empty())
            return false;
        if(list[0] == "max")
            layers.push_back(std::make_shared<max_layer>());
        else
        if(list[0] == "soft_max")
            layers.push_back(std::make_shared<soft_max_layer>());
        else
        if(list[0] == "full")
            layers.push_back(std::make_shared<fully_connected_layer>());
        else
        if(list[0].find("max_pooling") == 0)
        {
            int param = list[0].back()-'0';
            if(param <= 0 || param > 9)
                return false;
            layers.push_back(std::make_shared<max_pooling_layer>(param));
        }
        else
        if(list[0].find("conv") == 0)
        {
            int param = list[0].back()-'0';
            if(param <= 0 || param > 9)
                return false;
            layers.push_back(std::make_shared<convolutional_layer>(param));
        }
        else
            return false;

        for(int i = 1;i < list.size();++i)
        {
            if(list[i] == "relu")
                layers.back()->af.push_back(std::make_shared<relu_layer>());
            if(list[i] == "lrelu")
                layers.back()->af.push_back(std::make_shared<leaky_relu_layer>());
            if(list[i] == "push")
                layers.back()->af.push_back(std::make_shared<push_layer>());
            if(list[i] == "pull")
                layers.back()->af.push_back(std::make_shared<pull_layer>());
        }
        return true;
    }
    std::shared_ptr<basic_layer> get_layer(int i) const{return layers[i];}
    const tipl::shape<3>& get_geo(int i) const{return geo[i];}
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
                if(dynamic_cast<convolutional_layer*>(layers[i].get()))
                    out << "conv" << dynamic_cast<convolutional_layer*>(layers[i].get())->kernel_size;
                if(dynamic_cast<max_pooling_layer*>(layers[i].get()))
                    out << "max_pooling" << dynamic_cast<max_pooling_layer*>(layers[i].get())->pool_size;
                if(dynamic_cast<fully_connected_layer*>(layers[i].get()))
                    out << "full";
                if(dynamic_cast<soft_max_layer*>(layers[i].get()))
                    out << "soft_max";
                if(dynamic_cast<max_layer*>(layers[i].get()))
                    out << "max";
                for(auto f : layers[i]->af)
                {
                    if(dynamic_cast<relu_layer*>(f.get()))
                        out << ",relu";
                    if(dynamic_cast<leaky_relu_layer*>(f.get()))
                        out << ",lrelu";
                    if(dynamic_cast<push_layer*>(f.get()))
                        out << ",push";
                    if(dynamic_cast<pull_layer*>(f.get()))
                        out << ",pull";
                }

            }
        }
        return out.str();
    }
    template<typename io_type>
    bool save_to_file(const char* file_name)
    {
        io_type out;
        if(!out.open(file_name))
            return false;
        std::string nn_text = get_layer_text();
        unsigned int nn_text_length = nn_text.length();
        out.write((const char*)&nn_text_length,sizeof(nn_text_length));
        out.write((const char*)&*nn_text.begin(),nn_text_length);
        for(auto layer : layers)
            layer->write(out);
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
            layer->read(in);
        initialized = true;
        set_test_mode(true);
        return !!in;
    }




    void forward_propagation(const float* input,float* out_ptr) const
    {
        for(int k = 0;k < layers.size();++k)
        {    
            layers[k]->forward_propagation(input,out_ptr);
            for(int m = 0;m < layers[k]->af.size();++m)
                layers[k]->af[m]->forward_af(out_ptr);
            input = out_ptr;
            out_ptr += layers[k]->output_size;
        }
    }


    void back_propagation(const float* out_ptr2,float* df_ptr) const
    {
        for(int k = (int)layers.size()-1;k >= 0;--k)
        {
            for(int m = layers[k]->af.size();m >= 1;--m)
                layers[k]->af[m-1]->back_af(df_ptr,out_ptr2);
            const float* next_out_ptr = out_ptr2 - layers[k]->input_size;
            float* next_df_ptr = df_ptr - layers[k]->input_size;
            if(k)
                layers[k]->back_propagation(df_ptr,next_df_ptr,next_out_ptr);
            out_ptr2 = next_out_ptr;
            df_ptr = next_df_ptr;
        }

    }

    void forward_propagation(std::vector<float>& in) const
    {
        std::vector<float> out(data_size);
        forward_propagation(&in[0],&out[0]);
        in.resize(output_size);
        std::copy(out.end()-in.size(),out.end(),in.begin());
    }
    float calculate_error(float label,float* df_ptr)
    {
        const float target_value_min = 0.2f;
        const float target_value_max = 0.8f;
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
            error += std::fabs(df_ptr[i]);
        }
        return error;
    }
    float calculate_error(const std::vector<float>& label,float* df_ptr)
    {
        float sum_error = 0.0;
        for(int i = 0;i < output_size;++i)
            sum_error += std::fabs(df_ptr[i] -= label[i]);
        return sum_error;
    }

    void calculate_dwdb(const float* data_entry,
                        const float* dOut,const float* x,
                        std::vector<std::vector<float> >& dweight,
                                                std::vector<std::vector<float> >& dbias)
    {
        layers[0]->calculate_dwdb(dOut,data_entry,dweight[0],dbias[0]);
        for(int k = 1;k < layers.size();++k)
        {
            dOut += layers[k]->input_size;
            if(!layers[k]->weight.empty())
                layers[k]->calculate_dwdb(dOut,x,dweight[k],dbias[k]);
            x += layers[k]->input_size;
        }
    }

    void set_test_mode(bool test)
    {
        for(auto layer : layers)
            layer->status = (test ? testing:training);
    }

    unsigned int predict(std::vector<float>& in)const
    {
        unsigned int label = 0;
        predict(&in[0],label);
        return label;
    }

    template<typename output_type>
    void predict(std::vector<float>& in,output_type& output)const
    {
        predict(&in[0],output);
    }

    template<typename output_type>
    void predict(const float* in,output_type& output)const
    {
        std::vector<float> result(data_size);
        forward_propagation(in,&result[0]);
        if(output_size == 1)
            output = result.back();
        else
            output = std::max_element(result.end()-output_size,result.end())-result.end()+output_size;
    }
    void predict(const float* in,std::vector<float>& output)const
    {
        std::vector<float> result(data_size);
        forward_propagation(in,&result[0]);
        output = std::vector<float>(result.end()-output_size,result.end());
    }

    template<typename label_type,typename result_type>
    void predict(const network_data_proxy<label_type>& data,result_type& test_result) const
    {
        test_result.resize(data.size());
        par_for((int)data.size(), [&](int i)
        {
            predict(data.get_data(i),test_result[i]);
        });
    }
};

inline bool operator << (network& n, const tipl::shape<3>& dim)
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
    float learning_rate = 0.01f;
    float w_decay = 0.0f;
    float rate_decay = 1.0f;
    float momentum = 0.9f;
    float bias_cap = 10.0f;
    float weight_cap = 100.0f;
    int batch_size = 64;
    int epoch= 100;
public:
    std::vector<unsigned int> error_table;
    unsigned int training_count = 0;
    unsigned int training_error_count = 0;
    float training_error_value = 0.0;
public:
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
        dweight.clear();
        dbias.clear();
        training_count = 0;
        training_error_count = 0;
        training_error_value = 0.0f;
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

    void accumulate_error_table(float label,const float* ptr,int output_size)
    {
        // accumulate error table
        if(output_size != 1)
        {
            size_t predicted_label = std::max_element(ptr,ptr+output_size)-ptr;
            if(label != predicted_label)
                ++training_error_count;
            if(!error_table.empty())
            {
                size_t pos = output_size*label + predicted_label;
                if(pos < error_table.size())
                    ++error_table[pos];
            }
        }
    }
    void accumulate_error_table(const std::vector<float>&,const float*,int)
    {
    }

    template <class network_data_type>
    void train_batch(network& nn,network_data_type& data,bool &terminated)
    {
        nn.set_test_mode(false);
        int output_pos = nn.data_size - nn.output_size;
        training_count = 0;
        training_error_count = 0;
        training_error_value = 0.0f;
        if(!error_table.empty())
            std::fill(error_table.begin(),error_table.end(),0);
        for(int i = 0;i < data.size() && !terminated;i += batch_size)
        {
            // train a batch
            int cur_size = std::min<int>(batch_size,data.size()-i);
            par_for(cur_size,[&](int m, int thread_id)
            {
                ++training_count;
                int data_index = i+m;
                if(terminated)
                    return;
                nn.forward_propagation(data.get_data(data_index),in_out_ptr[thread_id]);
                const float* out_ptr2 = in_out_ptr[thread_id] + output_pos;
                float* df_ptr = back_df_ptr[thread_id] + output_pos;
                std::copy(out_ptr2,out_ptr2+nn.output_size,df_ptr);
                accumulate_error_table(data.get_label(data_index),out_ptr2,nn.output_size);
                training_error_value += nn.calculate_error(data.get_label(data_index),df_ptr);
                nn.back_propagation(out_ptr2,df_ptr);
                nn.calculate_dwdb(data.get_data(data_index),back_df_ptr[thread_id],
                                                            in_out_ptr[thread_id],
                                                            dweight[thread_id],dbias[thread_id]);
            });
            // update_weights
            par_for(nn.layers.size(),[this,&nn,cur_size](int j)
            {
                if(nn.layers[j]->weight.empty())
                    return;
                std::vector<float> dw(nn.layers[j]->weight.size());
                std::vector<float> db(nn.layers[j]->bias.size());
                par_for(dweight.size(),[this,&dw,&db,j](int k)
                {
                    tipl::add(dw,dweight[k][j]);
                    tipl::add(db,dbias[k][j]);
                    if(momentum > 0.0f)
                    {
                        tipl::multiply_constant(dweight[k][j],momentum);
                        tipl::multiply_constant(dbias[k][j],momentum);
                    }
                    else
                    {
                        std::fill(dweight[k][j].begin(),dweight[k][j].end(),0.0f);
                        std::fill(dbias[k][j].begin(),dbias[k][j].end(),0.0f);
                    }
                });

                nn.layers[j]->update(-learning_rate*rate_decay/float(cur_size),dw,-learning_rate*rate_decay/float(cur_size),db);

                tipl::upper_lower_threshold(nn.layers[j]->bias,-bias_cap,bias_cap);
                tipl::upper_lower_threshold(nn.layers[j]->weight,-weight_cap,weight_cap);

                if(dynamic_cast<fully_connected_layer*>(nn.layers[j].get()) == nullptr && w_decay != 0.0f)
                    tipl::multiply_constant(nn.layers[j]->weight,1.0f-w_decay);
            });
        }
    }

    template <typename label_type>
    void seed_search(network& nn,network_data_proxy<label_type>& data,bool &terminated,int search_count = 0)
    {
        float best_error = 0.0f;
        for(int seed = 0;seed < search_count && !terminated;++seed)
        {
            network tmp;
            tmp.add(nn.get_layer_text());
            tmp.init_weights(seed);
            initialize_training(tmp);
            for(int i = 0;i < 5 && !terminated;++i)
            {
                data.shuffle(rd_gen);
                train_batch(tmp,data,terminated);
            }
            if(get_training_error_value() < best_error || best_error == 0.0f)
            {
                best_error = get_training_error_value();
                nn = std::move(tmp);
            }
        }
    }
    template <typename label_type,typename iter_type>
    void train(network& nn,
               network_data_proxy<label_type>& data,
               bool &terminated,iter_type iter_fun,int seed = 0)
    {
        if(!nn.initialized)
            nn.init_weights(seed);
        initialize_training(nn);
        rate_decay = 1.0f;
        std::deque<float> train_errors;
        bool has_improved = true;
        for(int iter = 0; iter < epoch && !terminated;iter++ ,iter_fun())
        {
            data.shuffle(rd_gen);
            train_batch(nn,data,terminated);
            train_errors.push_back(get_training_error_value());
            if(train_errors.size() > 32)
            {
                double a,b,r2;
                std::vector<float> iter(train_errors.size());
                std::iota(iter.begin(), iter.end(), 0);
                tipl::linear_regression(iter.begin(),iter.end(),train_errors.begin(),a,b,r2);
                if(a > 0.0f)
                {
                    rate_decay *= 0.5f;
                    train_errors.clear();
                    if(!has_improved)
                        break;
                    has_improved = false;
                }
                else
                {
                    has_improved = true;
                    train_errors.pop_front();
                }
            }
        }
    }

};



template<typename layer_type>
void to_image(std::shared_ptr<layer_type> l,color_image& Is,int max_width)
{
    Is.clear();
    image<2,float> I;
    if(dynamic_cast<fully_connected_layer*>(l.get()))
    {
        fully_connected_layer* layer = dynamic_cast<fully_connected_layer*>(l.get());
        auto in_dim = layer->in_dim;
        if(in_dim.width() > 5000 || in_dim.height() > 1000)
        {
            I.clear();
            return;
        }
        std::vector<float> w(layer->weight),b(layer->bias);
        tipl::normalize_abs(w);
        tipl::normalize_abs(b);
        if(in_dim[1] == 1 || in_dim[0] == 1)
        {
            int width = in_dim.size()+3;
            I.resize(shape<2>(width,b.size()));
            for(int row = 0,row_pos = 0,w_pos = 0;row < b.size();++row,row_pos += width,w_pos += in_dim.size())
            {
                std::copy(&w[w_pos],&w[w_pos]+in_dim.size(),&I[row_pos]);
                I[row_pos+width-2] = b[row];
            }
        }
        else
        {
            int n = int(w.size()/in_dim.plane_size());
            int col = in_dim[2];
            int pad = 3;
            while(col < std::sqrt(n))
            {
                col += in_dim[2];
                pad += 3;
            }
            int row = n/col;
            I.resize(shape<2>(col* (in_dim.width()+1)+pad,row * (in_dim.height() +1) + 1));
            int b_pos = 0;
            for(int y = 0,index = 0;y < row;++y)
                for(int x = 0;x < col;++x,++index)
                {
                    tipl::draw(tipl::make_image(&w[0] + index*in_dim.plane_size(),
                               tipl::shape<2>(in_dim[0],in_dim[1])),
                                I,tipl::shape<2>(x*(in_dim.width()+1)+x/in_dim[2],y*(in_dim.height()+1)+1));
                    if((x+1)%in_dim[2] == 0)
                        I.at(x*(in_dim.width()+1)+x/in_dim[2]+in_dim.width(),y*(in_dim.height()+1)+1 + in_dim.height()/2) = b[b_pos++];
                }
        }
    }
    if(dynamic_cast<convolutional_layer*>(l.get()))
    {
        convolutional_layer* layer = dynamic_cast<convolutional_layer*>(l.get());
        std::vector<float> w(layer->weight),b(layer->bias);
        tipl::normalize_abs(w);
        tipl::normalize_abs(b);
        auto kernel_size = layer->kernel_size;
        auto kernel_size2 = layer->kernel_size2;
        I.resize(shape<2>(layer->out_dim.depth()* (kernel_size+1)+1,
                             layer->in_dim.depth() * (kernel_size+1) + 3));
        for(int x = 0,index = 0;x < layer->out_dim.depth();++x)
            for(int y = 0;y < layer->in_dim.depth();++y,++index)
            {
                tipl::draw(tipl::make_image(&w[0] + index*kernel_size2,tipl::shape<2>(kernel_size,kernel_size)),
                            I,tipl::shape<2>(x*(kernel_size+1),y*(kernel_size+1)+1));
            }

        for(int i = 0,pos = I.size()-I.width()*2+kernel_size/2;i < b.size();++i,pos += kernel_size+1)
            I[pos] = b[i];
    }
    if(I.empty())
        return;
    while(max_width && I.width() > max_width)
        tipl::downsampling(I);
    for(int j = 0;j < 2 && I.width() < max_width*0.5f;++j)
        tipl::upsampling_nearest(I);
    Is.resize(I.shape());
    std::fill(Is.begin(),Is.end(),tipl::rgb(0xFFFFFFFF));
    for(int j = 0;j < I.size();++j)
    {
        if(I[j] == 0)
            continue;
        unsigned char s(std::min<int>(255,((int)std::fabs(I[j]*1024.0f))));
        if(I[j] < 0) // red
            Is[j] = tipl::rgb(s,uint8_t(0),uint8_t(0));
        if(I[j] > 0) // blue
            Is[j] = tipl::rgb(uint8_t(0),uint8_t(0),s);
    }
}


template<typename label_type>
void to_image(network& nn,color_image& I,std::vector<float> in,label_type label,int layer_height = 20,int max_width = 600)
{
    std::vector<tipl::color_image> Is;
    Is.resize(nn.layers.size());
    tipl::par_for(nn.layers.size(),[&](int i)
    {
        to_image(nn.layers[i],Is[i],max_width);
    });
    int total_height = 0;
    for(int i = 0;i < Is.size();++i)
        total_height += std::max<int>(Is[i].height(),layer_height);

    int data_size = nn.data_size;
    int output_size = nn.output_size;
    int end = data_size-output_size;
    auto& geo = nn.geo;
    std::vector<tipl::color_image> values(geo.size());
    std::vector<float> out(data_size),back(data_size);
    float* out_buf = &out[0];
    float* back_buf = &back[0];
    float* in_buf = &in[0];
    nn.forward_propagation(in_buf,out_buf);
    std::copy(out_buf+end,out_buf+end+output_size,back_buf+end);
    nn.calculate_error(label,back_buf+end);
    nn.back_propagation(out_buf+end,back_buf+end);

    for(int i = 0;i < geo.size();++i)
    {
        if(geo[i].width() < 5000 && geo[i].height() < 1000)
        {
            int col = std::max<int>(1,(max_width-1)/(geo[i].width()+1));
            int row = std::max<int>(1,geo[i][2]/col+1);
            if(i == 0)
                values[i].resize(tipl::shape<2>(col*(geo[i].width()+1)+1,row*(geo[i].height()+1)+1));
            else
                values[i].resize(tipl::shape<2>(col*(geo[i].width()+1)+1,int(2.0f*row*(geo[i].height()+1)+2)));
            std::fill(values[i].begin(),values[i].end(),tipl::rgb(255,255,255));
            int draw_width = 0;
            for(int y = 0,j = 0;y < row;++y)
                for(int x = 0;j < geo[i][2] && x < col;++x,++j)
                {
                    auto v1 = tipl::make_image((i == 0 ? in_buf : out_buf)+geo[i].plane_size()*j,tipl::shape<2>(geo[i][0],geo[i][1]));
                    auto v2 = tipl::make_image((i == 0 ? in_buf : back_buf)+geo[i].plane_size()*j,tipl::shape<2>(geo[i][0],geo[i][1]));
                    tipl::normalize_abs(v1);
                    tipl::normalize_abs(v2);
                    tipl::color_image Iv1(v1.shape()),Iv2(v2.shape());
                    for(int j = 0;j < Iv1.size();++j)
                    {
                        unsigned char s1(std::min<int>(255,int(255.0f*std::fabs(v1[j]))));
                        if(v1[j] < 0) // red
                            Iv1[j] = tipl::rgb(s1,uint8_t(0),uint8_t(0));
                        if(v1[j] >= 0) // blue
                            Iv1[j] = tipl::rgb(uint8_t(0),uint8_t(0),s1);
                        unsigned char s2(std::min<int>(255,int(255.0f*std::fabs(v2[j]))));
                        if(v2[j] < 0) // red
                            Iv2[j] = tipl::rgb(s2,uint8_t(0),uint8_t(0));
                        if(v2[j] >= 0) // blue
                            Iv2[j] = tipl::rgb(uint8_t(0),uint8_t(0),s2);
                    }
                    tipl::draw(Iv1,values[i],tipl::shape<2>(x*(geo[i].width()+1)+1,y*(geo[i].height()+1)+1));
                    if(i)
                        tipl::draw(Iv2,values[i],tipl::shape<2>(x*(geo[i].width()+1)+1,row*(geo[i].height()+1)+1+y*(geo[i].height()+1)+1));
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
        }
        if(i)
        {
            back_buf += geo[i].size();
            out_buf += geo[i].size();
        }
    }


    I.resize(tipl::shape<2>(max_width,total_height));
    std::fill(I.begin(),I.end(),tipl::rgb(255,255,255));
    int cur_height = 0;
    for(int i = 0;i < geo.size();++i)
    {
        // input image
        tipl::draw(values[i],I,tipl::shape<2>(0,cur_height));
        cur_height += values[i].height();

        // network wieghts
        if(i < Is.size() && !Is.empty())
        {
            tipl::rgb b;
            if(Is[i].empty())
                b.from_hsl(0.5,0.5,0.85);
            else
                b = Is[i][0];
            tipl::fill_rect(I,tipl::shape<2>(0,cur_height),
                               tipl::shape<2>(max_width,cur_height+std::max<int>(Is[i].height(),layer_height)),b);
            tipl::draw(Is[i],I,tipl::shape<2>(1,cur_height +
                                                   (Is[i].height() < layer_height ? (layer_height- Is[i].height())/2: 0)));
            cur_height += std::max<int>(Is[i].height(),layer_height);
        }
    }
}

struct iterate_cnn_data{
    tipl::shape<3> dim;
    std::string str;
    int num_conv = 0;
    int depth = 0;
    enum {root = 1, conv = 2, max_pooling = 4, fully = 8} previous_layer;

};

template<typename str_list_type>
void iterate_cnn(
             const tipl::shape<3>& in_dim,
             const tipl::shape<3>& out_dim,
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
                    new_layer.str += std::string("max_pooling2|");
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
                    //new_layer.dim[0] -= kernel-1;
                    //new_layer.dim[1] -= kernel-1;
                    new_layer.dim[2] = width/2;
                    new_layer.str += std::string("conv")+std::to_string(kernel)+",relu|";
                    new_layer.previous_layer = iterate_cnn_data::conv;

                    int cost = new_layer.dim.size()*cur_layer.dim.depth()*kernel*kernel+layer_cost;
                    if(cur_cost+cost < max_cost)
                        candidates.insert(std::make_pair(cur_cost+cost,new_layer));
                }

                if(cur_layer.previous_layer != iterate_cnn_data::root)
                {
                    new_layer = cur_layer;
                    new_layer.dim[0] = 1;
                    new_layer.dim[1] = 1;
                    new_layer.dim[2] = width;
                    new_layer.str += std::string("full,relu|");
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
                std::string s = cur_layer.str + std::string("full|")+sout.str();
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
