#ifndef CNN3D_HPP
#define CNN3D_HPP
#include <vector>
#include <memory>
#include "../utility/pixel_index.hpp"
#include "../utility/basic_image.hpp"
#include "../numerical/statistics.hpp"
#include "../mt.hpp"



namespace tipl
{
namespace ml3d
{



class layer{
public:
    int in_channels_ = 1;
    int out_channels_ = 1;
    tipl::shape<3> dim;
public:
    layer(int channels_v):in_channels_(channels_v),out_channels_(channels_v){}
    layer(int in_channels_v,int out_channels_v):in_channels_(in_channels_v),out_channels_(out_channels_v){}
    virtual ~layer(void){}
    virtual std::vector<std::pair<float*,size_t> > parameters(void){return std::vector<std::pair<float*,size_t> >();}
    virtual void init_image(tipl::shape<3>& dim_){dim = dim_;}
public:
    virtual float* forward(float* in_ptr) = 0;
public:
    virtual void print(std::ostream& out)  const  = 0;
    virtual size_t in_size(void)  const {return dim.size()*in_channels_;}
    virtual size_t out_size(void)  const {return dim.size()*out_channels_;}
};


class relu : public layer {
private:
    size_t output_size_ = 0;
public:
    relu(int channels_v):layer(channels_v){}
    virtual void init_image(tipl::shape<3>& dim_)
    {
        dim = dim_;
        output_size_ = dim.size()*size_t(out_channels_);
    }
    virtual float* forward(float* in_ptr)
    {
        for(size_t i = 0;i < output_size_;++i)
            if(in_ptr[i] < 0.0f)
                in_ptr[i] = 0.0f;
        return in_ptr;
    }
    virtual void print(std::ostream& out) const {out << "relu " << out_channels_ << std::endl;}
    virtual size_t in_size(void) const {return output_size_;}
    virtual size_t out_size(void) const {return output_size_;}
};

class leakyrelu : public layer {
private:
    size_t output_size_ = 0;
    float slope = 1e-2f;
public:
    leakyrelu(int channels_v,float slope_ = 1e-2f):layer(channels_v),slope(slope_){}
    virtual void init_image(tipl::shape<3>& dim_)
    {
        dim = dim_;
        output_size_ = dim.size()*size_t(out_channels_);
    }
    virtual float* forward(float* in_ptr)
    {
        for(size_t i = 0;i < output_size_;++i)
            if(in_ptr[i] < 0.0f)
                 in_ptr[i] *= slope;
        return in_ptr;
    }
    virtual void print(std::ostream& out) const {out << "leaky relu " << out_channels_ << std::endl;}
    virtual size_t in_size(void) const {return output_size_;}
    virtual size_t out_size(void) const {return output_size_;}
};

class conv_3d : public layer {
private:
    int kernel_size_,kernel_size3,range;
    std::vector<int> kernel_shift;
public:
    std::vector<float> weight,bias,out;

    conv_3d(int in_channels_v,int out_channels_v,int kernel_size_v = 3):
        layer(in_channels_v,out_channels_v),kernel_size_(kernel_size_v)
    {
        kernel_size3 = kernel_size_*kernel_size_*kernel_size_;
        range = (kernel_size_-1)/2;
        weight.resize(kernel_size3* in_channels_*out_channels_);
        bias.resize(out_channels_);
    }
    virtual std::vector<std::pair<float*,size_t> > parameters(void)
    {
        std::vector<std::pair<float*,size_t> > params;
        params.push_back(std::make_pair(&weight[0],weight.size()));
        params.push_back(std::make_pair(&bias[0],bias.size()));
        return params;
    }
public:
    virtual void init_image(tipl::shape<3>& dim_)
    {
        dim = dim_;
        kernel_shift.resize(kernel_size3);
        for(int kz = -range,index = 0;kz <= range;++kz)
            for(int ky = -range;ky <= range;++ky)
                for(int kx = -range;kx <= range;++kx,++index)
                    kernel_shift[index] = (kz*int(dim.height())+ky)*int(dim.width())+kx;
        out.resize(dim.size()*out_channels_);
    }
    virtual float* forward(float* in)
    {
        // for each output channel
        size_t image_size = dim.size();
        size_t plane_size = dim.plane_size();
        size_t image_width = dim.width();

        size_t thread_count = tipl::max_thread_count;
        size_t thread_plane_size = dim.plane_size()*thread_count;
        // z dimensioni stratified multi-thread
        tipl::par_for(thread_count,[=](size_t thread)
        {
            size_t thread_base = plane_size*thread;
            auto out_ptr = out.data() + thread_base;
            auto w_ptr = weight.data();
            for(size_t outc = 0;outc < out_channels_;++outc,out_ptr += image_size)
            {
                // fill output with bias values
                {
                    auto out_plane = out_ptr;
                    for(int z = thread; z < dim[2]; z+=thread_count,out_plane += thread_plane_size)
                        std::fill(out_plane,out_plane+plane_size,bias[outc]);
                }
                auto in_ptr = in + thread_base;
                for(int inc = 0; inc < in_channels_; inc++, in_ptr += image_size,w_ptr += kernel_size3)
                {
                    for(int kz = -range,index = 0;kz <= range;++kz)
                        for(int ky = -range;ky <= range;++ky)
                            for(int kx = -range;kx <= range;++kx,++index)
                            {
                                float w = w_ptr[index];
                                auto in_plane = in_ptr + kernel_shift[index];
                                auto out_plane = out_ptr;
                                auto max_z = kz > 0 ? dim[2]-kz : dim[2];
                                auto max_y = ky > 0 ? dim[1]-ky : dim[1];
                                auto max_x = kx > 0 ? dim[0]-kx : dim[0];
                                for(int z = thread; z < max_z; z+=thread_count,in_plane += thread_plane_size,out_plane += thread_plane_size)
                                {
                                    if(z < -kz)
                                        continue;
                                    auto in_line = in_plane;
                                    auto out_line = out_plane;
                                    for(int y = 0; y < max_y; y++,in_line += image_width,out_line += image_width)
                                    {
                                        if(y < -ky)
                                            continue;
                                        int pad = std::max<int>(0,-kx);
                                        tipl::vec::axpy(out_line+pad,out_line+max_x,w,in_line+pad);
                                    }
                                }
                            }
                }
            }
        },thread_count);
        return out.data();
    }
    virtual void print(std::ostream& out) const {out << "conv3d " << in_channels_ << " " << out_channels_ << std::endl;}
};


class batch_norm_3d : public layer
{
public:
    std::vector<float> weight,bias;
public:
    batch_norm_3d(int channels_v):layer(channels_v)
    {
        weight.resize(out_channels_);
        bias.resize(out_channels_);
    }
    virtual std::vector<std::pair<float*,size_t> > parameters(void)
    {
        std::vector<std::pair<float*,size_t> > params;
        params.push_back(std::make_pair(&weight[0],weight.size()));
        params.push_back(std::make_pair(&bias[0],bias.size()));
        return params;
    }
    virtual float* forward(float* in)
    {
        tipl::par_for(out_channels_,[&](size_t outc)
        {
            auto in_ptr = in + dim.size()*outc;
            auto in_ptr_end = in_ptr + dim.size();
            auto m = tipl::mean(in_ptr,in_ptr_end);
            auto sd = tipl::standard_deviation(in_ptr,in_ptr_end);
            auto v = (sd == 0.0f ? 0.0f : weight[outc]/sd);
            auto b = bias[outc];
            for(size_t i = 0;i < dim.size();++i)
                in_ptr[i] = (in_ptr[i]-m)*v+b;
        },out_channels_);
        return in;
    }
    virtual void print(std::ostream& out) const {out << "batch_norm_3d " << out_channels_ << std::endl;}
};

class max_pool_3d : public layer
{
public:
    std::vector<std::vector<int> > o2i;
    tipl::shape<3> out_dim;
    std::vector<float> out;
public:
    int pool_size = 2;
public:
    max_pool_3d(int channels_v):layer(channels_v){}
    virtual void init_image(tipl::shape<3>& dim_)
    {
        dim = dim_;
        out_dim = tipl::shape<3>(dim[0]/pool_size,dim[1]/pool_size,dim[2]/pool_size);
        o2i.resize(out_dim.size());
        for(int z = 0,tz = 0,out_index = 0; z < out_dim.depth(); ++z,tz += pool_size)
            for(int y = 0,ty = 0; y < out_dim.height(); ++y,ty += pool_size)
                for(int x = 0,tx = 0; x < out_dim.width(); ++x,tx += pool_size,++out_index)
                {
                    for(int dz = 0,from_z = tz; dz < pool_size; dz++,from_z++)
                        for(int dy = 0,from_y = ty; dy < pool_size; dy++,from_y++)
                            for(int dx = 0,from_x = tx; dx < pool_size; dx++,from_x++)
                                if(from_x < dim.width() && from_y < dim.height() && from_z < dim.depth())
                                {
                                    int in_index = (from_z*dim.height() + from_y) * dim.width() + from_x;
                                    o2i[out_index].push_back(in_index);
                                }
                }
        dim_ = out_dim;
        out.resize(out_dim.size()*out_channels_);
    }
    virtual float* forward(float* in)
    {
        tipl::par_for(out_channels_,[&](size_t c)
        {
            auto in_ptr = in+c*dim.size();
            auto out_ptr = out.data()+c*out_dim.size();
            for(int i = 0; i < out_dim.size(); i++)
            {
                float max_value = std::numeric_limits<float>::lowest();
                for(auto j : o2i[i])
                {
                    if(in_ptr[j] > max_value)
                        max_value = in_ptr[j];
                }
                out_ptr[i] = max_value;
            }
        },out_channels_);
        return out.data();
    }
    virtual void print(std::ostream& out) const {out << "max_pool_3d " << out_channels_ << std::endl;    }
    virtual size_t out_size(void) const {return out_dim.size()*out_channels_;}
};

class upsample_3d : public layer
{
public:
    std::vector<int> o2i;
    tipl::shape<3> out_dim;
    std::vector<float> out;
public:
    int pool_size = 2;
public:
    upsample_3d(int channels_v):layer(channels_v)
    {
    }
    virtual void init_image(tipl::shape<3>& dim_)
    {
        dim = dim_;
        out_dim = tipl::shape<3>(dim[0]*pool_size,dim[1]*pool_size,dim[2]*pool_size);
        o2i.resize(out_dim.size());
        for(int z = 0,out_index = 0; z < out_dim.depth(); ++z)
            for(int y = 0; y < out_dim.height(); ++y)
                for(int x = 0; x < out_dim.width(); ++x,++out_index)
                {
                    auto from_x = x/pool_size;
                    auto from_y = y/pool_size;
                    auto from_z = z/pool_size;
                    o2i[out_index] = (from_z*dim.height() + from_y) * dim.width() + from_x;
                }
        dim_ = out_dim;
        out.resize(out_dim.size()*out_channels_);
    }
    virtual float* forward(float* in)
    {
        tipl::par_for(out_channels_,[&](size_t c)
        {
            auto in_ptr = in + c*dim.size();
            auto out_ptr = out.data()+c*out_dim.size();
            for(int i = 0; i < out_dim.size(); i++)
                out_ptr[i] = in_ptr[o2i[i]];
        },out_channels_);
        return out.data();
    }
    virtual void print(std::ostream& out) const {out << "upsample_3d " << out_channels_ << std::endl;}
    virtual size_t out_size(void)  const {return out_dim.size()*out_channels_;}
};

class network : public layer
{
public:
    network(void):layer(1,1){}
    std::vector<std::shared_ptr<layer> > layers;
    network& operator<<(layer* l)
    {
        layers.push_back(std::shared_ptr<layer>(l));
        in_channels_ = layers.front()->in_channels_;
        out_channels_ = layers.back()->out_channels_;
        return *this;
    }
    network& operator<<(std::shared_ptr<layer> l)
    {
        layers.push_back(l);
        in_channels_ = layers.front()->in_channels_;
        out_channels_ = layers.back()->out_channels_;
        return *this;
    }
    virtual std::vector<std::pair<float*,size_t> > parameters(void)
    {
        std::vector<std::pair<float*,size_t> > param;
        for(auto& each_layer : layers)
        {
            auto new_params = each_layer->parameters();
            param.insert(param.end(),new_params.begin(),new_params.end());
        }
        return param;
    }
    virtual void init_image(tipl::shape<3>& dim)
    {
        for(auto& each_layer : layers)
            each_layer->init_image(dim);
    }
    virtual float* forward(float* in)
    {
        for(auto& each_layer : layers)
            in = each_layer->forward(in);
        return in;
    }
    virtual void print(std::ostream& out) const
    {
        for(auto& each_layer : layers)
            each_layer->print(out);
    }
    virtual size_t in_size(void) const {return layers.front()->in_size();}
    virtual size_t out_size(void) const {return layers.back()->out_size();}
};


}//ml
}//image

#endif//CNN3D_HPP
