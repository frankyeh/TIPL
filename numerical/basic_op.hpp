//---------------------------------------------------------------------------
#ifndef BASIC_OP_HPP
#define BASIC_OP_HPP

#include "../def.hpp"
#include "../numerical/numerical.hpp"
#include "../utility/pixel_index.hpp"
#include "../utility/basic_image.hpp"
#include "../mt.hpp"
#include <algorithm>
#include <numeric>

namespace tipl
{

template<typename iterator_type1,typename iterator_type2,typename fun_type>
inline void for_each(iterator_type1 iter1,iterator_type1 end,iterator_type2 iter2,fun_type fun)
{
    for(; iter1 != end; ++iter1,++iter2)
        fun(*iter1,*iter2);
}

template <typename container_type,typename compare_type>
std::vector<unsigned int> arg_sort(const container_type& data,compare_type comp)
{
    size_t sz = data.size();
    std::vector<unsigned int> idx(sz);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&data,comp](size_t i1, size_t i2) {
        return comp(data[i1], data[i2]);
    });
    return idx;
}

template <typename container_type, typename compare_type>
auto rank(const container_type& data, compare_type comp)
{
    size_t sz = data.size();
    std::vector<unsigned int> idx(sz);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a,size_t b){ return less(data[a], data[b]); });
    std::vector<unsigned int> r(sz);
    for (unsigned int i = 0; i < sz; ++i)
        r[idx[i]] = i;
    return r;
}

template <typename container_type, typename compare_type>
auto rank_avg_tie(const container_type& data, compare_type comp)
{
    size_t sz = data.size();
    std::vector<size_t> idx(sz);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a,size_t b){ return comp(data[a], data[b]); });
    std::vector<float> r(sz);
    for(size_t i = 0; i < sz;)
    {
        size_t j = i + 1;
        while(j < sz && !comp(data[idx[i]],data[idx[j]]) && !comp(data[idx[j]],data[idx[i]]))
            ++j;
        float avg = 0.5f*(i + (j-1));
        for(size_t k = i; k < j; ++k)
            r[idx[k]] = avg;
        i = j;
    }
    return r;
}

template <typename compare_type>
std::vector<unsigned int> arg_sort(size_t size,compare_type comp)
{
    std::vector<unsigned int> idx(size);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),comp);
    return idx;
}

template <typename container_type>
size_t arg_max(const container_type& data)
{
    size_t sz = data.size();
    if(sz == 0) return 0;
    typename container_type::value_type m = data[0];
    size_t m_pos = 0;
    for(size_t i = 1; i < sz; ++i)
        if(data[i] > m) { m = data[i]; m_pos = i; }
    return m_pos;
}

template <typename container_type>
size_t arg_min(const container_type& data)
{
    size_t sz = data.size();
    if(sz == 0) return 0;
    typename container_type::value_type m = data[0];
    size_t m_pos = 0;
    for(size_t i = 1; i < sz; ++i)
        if(data[i] < m) { m = data[i]; m_pos = i; }
    return m_pos;
}

template<typename T>
inline auto get_sparse_index(const T& mask)
{
    std::vector<size_t> si2vi;
    size_t sz = mask.size();
    for(size_t index = 0; index < sz; ++index)
        if(mask[index])
            si2vi.push_back(index);
    return si2vi;
}

template<typename T>
void to_sparse_inplace(T& data,const std::vector<size_t>& si2vi)
{
    int64_t sz = si2vi.size();
    for(int64_t index = sz - 1; index >= 0; --index)
    {
        data[si2vi[index]] = data[index];
        if(si2vi[index] != static_cast<size_t>(index))
            data[index] = 0;
    }
}

template<typename T,typename U>
void to_sparse(const T& from,U& to,const std::vector<size_t>& si2vi)
{
    size_t sz = si2vi.size();
    for(size_t index = 0; index < sz; ++index)
        to[si2vi[index]] = from[index];
}

template<typename ImageType>
bool is_label_image(const ImageType& I)
{
    size_t sz = I.size();
    if(sz == 0) return true;

    if constexpr (std::is_floating_point_v<typename ImageType::value_type>)
        if (std::any_of(I.begin(), I.end(), [](auto v) { return std::floor(v) != v; }))
            return false;

    if (*std::max_element(I.begin(), I.end()) < 12)
        return true;

    int w = I.width();
    size_t ps = I.plane_size();
    int shift_base = 1;
    if constexpr(ImageType::dimension == 2) shift_base = w;
    if constexpr(ImageType::dimension == 3) shift_base = ps;

    size_t max_size = sz - shift_base;
    size_t thread_count = std::thread::hardware_concurrency();
    std::vector<size_t> same(thread_count), diff(thread_count);

    par_for(thread_count,[&](int thread)
    {
        for(size_t i = shift_base + thread; i < max_size; i += thread_count)
        {
            auto v = I[i];
            if(v == 0) continue;
            if(v == I[i+1]) ++same[thread]; else ++diff[thread];

            if constexpr(ImageType::dimension >= 2) {
                if(v == I[i+w]) ++same[thread]; else ++diff[thread];
            }
            if constexpr(ImageType::dimension >= 3) {
                if(v == I[i+ps]) ++same[thread]; else ++diff[thread];
            }
        }
    });
    return std::accumulate(same.begin(), same.end(), size_t(0)) > std::accumulate(diff.begin(), diff.end(), size_t(0));
}

template<typename T>
void expand_label_to_dimension(T& label,size_t label_count,bool skip_background = true)
{
    size_t sz = label.size();
    T out(label.shape().multiply(tipl::shape<3>::z, label_count));

    std::vector<size_t> offset(label_count);
    for(size_t i = 0; i < label_count; ++i) offset[i] = i * sz;

    auto it = label.begin();
    auto end = label.end();
    auto out_it = out.begin();

    for(; it != end; ++it, ++out_it)
    {
        int v = *it - skip_background;
        if(v >= 0 && v < static_cast<int>(label_count))
            *(out_it + offset[v]) = 1;
    }
    label.swap(out);
}

template<typename T>
void expand_label_to_images(const T& label,std::vector<T>& images,size_t max_v)
{
    max_v = std::min<size_t>(images.size(), max_v);
    auto shape = label.shape();
    for(size_t i = 0; i < max_v; ++i) images[i] = T(shape);

    size_t sz = label.size();
    for(size_t pos = 0; pos < sz; ++pos)
    {
        auto v = label[pos];
        if(v > 0 && v <= max_v)
            images[v-1][pos] = 255;
    }
}

template<typename ImageType,typename LabelImageType,typename fun_type>
void binary(const ImageType& I,LabelImageType& out,fun_type fun)
{
    out.resize(I.shape());
    auto iter = I.begin();
    auto end = I.end();
    auto out_iter = out.begin();
    for(; iter != end; ++iter, ++out_iter)
        *out_iter = fun(*iter) ? 1 : 0;
}

template<typename ImageType,typename fun_type>
ImageType& binary(ImageType& I,fun_type fun)
{
    auto iter = I.begin();
    auto end = I.end();
    for(; iter != end; ++iter)
        *iter = fun(*iter) ? 1 : 0;
    return I;
}


template<typename ImageType,typename LabelImageType>
LabelImageType& threshold(const ImageType& I,LabelImageType& out,typename ImageType::value_type threshold_value,
               typename LabelImageType::value_type foreground = 1,typename LabelImageType::value_type background = 0)
{
    out.resize(I.shape());
    auto iter = I.begin();
    auto end = I.end();
    auto out_iter = out.begin();
    for(; iter != end; ++iter, ++out_iter)
        *out_iter = (*iter > threshold_value) ? foreground : background;
    return out;
}

template<typename LabelImageType,typename ImageType>
inline auto threshold(const ImageType& I,typename ImageType::value_type threshold_value,
               typename LabelImageType::value_type foreground = 1,typename LabelImageType::value_type background = 0)
{
    LabelImageType out;
    threshold(I,out,threshold_value,foreground,background);
    return out;
}
template <int dim,typename vtype,template <typename...> typename stype>
inline auto operator>(const image<dim,vtype,stype>& I,vtype prob_threshold)
{
    tipl::image<dim,unsigned char,stype> mask;
    tipl::threshold(I,mask,prob_threshold);
    return mask;
}
template <int dim,typename vtype,template <typename...> typename stype>
inline auto operator<(const image<dim,vtype,stype>& I,vtype prob_threshold)
{
    tipl::image<dim,unsigned char,stype> mask;
    tipl::threshold(I,mask,prob_threshold,0,1);
    return mask;
}


template<typename ImageType>
ImageType& threshold(ImageType& I,typename ImageType::value_type threshold_value,
                     typename ImageType::value_type foreground = 1,typename ImageType::value_type background = 0)
{
    auto iter = I.begin();
    auto end = I.end();
    for(; iter != end; ++iter)
        *iter = (*iter > threshold_value) ? foreground : background;
    return I;
}

template<typename T,typename U>
inline T space2slice(unsigned char dim,const U& p)
{
    if constexpr(T::dimension == 3)
        return T(p[dim == 0 ? 1:0],p[dim == 2 ? 1:2],p[dim]);
    if constexpr(T::dimension == 2)
        return T(p[dim == 0 ? 1:0],p[dim == 2 ? 1:2]);
}

template<typename T>
inline T slice2space(unsigned char dim_index,typename T::value_type x,
                                             typename T::value_type y,
                                             typename T::value_type slice_index)
{
    if(dim_index == 2) return T(x,y,slice_index);
    if(dim_index == 1) return T(x,slice_index,y);
    return T(slice_index,x,y);
}

template<typename dim_type,typename GeoType,typename ResultType>
inline void get_slice_positions(dim_type dim,float pos,const GeoType& geo,ResultType& points)
{
    float x_max = float(geo[dim == 0 ? 1:0])-0.5f;
    float y_max = float(geo[dim == 2 ? 1:2])-0.5f;
    points[0] = tipl::slice2space<typename ResultType::value_type>(dim,-0.5f,-0.5f,pos);
    points[1] = tipl::slice2space<typename ResultType::value_type>(dim,x_max,-0.5f,pos);
    points[2] = tipl::slice2space<typename ResultType::value_type>(dim,-0.5f,y_max,pos);
    points[3] = tipl::slice2space<typename ResultType::value_type>(dim,x_max,y_max,pos);
}

template<typename T,typename U>
inline auto volume2points(const T& shape,U&& fun)
{
    std::vector<std::vector<tipl::vector<3,short> > > points(tipl::max_thread_count);
    tipl::par_for<dynamic_with_id>(shape,[&](const auto& index,unsigned int thread_id)
    {
        if (fun(index))
            points[thread_id].push_back(tipl::vector<3,short>(index.x(), index.y(),index.z()));
    });
    std::vector<tipl::vector<3,short> > region;
    tipl::aggregate_results(std::move(points),region);
    return region;
}

template<typename T>
inline auto volume2points(const T& mask)
{
    std::vector<std::vector<tipl::vector<T::dimension,short> > > points(tipl::max_thread_count);
    tipl::par_for<dynamic_with_id>(mask.shape(),[&](const auto& index,unsigned int thread_id)
    {
        if (mask[index.index()])
            points[thread_id].push_back(tipl::vector<T::dimension,short>(index.begin()));
    });
    std::vector<tipl::vector<T::dimension,short> > region;
    tipl::aggregate_results(std::move(points),region);
    return region;
}

template<typename T,typename U>
inline auto points2volume(const T& s,const U& points)
{
    size_t sz = points.size();
    tipl::image<T::dimension,unsigned char> mask(s);
    for(size_t i = 0; i < sz; ++i)
        if(s.is_valid(points[i]))
            mask.at(points[i]) = 1;
    return mask;
}

template<typename ImageType3D,typename ImageType2D,typename dim_type,typename slice_pos_type,
         typename std::enable_if<ImageType3D::dimension==3,bool>::type = true>
ImageType2D& volume2slice(const ImageType3D& slice,ImageType2D& I,dim_type dim,slice_pos_type slice_index)
{
    I.clear();
    const shape<3>& geo = slice.shape();
    if (dim == 2)
    {
        I.resize(shape<2>(geo[0],geo[1]));
        if(slice_index >= slice.depth()) return I;
        std::copy_n(slice.begin() + I.size() * slice_index, I.size(), I.begin());
    }
    else if (dim == 1)
    {
        I.resize(shape<2>(geo[0],geo[2]));
        if(slice_index >= slice.height()) return I;
        size_t wh = geo.plane_size();
        size_t sz = I.size();
        size_t w = geo[0];
        size_t sindex = size_t(slice_index) * w;
        for (size_t index = 0; index < sz; index += w, sindex += wh)
            std::copy_n(slice.begin() + sindex, w, I.begin() + index);
    }
    else if (dim == 0)
    {
        I.resize(shape<2>(geo[1],geo[2]));
        if(slice_index >= slice.width()) return I;
        size_t sz = I.size();
        size_t sindex = slice_index;
        size_t w = geo[0];
        for (size_t index = 0; index < sz; ++index, sindex += w)
            I[index] = slice[sindex];
    }
    return I;
}

template<typename ImageType3D,typename dim_type,typename slice_pos_type>
auto volume2slice(const ImageType3D& slice,dim_type dim,slice_pos_type slice_index)
{
    tipl::image<2,typename ImageType3D::value_type> I;
    volume2slice(slice,I,dim,slice_index);
    return I;
}

template<typename ImageType3D,typename ImageType2D,typename dim_type,typename slice_pos_type,
         typename std::enable_if<ImageType3D::dimension==2,bool>::type = true>
ImageType2D& volume2slice_scaled(const ImageType3D& slice,ImageType2D& I,dim_type,slice_pos_type,float scale)
{
    I.clear();
    I.resize(shape<2>(slice.width()*scale, slice.height()*scale));
    float ratio = 1.0f/scale;
    size_t sz = I.size();
    int w = slice.width() - 1, h = slice.height() - 1;
    for(pixel_index<2> pos(I.shape()); pos < sz; ++pos)
    {
        int x = std::min<int>(w, static_cast<int>(std::round(ratio*pos[0])));
        int y = std::min<int>(h, static_cast<int>(std::round(ratio*pos[1])));
        I[pos.index()] = slice.at(vector<2,int>(x,y));
    }
    return I;
}

template<typename ImageType3D,typename ImageType2D,typename dim_type,typename slice_pos_type,
         typename std::enable_if<ImageType3D::dimension==3,bool>::type = true>
ImageType2D& volume2slice_scaled(const ImageType3D& slice,ImageType2D& I,dim_type dim,slice_pos_type slice_index,float scale)
{
    const shape<3>& geo = slice.shape();
    I.clear();
    I.resize(shape<2>(geo[dim?0:1]*scale, geo[dim==2?1:2]*scale));
    if(slice_index >= geo[dim]) return I;

    float ratio = 1.0f/scale;
    size_t sz = I.size();
    int w = slice.width() - 1, h = slice.height() - 1, d = slice.depth() - 1;

    if (dim == 2)
    {
        for(pixel_index<2> pos(I.shape()); pos < sz; ++pos)
        {
            int x = std::min<int>(w, static_cast<int>(std::round(ratio*pos[0])));
            int y = std::min<int>(h, static_cast<int>(std::round(ratio*pos[1])));
            I[pos.index()] = slice.at(vector<3,int>(x, y, int(slice_index)));
        }
    }
    else if (dim == 1)
    {
        for(pixel_index<2> pos(I.shape()); pos < sz; ++pos)
        {
            int x = std::min<int>(w, static_cast<int>(std::round(ratio*pos[0])));
            int z = std::min<int>(d, static_cast<int>(std::round(ratio*pos[1])));
            I[pos.index()] = slice.at(vector<3,int>(x, int(slice_index), z));
        }
    }
    else if (dim == 0)
    {
        for(pixel_index<2> pos(I.shape()); pos < sz; ++pos)
        {
            int y = std::min<int>(h, static_cast<int>(std::round(ratio*pos[0])));
            int z = std::min<int>(d, static_cast<int>(std::round(ratio*pos[1])));
            I[pos.index()] = slice.at(vector<3,int>(int(slice_index), y, z));
        }
    }
    return I;
}

template<typename ImageType3D,typename dim_type,typename slice_pos_type>
auto volume2slice_scaled(const ImageType3D& slice,dim_type dim,slice_pos_type slice_index,float scale)
{
    tipl::image<2,typename ImageType3D::value_type> I;
    volume2slice_scaled(slice,I,dim,slice_index,scale);
    return I;
}

template<typename T,typename U>
inline bool draw_range(T from_w,T to_w,U& pos,int64_t& shift,int64_t& draw_size)
{
    if(pos < 0) { shift = -int64_t(pos); pos = 0; } else { shift = 0; }
    draw_size = std::min(int64_t(to_w)-int64_t(pos), int64_t(from_w)-shift);
    return draw_size > 0;
}

template<bool copy = true,typename T1,typename T2,typename PosType>
void draw(const T1& from_image,T2&& to_image,PosType pos)
{
    constexpr int dim = T1::dimension;
    int64_t fw = from_image.width(), fh = from_image.height();
    int64_t tw = to_image.width(), th = to_image.height();
    int64_t x_shift, y_shift, z_shift, x_width, y_height, z_depth;

    if(!draw_range(fw, tw, pos[0], x_shift, x_width) ||
       !draw_range(fh, th, pos[1], y_shift, y_height)) return;

    if constexpr(dim == 3)
    {
        if(!draw_range(from_image.depth(), to_image.depth(), pos[2], z_shift, z_depth)) return;
        for(int64_t z = 0; z < z_depth; ++z)
        {
            int64_t f_y = (z_shift+z)*fh + y_shift;
            int64_t t_y = (int64_t(pos[2])+z)*th + int64_t(pos[1]);

            auto iter = from_image.begin() + f_y * fw + x_shift;
            auto end = iter + int64_t(y_height-1) * fw;
            auto out = to_image.begin() + t_y * tw + int64_t(pos[0]);

            do {
                if constexpr(copy) std::copy_n(iter, x_width, out);
                else tipl::add(out, out+x_width, iter);
                if(iter >= end) break;
                iter += fw; out += tw;
            } while(1);
        }
    }
    else
    {
        auto iter = from_image.begin() + y_shift * fw + x_shift;
        auto end = iter + (y_height-1) * fw;
        auto out = to_image.begin() + pos[1] * tw + pos[0];

        do {
            if constexpr(copy) std::copy_n(iter, x_width, out);
            else tipl::add(out, out+x_width, iter);
            if(iter >= end) break;
            iter += fw; out += tw;
        } while(1);
    }
}

template<typename T1,typename T2,typename PosType>
void crop(const T1& from_image,T2&& to_image,PosType from,PosType to)
{
    constexpr int dim = T1::dimension;

    for(int i = 0; i < dim; ++i) if(from[i] >= to[i]) return;

    if(to_image.empty())
    {
        if constexpr(dim == 3)
            to_image.resize(tipl::shape<3>(to[0]-from[0],to[1]-from[1],to[2]-from[2]));
        else
            to_image.resize(tipl::shape<2>(to[0]-from[0],to[1]-from[1]));
    }

    PosType draw_pos;
    for(int i = 0; i < dim; ++i) draw_pos[i] = -from[i];
    draw(from_image,to_image,draw_pos);
}

template<typename T,typename U>
T& crop(T&& I,const U& from,const U& to)
{
    std::remove_reference_t<T> out_image;
    crop(I,out_image,from,to);
    I.swap(out_image);
    return I;
}

template<typename image_type,typename PosType,typename pixel_type>
void fill_rect(image_type&& I,PosType from,PosType to,pixel_type value)
{
    int64_t w = I.width();
    size_t line_pos = size_t(from[0]) + size_t(from[1]) * w;
    int line_width = int(to[0]) - int(from[0]);
    for(int y = from[1]; y < to[1]; ++y)
    {
        std::fill(I.begin()+line_pos, I.begin()+line_pos+line_width, value);
        line_pos += w;
    }
}

template<typename T,typename U,
         typename std::enable_if<!std::is_same_v<std::decay_t<U>,shape<T::dimension> >,bool>::type = true,
         typename std::enable_if<T::dimension==2,bool>::type = true>
void reshape(const T& I,U& I2)
{
    auto min_x = std::min(I.width(),I2.width());
    auto min_y = std::min(I.height(),I2.height());
    int64_t fw = I.width(), tw = I2.width();

    auto from2 = I.data();
    auto to2 = I2.data();
    for(size_t y = 0; y < min_y; ++y)
    {
        if(from2 != to2)
            for(size_t x = 0; x < min_x; ++x) to2[x] = from2[x];
        from2 += fw; to2 += tw;
    }
}

template<typename T,typename U,
         typename std::enable_if<!std::is_same_v<std::decay_t<U>,shape<T::dimension> >,bool>::type = true,
         typename std::enable_if<T::dimension==3,bool>::type = true>
void reshape(const T& I,U& I2)
{
    auto min_x = std::min(I.width(),I2.width());
    auto min_y = std::min(I.height(),I2.height());
    auto min_z = std::min(I.depth(),I2.depth());

    int64_t fw = I.width(), tw = I2.width();
    size_t fps = I.plane_size(), tps = I2.plane_size();
    auto from = I.data();
    auto to = I2.data();

    for(size_t z = 0; z < min_z; ++z)
    {
        auto from2 = from;
        auto to2 = to;
        for(size_t y = 0; y < min_y; ++y)
        {
            for(size_t x = 0; x < min_x; ++x) to2[x] = from2[x];
            from2 += fw; to2 += tw;
        }
        from += fps; to += tps;
    }
}

template<typename T>
void reshape(T& I,const shape<T::dimension>& new_shape)
{
    if(I.shape() == new_shape) return;
    if(I.width() < new_shape.width() || I.height() < new_shape.height() || I.size() < new_shape.size())
    {
        T new_I(new_shape);
        reshape(I,new_I);
        I.swap(new_I);
        return;
    }
    auto new_I = make_image(I.data(),new_shape);
    reshape(I,new_I);
    I.resize(new_shape);
}

template<bool copy = true,typename T1,typename T2>
inline void draw(const T1& from_image,T2&& to_image)
{
    draw<copy>(from_image,to_image,
         (tipl::vector<T1::dimension,int>(to_image.shape())-
         tipl::vector<T1::dimension,int>(to_image.shape()))/2);
}

template<bool copy = true,typename image_type,typename pos_type,typename shape_type>
void draw_rect(image_type&& to_image,pos_type pos,const shape_type& rect_sizes,typename std::remove_reference<image_type>::type::value_type value)
{
    using base_type = typename std::remove_reference<image_type>::type;
    constexpr int dim = base_type::dimension;
    int64_t tw = to_image.width(), th = to_image.height();
    int64_t x_shift, y_shift, z_shift, x_width, y_height, z_depth;

    if(!draw_range(rect_sizes[0], tw, pos[0], x_shift, x_width) ||
       !draw_range(rect_sizes[1], th, pos[1], y_shift, y_height)) return;

    if constexpr(dim == 3)
    {
        if(!draw_range(rect_sizes[2], to_image.depth(), pos[2], z_shift, z_depth)) return;
        for(int64_t z = 0; z < z_depth; ++z)
        {
            int64_t t_y = (int64_t(pos[2])+z) * th + int64_t(pos[1]);
            auto out = to_image.begin() + t_y * tw + int64_t(pos[0]);
            for(int64_t y = 0; y < y_height; ++y)
            {
                if constexpr(copy) std::fill(out, out+x_width, value);
                else tipl::add_constant(out, out+x_width, value);
                out += tw;
            }
        }
    }
    else
    {
        auto out = to_image.begin() + int64_t(pos[1]) * tw + int64_t(pos[0]);
        for(int64_t y = 0; y < y_height; ++y)
        {
            if constexpr(copy) std::fill(out, out+x_width, value);
            else tipl::add_constant(out, out+x_width, value);
            out += tw;
        }
    }
}

template<typename fun_type>
void draw_line(int x,int y,int x1,int y1,fun_type fun)
{
    int dx = x1-x;
    int dy = y1-y;
    int abs_dx = std::abs(dx);
    int abs_dy = std::abs(dy);
    if(abs_dx <= 1 && abs_dy <= 1) { fun(x,y); return; }

    if(abs_dx > abs_dy)
    {
        if(x1 < x) { std::swap(x1,x); std::swap(y1,y); dy = -dy; }
        for(int i = 0; i < abs_dx; ++i) fun(i+x, dy*i/abs_dx+y);
    }
    else
    {
        if(y1 < y) { std::swap(x1,x); std::swap(y1,y); dx = -dx; }
        for(int i = 0; i < abs_dy; ++i) fun(dx*i/abs_dy+x, i+y);
    }
}

template<typename T,typename U>
void mosaic(const T& source,U&& out,unsigned int mosaic_size,unsigned int skip = 1)
{
    unsigned slice_num = source.depth() / skip;
    unsigned int sw = source.width(), sh = source.height();
    out.clear();
    out.resize(tipl::shape<2>(sw * mosaic_size, sh * (std::ceil(float(slice_num)/float(mosaic_size)))));
    for(unsigned int z = 0; z < slice_num; ++z)
    {
        tipl::vector<2,int> pos(sw * (z % mosaic_size), sh * (z / mosaic_size));
        tipl::draw(source.slice_at(z*skip), out, pos);
    }
}

template<typename T,typename U>
T& move(T&& I,U pos)
{
    std::remove_reference_t<T> dest(I.shape());
    draw(I,dest,pos);
    dest.swap(I);
    return I;
}

template<typename ImageType,typename DimensionType,typename ValueType = typename ImageType::value_type>
bool bounding_box(const ImageType& I,DimensionType& range_min,DimensionType& range_max,ValueType background = 0,int margin = 0)
{
    auto shape = I.shape();
    size_t sz = I.size();
    for (unsigned int di = 0; di < ImageType::dimension; ++di)
    {
        range_min[di] = shape[di]-1;
        range_max[di] = 0;
    }
    for (pixel_index<ImageType::dimension> iter(shape); iter < sz; ++iter)
    {
        if (I[iter.index()] <= background) continue;
        for (unsigned int di = 0; di < ImageType::dimension; ++di)
        {
            if (iter[di] < range_min[di]) range_min[di] = iter[di];
            if (iter[di] > range_max[di]) range_max[di] = iter[di];
        }
    }
    bool has_bounding_box = true;
    for (unsigned int di = 0; di < ImageType::dimension; ++di)
    {
        if(range_max[di] == 0)
        {
            range_max[di] = range_min[di] = 0;
            has_bounding_box = false;
        }
        if(margin)
        {
            range_min[di] = std::max<int>(0, int(range_min[di])-margin);
            range_max[di] = std::min<int>(shape[di]-1, int(range_max[di])+margin);
        }
        ++range_max[di];
    }
    return has_bounding_box;
}

template<typename T,typename U>
unsigned char long_axis(const T& I,const U& vs)
{
    tipl::vector<3> range_min,range_max;
    if(!tipl::bounding_box(I,range_min,range_max)) return 1;
    range_max -= range_min;
    range_max.abs();
    range_max[0] *= vs[0]; range_max[1] *= vs[1]; range_max[2] *= vs[2];
    if(range_max[2] > range_max[1] && range_max[2] > range_max[0]) return 2;
    if(range_max[1] > range_max[0]) return 1;
    return 0;
}

template<typename T,typename U>
unsigned char symmetric_axis(const T& I_,const U& vs)
{
    tipl::vector<3,int> range_min,range_max;
    if(!tipl::bounding_box(I_,range_min,range_max)) return 0;
    typename T::buffer_type I(I_);
    tipl::crop(I,range_min,range_max);
    size_t dif_x = 0, dif_y = 0, dif_z = 0;
    size_t sz = I.size(), ps = I.plane_size();
    int w = I.width(), h = I.height(), d = I.depth();
    int w2 = w >> 1, h2 = h >> 1, d2 = d >> 1;

    for(tipl::pixel_index<3> pos(I.shape()); pos < sz; ++pos)
    {
        if(pos.x() < w2) {
            auto dx = int(I[pos.index()]) - int(I[pos.index() + w - 1 - pos.x() - pos.x()]);
            dif_x += dx*dx;
        }
        if(pos.y() < h2) {
            auto dy = int(I[pos.index()]) - int(I[pos.index() + (h - 1 - pos.y() - pos.y())*w]);
            dif_y += dy*dy;
        }
        if(pos.z() < d2) {
            auto dz = int(I[pos.index()]) - int(I[pos.index() + (d - 1 - pos.z() - pos.z())*ps]);
            dif_z += dz*dz;
        }
    }
    dif_x *= vs[0]; dif_y *= vs[1]; dif_z *= vs[2];
    if(dif_z < dif_y && dif_z < dif_x) return 2;
    if(dif_y < dif_x) return 1;
    return 0;
}

template<typename point_type>
void bounding_box(const std::vector<point_type>& points,point_type& max_value,point_type& min_value)
{
    if(points.empty()) return;
    unsigned int thread_count = tipl::max_thread_count;
    std::vector<point_type> max_values(thread_count), min_values(thread_count);
    for(unsigned int i = 0; i < thread_count; ++i)
    {
        max_values[i] = points[0];
        min_values[i] = points[0];
    }
    unsigned char dim = points[0].size();
    tipl::par_for<dynamic_with_id>(points.size(),[&](unsigned int index,unsigned int id)
    {
        for (unsigned char d = 0; d < dim; ++d)
            if (points[index][d] > max_values[id][d]) max_values[id][d] = points[index][d];
            else if (points[index][d] < min_values[id][d]) min_values[id][d] = points[index][d];
    });
    max_value = max_values[0]; min_value = min_values[0];

    for(unsigned int i = 0; i < thread_count; ++i)
    {
        for (unsigned char d = 0; d < dim; ++d)
        {
            if (max_values[i][d] > max_value[d]) max_value[d] = max_values[i][d];
            if (min_values[i][d] < min_value[d]) min_value[d] = min_values[i][d];
        }
    }
}

template<typename image_type>
bool has_mask(const image_type& img)
{
    size_t zero_count = 0;
    size_t sz = img.size();
    size_t threshold = sz / 5;
    auto shape = img.shape();

    for (size_t i = 0; i < sz && zero_count <= threshold; ++i)
        if (img[i] == 0) ++zero_count;

    if (zero_count > threshold) return true;

    tipl::vector<image_type::dimension,int> vmin, vmax;
    tipl::bounding_box(img, vmin, vmax, 0);

    for (int d = 0; d < image_type::dimension; ++d)
        if (vmin[d] > 0 && vmax[d] < shape[d]) return true;
    return false;
}

template<typename ImageType>
ImageType& trim(ImageType& I,typename ImageType::value_type background = 0)
{
    tipl::shape<ImageType::dimension> range_min,range_max;
    bounding_box(I,range_min,range_max,background);
    if (range_min[0] < range_max[0]) crop(I,range_min,range_max);
    return I;
}

template<typename iterator_type,typename dim_order_type,typename flip_type>
void get_orientation(iterator_type rotation_matrix,dim_order_type dim_order,flip_type flipped)
{
    iterator_type vec = rotation_matrix;
    for (int index = 0; index < 3; ++index, vec += 3)
    {
        dim_order[index] = 0;
        flipped[index] = vec[0] < 0;
        for(int j = 1; j < 3; ++j)
            if(std::abs(vec[j]) > std::abs(vec[dim_order[index]]))
            {
                dim_order[index] = j;
                flipped[index] = vec[j] < 0;
            }
    }
}

template<typename iterator_type,typename dim_order_type,typename flip_type>
void get_inverse_orientation(int dim,iterator_type rotation_matrix,dim_order_type dim_order,flip_type flipped)
{
    iterator_type vec = rotation_matrix;
    for (int index = 0; index < dim; ++index, ++vec)
    {
        dim_order[index] = 0;
        flipped[index] = vec[0] < 0;
        for(int j = 1; j < dim; ++j)
            if(std::abs(vec[j*dim]) > std::abs(vec[dim_order[index]*dim]))
            {
                dim_order[index] = j;
                flipped[index] = (vec[j*dim] < 0);
            }
    }
}

template<typename image_type1,typename image_type2>
void reorder(const image_type1& volume,image_type2& volume_out,int64_t origin[],int64_t shift[],uint8_t index_dim)
{
    uint64_t index = 0, base_index = 0;
    uint64_t sz = volume.size(), vw = volume.width(), vh = volume.height(), op = volume_out.plane_size();

    while(index < sz)
    {
        if(index_dim == 2)
        {
            uint64_t y_index = base_index + origin[1];
            for (uint64_t y = 0; y < vh; ++y)
            {
                uint64_t x_index = y_index + origin[0];
                for (uint64_t x = 0; x < vw; ++x, ++index)
                {
                    volume_out[x_index] = volume[index];
                    x_index += shift[0];
                }
                y_index += shift[1];
            }
            base_index += op;
        }
        if(index_dim == 3)
        {
            uint64_t vd = volume.depth();
            uint64_t z_index = base_index + origin[2];
            for (uint64_t z = 0; z < vd; ++z)
            {
                uint64_t y_index = z_index + origin[1];
                for (uint64_t y = 0; y < vh; ++y)
                {
                    uint64_t x_index = y_index + origin[0];
                    for (uint64_t x = 0; x < vw; ++x, ++index)
                    {
                        volume_out[x_index] = volume[index];
                        x_index += shift[0];
                    }
                    y_index += shift[1];
                }
                z_index += shift[2];
            }
            base_index += op * vd;
        }
    }
}

template<typename dim_order_type>
void reorient_vector(tipl::vector<3>& spatial_resolution,dim_order_type dim_order)
{
    tipl::vector<3> sr(spatial_resolution);
    for(unsigned int index = 0; index < 3; ++index)
        spatial_resolution[dim_order[index]] = sr[index];
}

template<typename iterator_type2,typename dim_order_type,typename flip_type>
void reorient_matrix(iterator_type2 orientation_matrix,dim_order_type dim_order,flip_type flip)
{
    float orientation_matrix_[9];
    std::copy_n(orientation_matrix,9,orientation_matrix_);
    for(unsigned int index = 0, ptr = 0; index < 3; ++index, ptr += 3)
        if(flip[index])
        {
            orientation_matrix_[ptr] = -orientation_matrix_[ptr];
            orientation_matrix_[ptr+1] = -orientation_matrix_[ptr+1];
            orientation_matrix_[ptr+2] = -orientation_matrix_[ptr+2];
        }
    for(unsigned int index = 0; index < 3; ++index)
        std::copy_n(orientation_matrix_ + index*3, 3, orientation_matrix + dim_order[index]*3);
}

template<typename geo_type,typename dim_order_type,typename flip_type,typename origin_type,typename shift_type>
bool reorder_shift_index(const geo_type& geo,
                         dim_order_type dim_order,
                         flip_type flip,
                         geo_type& new_geo,
                         origin_type origin_index,
                         shift_type shift_index)
{
    bool need_update = false;
    for (unsigned char index = 0; index < geo_type::dimension; ++index)
    {
        new_geo.set_dim(dim_order[index],geo[index]);
        if (dim_order[index] != index) need_update = true;
    }

    std::vector<int64_t> shift_vector(geo_type::dimension);
    shift_vector[0] = 1;
    for(unsigned char dim = 1; dim < geo_type::dimension; ++dim)
        shift_vector[dim] = shift_vector[dim-1]*int64_t(new_geo[dim-1]);

    for (unsigned int index = 0; index < geo_type::dimension; ++index)
    {
        if (flip[index])
        {
            origin_index[index] = shift_vector[dim_order[index]]*(new_geo[dim_order[index]]-1);
            shift_index[index] = -shift_vector[dim_order[index]];
            need_update = true;
        }
        else
        {
            origin_index[index] = 0;
            shift_index[index] = shift_vector[dim_order[index]];
        }
    }
    return need_update;
}

template<typename image_type1,typename image_type2,typename dim_order_type,typename flip_type>
void reorder(const image_type1& volume,image_type2& volume_out,dim_order_type dim_order,flip_type flip)
{
    tipl::shape<image_type1::dimension> new_geo;
    int64_t origin[image_type1::dimension];
    int64_t shift[image_type1::dimension];
    if (!reorder_shift_index(volume.shape(),dim_order,flip,new_geo,origin,shift))
    {
        volume_out = volume;
        return;
    }
    volume_out.resize(new_geo);
    if(!volume_out.empty()) reorder(volume,volume_out,origin,shift,image_type1::dimension);
}

template<typename image_type,typename dim_order_type,typename flip_type>
void reorder(image_type& volume,dim_order_type dim_order,flip_type flip)
{
    image_type volume_out;
    reorder(volume,volume_out,dim_order,flip);
    volume.swap(volume_out);
}

template<typename iterator_type>
void flip_block(iterator_type beg,iterator_type end,size_t block_size)
{
    tipl::par_for((end-beg)/block_size,[&](size_t i)
    {
        iterator_type from = beg + i * block_size;
        iterator_type to = from + block_size - 1;
        while (from < to)
        {
            std::swap(*from,*to);
            ++from; --to;
        }
    });
}

template<typename iterator_type>
void flip_block_line(iterator_type beg,iterator_type end,size_t block_size,unsigned int line_length)
{
    if(line_length == 1) flip_block(beg,end,block_size);
    else
        while (beg < end)
        {
            iterator_type from = beg;
            beg += block_size;
            iterator_type to = beg - line_length;
            while (from < to)
            {
                for(unsigned int index = 0; index < line_length; ++index)
                    std::swap(*(from+index),*(to+index));
                from += line_length;
                to -= line_length;
            }
        }
}

template<typename ImageType>
ImageType& flip_x(ImageType& I)
{
    if(I.empty()) return I;
    flip_block(I.begin(), I.end(), I.width());
    return I;
}

template<typename ImageType>
ImageType& flip_y(ImageType& I)
{
    if(I.empty()) return I;
    flip_block_line(I.begin(), I.end(), I.height() * I.width(), I.width());
    return I;
}

template<typename ImageType>
ImageType& flip_z(ImageType& I)
{
    if(I.empty()) return I;
    flip_block_line(I.begin(), I.end(), I.plane_size() * I.depth(), I.plane_size());
    return I;
}

template<typename ImageType>
ImageType& flip_xy(ImageType& I)
{
    if(I.empty()) return I;
    flip_block(I.begin(), I.end(), I.plane_size());
    return I;
}

template<typename ImageType>
ImageType& swap_xy(ImageType& I)
{
    size_t sz = I.size();
    if(sz == 0) return I;
    uint32_t w = I.width(), h = I.height();
    size_t ps = I.plane_size();

    if(w == h)
    {
        size_t w_1 = w + 1;
        for(size_t i = 0; i < sz; i += ps)
        {
            auto plane_ptr = &I[i];
            for(uint32_t y = 0, pos = 0; y < h; ++y, pos += w_1)
            {
                size_t pos_x = pos + 1, pos_y = pos + w;
                for(uint32_t x = y + 1; x < w; ++x)
                {
                    std::swap(plane_ptr[pos_x], plane_ptr[pos_y]);
                    ++pos_x; pos_y += w;
                }
            }
        }
        return I;
    }

    tipl::image<2,typename ImageType::value_type> plane(tipl::shape<2>(w, h));
    size_t plane_sz = plane.size();
    for(size_t i = 0; i < sz; i += plane_sz)
    {
        auto plane_ptr = &I[i];
        std::copy_n(plane_ptr, ps, &plane[0]);
        for(size_t y = 0, p1 = 0; y < h; ++y)
            for(size_t x = 0, p2 = y; x < w; ++x, ++p1, p2 += h)
                plane_ptr[p2] = plane[p1];
    }
    I.resize(I.shape().swap_dim(0,1));
    return I;
}

template<typename ImageType>
ImageType& swap_xz(ImageType& I)
{
    tipl::shape<ImageType::dimension> new_geo(I.shape().swap_dim(0,2));
    tipl::image<ImageType::dimension,typename ImageType::value_type> new_volume(new_geo);

    int64_t origin[3] = {0,0,0};
    int64_t shift[3];
    shift[0] = new_geo.plane_size();
    shift[1] = new_geo.width();
    shift[2] = 1;
    reorder(I,new_volume,origin,shift,3);

    I.resize(new_geo);
    std::copy(new_volume.begin(),new_volume.end(),I.begin());
    return I;
}

template<typename ImageType>
ImageType& swap_yz(ImageType& I)
{
    size_t sz = I.size();
    if(sz == 0) return I;

    uint32_t w = I.width(), h = I.height(), d = I.depth();
    size_t ps = I.plane_size();
    size_t volume_size = size_t(w) * size_t(h) * size_t(d);
    tipl::shape<ImageType::dimension> new_geo(I.shape().swap_dim(1,2));

    for(size_t v = 0; v < sz; v += volume_size)
    for(size_t x = 0; x < w; ++x)
    {
        size_t start_pos = x + v;
        if(h == d)
        {
            for(uint16_t z = 0; z < d; ++z, start_pos += w + ps)
            {
                size_t pos_y = start_pos + w;
                size_t pos_z = start_pos + ps;
                for(uint16_t y = z + 1; y < h; ++y)
                {
                    std::swap(I[pos_y], I[pos_z]);
                    pos_y += w; pos_z += ps;
                }
            }
        }
        else
        {
            tipl::image<2,typename ImageType::value_type> plane(tipl::shape<2>(h, d));
            {
                size_t index = 0, pos = start_pos;
                for(uint32_t z = 0; z < d; ++z)
                    for(uint32_t y = 0; y < h; ++y, ++index, pos += w)
                        plane[index] = I[pos];
            }
            {
                size_t index = 0, new_pos = start_pos;
                uint32_t new_w = new_geo.width();
                size_t new_ps = new_geo.plane_size();
                for(uint16_t z = 0; z < d; ++z, new_pos += new_w)
                {
                    size_t pos = new_pos;
                    for(uint16_t y = 0; y < h; ++y, ++index, pos += new_ps)
                        I[pos] = plane[index];
                }
            }
        }
    }

    if(h != d) I.resize(new_geo);
    return I;
}

template<typename ImageType>
ImageType& flip(ImageType&& I,unsigned char dim)
{
    switch(dim)
    {
    case 0: flip_x(I); break;
    case 1: flip_y(I); break;
    case 2: flip_z(I); break;
    case 3: swap_xy(I); break;
    case 4: swap_yz(I); break;
    case 5: swap_xz(I); break;
    }
    return I;
}

template<typename iterator_type,typename value_type>
void negate(iterator_type iter,iterator_type end,value_type maximum)
{
    for (; iter != end; ++iter) *iter = maximum - *iter;
}

template<typename ImageType,typename value_type>
ImageType& negate(ImageType& I,value_type maximum)
{
    auto iter = I.begin(), end = I.end();
    for (; iter != end; ++iter) *iter = maximum - *iter;
    return I;
}

template<typename ImageType>
ImageType& negate(ImageType& I)
{
    negate(I,*std::max_element(I.begin(),I.end()));
    return I;
}

template<typename ImageType1,typename ImageType2,typename PixelType2>
void paint(const ImageType1& image1,ImageType2& image2,PixelType2 paint_value)
{
    auto iter1 = image1.begin(), end = image1.end();
    auto iter2 = image2.begin();
    for (; iter1 != end; ++iter1, ++iter2)
        if (*iter1) *iter2 = paint_value;
}

template<typename PixelType1,typename PixelType2,typename LocationType,typename DetermineType>
void draw_if(const tipl::image<2,PixelType1>& src,
             tipl::image<2,PixelType2>& des,LocationType place,DetermineType pred_background)
{
    int64_t x_src = 0, y_src = 0;
    int64_t x_des = place[0], y_des = place[1];
    if (x_des < 0) { x_src = -x_des; x_des = 0; }
    if (y_des < 0) { x_src = -y_des; y_des = 0; }

    int64_t sw = src.width(), dw = des.width();
    int64_t draw_width = sw - x_src;
    int64_t draw_height = src.height() - y_src;

    if (x_des + draw_width > dw) draw_width = dw - x_des;
    if (y_des + draw_height > des.height()) draw_height = des.height() - y_des;

    const PixelType1* src_iter = src.begin() + y_src * sw + x_src;
    const PixelType1* src_end = src_iter + draw_height * sw;
    PixelType2* des_iter = des.begin() + y_des * dw + x_des;

    for(; src_iter != src_end; des_iter += dw)
    {
        const PixelType1* from = src_iter;
        const PixelType1* to = src_iter + sw;
        PixelType2* des_p = des_iter;
        for(; from != to; ++from, ++des_p)
            if(!pred_background(*from))
                *des_p = *from;
        src_iter = to;
    }
}

template<typename PixelType1,typename OutImageType>
void project(const tipl::image<2,PixelType1>& src,OutImageType& result,unsigned int dim)
{
    if(dim == 0) // project x
    {
        int sh = src.height(), sw = src.width();
        result.resize(sh);
        size_t index = 0;
        for(int y = 0; y < sh; ++y, index += sw)
            result[y] = std::accumulate(src.begin()+index, src.begin()+index+sw, typename OutImageType::value_type(0));
    }
    else//project y
    {
        result.clear();
        result.resize(src.width());
        size_t sz = src.size();
        for(pixel_index<2> index(src.shape()); index < sz; ++index)
            result[index.x()] += src[index.index()];
    }
}

template <typename image_type,typename output_type>
void project_x(const image_type& I,output_type& P)
{
    typedef typename output_type::value_type value_type;
    int h = I.height(), w = I.width();
    P.resize(tipl::shape<2>(h, I.depth()));
    tipl::par_for(P.shape(),[&](const auto& index)
    {
        size_t pos = (index[0] + index[1] * h) * w;
        P[index.index()] = std::accumulate(I.begin()+pos, I.begin()+pos+w, value_type(0));
    });
}

template <typename image_type,typename output_type>
void project_y(const image_type& I,output_type& P)
{
    typedef typename output_type::value_type value_type;
    int h = I.height(), w = I.width();
    size_t ps = I.plane_size();
    P.resize(tipl::shape<2>(w, I.depth()));
    tipl::par_for(P.shape(),[&](tipl::pixel_index<2> index)
    {
        size_t pos = index[0] + index[1] * ps;
        value_type v(0);
        for(int y = 0; y < h; ++y, pos += w) v += I[pos];
        P[index.index()] = v;
    });
}

template<typename ImageType,typename HisType>
void histogram(const ImageType& src,HisType& hist,
               typename ImageType::value_type min_value,
               typename ImageType::value_type max_value,unsigned int resolution_count = 256)
{
    if(min_value >= max_value) return;
    float range = max_value - min_value;
    if(range == 0.0f) range = 1.0f;
    hist.clear();
    range = float(resolution_count)/range;
    hist.resize(resolution_count);

    auto iter = src.begin(), end = src.end();
    for(; iter != end; ++iter)
    {
        float value = (*iter - min_value) * range;
        int index = int(std::floor(value));
        if(index < 0) index = 0;
        if(index >= int(hist.size())) index = int(hist.size()) - 1;
        ++hist[uint32_t(index)];
    }
}

template<typename ImageType,typename HistType>
void histogram_sharpening(
                       const ImageType& src,
                       HistType& hist,
                       typename ImageType::value_type mn,
                       typename ImageType::value_type mx,
                       unsigned int     resolution_count = 256,
                       double           sigma            = 0.05,
                       double           noise            = 1e-3)
{
    if (mn >= mx) return;

    tipl::histogram(src, hist, mn, mx, resolution_count);

    std::vector<double> hist_blur(resolution_count);
    int rad = int(std::ceil(3.0 * sigma * resolution_count));
    double twoSigma2 = 2.0 * (sigma * resolution_count) * (sigma * resolution_count);

    std::vector<double> kern(2*rad+1);
    double kw = 0;
    for (int k = -rad; k <= rad; ++k)
        kw += (kern[k+rad] = std::exp(-k*k / twoSigma2));
    for (auto &w : kern) w /= kw;

    for (int i = 0; i < int(resolution_count); ++i)
    {
        double v = 0;
        for (int k = -rad; k <= rad; ++k)
        {
            int j = i + k;
            if (j >= 0 && j < int(resolution_count))
                v += hist[j] * kern[k+rad];
        }
        hist_blur[i] = v;
    }

    for (int i = 0; i < int(resolution_count); ++i)
    {
        double H = hist_blur[i];
        H *= H;
        hist[i] *= H / (H + noise);
    }
}

template<typename ImageType>
void histogram_sharpening(
                       ImageType&       src,
                       unsigned int     resolution_count = 256,
                       double           sigma            = 0.05,
                       double           noise            = 1e-3)
{
    typename ImageType::value_type mn, mx;
    minmax_value(src.begin(), src.end(), mn, mx);
    if (mn >= mx) return;

    std::vector<double> hist_sharp;
    tipl::histogram_sharpening(src, hist_sharp, mn, mx, resolution_count);

    std::vector<double> cdf(resolution_count);
    cdf[0] = hist_sharp[0];
    for (int i = 1; i < int(resolution_count); ++i)
        cdf[i] = cdf[i-1] + hist_sharp[i];
    if (cdf.back() == 0) return;

    tipl::divide_constant(cdf.begin(), cdf.end(), cdf.back());
    double range = double(mx) - double(mn);
    size_t sz = src.size();

    for(size_t i = 0; i < sz; ++i)
        src[i] = typename ImageType::value_type(double(mn) +
             cdf[std::clamp(int((double(src[i]) - mn)/range * (resolution_count-1) + 0.5), 0, int(resolution_count-1))] * range);
}

template<typename image_type1,typename image_type2>
void hist_norm(const image_type1& I1,image_type2& I2,unsigned int bin_count)
{
    typename image_type1::value_type min_v = *std::min_element(I1.begin(),I1.end());
    typename image_type1::value_type max_v = *std::max_element(I1.begin(),I1.end());

    std::vector<unsigned int> hist;
    tipl::histogram(I1,hist,min_v,max_v,bin_count);

    for(unsigned int i = 1; i < hist.size(); ++i)
        hist[i] += hist[i-1];

    size_t sz = I1.size();
    if(I2.size() != sz) I2.resize(I1.shape());

    float range = max_v - min_v;
    if(range == 0.0f) range = 1.0f;
    float r = (hist.size()+1) / range;

    for(size_t i = 1; i < sz; ++i)
    {
        int rank = std::floor(float(I1[i]-min_v) * r);
        if(rank <= 0) I2[i] = min_v;
        else
        {
            --rank;
            if(rank >= int(hist.size())) rank = int(hist.size()) - 1;
            I2[i] = range * float(hist[rank]) / float(hist.back()) + min_v;
        }
    }
}

template<typename image_type>
image_type& hist_norm(image_type& I,unsigned int bin_count)
{
    hist_norm(I,I,bin_count);
    return I;
}

template<typename container_type>
void softmax(container_type&& eval_output, size_t spatial_size, size_t model_out_count)
{
    if (model_out_count <= 1 || spatial_size == 0)
        return;

    size_t sz = eval_output.size();
    tipl::par_for(spatial_size, [&](size_t pos)
    {
        auto max_val = eval_output[pos];
        for(size_t offset = pos + spatial_size; offset < sz; offset += spatial_size)
            if(eval_output[offset] > max_val)
                max_val = eval_output[offset];

        double sum_exp(0);
        for(size_t offset = pos; offset < sz; offset += spatial_size)
        {
            eval_output[offset] = std::exp(eval_output[offset] - max_val);
            sum_exp += eval_output[offset];
        }

        sum_exp = 1.0/sum_exp;
        for(size_t offset = pos; offset < sz; offset += spatial_size)
            eval_output[offset] *= sum_exp;
    });
}

template<typename container_type>
void softmax(container_type&& img)
{
    size_t spatial_size = 1;
    for(size_t i = 0; i < container_type::dimension - 1; ++i)
        spatial_size *= img.shape()[i];

    softmax(img, spatial_size, img.shape()[container_type::dimension - 1]);
}


template <typename ImageType, typename MaskType>
auto argmax(const ImageType& label_prob, const MaskType& mask)
{
    tipl::image<3, unsigned char> I(mask.shape());
    size_t s = mask.size();
    size_t total_size = label_prob.size();
    tipl::par_for(s, [&](size_t pos)
    {
        if (!mask[pos])
            return;
        auto m = label_prob[pos];
        unsigned char max_label = 1;
        for (size_t i = pos + s, label = 2; i < total_size; i += s, ++label)
        {
            if (label_prob[i] > m)
            {
                m = label_prob[i];
                max_label = static_cast<unsigned char>(label);
            }
        }
        I[pos] = max_label;
    });
    return I;
}

template<typename U>
auto remove_channel(U& label_prob,const shape<3>& image_dim,size_t bg_channel = 0)
{
    std::copy(label_prob.begin() + (bg_channel+1)*image_dim.size(), label_prob.end(), label_prob.begin() + bg_channel*image_dim.size());
    label_prob.resize(image_dim.multiply(tipl::shape<3>::z, label_prob.depth()/image_dim[2]-1));
}

template<typename type>
void change_endian(type& value)
{
    type data = value;
    unsigned char* temp = reinterpret_cast<unsigned char*>(&value);
    unsigned char* pdata = reinterpret_cast<unsigned char*>(&data)+sizeof(type)-1;
    for (unsigned char i = 0; i < sizeof(type); ++i, --pdata)
        temp[i] = *pdata;
}

inline void change_endian(unsigned short& data)
{
    unsigned char* h = reinterpret_cast<unsigned char*>(&data);
    std::swap(*h,*(h+1));
}

inline void change_endian(short& data)
{
    unsigned char* h = reinterpret_cast<unsigned char*>(&data);
    std::swap(*h,*(h+1));
}

inline void change_endian(unsigned int& data)
{
    unsigned char* h = reinterpret_cast<unsigned char*>(&data);
    std::swap(*h,*(h+3));
    std::swap(*(h+1),*(h+2));
}

inline void change_endian(int& data)
{
    unsigned char* h = reinterpret_cast<unsigned char*>(&data);
    std::swap(*h,*(h+3));
    std::swap(*(h+1),*(h+2));
}

inline void change_endian(float& data)
{
    change_endian(*reinterpret_cast<int*>(&data));
}

template<typename datatype>
inline void change_endian(void* data_,size_t count)
{
    auto data = reinterpret_cast<datatype*>(data_);
    for (size_t index = 0; index < count; ++index)
        change_endian(data[index]);
}

template<typename datatype>
inline void change_endian(datatype* data,size_t count)
{
    for (size_t index = 0; index < count; ++index)
        change_endian(data[index]);
}

}
#endif
