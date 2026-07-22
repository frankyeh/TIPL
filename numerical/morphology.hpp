//---------------------------------------------------------------------------
#ifndef MORPHOLOGY_HPP_INCLUDED
#define MORPHOLOGY_HPP_INCLUDED
#include <map>
#include <list>
#include <set>
#include <unordered_map>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <string>

#include "../numerical/basic_op.hpp"
#include "../utility/basic_image.hpp"
#include "../utility/pixel_index.hpp"
#include "../numerical/index_algorithm.hpp"
#include "../numerical/window.hpp"
#include "../numerical/statistics.hpp"


namespace tipl
{

namespace morphology
{

template<typename T,typename F>
void for_each_label(T& data,F&& fun)
{
    auto max_v = tipl::max_value(data);
    if(max_v <= 1)
    {
        tipl::image<3,char> mask(data);
        fun(mask);
        data = mask;
        return;
    }
    auto shape = data.shape();
    size_t sz = data.size();
    T result_data(shape);
    tipl::par_for<sequential>(uint32_t(max_v)+1,[&](uint32_t index)
    {
        if(!index)
            return;
        tipl::image<3,char> mask(shape);
        for(size_t i = 0; i < sz; ++i)
            if(uint32_t(data[i]) == index)
                mask[i] = 1;
        fun(mask);
        for(size_t i = 0; i < sz; ++i)
            if(mask[i] && result_data[i] < typename T::value_type(index))
                result_data[i] = typename T::value_type(index);
    });
    data.swap(result_data);
}

template<typename ImageType>
ImageType& erosion(ImageType& I)
{
    auto s = I.shape();
    std::vector<size_t> rm;
    if constexpr(ImageType::dimension == 2) rm.reserve(I.width()); else rm.reserve(I.plane_size());
    for(tipl::pixel_index<ImageType::dimension> p(s);p < I.size();++p)
        if(!I[p.index()])
            tipl::for_each_connected_neighbors(p,s,[&](const auto& q)
                {rm.push_back(q.index());});
    for(auto i : rm)
        I[i] = 0;
    return I;
}

template<typename ImageType,typename RefType,typename threshold_type>
void erosion_by_threshold(ImageType& I,const RefType& V,threshold_type t)
{
    auto s = I.shape();
    auto pass = [&](auto v){return t < 0 ? v < -t : v > t;};
    std::vector<size_t> rm;
    if constexpr(ImageType::dimension == 2) rm.reserve(I.width()); else rm.reserve(I.plane_size());
    for(tipl::pixel_index<ImageType::dimension> p(s);p < I.size();++p)
        if(auto i = p.index(); !I[i] && pass(V[i]))
            tipl::for_each_connected_neighbors(p,s,[&](const auto& q)
                {if(pass(V[q.index()])) rm.push_back(q.index());});
    for(auto i : rm)
        I[i] = 0;
}

template<typename ImageType>
ImageType& dilation(ImageType& I)
{
    auto s = I.shape();
    std::vector<std::pair<size_t,typename ImageType::value_type> > add;
    if constexpr(ImageType::dimension == 2) add.reserve(I.width()); else add.reserve(I.plane_size());
    for(tipl::pixel_index<ImageType::dimension> p(s);p < I.size();++p)
        if(auto x = I[p.index()])
            tipl::for_each_connected_neighbors(p,s,[&](const auto& q)
                {add.push_back({q.index(),x});});
    for(auto [i,x] : add)
        I[i] |= x;
    return I;
}

template<typename ImageType,typename RefType,typename threshold_type>
void dilation_by_threshold(ImageType& I,const RefType& V,threshold_type t)
{
    auto s = I.shape();
    auto pass = [&](auto v){return t < 0 ? v < -t : v > t;};
    std::vector<std::pair<size_t,typename ImageType::value_type> > add;
    if constexpr(ImageType::dimension == 2) add.reserve(I.width()); else add.reserve(I.plane_size());
    for(tipl::pixel_index<ImageType::dimension> p(s);p < I.size();++p)
        if(auto i = p.index(); I[i] && pass(V[i]))
            tipl::for_each_connected_neighbors(p,s,[&](const auto& q)
                {if(pass(V[q.index()])) add.push_back({q.index(),I[i]});});
    for(auto [i,x] : add)
        I[i] |= x;
}



template<bool dilate,typename ImageType>
ImageType& morphology_by_radius(ImageType& I,unsigned int radius)
{
    if(radius == 0 || I.empty())
        return I;

    using value_type = typename ImageType::value_type;
    ImageType src(I);

    const int w = int(I.width()), h = int(I.height()), d = int(I.depth()), wh = w*h;
    const int r = int(radius), r2 = r*r;
    int x0 = w, y0 = h, z0 = d, x1 = -1, y1 = -1, z1 = -1;

    for(size_t p = 0;p < src.size();++p)
        if(src[p])
        {
            int z = int(p)/wh, rem = int(p)%wh, y = rem/w, x = rem-y*w;
            x0 = std::min(x0,x); y0 = std::min(y0,y); z0 = std::min(z0,z);
            x1 = std::max(x1,x); y1 = std::max(y1,y); z1 = std::max(z1,z);
        }

    if(x1 < 0)
        return I;

    struct span_type{int dz,dy,dx,off;};
    std::vector<span_type> span;
    for(int dz = -r;dz <= r;++dz)
        for(int dy = -r;dy <= r;++dy)
            if(int rr = r2-dz*dz-dy*dy; rr >= 0)
                span.push_back({dz,dy,int(std::sqrt(float(rr))),dz*wh+dy*w});

    const int bx0 = dilate ? std::max(0,x0-r) : x0;
    const int by0 = dilate ? std::max(0,y0-r) : y0;
    const int bz0 = dilate ? std::max(0,z0-r) : z0;
    const int bx1 = dilate ? std::min(w-1,x1+r) : x1;
    const int by1 = dilate ? std::min(h-1,y1+r) : y1;
    const int bz1 = dilate ? std::min(d-1,z1+r) : z1;

    tipl::par_for(I.shape(),[&](const tipl::pixel_index<3>& p)
    {
        const int x = p[0], y = p[1], z = p[2];
        if(x < bx0 || x > bx1 || y < by0 || y > by1 || z < bz0 || z > bz1)
            return;

        const size_t pos = p.index();
        if constexpr(dilate)
        {
            if(src[pos])
                return;
        }
        else
        {
            if(!src[pos])
                return;
        }

        for(const auto& s : span)
        {
            int zz = z+s.dz, yy = y+s.dy;
            if(zz < 0 || yy < 0 || zz >= d || yy >= h)
            {
                if constexpr(!dilate)
                    I[pos] = value_type(0);
                if constexpr(!dilate)
                    break;
                continue;
            }

            if constexpr(!dilate)
                if(x-s.dx < 0 || x+s.dx >= w)
                {
                    I[pos] = value_type(0);
                    break;
                }

            int xx0 = dilate ? std::max(0,x-s.dx) : x-s.dx;
            int xx1 = dilate ? std::min(w-1,x+s.dx) : x+s.dx;
            auto p0 = src.begin() + pos + s.off + (xx0-x);
            auto p1 = src.begin() + pos + s.off + (xx1-x) + 1;

            bool hit = dilate ?
                           std::find_if(p0,p1,[](const auto& v){return v;}) != p1 :
                           std::find(p0,p1,value_type(0)) != p1;

            if(hit)
            {
                I[pos] = value_type(dilate);
                break;
            }
        }
    });
    return I;
}

template<typename ImageType>
ImageType& dilation_by_radius(ImageType& I,unsigned int radius)
{
    return morphology_by_radius<true>(I,radius);
}

template<typename ImageType>
ImageType& erosion_by_radius(ImageType& I,unsigned int radius)
{
    return morphology_by_radius<false>(I,radius);
}



template<typename ImageType,typename LabelType,typename ShiftType>
void edge(const ImageType& I,LabelType& act,const ShiftType& shift_list)
{
    auto shape = I.shape();
    size_t sz = I.size();
    act.resize(shape);
    for (int64_t shift : shift_list)
    {
        if (shift > 0)
        {
            auto iter1 = act.data() + shift;
            auto iter2 = I.data();
            auto iter3 = I.data()+shift;
            auto end = act.data() + sz;
            for (;iter1 < end;++iter1,++iter2,++iter3)
                if (*iter2 != *iter3)
                    *iter1 = 1;
        }
        else if (shift < 0)
        {
            auto iter1 = act.data();
            auto iter2 = I.data() - shift;
            auto iter3 = I.data();
            auto end = I.data() + sz;
            for (;iter2 < end;++iter1,++iter2,++iter3)
                if (*iter2 != *iter3)
                    *iter1 = 1;
        }
    }
}
template<typename ImageType,typename LabelType>
void edge(const ImageType& I,LabelType& act)
{
    neighbor_index_shift<ImageType::dimension> neighborhood(I.shape());
    edge(I,act,neighborhood.index_shift);
}
template<typename ImageType>
void edge(ImageType& I)
{
    ImageType out;
    edge(I,out);
    I = out;
}

template<typename ImageType>
void edge_thin(ImageType& I)
{
    ImageType out;
    neighbor_index_shift_narrow<ImageType::dimension> neighborhood(I.shape());
    neighborhood.index_shift.resize(neighborhood.index_shift.size()/2);
    edge(I,out,neighborhood.index_shift);
    I = out;
}
template<typename ImageType>
void edge_xy(ImageType& I)
{
    ImageType out;
    std::vector<int64_t> index_shift;
    index_shift.push_back(-1);
    index_shift.push_back(-int64_t(I.width()));
    edge(I,out,index_shift);
    I = out;
}

template<typename ImageType>
void edge_yz(ImageType& I)
{
    ImageType out;
    std::vector<int64_t> index_shift;
    index_shift.push_back(-int64_t(I.width()));
    index_shift.push_back(-int64_t(I.plane_size()));
    edge(I,out,index_shift);
    I = out;
}

template<typename ImageType>
void edge_xz(ImageType& I)
{
    ImageType out;
    std::vector<int64_t> index_shift;
    index_shift.push_back(-1);
    index_shift.push_back(-int64_t(I.plane_size()));
    edge(I,out,index_shift);
    I = out;
}

template<typename ImageType,typename LabelType>
void inner_edge(const ImageType& I,LabelType& act)
{
    auto shape = I.shape();
    size_t sz = I.size();
    act.resize(shape);
    neighbor_index_shift<ImageType::dimension> neighborhood(shape);
    for (int64_t shift : neighborhood.index_shift)
    {
        if (shift > 0)
        {
            auto iter1 = act.data() + shift;
            auto iter2 = I.data();
            auto iter3 = I.data()+shift;
            auto end = act.data() + sz;
            for (;iter1 < end;++iter1,++iter2,++iter3)
                if (*iter2 < *iter3)
                    *iter1 = 1;
        }
        else if (shift < 0)
        {
            auto iter1 = act.data();
            auto iter2 = I.data() - shift;
            auto iter3 = I.data();
            auto end = I.data() + sz;
            for (;iter2 < end;++iter1,++iter2,++iter3)
                if (*iter2 < *iter3)
                    *iter1 = 1;
        }
    }
}

template<typename ImageType>
void inner_edge(ImageType& I)
{
    ImageType out;
    inner_edge(I,out);
    I = out;
}

template<typename ImageType>
bool is_edge(ImageType& I,tipl::pixel_index<2> index)
{
    size_t idx = index.index();
    typename ImageType::value_type center = I[idx];
    unsigned int w = I.width();
    unsigned int h = I.height();
    bool have_left = index.x() >= 1;
    bool have_right = index.x()+1 < w;
    if (index.y() >= 1)
    {
        size_t base_index = idx - w;
        if ((have_left && I[base_index-1] != center) ||
             I[base_index] != center ||
            (have_right && I[base_index+1] != center))
            return true;
    }

    if ((have_left && I[idx-1] != center) ||
        (have_right && I[idx+1] != center))
        return true;

    if (index.y()+1 < h)
    {
        size_t base_index = idx + w;
        if ((have_left && I[base_index-1] != center) ||
             I[base_index] != center ||
            (have_right && I[base_index+1] != center))
            return true;
    }
    return false;
}

template<typename ImageType,typename std::enable_if<ImageType::dimension==3,bool>::type = true>
bool is_edge(ImageType& I,tipl::pixel_index<ImageType::dimension> index)
{
    size_t idx = index.index();
    typename ImageType::value_type center = I[idx];
    size_t z_offset = I.plane_size();
    size_t y_offset = I.width();

    bool have_left = index.x() >= 1;
    bool have_right = index.x()+1 < I.width();
    bool has_top = index.y() >= 1;
    bool has_bottom = index.y()+1 < I.height();

    if (index.z() >= 1)
    {
        if (has_top)
        {
            if ((have_left  && I[idx-1-y_offset-z_offset] != center) ||
                    I[idx  -y_offset-z_offset] != center  ||
                    (have_right && I[idx+1-y_offset-z_offset] != center))
                return true;
        }
        if ((have_left  && I[idx-1-z_offset] != center) ||
                I[idx  -z_offset] != center  ||
                (have_right && I[idx+1-z_offset] != center))
            return true;

        if (has_bottom)
        {
            if ((have_left  && I[idx-1+y_offset-z_offset] != center) ||
                    I[idx  +y_offset-z_offset] != center  ||
                    (have_right && I[idx+1+y_offset-z_offset] != center))
                return true;
        }
    }
    {
        if (has_top)
        {
            if ((have_left  && I[idx-1-y_offset] != center) ||
                    I[idx  -y_offset] != center  ||
                    (have_right && I[idx+1-y_offset] != center))
                return true;
        }
        {
            if ((have_left  && I[idx-1] != center) ||
                    (have_right && I[idx+1] != center))
                return true;
        }
        if (has_bottom)
        {
            if ((have_left  && I[idx-1+y_offset] != center) ||
                    I[idx  +y_offset] != center  ||
                    (have_right && I[idx+1+y_offset] != center))
                return true;
        }
    }
    if (index.z()+1 < I.depth())
    {
        if (has_top)
        {
            if ((have_left  && I[idx-1-y_offset+z_offset] != center) ||
                    I[idx  -y_offset+z_offset] != center  ||
                    (have_right && I[idx+1-y_offset+z_offset] != center))
                return true;
        }

        if ((have_left  && I[idx-1+z_offset] != center) ||
                I[idx  +z_offset] != center  ||
                (have_right && I[idx+1+z_offset] != center))
            return true;

        if (has_bottom)
        {
            if ((have_left  && I[idx-1+y_offset+z_offset] != center) ||
                    I[idx  +y_offset+z_offset] != center  ||
                    (have_right && I[idx+1+y_offset+z_offset] != center))
                return true;
        }
    }
    return false;
}

template<typename ImageType>
auto get_neighbor_count_multiple_region(const ImageType& I)
{
    size_t sz = I.size();
    std::vector<std::unordered_map<int, char>> region_count(sz);
    neighbor_index_shift<ImageType::dimension> neighborhood(I.shape());
    tipl::par_for<sequential>(sz,[&](int64_t index)
    {
        for(int64_t pos : neighborhood.index_shift)
        {
            pos += index;
            if(pos < 0 || pos >= static_cast<int64_t>(sz))
                continue;
            ++region_count[index][I[pos]];
        }
    });
    return region_count;
}

template<typename ImageType>
auto get_neighbor_count(const ImageType& I)
{
    auto shape = I.shape();
    std::vector<char> count(I.size());
    tipl::par_for<sequential>(shape,[&](const auto& p)
    {
        char n = (I[p.index()] != 0);
        tipl::for_each_neighbors(p,shape,[&](const auto& q){n += I[q.index()] != 0;});
        count[p.index()] = n;
    });
    return count;
}

template<bool close,typename ImageType>
size_t opening_closing(ImageType& I,char threshold_shift)
{
    auto shape = I.shape();
    auto act = get_neighbor_count(I);
    const char threshold = (((ImageType::dimension == 2) ? 9 : 27) >> 1) + threshold_shift;
    using index_type = tipl::pixel_index<ImageType::dimension>;
    std::vector<index_type> cur,next,test;
    std::vector<char> queued(I.size());
    size_t total = 0;

    auto hit = [&](size_t i)
    {
        if constexpr(close) return !I[i] && act[i] > threshold;
        else                return  I[i] && act[i] < threshold;
    };

    for(index_type p(shape);p < I.size();++p)
        if(hit(p.index()))
            cur.push_back(p);

    while(!cur.empty())
    {
        for(const auto& p : cur)
            I[p.index()] = close;
        total += cur.size();

        test.clear();
        for(const auto& p : cur)
            tipl::for_each_neighbors(p,shape,[&](const auto& q)
            {
                size_t i = q.index();
                if constexpr(close)
                {
                    if(I[i]) return;
                    ++act[i];
                }
                else
                {
                    if(!I[i]) return;
                    --act[i];
                }
                if(!queued[i])
                    queued[i] = 1,test.push_back(q);
            });

        next.clear();
        for(const auto& p : test)
            if(queued[p.index()] = 0; hit(p.index()))
                next.push_back(p);
        cur.swap(next);
    }
    return total;
}

template<typename ImageType>
ImageType& closing(ImageType& I)
{
    for(char t = ((ImageType::dimension == 2) ? 4 : 8);t >= 0;--t)
        if(size_t c = opening_closing<true>(I,t))
            return I;
    return I;
}

template<typename ImageType>
ImageType& opening(ImageType& I)
{
    for(char t = ((ImageType::dimension == 2) ? 4 : 8);t >= 0;--t)
        if(size_t c = opening_closing<false>(I,-t))
            return I;
    return I;
}

template<typename ImageType>
ImageType& negate(ImageType& I)
{
    size_t sz = I.size();
    for (size_t index = 0; index < sz; ++index)
        I[index] = I[index] ? 0:1;
    return I;
}

template<typename ImageType>
ImageType& smoothing(ImageType& I)
{
    auto act = get_neighbor_count(I);
    constexpr char threshold = ((ImageType::dimension == 2) ? 9 : 27) >> 1;
    size_t sz = I.size();
    for (size_t index = 0; index < sz; ++index)
    {
        if (act[index] > threshold)
            I[index] = 1;
        if (act[index] < threshold)
            I[index] = 0;
    }
    return I;
}

template<typename ImageType>
void smoothing_multiple_region(ImageType& I)
{
    auto region_count = get_neighbor_count_multiple_region(I);
    constexpr char threshold = ((ImageType::dimension == 2) ? 9 : 27) >> 1;
    size_t sz = I.size();

    tipl::par_for<sequential>(sz,[&](size_t index)
    {
        int max_count = 0;
        int dominant_region = I[index];
        for (const auto& [region, count] : region_count[index])
        {
            if (count > max_count)
            {
                max_count = count;
                dominant_region = region;
            }
        }
        if (max_count > threshold)
            I[index] = dominant_region;
    });
}

template<typename ImageType,typename RefImageType>
void fit(ImageType& I,const RefImageType& ref,char weight = 2)
{
    auto act = get_neighbor_count(I);
    size_t sz = I.size();
    constexpr char threshold = ((ImageType::dimension == 2) ? 9 : 27) >> 1;
    char upper_threshold = threshold+ImageType::dimension+ImageType::dimension;
    char lower_threshold = threshold-ImageType::dimension-ImageType::dimension;
    tipl::par_for(I.shape(),[&](auto pos)
    {
        if(act[pos.index()] < lower_threshold || act[pos.index()] > upper_threshold)
            return;
        typename ImageType::value_type Iv[get_window_size<1,ImageType::dimension>::value];
        float refv[get_window_size<1,RefImageType::dimension>::value];
        get_window_at_width<1>(pos,I,Iv);
        auto size = get_window_at_width<1>(pos,ref,refv);
        for(size_t i = 1; i < size; ++i)
            refv[i] = std::fabs(refv[i]-refv[0]);
        std::vector<unsigned int> idx(size-1);
        std::iota(idx.begin(), idx.end(), 1);
        std::sort(idx.begin(), idx.end(),[&](size_t i1, size_t i2){return refv[i1] < refv[i2];});
        size >>= 1;
        for(size_t i = 1; i < size; ++i)
            if(Iv[idx[i]])
                act[pos.index()] += weight;
            else
                act[pos.index()] -= weight;
    },4);

    for (size_t index = 0; index < sz; ++index)
    {
        if (act[index] > threshold)
            I[index] = 1;
        if (act[index] < threshold)
            I[index] = 0;
    }
}

template<typename ImageType>
bool smoothing_fill(ImageType& I)
{
    bool filled = false;
    auto act = get_neighbor_count(I);
    size_t sz = I.size();
    constexpr char threshold = ((ImageType::dimension == 2) ? 9 : 27) >> 1;
    for (size_t index = 0; index < sz; ++index)
        if (act[index] > threshold)
        {
            if (!I[index])
            {
                I[index] = 1;
                filled = true;
            }
        }
    return filled;
}

template<typename ImageType>
void recursive_smoothing(ImageType& I,unsigned int max_iteration = 100)
{
    size_t sz = I.size();
    for(unsigned int iter = 0; iter < max_iteration; ++iter)
    {
        bool has_change = false;
        auto act = get_neighbor_count(I);
        constexpr char threshold = ((ImageType::dimension == 2) ? 9 : 27) >> 1;
        for (size_t index = 0; index < sz; ++index)
        {
            if (act[index] > threshold)
            {
                if (!I[index])
                {
                    I[index] = 1;
                    has_change = true;
                }
            }
            if (act[index] < threshold)
            {
                if (I[index])
                {
                    I[index] = 0;
                    has_change = true;
                }
            }
        }
        if(!has_change)
            break;
    }
}

template<typename ImageType,typename IndexType,typename GrowFunc>
void region_growing(const ImageType& I,const IndexType& seed_point,
                    std::vector<IndexType>& grown_region,GrowFunc grow)
{
    auto shape = I.shape();
    std::vector<unsigned char> label_map(I.size());
    std::vector<IndexType> seeds;
    seeds.push_back(seed_point);
    label_map[seed_point.index()] = 1;
    for (unsigned int index = 0; index < seeds.size(); ++index)
    {
        IndexType active_point = seeds[index];
        for_each_neighbors(active_point,shape,[&](const auto& pos)
        {
            auto cur_neighbor_index = pos.index();
            if (label_map[cur_neighbor_index])
                return;
            if (grow(I[active_point.index()],I[cur_neighbor_index]))
            {
                seeds.push_back(pos);
                label_map[cur_neighbor_index] = 1;
            }
        });
    }
    seeds.swap(grown_region);
}

template<typename ImageType>
void convex_xy(ImageType& I)
{
    tipl::shape<ImageType::dimension> range_min,range_max;
    bounding_box(I,range_min,range_max);

    auto shape = I.shape();
    size_t sz = I.size();
    size_t w = shape[0];

    int dirs[8][2] = {{1,0},{2,1},{1,1},{1,2},{0,1},{-1,2},{-1,1},{-2,1}};
    std::vector<unsigned int> fill_buf;
    for(unsigned int i = 0; i < 8; ++i)
    {
        int shift = dirs[i][0] + w*dirs[i][1];
        if(shift <= 0)
            continue;
        std::vector<unsigned char> label(sz);
        for(pixel_index<ImageType::dimension> index(shape);
            index.is_valid(shape);
            index.next(shape))
        {
            size_t idx = index.index();
            if(index[0] < range_min[0] || index[0] >= range_max[0] ||
               index[1] < range_min[1] || index[1] >= range_max[1] ||
                    label[idx])
                continue;

            bool has_first = false;
            fill_buf.clear();
            for(pixel_index<ImageType::dimension> index2(index);
                index2.is_valid(shape);)
            {
                size_t idx2 = index2.index();
                label[idx2] = 1; // FIX: Mark trace as visited to prevent redundant work
                if(I[idx2])
                {
                    if(!has_first)
                        has_first = true;
                    else
                    {
                        for(unsigned int fill_idx : fill_buf)
                            I[fill_idx] = 1;
                        fill_buf.clear();
                    }
                }
                else
                {
                    if(has_first)
                        fill_buf.push_back(idx2);
                }
                index2[0] += dirs[i][0];
                index2[1] += dirs[i][1];
                index2.index() += shift;
                if(index2[0] < range_min[0] || index2[0] >= range_max[0] ||
                   index2[1] < range_min[1] || index2[1] >= range_max[1])
                    break;
            }
        }
    }
}

template<typename ImageType>
void convex_x(ImageType& I,typename ImageType::value_type assign_value = 1)
{
    auto iter = I.begin();
    auto end = iter + I.size();
    size_t w = I.width();
    while(iter != end)
    {
        auto next_iter = iter + w;
        auto first = next_iter;
        auto last = next_iter;
        for(;iter != next_iter;++iter)
            if(*iter > 0)
            {
                if(first == next_iter)
                    first = iter;
                else
                    last = iter;
            }
        if(first != next_iter && last != next_iter)
            std::fill(first,last,assign_value);
    }
}

template<typename ImageType>
void convex_y(ImageType& I)
{
    size_t plane_size = I.plane_size();
    size_t w = I.width();
    size_t sz = I.size();
    for(size_t iter_plane = 0; iter_plane < sz; iter_plane += plane_size)
    {
        for(size_t iter_x = iter_plane,iter_x_end = iter_x + w
                ;iter_x < iter_x_end;++iter_x)
        {
            int64_t iter_y = iter_x;
            int64_t iter_y_end = iter_y+(plane_size-w);
            int64_t first=0,last=0;
            int64_t find_count = 0;
            for(;iter_y <= iter_y_end;iter_y += w)
                if(I[iter_y] > 0)
                {
                    ++find_count;
                    if(find_count == 1)
                        first = iter_y;
                    else
                        last = iter_y;
                }
            if(find_count >= 2)
            {
                for(first += w;first != last;first += w)
                    I[first] = 1;
            }
        }
    }
}


template<typename LabelImageType>
void get_region_bounding_box(const LabelImageType& labels,
                             const std::vector<std::vector<size_t> >& regions,
                             std::vector<tipl::vector<2,int> >& min_pos,
                             std::vector<tipl::vector<2,int> >& max_pos)
{
    min_pos.clear();
    min_pos.resize(regions.size());
    max_pos.clear();
    max_pos.resize(regions.size());
    auto shape = labels.shape();
    size_t sz = labels.size();
    std::fill(min_pos.begin(),min_pos.end(),tipl::vector<2,float>(shape[0],shape[1]));
    for(tipl::pixel_index<2> index(shape); index < sz; ++index)
        if (labels[index.index()])
        {
            size_t region_id = labels[index.index()]-1;
            if (regions[region_id].empty())
                continue;
            max_pos[region_id][0] = std::max<int>(index[0],max_pos[region_id][0]);
            max_pos[region_id][1] = std::max<int>(index[1],max_pos[region_id][1]);
            min_pos[region_id][0] = std::min<int>(index[0],min_pos[region_id][0]);
            min_pos[region_id][1] = std::min<int>(index[1],min_pos[region_id][1]);
        }
}

template<typename LabelImageType>
void get_region_bounding_size(const LabelImageType& labels,
                              const std::vector<std::vector<size_t> >& regions,
                              std::vector<int>& size_x,
                              std::vector<int>& size_y)
{
    std::vector<tipl::vector<2,int> > max_pos,min_pos;
    tipl::morphology::get_region_bounding_box(labels,regions,min_pos,max_pos);
    size_x.clear();
    size_x.resize(regions.size());
    size_y.clear();
    size_y.resize(regions.size());

    for(size_t index = 0; index < regions.size(); ++index)
        if(!regions[index].empty())
        {
            size_x[index] = max_pos[index][0]-min_pos[index][0];
            size_y[index] = max_pos[index][1]-min_pos[index][1];
        }
}

template<typename LabelImageType>
void get_region_center(const LabelImageType& labels,
                       const std::vector<std::vector<int64_t> >& regions,
                       std::vector<tipl::vector<2,float> >& center_of_mass)
{
    center_of_mass.clear();
    center_of_mass.resize(regions.size());
    auto shape = labels.shape();
    size_t sz = labels.size();
    for(tipl::pixel_index<2> index(shape); index < sz; ++index)
        if (labels[index.index()])
        {
            size_t region_id = labels[index.index()]-1;
            if (regions[region_id].empty())
                continue;
            center_of_mass[region_id] += tipl::vector<2,float>(index);
        }

    for(size_t index = 0; index < regions.size(); ++index)
        if(!regions[index].empty())
            center_of_mass[index] /= regions[index].size();
}

namespace detail
{

struct component_run
{
    size_t first,last,parent,size;
};

struct component_data
{
    std::vector<component_run> runs;
    std::vector<size_t> roots;
};

template<bool zero = false,typename ImageType>
component_data connected_component_runs(const ImageType& I)
{
    component_data data;
    if(I.empty())
        return data;

    auto active = [&](size_t i)
    {
        if constexpr(zero)
            return !I[i];
        else
            return bool(I[i]);
    };

    const size_t w = I.width(),line_count = I.size()/w;
    size_t h = 1,d = 1;
    if constexpr(ImageType::dimension >= 2) h = I.height();
    if constexpr(ImageType::dimension >= 3) d = I.depth();

    std::vector<size_t> line_pos(line_count+1),run_pos(line_count+1);
    for(size_t line = 0;line < line_count;++line)
        line_pos[line+1] = line_pos[line]+w;

    serial_or_parallel(I.size(),line_count,[&](size_t line)
    {
        size_t p = line_pos[line],end = line_pos[line+1],count = 0;
        while(p < end)
        {
            while(p < end && !active(p))
                ++p;
            if(p == end)
                break;
            ++count;
            while(p < end && active(p))
                ++p;
        }
        run_pos[line+1] = count;
    });

    for(size_t line = 0;line < line_count;++line)
        run_pos[line+1] += run_pos[line];

    data.runs.resize(run_pos.back());
    serial_or_parallel(I.size(),line_count,[&](size_t line)
    {
        size_t p = line_pos[line],end = line_pos[line+1];
        size_t run = run_pos[line];

        while(p < end)
        {
            while(p < end && !active(p))
                ++p;
            if(p == end)
                break;

            size_t first = p;
            while(p < end && active(p))
                ++p;

            data.runs[run] = {first,p,run,p-first};
            ++run;
        }
    });

    auto root = [&](size_t i)
    {
        while(data.runs[i].parent != i)
        {
            data.runs[i].parent =
                data.runs[data.runs[i].parent].parent;
            i = data.runs[i].parent;
        }
        return i;
    };

    auto join = [&](size_t a,size_t b)
    {
        a = root(a);
        b = root(b);
        if(a == b)
            return;

        if(data.runs[a].size < data.runs[b].size)
            std::swap(a,b);

        data.runs[b].parent = a;
        data.runs[a].size += data.runs[b].size;
    };

    auto join_lines = [&](size_t line1,size_t line2)
    {
        size_t i = run_pos[line1],i_end = run_pos[line1+1];
        size_t j = run_pos[line2],j_end = run_pos[line2+1];
        const size_t base1 = line_pos[line1],base2 = line_pos[line2];

        while(i < i_end && j < j_end)
        {
            size_t a0 = data.runs[i].first-base1;
            size_t a1 = data.runs[i].last-base1;
            size_t b0 = data.runs[j].first-base2;
            size_t b1 = data.runs[j].last-base2;

            if(a0 < b1 && b0 < a1)
                join(i,j);

            if(a1 <= b1)
                ++i;
            if(b1 <= a1)
                ++j;
        }
    };

    if constexpr(ImageType::dimension >= 2)
    {
        size_t slice = 0;
        for(size_t z = 0;z < d;++z,slice += h)
            for(size_t line = slice+1,end = slice+h;line < end;++line)
                join_lines(line,line-1);
    }

    if constexpr(ImageType::dimension >= 3)
        for(size_t line = h;line < line_count;++line)
            join_lines(line,line-h);

    for(size_t i = 0;i < data.runs.size();++i)
        data.runs[i].parent = root(i);

    for(size_t i = 0;i < data.runs.size();++i)
        if(data.runs[i].parent == i)
            data.roots.push_back(i);

    return data;
}

template<typename ImageType>
auto component_regions(const ImageType& I,const component_data& data)
{
    std::vector<std::vector<size_t> > regions(data.roots.size());
    if(data.roots.empty())
        return regions;

    constexpr size_t invalid = size_t(-1);
    std::vector<size_t> id(data.runs.size(),invalid),
        offset(data.runs.size()),
        used(data.roots.size());

    for(size_t i = 0;i < data.roots.size();++i)
    {
        size_t root = data.roots[i];
        id[root] = i;
        regions[i].resize(data.runs[root].size);
    }

    for(size_t i = 0;i < data.runs.size();++i)
    {
        size_t region = id[data.runs[i].parent];
        offset[i] = used[region];
        used[region] += data.runs[i].last-data.runs[i].first;
    }

    serial_or_parallel(I.size(),data.runs.size(),[&](size_t i)
                  {
                      const auto& run = data.runs[i];
                      auto& region = regions[id[run.parent]];
                      std::iota(region.begin()+offset[i],
                                region.begin()+offset[i]+run.last-run.first,
                                run.first);
                  });
    return regions;
}

} // namespace detail

template<typename ImageType,typename LabelImageType>
void connected_component_labeling(
    const ImageType& I,
    LabelImageType& labels,
    std::vector<std::vector<size_t> >& regions)
{
    auto data = detail::connected_component_runs(I);
    regions = detail::component_regions(I,data);

    labels.resize(I.shape());
    labels = 0;

    using label_type = typename LabelImageType::value_type;
    std::vector<size_t> id(data.runs.size(),size_t(-1));

    for(size_t i = 0;i < data.roots.size();++i)
        id[data.roots[i]] = i;

    tipl::serial_or_parallel(I.size(),data.runs.size(),[&](size_t i)
    {
        const auto& run = data.runs[i];
        std::fill(labels.begin()+run.first,labels.begin()+run.last,label_type(id[run.parent]+1));
    });
}

template<typename ImageType>
auto connected_component_labeling(const ImageType& I)
{
    auto data = detail::connected_component_runs(I);
    return detail::component_regions(I,data);
}

namespace detail
{

template<bool zero,typename ImageType>
void keep_largest(ImageType& I,const component_data& data)
{
    if(data.roots.size() < 2)
        return;

    size_t keep = data.roots[0];
    for(size_t root : data.roots)
        if(data.runs[root].size > data.runs[keep].size)
            keep = root;

    serial_or_parallel(I.size(),data.runs.size(),[&](size_t i)
    {
        const auto& run = data.runs[i];
        if(run.parent != keep)
            std::fill(I.begin()+run.first,I.begin()+run.last,typename ImageType::value_type(zero));
    });
}

template<bool zero,typename ImageType>
ImageType& defragment_impl(ImageType& I)
{
    keep_largest<zero>(I,connected_component_runs<zero>(I));
    return I;
}

}

template<typename ImageType>
inline ImageType& defragment(ImageType& I)
{
    return detail::defragment_impl<false>(I);
}

template<typename ImageType>
inline ImageType& fill_holes(ImageType& I)
{
    return detail::defragment_impl<true>(I);
}

template<typename ImageType>
ImageType& defragment_and_fill_holes(ImageType& I)
{
    if(I.empty())
        return I;

    detail::component_data foreground,background;

    if(I.size() < 1024*1024 || max_thread_count < 2)
    {
        foreground = detail::connected_component_runs(I);
        background = detail::connected_component_runs<true>(I);
        detail::keep_largest<false>(I,foreground);
        detail::keep_largest<true>(I,background);
        return I;
    }

    auto threads = max_thread_count;
    max_thread_count = std::max<size_t>(1,threads >> 1);

    std::thread worker([&]{foreground = detail::connected_component_runs(I);});
    background = detail::connected_component_runs<true>(I);
    worker.join();

    worker = std::thread([&]{detail::keep_largest<false>(I,foreground);});
    detail::keep_largest<true>(I,background);
    worker.join();

    max_thread_count = threads;
    return I;
}

template<typename ImageType>
inline ImageType& dndnco(ImageType& mask)
{
    return opening(closing(defragment_and_fill_holes(mask)));
}
template<typename ImageType>
void fill_holes_by_slice(ImageType& I)
{
    tipl::par_for(I.depth(),[&](size_t z)
    {
        auto slice = I.slice_at(z);
        auto data = detail::connected_component_runs<true>(slice);
        std::vector<unsigned char> outside(data.runs.size());

        const size_t w = slice.width(),sz = slice.size();
        for(const auto& run : data.runs)
            if(run.first < w || run.first >= sz-w ||
                run.first%w == 0 || run.last%w == 0)
                outside[run.parent] = 1;

        for(const auto& run : data.runs)
            if(!outside[run.parent])
                std::fill(slice.begin()+run.first,slice.begin()+run.last,
                          typename ImageType::value_type(1));
    });
}

template<typename ImageType>
ImageType& defragment_by_size_ratio(ImageType& I,float ratio = 0.05f)
{
    auto data = detail::connected_component_runs(I);
    size_t max_size = 0;
    for(size_t root : data.roots)
        max_size = std::max(max_size,data.runs[root].size);

    const size_t threshold = size_t(double(max_size)*ratio);
    tipl::serial_or_parallel(I.size(),data.runs.size(),[&](size_t i)
    {
        const auto& run = data.runs[i];
        if(data.runs[run.parent].size <= threshold)
            std::fill(I.begin()+run.first,I.begin()+run.last,typename ImageType::value_type());
    });
    return I;
}

template<typename ImageType,typename PixelIndexType,typename ValueType>
void fill(ImageType& I,PixelIndexType seed_point,ValueType new_value)
{
    auto shape = I.shape();
    std::deque<PixelIndexType> seeds;
    seeds.push_back(seed_point);
    ValueType old_value = I[seed_point.index()];
    I[seed_point.index()] = new_value;
    while (!seeds.empty())
    {
        PixelIndexType active_point = seeds.front();
        seeds.pop_front();
        for_each_neighbors(active_point,shape,[&](const auto& pos)
        {
            if (I[pos.index()] != old_value)
                return;
            seeds.push_back(pos);
            I[pos.index()] = new_value;
        });
    }
}

template<typename out_type,typename mask_type,typename image_type>
void fill_and_smooth_labels(const mask_type& mask,image_type& atlas_I,size_t max_growing_iteration = 64,size_t max_smoothing_iteration = 12)
{
    size_t sz = mask.size();
    auto shape = mask.shape();
    std::vector<tipl::pixel_index<3>> missing_voxel;
    for(size_t pos = 0; pos < sz; ++pos)
        if(mask[pos] && atlas_I[pos] == 0)
            missing_voxel.push_back(tipl::pixel_index<3>(pos,shape));

    std::string grow_report;
    std::vector<unsigned short> updates;

    for(size_t iter = 0; iter < max_growing_iteration; ++iter)
    {
        if(missing_voxel.empty())
            break;

        size_t missing_sz = missing_voxel.size();
        if constexpr(!std::is_same_v<out_type,void>)
        {
            if(!grow_report.empty())
                grow_report += ", ";
            grow_report += std::to_string(missing_sz);
        }

        updates.assign(missing_sz,0);

        tipl::par_for(missing_sz,[&](size_t idx)
        {
            auto pos = missing_voxel[idx];
            std::pair<unsigned short,size_t> candidates[27];
            size_t cand_count = 0;

            tipl::for_each_neighbors(pos,shape,[&](const tipl::pixel_index<3>& n_pos)
            {
                size_t n_idx = n_pos.index();
                if(mask[n_idx] && atlas_I[n_idx] > 0)
                {
                    auto val = atlas_I[n_idx];
                    size_t c = 0;
                    for(;c < cand_count;++c)
                        if(candidates[c].first == val)
                        {
                            candidates[c].second++;
                            break;
                        }

                    if(c == cand_count)
                        candidates[cand_count++] = {val,1};
                }
            });

            if(cand_count > 0)
            {
                size_t best_idx = 0;
                for(size_t c = 1; c < cand_count; ++c)
                    if(candidates[c].second > candidates[best_idx].second)
                        best_idx = c;
                updates[idx] = candidates[best_idx].first;
            }
        });

        std::vector<tipl::pixel_index<3>> next_missing;
        next_missing.reserve(missing_sz);
        bool changed = false;

        for(size_t idx = 0; idx < missing_sz; ++idx)
            if(updates[idx] > 0)
            {
                atlas_I[missing_voxel[idx].index()] = updates[idx];
                changed = true;
            }
            else
                next_missing.push_back(missing_voxel[idx]);

        if(!changed)
            break;

        missing_voxel.swap(next_missing);
    }

    if constexpr(!std::is_same_v<out_type,void>)
        if(!grow_report.empty())
            out_type() << "growing iterations (missing voxels): " << grow_report;

    std::vector<size_t> tissue_voxels;
    for(size_t pos = 0; pos < sz; ++pos)
        if(mask[pos] && atlas_I[pos] > 0)
            tissue_voxels.push_back(pos);

    std::string smooth_report;
    for(size_t iter = 0; iter < max_smoothing_iteration; ++iter)
    {
        size_t tissue_sz = tissue_voxels.size();
        updates.assign(tissue_sz,0);

        tipl::par_for(tissue_sz,[&](size_t idx)
        {
            tipl::pixel_index<3> pos(tissue_voxels[idx],shape);
            unsigned short current_id = atlas_I[pos.index()];
            std::pair<unsigned short,size_t> candidates[27];
            size_t cand_count = 0;
            size_t valid_neighbors = 0;

            tipl::for_each_neighbors(pos,shape,[&](const tipl::pixel_index<3>& n_pos)
            {
                size_t n_idx = n_pos.index();
                if(mask[n_idx] && atlas_I[n_idx] > 0)
                {
                    ++valid_neighbors;
                    auto val = atlas_I[n_idx];
                    size_t c = 0;
                    for(;c < cand_count;++c)
                        if(candidates[c].first == val)
                        {
                            candidates[c].second++;
                            break;
                        }

                    if(c == cand_count)
                        candidates[cand_count++] = {val,1};
                }
            });

            if(valid_neighbors > 0)
            {
                size_t best_idx = 0;
                for(size_t c = 1; c < cand_count; ++c)
                    if(candidates[c].second > candidates[best_idx].second)
                        best_idx = c;

                if(candidates[best_idx].first != current_id && candidates[best_idx].second*2 > valid_neighbors)
                    updates[idx] = candidates[best_idx].first;
            }
        });

        size_t flipped_count = 0;
        for(size_t idx = 0; idx < tissue_sz; ++idx)
            if(updates[idx] > 0)
            {
                atlas_I[tissue_voxels[idx]] = updates[idx];
                ++flipped_count;
            }

        if constexpr(!std::is_same_v<out_type,void>)
        {
            if(!smooth_report.empty())
                smooth_report += ", ";
            smooth_report += std::to_string(flipped_count);
        }

        if(flipped_count == 0)
            break;
    }

    if constexpr(!std::is_same_v<out_type,void>)
        if(!smooth_report.empty())
            out_type() << "smoothing iterations (voxels flipped): " << smooth_report;
}

/*
Reclassify edge voxels using reference-image likelihood and a local spatial prior.

current_weight controls how hard it is to flip the existing center label:
    4.0f   aggressive
    8.0f   default
    12.0f  conservative

Internal spatial weights:

spatial_weight controls spatial prior relative to signal likelihood:

current label:       from 12 decreased to 4
6-neighbor support:  2.0
20-neighbor support: 0.5
pseudo-count:        0.1

Only labels found in the 6-connected neighbors are considered candidates.
The remaining 20 neighbors only adjust the prior of existing candidates.
*/


template<typename label_image_type,typename ref_image_type>
size_t refine_label(label_image_type& label,const ref_image_type& ref,float final_weight = 5.0f)
{
    if(label.shape() != ref.shape())
        return 0;

    float spatial_weight = 4.0f;
    unsigned int width = 4;
    if(label.plane_size() <= 256*256) spatial_weight = 2.0f,width = 2;
    if(label.plane_size() <= 128*128) spatial_weight = 1.0f,width = 1;

    using label_type = typename label_image_type::value_type;
    auto shape = label.shape();
    constexpr double eps = 1.0e-6,fw = 1.5,dw = 0.5,sw = 0.1;
    tipl::neighbor_index_shift_narrow<3> shift(shape);
    tipl::image<3,unsigned char> edge_mask(shape);
    std::vector<size_t> edge_voxels;
    std::vector<label_type> next;
    size_t total = 0;
    float current_weight = 12.0f;

    for(unsigned int iter = 0;iter < 100;)
    {
        edge_mask = 0;
        tipl::morphology::edge(label,edge_mask,shift.index_shift);

        edge_voxels.clear();
        for(size_t i = 0;i < label.size();++i)
            if(edge_mask[i] && label[i])
                edge_voxels.push_back(i);
        if(edge_voxels.empty())
            break;

        double cw = std::max<double>(current_weight,sw);
        next.resize(edge_voxels.size());

        tipl::par_for(edge_voxels.size(),[&](size_t j)
        {
            tipl::pixel_index<3> pos(edge_voxels[j],shape);
            size_t index = pos.index(),cand_count = 0;
            label_type cur = label[index],cand[6] = {},best_label = cur;
            double pc[6] = {};

            tipl::for_each_connected_neighbors(pos,shape,[&](const auto& n_pos)
            {
                label_type v = label[n_pos.index()];
                if(!v)
                    return;
                auto p = std::find(cand,cand+cand_count,v);
                if(p == cand+cand_count)
                    cand[cand_count] = v,pc[cand_count] = v == cur ? cw : sw,p = cand+cand_count++;
                pc[p-cand] += fw;
            });

            if(cand_count < 2)
                return next[j] = cand_count ? cand[0] : cur,void();

            tipl::for_each_neighbors(pos,shape,[&](const auto& n_pos)
            {
                label_type v = label[n_pos.index()];
                if(!v)
                    return;
                auto p = std::find(cand,cand+cand_count,v);
                if(p != cand+cand_count)
                    pc[p-cand] += dw;
            });

            auto lw = tipl::get_window(pos,label,width);
            auto rw = tipl::get_window(pos,ref,width);
            double s[6] = {},s2[6] = {},nv[6] = {};
            for(size_t i = 0;i < rw.size();++i)
            {
                auto p = std::find(cand,cand+cand_count,lw[i]);
                if(p == cand+cand_count)
                    continue;
                double v = rw[i];
                size_t c = p-cand;
                s[c] += v;
                s2[c] += v*v;
                ++nv[c];
            }

            double x = ref[index],sum_pc = std::accumulate(pc,pc+cand_count,0.0);
            double best = -std::numeric_limits<double>::infinity();

            for(size_t c = 0;c < cand_count;++c)
            {
                if(nv[c] < 2.0)
                    continue;
                double mean = s[c]/nv[c],var = std::max(s2[c]/nv[c]-mean*mean,eps);
                double score = spatial_weight*std::log(pc[c]/sum_pc)
                               - 0.5*std::log(var)
                               - 0.5*(x-mean)*(x-mean)/var;
                if(score > best)
                    best = score,best_label = cand[c];
            }
            next[j] = best_label;
        });

        size_t changed = 0;
        for(size_t j = 0;j < edge_voxels.size();++j)
            if(next[j] != label[edge_voxels[j]])
                label[edge_voxels[j]] = next[j],++changed;

        if(!changed)
        {
            current_weight -= 1.0f;
            if(current_weight < final_weight)
                break;
            iter = 0;
        }
        total += changed;
        ++iter;
    }
    return total;
}

template<typename ImageType>
std::enable_if_t<!std::is_lvalue_reference_v<ImageType>,ImageType&&>
defragment(ImageType&& I){defragment(static_cast<ImageType&>(I));return std::move(I);}

template<typename ImageType>
std::enable_if_t<!std::is_lvalue_reference_v<ImageType>,ImageType&&>
fill_holes(ImageType&& I){fill_holes(static_cast<ImageType&>(I));return std::move(I);}

template<typename ImageType>
std::enable_if_t<!std::is_lvalue_reference_v<ImageType>,ImageType&&>
defragment_and_fill_holes(ImageType&& I){defragment_and_fill_holes(static_cast<ImageType&>(I));return std::move(I);}

template<typename ImageType>
std::enable_if_t<!std::is_lvalue_reference_v<ImageType>,ImageType&&>
smoothing(ImageType&& I){smoothing(static_cast<ImageType&>(I));return std::move(I);}

template<typename ImageType>
std::enable_if_t<!std::is_lvalue_reference_v<ImageType>,ImageType&&>
dilation(ImageType&& I){dilation(static_cast<ImageType&>(I));return std::move(I);}

template<typename ImageType>
std::enable_if_t<!std::is_lvalue_reference_v<ImageType>,ImageType&&>
erosion(ImageType&& I){erosion(static_cast<ImageType&>(I));return std::move(I);}

template<typename ImageType>
std::enable_if_t<!std::is_lvalue_reference_v<ImageType>,ImageType&&>
opening(ImageType&& I){opening(static_cast<ImageType&>(I));return std::move(I);}

template<typename ImageType>
std::enable_if_t<!std::is_lvalue_reference_v<ImageType>,ImageType&&>
closing(ImageType&& I){closing(static_cast<ImageType&>(I));return std::move(I);}

template<typename ImageType>
std::enable_if_t<!std::is_lvalue_reference_v<ImageType>,ImageType&&>
negate(ImageType&& I){negate(static_cast<ImageType&>(I));return std::move(I);}

template<typename ImageType>
std::enable_if_t<!std::is_lvalue_reference_v<ImageType>,ImageType&&>
dilation_by_radius(ImageType&& I,unsigned int r){tipl::morphology::dilation_by_radius(static_cast<ImageType&>(I),r);return std::move(I);}

template<typename ImageType>
std::enable_if_t<!std::is_lvalue_reference_v<ImageType>,ImageType&&>
erosion_by_radius(ImageType&& I,unsigned int r){tipl::morphology::erosion_by_radius(static_cast<ImageType&>(I),r);return std::move(I);}

}
}
#endif
