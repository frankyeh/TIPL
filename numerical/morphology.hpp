//---------------------------------------------------------------------------
#ifndef HPP
#define HPP
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
    auto shape = data.shape();
    size_t sz = data.size();
    T result_data(shape);
    tipl::par_for<sequential>(uint32_t(tipl::max_value(data))+1,[&](uint32_t index)
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
void erosion(ImageType& I,const std::vector<int64_t>& index_shift)
{
    size_t sz = I.size();
    std::vector<typename ImageType::value_type> act(sz);
    for (int64_t shift : index_shift)
    {
        if (shift > 0)
        {
            auto iter1 = act.data() + shift;
            auto iter2 = I.data();
            auto end = act.data() + sz;
            for (;iter1 < end;++iter1,++iter2)
                if (*iter2 == 0)
                    *iter1 = 1;
        }
        else if (shift < 0)
        {
            auto iter1 = act.data();
            auto iter2 = I.data() - shift;
            auto end = I.data() + sz;
            for (;iter2 < end;++iter1,++iter2)
                if (*iter2 == 0)
                    *iter1 = 1;
        }
    }

    for (size_t index = 0; index < sz; ++index)
        if (act[index])
            I[index] = 0;
}

template<typename ImageType>
void erosion(ImageType& I)
{
    neighbor_index_shift_narrow<ImageType::dimension> neighborhood(I.shape());
    erosion(I,neighborhood.index_shift);
}

template<typename ImageType>
void erosion2(ImageType& I,int radius)
{
    neighbor_index_shift<ImageType::dimension> neighborhood(I.shape(),radius);
    erosion(I,neighborhood.index_shift);
}

template<typename ImageType>
void dilation(ImageType& I,const std::vector<int64_t>& index_shift)
{
    size_t sz = I.size();
    std::vector<typename ImageType::value_type> act(sz);
    tipl::par_for<sequential>(index_shift.size(),[&](unsigned int index)
    {
        int64_t shift = index_shift[index];
        if (shift > 0)
        {
            auto iter1 = act.data() + shift;
            auto iter2 = I.data();
            auto end = act.data() + sz;
            for (;iter1 < end;++iter1,++iter2)
                *iter1 |= *iter2;
        }
        else if (shift < 0)
        {
            auto iter1 = act.data();
            auto iter2 = I.data() - shift;
            auto end = I.data() + sz;
            for (;iter2 < end;++iter1,++iter2)
                *iter1 |= *iter2;
        }
    });
    for (size_t index = 0; index < sz; ++index)
        I[index] |= act[index];
}

template<typename T>
void dilation(T&& I)
{
    neighbor_index_shift_narrow<std::remove_reference_t<T>::dimension> neighborhood(I.shape());
    dilation(I,neighborhood.index_shift);
}

template<typename ImageType>
void dilation2(ImageType& I,int radius)
{
    neighbor_index_shift<ImageType::dimension> neighborhood(I.shape(),radius);
    dilation(I,neighborhood.index_shift);
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
auto get_neighbor_count(ImageType& I)
{
    size_t sz = I.size();
    std::vector<char> region_count(sz);
    neighbor_index_shift<ImageType::dimension> neighborhood(I.shape());
    tipl::par_for<sequential>(sz,[&](int64_t index)
    {
        for(int64_t pos : neighborhood.index_shift)
        {
            pos += index;
            if(pos < 0 || pos >= static_cast<int64_t>(sz))
                continue;
            if(I[pos])
                ++region_count[index];
        }
    });
    return region_count;
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
size_t closing(ImageType& I,char threshold_shift = 0)
{
    auto act = get_neighbor_count(I);
    char threshold = ((ImageType::dimension == 2) ? 9 : 27) >> 1;
    threshold += threshold_shift;
    size_t count = 0;
    size_t sz = I.size();
    for (size_t index = 0; index < sz; ++index)
        if (!I[index] && act[index] > threshold)
        {
            I[index] = 1;
            ++count;
        }
    return count;
}

template<typename ImageType>
size_t opening(ImageType& I,char threshold_shift = 0)
{
    auto act = get_neighbor_count(I);
    char threshold = ((ImageType::dimension == 2) ? 9 : 27) >> 1;
    threshold += threshold_shift;
    size_t count = 0;
    size_t sz = I.size();
    for (size_t index = 0; index < sz; ++index)
        if (I[index] && act[index] < threshold)
        {
            I[index] = 0;
            ++count;
        }
    return count;
}

template<typename ImageType>
void negate(ImageType& I)
{
    size_t sz = I.size();
    for (size_t index = 0; index < sz; ++index)
        I[index] = I[index] ? 0:1;
}

template<typename ImageType>
void smoothing(ImageType& I)
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
    auto shape = I.shape();
    size_t sz = I.size();
    constexpr char threshold = ((ImageType::dimension == 2) ? 9 : 27) >> 1;
    char upper_threshold = threshold+ImageType::dimension+ImageType::dimension;
    char lower_threshold = threshold-ImageType::dimension-ImageType::dimension;
    tipl::par_for(begin_index(shape),end_index(shape),[&](auto pos)
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

template<typename ImageType,typename LabelImageType>
void connected_component_labeling_pass(const ImageType& I,
                                       LabelImageType& labels,
                                       std::vector<std::vector<size_t> >& regions,
                                       size_t shift)
{
    size_t sz = I.size();
    if(sz == 0)
        return;
    typedef typename std::vector<size_t>::const_iterator region_iterator;
    if (shift == 1)
    {
        regions.clear();
        labels.resize(I.shape());
        std::mutex add_lock;

        size_t width = I.width();
        tipl::par_for(sz/width,[&,width](size_t y)
        {
            size_t index = size_t(y)*width;
            size_t end_index = index+width;
            while (index < end_index)
            {
                if (I[index] == 0)
                {
                    labels[index] = 0;
                    ++index;
                    continue;
                }
                size_t start_index = index;
                do{
                    ++index;
                }while(index < end_index && I[index] != 0);
                std::vector<size_t> voxel_pos(index-start_index);
                std::iota(voxel_pos.begin(),voxel_pos.end(),start_index);
                unsigned int group_id;
                {
                    std::lock_guard<std::mutex> lock(add_lock);
                    regions.push_back(std::move(voxel_pos));
                    group_id = regions.size();
                }
                std::fill(labels.begin()+start_index,labels.begin()+index,group_id);
            }
        });
    }
    else
    {
        for (size_t x = 0;x < shift;++x)
        {
            unsigned int group_id = 0;
            for (size_t index = x;index < sz;index += shift)
            {
                if (group_id && labels[index] != 0 && group_id != labels[index])
                {
                    unsigned int from_id = group_id-1;
                    unsigned int to_id = labels[index]-1;
                    if (regions[from_id].size() > regions[to_id].size())
                        std::swap(from_id,to_id);

                    {
                        region_iterator end = regions[from_id].end();
                        unsigned int new_id = to_id +1;
                        for (region_iterator iter = regions[from_id].begin();iter != end;++iter)
                            labels[*iter] = new_id;
                    }
                    {
                        regions[to_id].insert(regions[to_id].end(),regions[from_id].begin(),regions[from_id].end());
                        regions[from_id] = std::vector<size_t>();
                    }
                }
                group_id = labels[index];
            }
        }
    }
}

template<typename T1,typename T2,typename std::enable_if<T1::dimension==1,bool>::type = true>
void connected_component_labeling(const T1& I,T2& labels,std::vector<std::vector<size_t> >& regions)
{
    connected_component_labeling_pass(I,labels,regions,1);
}

template<typename T1,typename T2,typename std::enable_if<T1::dimension==2,bool>::type = true>
void connected_component_labeling(const T1& I,T2& labels,std::vector<std::vector<size_t> >& regions)
{
    connected_component_labeling_pass(I,labels,regions,1);
    connected_component_labeling_pass(I,labels,regions,I.width());
}


template<typename T1,typename T2,typename std::enable_if<T1::dimension==3,bool>::type = true>
void connected_component_labeling(const T1& I,T2& labels,std::vector<std::vector<size_t> >& regions)
{
    connected_component_labeling_pass(I,labels,regions,1);
    connected_component_labeling_pass(I,labels,regions,I.width());
    connected_component_labeling_pass(I,labels,regions,I.plane_size());
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

template<typename ImageType>
void defragment(ImageType& I)
{
    size_t sz = I.size();
    if(sz == 0)
        return;
    tipl::image<ImageType::dimension,unsigned int> labels(I.shape());
    std::vector<std::vector<size_t> > regions;

    connected_component_labeling(I,labels,regions);

    unsigned int max_size_group_id = 1;
    if (!regions.empty())
    {
        unsigned int max_size = regions[0].size();
        for (unsigned int index = 1;index < regions.size();++index)
            if (regions[index].size() > max_size)
            {
                max_size = regions[index].size();
                max_size_group_id = index+1;
            }
    }

    for (size_t index = 0; index < sz; ++index)
        if (I[index] && labels[index] != max_size_group_id)
            I[index] = 0;
}
template<typename ImageType>
void defragment_slice(ImageType& I)
{
    tipl::morphology::negate(I);
    size_t d = I.depth();
    tipl::par_for(d,[&](size_t z)
    {
        auto slice = I.slice_at(z);
        std::vector<std::vector<size_t> > regions;
        tipl::image<2,size_t> labels;
        connected_component_labeling(slice,labels,regions);
        size_t sz = labels.size();
        for(size_t i = 0; i < sz; ++i)
            if(labels[i] != labels[0])
                slice[i] = 1;
            else
                slice[i] = 0;
    });
}

template<typename ImageType>
void defragment_by_size_ratio(ImageType& I,float area_ratio = 0.05f)
{
    tipl::image<ImageType::dimension,unsigned int> labels(I.shape());
    std::vector<std::vector<size_t> > regions;

    connected_component_labeling(I,labels,regions);

    if(regions.empty())
        return;

    unsigned int max_size = regions[0].size();
    for (unsigned int index = 1;index < regions.size();++index)
        if (regions[index].size() > max_size)
            max_size = regions[index].size();
    size_t size_threshold = size_t(float(max_size)*area_ratio);

    std::vector<unsigned char> region_filter(regions.size()+1);

    for (unsigned int index = 0;index < regions.size();++index)
        region_filter[index+1] = regions[index].size() > size_threshold;

    size_t sz = I.size();
    for (size_t index = 0; index < sz; ++index)
        if (I[index] && !region_filter[labels[index]])
            I[index] = 0;
}

template<typename ImageType>
void defragment_by_radius(ImageType& I,int radius = 3)
{
    tipl::image<ImageType::dimension,unsigned char> mask(I.shape()),mask2;
    size_t sz = I.size();
    for (size_t index = 0; index < sz; ++index)
        mask[index] = I[index] > 0 ? 1 : 0;
    erosion2(mask,radius);
    mask2 = mask;
    defragment(mask);
    for (size_t index = 0; index < sz; ++index)
        if (mask2[index] && !mask[index])
            mask2[index] = 1;
        else
            mask2[index] = 0;
    dilation2(mask2,radius);

    for (size_t index = 0; index < sz; ++index)
        if (I[index] && mask2[index])
            I[index] = 0;
}
template<typename ImageType>
void defragment_by_threshold(ImageType& I,typename ImageType::value_type threshold)
{
    tipl::image<3,char> mask = I > threshold;
    tipl::morphology::defragment(mask);
    tipl::preserve(I.begin(),I.end(),mask.begin());
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

template<typename out_type,typename template_type,typename image_type>
void reclassify_labels_by_template(const template_type& template_I,image_type& atlas_I)
{
    size_t template_region_count = tipl::max_value(template_I) + 1;
    size_t atlas_region_count = tipl::max_value(atlas_I);
    std::vector<size_t> tissue_votes((atlas_region_count+1)*template_region_count,0);

    size_t sz = atlas_I.size();
    for(size_t pos = 0; pos < sz; ++pos)
    {
        auto a = atlas_I[pos];
        auto t = template_I[pos];
        if(a > 0 && t < template_region_count)
            tissue_votes[a*template_region_count + t]++;
    }

    std::vector<size_t> region_majority_tissue(atlas_region_count+1,0);
    tipl::par_for(atlas_region_count,[&](size_t i)
    {
        ++i;
        auto begin_it = tissue_votes.begin() + i*template_region_count;
        auto best_tissue = std::max_element(begin_it,begin_it + template_region_count);
        region_majority_tissue[i] = std::distance(begin_it,best_tissue);
    });

    std::vector<size_t> region_erased(atlas_region_count+1,0);
    for(size_t pos = 0; pos < sz; ++pos)
    {
        auto a = atlas_I[pos];
        if(a > 0 && template_I[pos] != region_majority_tissue[a])
        {
            atlas_I[pos] = 0;
            region_erased[a]++;
        }
    }

    if constexpr(!std::is_same_v<out_type,void>)
    {
        std::string erased_report;
        for(size_t i = 1; i <= atlas_region_count; ++i)
            if(region_majority_tissue[i] > 0)
            {
                if(!erased_report.empty())
                    erased_report += ", ";
                erased_report += std::to_string(region_erased[i]);
            }

        if(!erased_report.empty())
            out_type() << " voxel erased based on tissue classification: " << erased_report;
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

}
}
#endif
