//---------------------------------------------------------------------------
#ifndef HPP
#define HPP
#include <map>
#include <list>
#include <set>
#include "image/numerical/basic_op.hpp"
#include "image/utility/basic_image.hpp"
#include "image/utility/pixel_index.hpp"
#include "image/numerical/index_algorithm.hpp"
#include "image/numerical/window.hpp"


namespace image
{

namespace morphology
{

template<class ImageType>
void erosion(ImageType& image,const std::vector<int>& index_shift)
{
    std::vector<typename ImageType::value_type> act(image.size());
    for (unsigned int index = 0;index < index_shift.size();++index)
    {
        int shift = index_shift[index];
        if (shift > 0)
        {
            typename ImageType::value_type* iter1 = &*act.begin() + shift;
            typename ImageType::value_type* iter2 = &*image.begin();
            typename ImageType::value_type* end = &*act.begin() + act.size();
            for (;iter1 < end;++iter1,++iter2)
                if (*iter2 == 0)
                    *iter1 = 1;
        }
        if (shift < 0)
        {
            typename ImageType::value_type* iter1 = &*act.begin();
            typename ImageType::value_type* iter2 = &*image.begin() - shift;
            typename ImageType::value_type* end = &*image.begin() + image.size();
            for (;iter2 < end;++iter1,++iter2)
                if (*iter2 == 0)
                    *iter1 = 1;
        }
    }

    for (unsigned int index = 0;index < image.size();++index)
        if (act[index])
            image[index] = 0;
}

template<class ImageType>
void erosion(ImageType& image)
{
    neighbor_index_shift_narrow<ImageType::dimension> neighborhood(image.geometry());
    erosion(image,neighborhood.index_shift);
}

template<class ImageType>
void erosion2(ImageType& image,int radius)
{
    neighbor_index_shift<ImageType::dimension> neighborhood(image.geometry(),radius);
    erosion(image,neighborhood.index_shift);
}

template<class ImageType>
void dilation(ImageType& image,const std::vector<int>& index_shift)
{
    std::vector<typename ImageType::value_type> act(image.size());
    for (unsigned int index = 0;index < index_shift.size();++index)
    {
        int shift = index_shift[index];
        if (shift > 0)
        {
            typename ImageType::value_type* iter1 = &*act.begin() + shift;
            typename ImageType::value_type* iter2 = &*image.begin();
            typename ImageType::value_type* end = &*act.begin() + act.size();
            for (;iter1 < end;++iter1,++iter2)
                *iter1 |= *iter2;
        }
        if (shift < 0)
        {
            typename ImageType::value_type* iter1 = &*act.begin();
            typename ImageType::value_type* iter2 = &*image.begin() - shift;
            typename ImageType::value_type* end = &*image.begin() + image.size();
            for (;iter2 < end;++iter1,++iter2)
                *iter1 |= *iter2;
        }
    }

    for (unsigned int index = 0;index < image.size();++index)
        image[index] |= act[index];
}

template<class ImageType>
void dilation(ImageType& image)
{
    neighbor_index_shift_narrow<ImageType::dimension> neighborhood(image.geometry());
    dilation(image,neighborhood.index_shift);
}

template<class ImageType>
void dilation2(ImageType& image,int radius)
{
    neighbor_index_shift<ImageType::dimension> neighborhood(image.geometry(),radius);
    dilation(image,neighborhood.index_shift);
}

/*
template<class ImageType>
void opening(ImageType& image)
{
    erosion(image);
    dilation(image);
}

template<class ImageType>
void closing(ImageType& image)
{
    dilation(image);
    erosion(image);
}
*/

template<class ImageType,class LabelType>
void edge(const ImageType& image,LabelType& act)
{
    act.resize(image.geometry());
    neighbor_index_shift<ImageType::dimension> neighborhood(image.geometry());
    for (unsigned int index = 0;index < neighborhood.index_shift.size();++index)
    {
        int shift = neighborhood.index_shift[index];
        if (shift > 0)
        {
            typename LabelType::value_type* iter1 = &*act.begin() + shift;
            const typename ImageType::value_type* iter2 = &*image.begin();
            const typename ImageType::value_type* iter3 = &*image.begin()+shift;
            typename LabelType::value_type* end = &*act.begin() + act.size();
            for (;iter1 < end;++iter1,++iter2,++iter3)
                if (*iter2 != *iter3)
                    *iter1 = 1;
        }
        if (shift < 0)
        {
            typename LabelType::value_type* iter1 = &*act.begin();
            const typename ImageType::value_type* iter2 = &*image.begin() - shift;
            const typename ImageType::value_type* iter3 = &*image.begin();
            const typename ImageType::value_type* end = &*image.begin() + image.size();
            for (;iter2 < end;++iter1,++iter2,++iter3)
                if (*iter2 != *iter3)
                    *iter1 = 1;
        }
    }
}

template<class ImageType>
void edge(ImageType& image)
{
	ImageType out;
	edge(image,out);
	image = out;
}

template<class ImageType,class LabelType>
void inner_edge(const ImageType& image,LabelType& act)
{
    act.resize(image.geometry());
    neighbor_index_shift<ImageType::dimension> neighborhood(image.geometry());
    for (unsigned int index = 0;index < neighborhood.index_shift.size();++index)
    {
        int shift = neighborhood.index_shift[index];
        if (shift > 0)
        {
            typename LabelType::value_type* iter1 = &*act.begin() + shift;
            const typename ImageType::value_type* iter2 = &*image.begin();
            const typename ImageType::value_type* iter3 = &*image.begin()+shift;
            typename LabelType::value_type* end = &*act.begin() + act.size();
            for (;iter1 < end;++iter1,++iter2,++iter3)
                if (*iter2 < *iter3)
                    *iter1 = 1;
        }
        if (shift < 0)
        {
            typename LabelType::value_type* iter1 = &*act.begin();
            const typename ImageType::value_type* iter2 = &*image.begin() - shift;
            const typename ImageType::value_type* iter3 = &*image.begin();
            const typename ImageType::value_type* end = &*image.begin() + image.size();
            for (;iter2 < end;++iter1,++iter2,++iter3)
                if (*iter2 < *iter3)
                    *iter1 = 1;
        }
    }
}

template<class ImageType>
void inner_edge(ImageType& image)
{
        ImageType out;
        inner_edge(image,out);
        image = out;
}

template<class ImageType>
bool is_edge(ImageType& image,image::pixel_index<2> index)
{
    typename ImageType::value_type center = image[index.index()];
    unsigned int width = image.width();
    unsigned int height = image.height();
    bool have_left = index.x() >= 1;
    bool have_right = index.x()+1 < width;
    if (index.y() >= 1)
    {
        unsigned int base_index = index.index()-width;
        if ((have_left && image[base_index-1] != center) ||
                image[base_index] != center                  ||
                (have_right && image[base_index+1] != center))
            return true;
    }

    if ((have_left && image[index.index()-1] != center) ||
            (have_right && image[index.index()+1] != center))
        return true;

    if (index.y()+1 < height)
    {
        unsigned int base_index = index.index()+width;
        if ((have_left && image[base_index-1] != center) ||
                image[base_index] != center                  ||
                (have_right && image[base_index+1] != center))
            return true;
    }
    return false;
}

template<class ImageType>
bool is_edge(ImageType& image,image::pixel_index<3> index)
{
    typename ImageType::value_type center = image[index.index()];
    unsigned int z_offset = image.geometry().plane_size();
    unsigned int y_offset = image.width();
    bool have_left = index.x() >= 1;
    bool have_right = index.x()+1 < image.width();
    bool has_top = index.y() >= 1;
    bool has_bottom = index.y()+1 < image.height();
    if (index.z() >= 1)
    {
        if (has_top)
        {
            if ((have_left  && image[index.index()-1-y_offset-z_offset] != center) ||
                    image[index.index()  -y_offset-z_offset] != center  ||
                    (have_right && image[index.index()+1-y_offset-z_offset] != center))
                return true;
        }

        if ((have_left  && image[index.index()-1-z_offset] != center) ||
                image[index.index()  -z_offset] != center  ||
                (have_right && image[index.index()+1-z_offset] != center))
            return true;

        if (has_bottom)
        {
            if ((have_left  && image[index.index()-1+y_offset-z_offset] != center) ||
                    image[index.index()  +y_offset-z_offset] != center  ||
                    (have_right && image[index.index()+1+y_offset-z_offset] != center))
                return true;
        }
    }

    {
        if (has_top)
        {
            if ((have_left  && image[index.index()-1-y_offset] != center) ||
                    image[index.index()  -y_offset] != center  ||
                    (have_right && image[index.index()+1-y_offset] != center))
                return true;
        }
        {
            if ((have_left  && image[index.index()-1] != center) ||
                    (have_right && image[index.index()+1] != center))
                return true;
        }
        if (has_bottom)
        {
            if ((have_left  && image[index.index()-1+y_offset] != center) ||
                    image[index.index()  +y_offset] != center  ||
                    (have_right && image[index.index()+1+y_offset] != center))
                return true;
        }

    }
    if (index.z()+1 < image.depth())
    {
        if (has_top)
        {
            if ((have_left  && image[index.index()-1-y_offset+z_offset] != center) ||
                    image[index.index()  -y_offset+z_offset] != center  ||
                    (have_right && image[index.index()+1-y_offset+z_offset] != center))
                return true;
        }

        if ((have_left  && image[index.index()-1+z_offset] != center) ||
                image[index.index()  +z_offset] != center  ||
                (have_right && image[index.index()+1+z_offset] != center))
            return true;

        if (has_bottom)
        {
            if ((have_left  && image[index.index()-1+y_offset+z_offset] != center) ||
                    image[index.index()  +y_offset+z_offset] != center  ||
                    (have_right && image[index.index()+1+y_offset+z_offset] != center))
                return true;
        }
    }
    return false;
}

template<class ImageType>
unsigned char get_neighbor_count(ImageType& image,std::vector<unsigned char>& act)
{
    act.resize(image.size());
    neighbor_index_shift<ImageType::dimension> neighborhood(image.geometry());
    for (unsigned int index = 0;index < neighborhood.index_shift.size();++index)
    {
        int shift = neighborhood.index_shift[index];
        if (shift > 0)
        {
            unsigned char* iter1 = &*act.begin() + shift;
            typename ImageType::value_type* iter2 = &*image.begin();
            unsigned char* end = &*act.begin() + act.size();
            for (;iter1 < end;++iter1,++iter2)
                if (*iter2)
                    (++*iter1);
        }
        if (shift < 0)
        {
            unsigned char* iter1 = &*act.begin();
            typename ImageType::value_type* iter2 = &*image.begin() - shift;
            typename ImageType::value_type* end = &*image.begin() + image.size();
            for (;iter2 < end;++iter1,++iter2)
                if (*iter2)
                    (++*iter1);
        }
    }
    return neighborhood.index_shift.size();
}

template<class ImageType>
void closing(ImageType& I,int threshold_shift = 0)
{
    std::vector<unsigned char> act;
    unsigned int threshold = get_neighbor_count(I,act) >> 1;
    threshold += threshold_shift;
    for (unsigned int index = 0;index < I.size();++index)
    {
        if (act[index] > threshold)
        {
            if (!I[index])
                I[index] = 1;
        }
    }
}

template<class ImageType>
void opening(ImageType& I,int threshold_shift = 0)
{
    std::vector<unsigned char> act;
    unsigned int threshold = get_neighbor_count(I,act) >> 1;
    threshold += threshold_shift;
    for (unsigned int index = 0;index < I.size();++index)
    {
        if (act[index] < threshold)
        {
            if (I[index])
                I[index] = 0;
        }
    }
}

template<class ImageType>
void negate(ImageType& I)
{
    for (unsigned int index = 0;index < I.size();++index)
        I[index] = I[index] ? 0:1;
}

template<class ImageType>
void smoothing(ImageType& I)
{
    std::vector<unsigned char> act;
    unsigned int threshold = get_neighbor_count(I,act) >> 1;
    for (unsigned int index = 0;index < I.size();++index)
    {
        if (act[index] > threshold)
        {
            if (!I[index])
                I[index] = 1;
        }
        if (act[index] < threshold)
        {
            if (I[index])
                I[index] = 0;
        }

    }
}

template<class ImageType>
bool smoothing_fill(ImageType& I)
{
    bool filled = false;
    std::vector<unsigned char> act;
    unsigned int threshold = get_neighbor_count(I,act) >> 1;
    for (unsigned int index = 0;index < I.size();++index)
    {
        if (act[index] > threshold)
        {
            if (!I[index])
            {
                I[index] = 1;
                filled = true;
            }
        }
    }
    return filled;
}

template<class ImageType>
void recursive_smoothing(ImageType& I,unsigned int max_iteration = 100)
{
    for(unsigned int iter = 0;iter < max_iteration;++iter)
    {
        bool has_change = false;
        std::vector<unsigned char> act;
        unsigned int threshold = get_neighbor_count(I,act) >> 1;
        for (unsigned int index = 0;index < I.size();++index)
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


/**
//  grow can be std::equal_to, std::less, std::greater  std::greater_equal std::less_equal
//
*/
//-------------------------------------------------------------------------------
template<class ImageType,class IndexType,class GrowFunc>
void region_growing(const ImageType& image,const IndexType& seed_point,
                    std::vector<IndexType>& grown_region,GrowFunc grow)
{
    std::vector<unsigned char> label_map(image.size());
    std::vector<IndexType> seeds;
    std::vector<IndexType> neighbor;
    seeds.push_back(seed_point);
    label_map[seed_point.index()] = 1;
    for (unsigned int index = 0;index < seeds.size();++index)
    {
        IndexType active_point = seeds[index];
        get_neighbors(active_point,image.geometry(),neighbor);
        for (unsigned int index = 0;index < neighbor.size();++index)
        {
            unsigned int cur_neighbor_index = neighbor[index].index();
            if (label_map[cur_neighbor_index])
                continue;
            if (grow(image[active_point.index()],image[cur_neighbor_index]))
            {
                seeds.push_back(neighbor[index]);
                label_map[cur_neighbor_index] = 1;
            }
        }
    }
    seeds.swap(grown_region);
}

template<class ImageType>
void convex_xy(ImageType& I)
{
    image::geometry<ImageType::dimension> range_min,range_max;
    bounding_box(I,range_min,range_max);
    // get the bounding box first
    int dirs[8][2] = {{1,0},{2,1},{1,1},{1,2},{0,1},{-1,2},{-1,1},{-2,1}};
    std::vector<unsigned int> fill_buf;
    for(unsigned int i = 0;i < 8;++i)
    {
        int shift = dirs[i][0] + I.width()*dirs[i][1];
        if(shift <= 0)
            continue;
        std::vector<unsigned char> label(I.size());
        for(pixel_index<ImageType::dimension> index;
            index.is_valid(I.geometry());
            index.next(I.geometry()))
        {
            if(index[0] < range_min[0] || index[0] >= range_max[0] ||
               index[1] < range_min[1] || index[1] >= range_max[1] ||
                    label[index.index()])
                continue;
            bool has_first = false;
            fill_buf.clear();
            for(pixel_index<ImageType::dimension> index2(index);
                index.is_valid(I.geometry());)
            {
                if(I[index2.index()])
                {
                    if(!has_first)
                        has_first = true;
                    else
                    {
                        for(unsigned int i = 0;i < fill_buf.size();++i)
                            I[fill_buf[i]] = 1;
                        fill_buf.clear();
                    }
                }
                else
                {
                    if(has_first)
                        fill_buf.push_back(index2.index());
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

/*
 convex in x direction
*/
template<class ImageType>
void convex_x(ImageType& image,class ImageType::value_type assign_value = 1)
{
    typename ImageType::iterator iter = image.begin();
    typename ImageType::iterator end = iter+image.size();
    while(iter != end)
    {
        typename ImageType::iterator next_iter = iter + image.width();
        typename ImageType::iterator first = next_iter;
        typename ImageType::iterator last = next_iter;
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

template<class ImageType>
void convex_y(ImageType& image)
{
    unsigned int plane_size = image.plane_size();
    for(unsigned int iter_plane = 0;iter_plane < image.size();iter_plane += plane_size)
    {
        for(int iter_x = iter_plane,iter_x_end = iter_x + image.width()
                ;iter_x < iter_x_end;++iter_x)
        {
            int iter_y = iter_x;
            int iter_y_end = iter_y+(plane_size-image.width());
            int first,last;
            int find_count = 0;
            for(;iter_y <= iter_y_end;iter_y += image.width())
                if(image[iter_y] > 0)
                {
                    ++find_count;
                    if(find_count == 1)
                        first = iter_y;
                    else
                        last = iter_y;
                }
            if(find_count >= 2)
            {
                for(first += image.width();first != last;first += image.width())
                    image[first] = 1;
            }
        }
    }
}


/*
perform region growing in one dimension
shift = 1 : grow in x dimension
shift = image.width() : grow in y dimension
shift = image.width()*image.height() : grow in z dimension
*/

template<class ImageType,class LabelImageType>
void connected_component_labeling_pass(const ImageType& image,
                                       LabelImageType& labels,
                                       std::vector<std::vector<unsigned int> >& regions,
                                       unsigned int shift)
{
    typedef typename std::vector<unsigned int>::const_iterator region_iterator;
    if (shift == 1) // growing in one dimension
    {
        regions.clear();
        labels.resize(image.geometry());

        unsigned int group_id = 0;
        unsigned int width = image.width();
        for (unsigned int index = 0,x = 0;index < image.size();++index,++x)
        {
            if (x >= width)
            {
                x = 0;
                group_id = 0;
            }
            if (image[index] == 0)
            {
                group_id = 0;
                labels[index] = 0;
                continue;
            }
            if (!group_id)
            {
                regions.push_back(std::vector<unsigned int>());
                group_id = regions.size();
            }
            regions.back().push_back(index);
            labels[index] = group_id;
        }
    }
    else
        // growing in higher dimension
    {
        for (unsigned int x = 0;x < shift;++x)
        {
            for (unsigned int index = x,group_id = 0;index < image.size();index += shift)
            {
                if (group_id && labels[index] != 0 && group_id != labels[index])
                {
                    unsigned int from_id = group_id-1;
                    unsigned int to_id = labels[index]-1;
                    if (regions[from_id].size() > regions[to_id].size())
                        std::swap(from_id,to_id);

                    // change the labeling of the merged region
                    {
                        region_iterator end = regions[from_id].end();
                        unsigned int new_id = to_id +1;
                        for (region_iterator iter = regions[from_id].begin();iter != end;++iter)
                            labels[*iter] = new_id;
                    }

                    // merge the region information
                    {
                        regions[to_id].insert(regions[to_id].end(),regions[from_id].begin(),regions[from_id].end());
                        regions[from_id] = std::vector<unsigned int>();
                    }
                }
                group_id = labels[index];
            }
        }
    }
}

template<class PixelType,class StorageType,class LabelImageType>
void connected_component_labeling(const basic_image<PixelType,1,StorageType>& image,
                                  LabelImageType& labels,
                                  std::vector<std::vector<unsigned int> >& regions)
{
    connected_component_labeling_pass(image,labels,regions,1);
}

template<class PixelType,class StorageType,class LabelImageType>
void connected_component_labeling(const basic_image<PixelType,2,StorageType>& image,
                                  LabelImageType& labels,
                                  std::vector<std::vector<unsigned int> >& regions)
{
    connected_component_labeling_pass(image,labels,regions,1);
    connected_component_labeling_pass(image,labels,regions,image.width());
}


template<class PixelType,class StorageType,class LabelImageType>
void connected_component_labeling(const basic_image<PixelType,3,StorageType>& image,
                                  LabelImageType& labels,
                                  std::vector<std::vector<unsigned int> >& regions)
{
    connected_component_labeling_pass(image,labels,regions,1);
    connected_component_labeling_pass(image,labels,regions,image.width());
    connected_component_labeling_pass(image,labels,regions,image.geometry().plane_size());
}

template<class LabelImageType>
void get_region_bounding_box(const LabelImageType& labels,
                             const std::vector<std::vector<unsigned int> >& regions,
                             std::vector<image::vector<2,int> >& min_pos,
                             std::vector<image::vector<2,int> >& max_pos)
{
    min_pos.clear();
    min_pos.resize(regions.size());
    max_pos.clear();
    max_pos.resize(regions.size());
    std::fill(min_pos.begin(),min_pos.end(),image::vector<2,float>(labels.geometry()[0],labels.geometry()[1]));
    for(image::pixel_index<2> index(labels.geometry());index < labels.size();++index)
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

template<class LabelImageType>
void get_region_bounding_size(const LabelImageType& labels,
                              const std::vector<std::vector<unsigned int> >& regions,
                              std::vector<int>& size_x,
                              std::vector<int>& size_y)
{
    std::vector<image::vector<2,int> > max_pos,min_pos;
    image::morphology::get_region_bounding_box(labels,regions,min_pos,max_pos);
    size_x.clear();
    size_x.resize(regions.size());
    size_y.clear();
    size_y.resize(regions.size());

    for(size_t index = 0;index < regions.size();++index)
        if(!regions[index].empty())
        {
            size_x[index] = max_pos[index][0]-min_pos[index][0];
            size_y[index] = max_pos[index][1]-min_pos[index][1];
        }
}

template<class LabelImageType>
void get_region_center(const LabelImageType& labels,
                       const std::vector<std::vector<unsigned int> >& regions,
                       std::vector<image::vector<2,float> >& center_of_mass)
{
    center_of_mass.clear();
    center_of_mass.resize(regions.size());
    for(image::pixel_index<2> index(labels.geometry());index < labels.size();++index)
        if (labels[index.index()])
        {
            size_t region_id = labels[index.index()]-1;
            if (regions[region_id].empty())
                continue;
            center_of_mass[region_id] += image::vector<2,float>(index);
        }

    for(size_t index = 0;index < regions.size();++index)
        if(!regions[index].empty())
            center_of_mass[index] /= regions[index].size();
}

template<class ImageType>
void defragment(ImageType& image)
{
    image::basic_image<unsigned int,ImageType::dimension> labels(image.geometry());
    std::vector<std::vector<unsigned int> > regions;

    connected_component_labeling(image,labels,regions);

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

    for (unsigned int index = 0;index < image.size();++index)
        if (image[index] && labels[index] != max_size_group_id)
            image[index] = 0;
}

template<class ImageType>
void defragment(ImageType& image,float fragment_percentage)
{
    image::basic_image<unsigned int,ImageType::dimension> labels(image.geometry());
    std::vector<std::vector<unsigned int> > regions;

    connected_component_labeling(image,labels,regions);

    std::vector<unsigned char> region_filter(regions.size()+1);

    unsigned int area_threshold = image.size() * fragment_percentage;
    for (unsigned int index = 0;index < regions.size();++index)
        region_filter[index+1] = regions[index].size() > area_threshold;

    for (unsigned int index = 0;index < image.size();++index)
        if (image[index] && !region_filter[labels[index]])
            image[index] = 0;
}

template<class ImageType,class PixelIndexType,class ValueType>
void fill(ImageType& image,PixelIndexType seed_point,ValueType new_value)
{
    std::deque<PixelIndexType> seeds;
    seeds.push_back(seed_point);
    ValueType old_value = image[seed_point.index()];
    image[seed_point.index()] = new_value;
    while (seeds.size())
    {
        PixelIndexType active_point = seeds.front();
        seeds.pop_front();
        std::vector<PixelIndexType> neighbor;
        get_neighbors(active_point,image.geometry(),neighbor);
        for (unsigned int index = 0;index < neighbor.size();++index)
        {
            if (image[neighbor[index].index()] != old_value)
                continue;
            seeds.push_back(neighbor[index]);
            image[neighbor[index].index()] = new_value;
        }
    }
}

}
}
#endif
