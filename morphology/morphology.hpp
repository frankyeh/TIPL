//---------------------------------------------------------------------------
#ifndef HPP
#define HPP
#include <map>
#include <list>
#include <set>
#include "image/utility/basic_image.hpp"
#include "image/utility/pixel_index.hpp"
#include "image/numerical/index_algorithm.hpp"
#include "image/numerical/window.hpp"


namespace image
{

namespace morphology
{

template<typename ImageType>
void erosion(ImageType& image)
{
    std::vector<typename ImageType::value_type> act(image.size());
    neighbor_index_shift_narrow<ImageType::dimension> neighborhood(image.geometry());
    for (unsigned int index = 0;index < neighborhood.index_shift.size();++index)
    {
        int shift = neighborhood.index_shift[index];
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

template<typename ImageType>
void dilation(ImageType& image)
{
    std::vector<typename ImageType::value_type> act(image.size());
    neighbor_index_shift_narrow<ImageType::dimension> neighborhood(image.geometry());
    for (unsigned int index = 0;index < neighborhood.index_shift.size();++index)
    {
        int shift = neighborhood.index_shift[index];
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

/*
template<typename ImageType>
void opening(ImageType& image)
{
    erosion(image);
    dilation(image);
}

template<typename ImageType>
void closing(ImageType& image)
{
    dilation(image);
    erosion(image);
}
*/

template<typename ImageType,typename LabelType>
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

template<typename ImageType>
void edge(ImageType& image)
{
	ImageType out;
	edge(image,out);
	image = out;
}

template<typename ImageType>
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

template<typename ImageType>
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


template<typename ImageType>
void closing(ImageType& image,typename ImageType::value_type assign_value = 1,int threshold_shift = 0)
{
    std::vector<typename ImageType::value_type> act(image.size());
    neighbor_index_shift<ImageType::dimension> neighborhood(image.geometry());
    unsigned int threshold = neighborhood.index_shift.size() >> 1;
    threshold += threshold_shift;
    for (unsigned int index = 0;index < neighborhood.index_shift.size();++index)
    {
        int shift = neighborhood.index_shift[index];
        if (shift > 0)
        {
            typename ImageType::value_type* iter1 = &*act.begin() + shift;
            typename ImageType::value_type* iter2 = &*image.begin();
            typename ImageType::value_type* end = &*act.begin() + act.size();
            for (;iter1 < end;++iter1,++iter2)
                if (*iter2)
                    (++*iter1);
        }
        if (shift < 0)
        {
            typename ImageType::value_type* iter1 = &*act.begin();
            typename ImageType::value_type* iter2 = &*image.begin() - shift;
            typename ImageType::value_type* end = &*image.begin() + image.size();
            for (;iter2 < end;++iter1,++iter2)
                if (*iter2)
                    (++*iter1);
        }
    }

    for (unsigned int index = 0;index < image.size();++index)
    {
        if (act[index] > threshold)
        {
            if (!image[index])
                image[index] = assign_value;
        }
    }
}

template<typename ImageType>
void opening(ImageType& image,int threshold_shift = 0)
{
    std::vector<typename ImageType::value_type> act(image.size());
    neighbor_index_shift<ImageType::dimension> neighborhood(image.geometry());
    unsigned int threshold = neighborhood.index_shift.size() >> 1;
    threshold += threshold_shift;
    for (unsigned int index = 0;index < neighborhood.index_shift.size();++index)
    {
        int shift = neighborhood.index_shift[index];
        if (shift > 0)
        {
            typename ImageType::value_type* iter1 = &*act.begin() + shift;
            typename ImageType::value_type* iter2 = &*image.begin();
            typename ImageType::value_type* end = &*act.begin() + act.size();
            for (;iter1 < end;++iter1,++iter2)
                if (*iter2)
                    (++*iter1);
        }
        if (shift < 0)
        {
            typename ImageType::value_type* iter1 = &*act.begin();
            typename ImageType::value_type* iter2 = &*image.begin() - shift;
            typename ImageType::value_type* end = &*image.begin() + image.size();
            for (;iter2 < end;++iter1,++iter2)
                if (*iter2)
                    (++*iter1);
        }
    }

    for (unsigned int index = 0;index < image.size();++index)
    {
        if (act[index] < threshold)
        {
            if (image[index])
                image[index] = 0;
        }
    }
}

template<typename ImageType>
void smoothing(ImageType& image,typename ImageType::value_type assign_value = 1)
{
    std::vector<typename ImageType::value_type> act(image.size());
    neighbor_index_shift<ImageType::dimension> neighborhood(image.geometry());
    unsigned int threshold = neighborhood.index_shift.size() >> 1;
    for (unsigned int index = 0;index < neighborhood.index_shift.size();++index)
    {
        int shift = neighborhood.index_shift[index];
        if (shift > 0)
        {
            typename ImageType::value_type* iter1 = &*act.begin() + shift;
            typename ImageType::value_type* iter2 = &*image.begin();
            typename ImageType::value_type* end = &*act.begin() + act.size();
            for (;iter1 < end;++iter1,++iter2)
                if (*iter2)
                    (++*iter1);
        }
        if (shift < 0)
        {
            typename ImageType::value_type* iter1 = &*act.begin();
            typename ImageType::value_type* iter2 = &*image.begin() - shift;
            typename ImageType::value_type* end = &*image.begin() + image.size();
            for (;iter2 < end;++iter1,++iter2)
                if (*iter2)
                    (++*iter1);
        }
    }

    for (unsigned int index = 0;index < image.size();++index)
    {
        if (act[index] > threshold)
        {
            if (!image[index])
                image[index] = assign_value;
        }
        if (act[index] < threshold)
        {
            if (image[index])
                image[index] = 0;
        }

    }
}


/**
//  grow can be std::equal_to, std::less, std::greater  std::greater_equal std::less_equal
//
*/
//-------------------------------------------------------------------------------
template<typename ImageType,typename IndexType,typename GrowFunc>
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
/*
 convex in x direction
*/
template<typename ImageType>
void convex_x(ImageType& image,typename ImageType::value_type assign_value = 1)
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

template<typename ImageType>
void convex_y(ImageType& image,typename ImageType::value_type assign_value = 1)
{
    unsigned int plane_size = image.plane_size();
    typename ImageType::iterator iter_plane = image.begin();
    typename ImageType::iterator end = iter_plane+image.size();
    for(;iter_plane != end;iter_plane += plane_size)
    {
        typename ImageType::iterator iter_x = iter_plane;
        typename ImageType::iterator iter_x_end = iter_x + image.width();
        for(;iter_x != iter_x_end;++iter_x)
        {
            typename ImageType::iterator iter_y = iter_x;
            typename ImageType::iterator iter_y_end = iter_y+(plane_size-image.width());
            typename ImageType::iterator first,last;
            int find_count = 0;
            for(;iter_y <= iter_y_end;iter_y += image.width())
                if(*iter_y > 0)
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
                    *first = assign_value;
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

template<typename ImageType,typename LabelImageType>
void connected_component_labeling_pass(const ImageType& image,
                                       LabelImageType& labels,
                                       std::vector<std::vector<unsigned int> >& regions,
                                       unsigned int shift,
                                       typename ImageType::value_type background = 0)
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
            if (image[index] == background)
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
                if (group_id && labels[index] != background && group_id != labels[index])
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

template<typename PixelType,typename StorageType,typename LabelImageType>
void connected_component_labeling(basic_image<PixelType,1,StorageType>& image,
                                  LabelImageType& labels,
                                  std::vector<std::vector<unsigned int> >& regions,
                                  PixelType background = 0)
{
    connected_component_labeling_pass(image,labels,regions,1,background);
}

template<typename PixelType,typename StorageType,typename LabelImageType>
void connected_component_labeling(basic_image<PixelType,2,StorageType>& image,
                                  LabelImageType& labels,
                                  std::vector<std::vector<unsigned int> >& regions,
                                  PixelType background = 0)
{
    connected_component_labeling_pass(image,labels,regions,1,background);
    connected_component_labeling_pass(image,labels,regions,image.width(),background);
}


template<typename PixelType,typename StorageType,typename LabelImageType>
void connected_component_labeling(basic_image<PixelType,3,StorageType>& image,
                                  LabelImageType& labels,
                                  std::vector<std::vector<unsigned int> >& regions,
                                  PixelType background = 0)
{
    connected_component_labeling_pass(image,labels,regions,1,background);
    connected_component_labeling_pass(image,labels,regions,image.width(),background);
    connected_component_labeling_pass(image,labels,regions,image.geometry().plane_size(),background);
}

template<typename ImageType>
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

template<typename ImageType>
void defragment(ImageType& image,float fragment_percentage)
{
    image::basic_image<unsigned int,ImageType::dimension> labels(image.geometry());
    std::vector<std::vector<unsigned int> > regions;

    connected_component_labeling(image,labels,regions);

    std::vector<unsigned char> region_filter(regions.size());

    unsigned int area_threshold = image.size() * fragment_percentage;
    for (unsigned int index = 0;index < regions.size();++index)
        region_filter[index+1] = regions[index].size() > area_threshold;

    for (unsigned int index = 0;index < image.size();++index)
        if (image[index] && !region_filter[labels[index]])
            image[index] = 0;
}

template<typename ImageType,typename PixelIndexType,typename ValueType>
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
