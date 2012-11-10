#include <vector>
#include <list>
#include "image/numerical/index_algorithm.hpp"
#include "image/numerical/numerical.hpp"
#include "image/morphology/morphology.hpp"
#include "image/segmentation/otsu.hpp"

#ifdef DEBUG
#include "image/io/bitmap.hpp"
#include <sstream>
#endif
namespace image
{

namespace segmentation
{


template<typename ImageType,typename LabelImageType>
void watershed(const ImageType& input_image,LabelImageType& label)
{
    label.clear();
    label.resize(input_image.geometry());
    image::basic_image<unsigned char,ImageType::dimension> image(input_image.geometry());
    image::normalize(input_image,image);

    std::vector<std::list<pixel_index<ImageType::dimension> > > presort_table(256);
    for (pixel_index<ImageType::dimension> index; index.valid(image.geometry()); index.next(image.geometry()))
        presort_table[image[index.index()]].push_back(index);

    std::list<std::pair<pixel_index<ImageType::dimension>,size_t> > flooding_points;
    size_t basin_id = 1;
    for (typename ImageType::value_type intensity = 0; intensity < presort_table.size(); ++intensity)
    {
        if(presort_table[intensity].empty())
            continue;
        // new basin
        {
            typename std::list<pixel_index<ImageType::dimension> >::const_iterator iter = presort_table[intensity].begin();
            typename std::list<pixel_index<ImageType::dimension> >::const_iterator end = presort_table[intensity].end();
            for (; iter != end; ++iter)
            {
                if (label[iter->index()])
                    continue;
                flooding_points.push_back(std::make_pair(*iter,size_t(0)));
            }
            presort_table[intensity].clear();
        }
        typename std::list<std::pair<pixel_index<ImageType::dimension>,size_t> >::iterator iter = flooding_points.begin();
        typename std::list<std::pair<pixel_index<ImageType::dimension>,size_t> >::iterator end = flooding_points.end();
        while (iter != end)
        {
            if(image[iter->first.index()] != intensity)
            {
                ++iter;
                continue;
            }
            if (label[iter->first.index()] == 0)
            {

                size_t cur_basin;
                if(iter->second == 0)
                {
                    cur_basin = basin_id;
                    ++basin_id;
                }
                else
                    cur_basin = iter->second;
                label[iter->first.index()] = cur_basin;

                pixel_index<ImageType::dimension> active_point;
                std::vector<pixel_index<ImageType::dimension> > front,neighbor_points;
                front.push_back(iter->first);
                while(!front.empty())
                {
                    active_point = front.back();
                    front.pop_back();
                    get_connected_neighbors(active_point,image.geometry(),neighbor_points);
                    for(size_t index = 0; index < neighbor_points.size(); ++index)
                    {
                        size_t cur_index = neighbor_points[index].index();
                        if(image[cur_index] == intensity)
                        {
                            if(label[cur_index] != cur_basin)
                            {
                                front.push_back(neighbor_points[index]);
                                label[cur_index] = cur_basin;
                            }
                        }
                        else if(!label[cur_index])
                            flooding_points.insert(iter,std::make_pair(neighbor_points[index],cur_basin));
                    }
                }
            }
            flooding_points.erase(iter++);
        }
    }
}


template<typename ImageType,typename LabelImageType>
void watershed2(const ImageType& input_image,LabelImageType& label,unsigned int size_threshold,double detail_level = 1.0)
{
    typedef image::pixel_index<ImageType::dimension> pixel_type;
    label.clear();
    label.resize(input_image.geometry());
    ImageType I(input_image);

    float level = *std::max_element(input_image.begin(),input_image.end());
    float otsu_level = image::segmentation::otsu_threshold(input_image)*detail_level;
    unsigned int cur_region_num = 0;
    for(double L = 0.9;level*L > otsu_level;L -= 0.05)
    {
        std::vector<std::vector<unsigned int> > regions;
        image::basic_image<unsigned char,ImageType::dimension> mask;
        LabelImageType cur_label;
        image::threshold(I,mask,level*L);
        image::morphology::connected_component_labeling(mask,cur_label,regions);

        // merge
        if(L != 2.0)
        while(1)
        {
            std::vector<unsigned int> grow_pos;
            std::vector<unsigned int> grow_index;
            for(pixel_type pos;label.geometry().is_valid(pos);pos.next(label.geometry()))
                if(cur_label[pos.index()])
                {
                    std::vector<pixel_type> neighbor_points;
                    get_connected_neighbors(pos,label.geometry(),neighbor_points);
                    for(unsigned int index = 0;index < neighbor_points.size();++index)
                        if(label[neighbor_points[index].index()])
                        {
                            grow_pos.push_back(pos.index());
                            grow_index.push_back(label[neighbor_points[index].index()]);
                            cur_label[pos.index()] = 0;
                            break;
                        }

                }
            if(grow_pos.empty())
                break;
            for(unsigned int index = 0;index < grow_pos.size();++index)
                label[grow_pos[index]] = grow_index[index];
        }

        // new regions
        for(unsigned int pos = 0;pos < cur_label.size();++pos)
            if(cur_label[pos])
            {
                if(regions[cur_label[pos]-1].size() < size_threshold)
                    continue;
                ++cur_region_num;
                unsigned int region_id = cur_label[pos]-1;
                for(unsigned int i = 0;i < regions[region_id].size();++i)
                {
                    unsigned int cur_pos = regions[region_id][i];
                    cur_label[cur_pos] = 0;
                    label[cur_pos] = cur_region_num;
                    I[cur_pos] = 0;
                }
            }
        #ifdef DEBUG
        std::ostringstream name;
        name << L << ".bmp";
        image::basic_image<unsigned char,2> out;
        image::normalize(label,out);
        image::io::bitmap bmp;
        bmp << out;
        bmp.save_to_file(name.str().c_str());
        #endif
    }
}

}

}
