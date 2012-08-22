#include <vector>
#include <list>
#include "image/numerical/index_algorithm.hpp"
#include "image/numerical/numerical.hpp"

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


}

}
