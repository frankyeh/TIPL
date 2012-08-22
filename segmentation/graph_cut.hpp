#ifndef TIPL_SEGMENTATION_GRAPH_CUT
#define TIPL_SEGMENTATION_GRAPH_CUT
#include <vector>
#include <algorithm>
#include "disjoint_set.hpp"

namespace image{

namespace segmentation{


struct graph_edge{
    unsigned int n1,n2;
    float w;
    bool operator < (const graph_edge& rhs) const
    {
        return w < rhs.w;
    }
};

/**
Efficient Graph-Based Image Segmentation
Pedro F. Felzenszwalb and Daniel P. Huttenlocher
International Journal of Computer Vision, Volume 59, Number 2, September 2004

c uses relative scale from 0~1.0
*/
template<typename image_type,typename label_type>
void graph_cut(const image_type& I,label_type& out,float c,unsigned int min_size)
{
     typedef image::pixel_index<image_type::dimension> index_type;
     typedef typename image_type::value_type value_type;
     std::vector<graph_edge> edges;
     edges.reserve(I.size() * 2* image_type::dimension);
     float max_w = 0.0;
     for(index_type index;I.geometry().is_valid(index);index.next(I.geometry()))
         {
             std::vector<index_type> neighbor;
             image::get_neighbors(index,I.geometry(),neighbor);
             for(int i = 0;i < neighbor.size();++i)
             if(index.index() > neighbor[i].index())
             {
                 edges.push_back(graph_edge());
                 edges.back().n1 = index.index();
                 edges.back().n2 = neighbor[i].index();
                 edges.back().w = I[edges.back().n1]-I[edges.back().n2];
		 if(edges.back().w < 0)
		     edges.back().w = -edges.back().w;
                 if(edges.back().w > max_w)
                     max_w = edges.back().w;
	
             }
         }
     c *= max_w;
     std::sort(edges.begin(),edges.end());
     disjoint_set dset;
     dset.label.resize(I.size());
     dset.rank.resize(I.size());
     for(unsigned int index = 0;index < I.size();++index)
            dset.label[index] = index;
     {
         std::vector<unsigned int> size(I.size());
         std::vector<float> threshold(I.size());
         std::fill(size.begin(),size.end(),1);
         std::fill(threshold.begin(),threshold.end(),c);
         for (int i = 0; i < edges.size(); i++)
         {
             unsigned int s1 = dset.find_set(edges[i].n1);
             unsigned int s2 = dset.find_set(edges[i].n2);
             float w = edges[i].w;
             if(s1 == s2)
                 continue;
             if(w <= threshold[s1] && w <= threshold[s2])
             {
                 unsigned int total_size = size[s1]+size[s2];
                 unsigned int s12 = dset.join_set(s1,s2);
                 size[s12] = total_size;
                 threshold[s12] = w + c/total_size;
             }
         }

         for (int i = 0; i < edges.size(); i++)
         {
             unsigned int s1 = dset.find_set(edges[i].n1);
             unsigned int s2 = dset.find_set(edges[i].n2);
             if(s1 == s2)
                 continue;
             if(size[s1] < min_size || size[s2] < min_size)
                dset.join_set(s1,s2);
         }
     }
     // re-labeling
     out.clear();
     out.resize(I.geometry());
     unsigned int num_region = 0;
     for(unsigned int index = 0;index < out.size();++index)
        {
             unsigned int s1 = dset.find_set(index);
             if(out[s1] == 0)
             {
                 ++num_region;
                 out[s1] = num_region;
             }
             out[index] = out[s1];
        }
}


}
}

#endif
