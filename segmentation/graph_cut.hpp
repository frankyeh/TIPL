#ifndef TIPL_SEGMENTATION_GRAPH_CUT
#define TIPL_SEGMENTATION_GRAPH_CUT
#include <vector>
#include <algorithm>
#include "disjoint_set.hpp"

namespace image
{

namespace segmentation
{


struct graph_edge
{
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
template<class value_type>
struct graph_cut_dis{

    float operator()(value_type lhs,value_type rhs)
    {
        return std::fabs(float(lhs)-float(rhs));
    }
};
template<>
struct graph_cut_dis<image::rgb_color>{

    float operator()(image::rgb_color lhs,image::rgb_color rhs)
    {
        return std::fabs(float(lhs[0])-float(rhs[0]))+std::fabs(float(lhs[1])-float(rhs[1]))+std::fabs(float(lhs[2])-float(rhs[2]));
    }
};

template<class image_type,class label_type>
void graph_cut(const image_type& I,label_type& out,float c,unsigned int min_size)
{
    typedef image::pixel_index<image_type::dimension> index_type;
    typedef typename image_type::value_type value_type;
    std::vector<graph_edge> edges;
    edges.reserve(I.size() * 2* image_type::dimension);
    float max_w = 0.0;
    for(index_type index; index.is_valid(I.geometry()); index.next(I.geometry()))
    {
        std::vector<index_type> neighbor;
        image::get_neighbors(index,I.geometry(),neighbor);
        for(int i = 0; i < neighbor.size(); ++i)
            if(index.index() > neighbor[i].index())
            {
                edges.push_back(graph_edge());
                edges.back().n1 = index.index();
                edges.back().n2 = neighbor[i].index();
                edges.back().w = graph_cut_dis<value_type>()(I[edges.back().n1],I[edges.back().n2]);
                if(edges.back().w > max_w)
                    max_w = edges.back().w;

            }
    }
    c *= max_w;
    std::sort(edges.begin(),edges.end());
    disjoint_set dset;
    dset.label.resize(I.size());
    dset.rank.resize(I.size());
    for(unsigned int index = 0; index < I.size(); ++index)
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
    for(unsigned int index = 0; index < out.size(); ++index)
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

template<class label_type1,class label_type2>
void refine_contour(const label_type1& I,label_type2& out)
{
    std::vector<unsigned int> region_list_yes((*std::max_element(I.begin(),I.end()))+1);
    std::vector<unsigned int> region_list_no(region_list_yes.size());
    for(unsigned int index = 0;index < I.size();++index)
        if(out[index])
            ++region_list_yes[I[index]];
        else
            ++region_list_no[I[index]];

    for(unsigned int index = 0;index < I.size();++index)
        if(region_list_yes[I[index]] > region_list_no[I[index]])
            out[index] = 1;
        else
            out[index] = 0;
}


}
}

#endif
