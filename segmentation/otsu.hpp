#ifndef OTSU_HPP
#define OTSU_HPP
#include <vector>
#include "../numerical/numerical.hpp"
#include "../numerical/basic_op.hpp"

namespace tipl{

namespace segmentation{

template<typename ImageType>
float otsu_threshold(const ImageType& src)
{
    auto min_max = minmax_value(src.begin(),src.end());
    std::vector<unsigned int> hist;
    histogram(src,hist,min_max.first,min_max.second);
    if(hist.empty())
        return min_max.first;
    std::vector<unsigned int> w(hist.size());
    std::vector<float> sum(hist.size());
    w[0] = hist[0];
    sum[0] = 0.0;
    for (unsigned int index = 1,last_w = hist[0],last_sum = 0.0; index < hist.size(); ++index)
    {
        w[index] = last_w + hist[index];
        sum[index] = last_sum + hist[index]*index;
        last_w = w[index];
        last_sum = sum[index];
    }
    float total_sum = sum.back();
    float total_w = w.back();

    float max_sig_b = 0.0;
    unsigned int optimal_threshold = 0;

    for (unsigned int index = 0; index < hist.size()-1; ++index)
    {
        if (!w[index])
            continue;
        unsigned int w2 = (total_w-w[index]);
        if (!w2)
            continue;

        float d = sum[index]*(float)total_w;
        d -= total_sum*(float)w[index];
        d *= d;
        d /= w[index];
        d /= w2;
        if (d > max_sig_b)
        {
            max_sig_b = d;
            optimal_threshold = index;
        }
    }
    float optimal_threshold_value = optimal_threshold;
    optimal_threshold_value /= hist.size();
    optimal_threshold_value *= min_max.second - min_max.first;
    optimal_threshold_value += min_max.first;
    return optimal_threshold_value;
}

template<typename ImageType,typename LabelImageType>
LabelImageType& otsu(const ImageType& src,LabelImageType& label,typename LabelImageType::value_type foreground = 1,typename LabelImageType::value_type background = 0)
{
    return threshold(src,label,otsu_threshold(src),foreground,background);
}

template<typename ImageType>
unsigned int otsu_count(const ImageType& src,float level = 0.6f)
{
    float threshold = otsu_threshold(src)*level;
    return std::count_if(src.begin(),src.end(),[threshold](float v){return v > threshold;});
}



}
}

#endif
