#ifndef OTSU_HPP
#define OTSU_HPP
#include <vector>
#include <algorithm>
#include "../numerical/numerical.hpp"
#include "../numerical/statistics.hpp"
#include "../numerical/basic_op.hpp"

namespace tipl{

namespace segmentation{


template<typename hist_value_type,typename value_type>
float otsu_threshold_from_hist(const std::vector<hist_value_type>& hist,
                               value_type min_v, value_type max_v)
{
    size_t hist_sz = hist.size();
    if(hist_sz == 0)
        return static_cast<float>(min_v);

    float total_w = 0.0f;
    float total_sum = 0.0f;

    // Accumulate total weights and scaled sums in one pass
    for (size_t index = 0; index < hist_sz; ++index)
    {
        float val = static_cast<float>(hist[index]);
        total_w += val;
        total_sum += val * static_cast<float>(index);
    }

    if (total_w == 0.0f)
        return static_cast<float>(min_v);

    float max_sig_b = 0.0f;
    size_t optimal_threshold = 0;
    float w_b = 0.0f;
    float sum_b = 0.0f;

    // Stream running variables without allocating O(N) arrays
    for (size_t index = 0; index < hist_sz - 1; ++index)
    {
        w_b += static_cast<float>(hist[index]);
        if (w_b == 0.0f)
            continue;

        float w_f = total_w - w_b;
        if (w_f == 0.0f)
            break;

        sum_b += static_cast<float>(hist[index]) * static_cast<float>(index);

        float d = (sum_b * total_w) - (total_sum * w_b);
        float sig_b = (d * d) / (w_b * w_f);

        if (sig_b > max_sig_b)
        {
            max_sig_b = sig_b;
            optimal_threshold = index;
        }
    }

    return static_cast<float>(min_v) + static_cast<float>(optimal_threshold) * static_cast<float>(max_v - min_v) / static_cast<float>(hist_sz);
}

template<typename ImageType>
float otsu_threshold(const ImageType& src,size_t skip = 1)
{
    if(src.empty())
        return 0.0f;
    typename ImageType::value_type min_v,max_v;
    minmax_value(src,min_v,max_v);
    std::vector<unsigned int> hist;
    histogram(src,hist,min_v,max_v);
    if(hist.empty())
        return min_v;

    std::fill(hist.begin(), hist.begin() + std::min<size_t>(skip, hist.size()), 0);
    return otsu_threshold_from_hist(hist,min_v,max_v);
}

template<typename ImageType>
float otsu_threshold_sharpening(const ImageType& src)
{
    if(src.empty())
        return 0.0f;
    typename ImageType::value_type min_v,max_v;
    minmax_value(src,min_v,max_v);
    std::vector<double> hist;
    histogram_sharpening(src,hist,min_v,max_v);
    return otsu_threshold_from_hist(hist,min_v,max_v);
}

template<typename image_type>
float otsu_median(const image_type& I)
{
    float ot = otsu_threshold(I);
    std::vector<float> buf;
    size_t sz = I.size();
    buf.reserve(sz);
    for(size_t i = 0; i < sz; ++i)
        if(I[i] > ot)
            buf.push_back(I[i]);
    if(buf.empty())
        return ot;
    return tipl::median(buf);
}

template<typename image_type>
void normalize_otsu_median(image_type& I,float upper_limit = 1.0f)
{
    size_t sz = I.size();
    if constexpr(std::is_integral<typename image_type::value_type>::value)
    {
        std::vector<float> buf(sz);
        std::copy(I.begin(),I.end(),buf.begin());
        normalize_otsu_median(buf,255.99f);
        std::copy(buf.begin(),buf.end(),I.begin());
    }
    else
    {
        float ot = otsu_median(I);
        if(ot == 0.0f)
            return;
        tipl::multiply_constant(I,upper_limit*0.5f/ot);
        if(std::count_if(I.begin(),I.end(),[&](auto v){return v > 0.0f;}) > 0.5f*sz)
            tipl::minus_constant(I,upper_limit*0.1f);
        tipl::upper_lower_threshold(I,0.0f,upper_limit);
    }
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
    return static_cast<unsigned int>(std::count_if(src.begin(),src.end(),[threshold](float v){return v > threshold;}));
}

}
}

#endif
