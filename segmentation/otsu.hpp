#ifndef OTSU_HPP
#define OTSU_HPP
#include <vector>
#include "../numerical/numerical.hpp"
#include "../numerical/statistics.hpp"
#include "../numerical/basic_op.hpp"

namespace tipl{

namespace segmentation{


template<typename hist_value_type,typename value_type>
float otsu_threshold_from_hist(const std::vector<hist_value_type>& hist,
                                    value_type min_v,value_type max_v)
{
    if(hist.empty())
        return min_v;
    std::vector<hist_value_type> w(hist.size());
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
    optimal_threshold_value *= max_v - min_v;
    optimal_threshold_value += min_v;
    return optimal_threshold_value;
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
    std::fill(hist.begin(),hist.begin() + skip,0);
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
    buf.reserve(I.size());
    for(size_t i = 0;i < I.size();++i)
        if(I[i] > ot)
            buf.push_back(I[i]);
    if(buf.empty())
        return ot;
    return tipl::median(buf);
}

template<typename image_type>
void normalize_otsu_median(image_type& I,float upper_limit = 1.0f)
{
    if constexpr(std::is_integral<typename image_type::value_type>::value)
    {
        std::vector<float> buf(I.size());
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
        if(std::count_if(I.begin(),I.end(),[&](auto v){return v > 0.0f;}) > 0.5f*I.size())
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
    return std::count_if(src.begin(),src.end(),[threshold](float v){return v > threshold;});
}



}
}

#endif
