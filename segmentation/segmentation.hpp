//---------------------------------------------------------------------------
#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP


#include "otsu.hpp"
#include "watershed.hpp"
#include "graph_cut.hpp"
#include "fast_marching.hpp"
#include "stochastic_competition.hpp"

namespace image{
    namespace segmentation{
        template<typename ImageType1,typename ImageType2,typename LabelType>
        void resample(const ImageType1& data,const LabelType& label,ImageType2& out)
        {
            out.resize(data.geometry());
            std::vector<float> sum_intensity;
            std::vector<unsigned int> num;
            for(unsigned int index = 0;index < label.size();++index)
            {
                typename LabelType::value_type cur_label = label[index];
                if(sum_intensity.size() <= cur_label)
                    sum_intensity.resize(cur_label+1);
                if(num.size() <= cur_label)
                    num.resize(cur_label+1);
                sum_intensity[cur_label] += data[index];
                ++num[cur_label];
            }
            for(unsigned int index = 0;index < sum_intensity.size();++index)
                sum_intensity[index] /= num[index];
            for(unsigned int index = 0;index < out.size();++index)
                out[index] = sum_intensity[label[index]];
        }
    }
}



#endif
