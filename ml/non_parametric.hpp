#ifndef ML_NON_PARAMETRIC_HPP
#define ML_NON_PARAMETRIC_HPP
#include <map>
#include <limits>
#include "utility.hpp"


namespace image{

namespace ml{

struct weight_function_inverse_distance
{
    template<class classification_type>
    double operator()(const std::map<double,size_t>& neighbor_list,const std::vector<classification_type>& classification) const
    {
        double predict = 0.0;
        double sum_w = 0.0;
        std::map<double,size_t>::const_iterator iter = neighbor_list.begin();
        std::map<double,size_t>::const_iterator end = neighbor_list.end();
        for (;iter != end;++iter)
        {
            if (iter->first == 0.0)
                return classification[iter->second];
            double w = 1.0/std::sqrt(iter->first);
            predict += ((double)classification[iter->second])*w;
            sum_w += w;
        }
        predict/=sum_w;
        return predict;
    }
};

template<size_t width>
struct weight_function_gaussian
{
    template<class classification_type>
    double operator()(const std::map<double,size_t>& neighbor_list,const std::vector<classification_type>& classification) const
    {
        double predict = 0.0;
        double sum_w = 0.0;
        std::map<double,size_t>::const_iterator iter = neighbor_list.begin();
        std::map<double,size_t>::const_iterator end = neighbor_list.end();
        for (;iter != end;++iter)
        {
            double w = std::exp(-(iter->first)/((double)width));
            predict += ((double)classification[iter->second])*w;
            sum_w += w;
        }
        predict/=sum_w;
        return predict;
    }
};

struct weight_function_average
{
    template<class classification_type>
    double operator()(const std::map<double,size_t>& neighbor_list,const std::vector<classification_type>& classification) const
    {
        double predict = 0.0;
        std::map<double,size_t>::const_iterator iter = neighbor_list.begin();
        std::map<double,size_t>::const_iterator end = neighbor_list.end();
        for (;iter != end;++iter)
            predict += ((double)classification[iter->second]);
        predict/=(double)neighbor_list.size();
        return predict;
    }
};

template<class attribute_type,class classification_type,class weight_function_type>
class nearest_neighbor
{
protected:
    normalized_attributes<attribute_type> attributes;
    std::vector<classification_type> classification;
    size_t attribute_dimension;
    size_t sample_size;
    size_t neighbor_count;
    weight_function_type weighted;
public:
    nearest_neighbor(size_t neighbor_count_ = 3):neighbor_count(neighbor_count_) {}

    template<class attributes_iterator_type>
    void unlearn(attributes_iterator_type attributes_from)
    {
        std::vector<attribute_type> unlearn_attributes(attributes_from,attributes_from+attribute_dimension);
        attributes.normalize(unlearn_attributes.begin());
        for (size_t i = 0;i < sample_size;++i)
        {
            std::vector<attribute_type>& Si = attributes[i];
            bool match = true;
            for (size_t index = 0;index < attribute_dimension;++index)
                if (unlearn_attributes[index] != Si[index])
                {
                    match = false;
                    break;
                }
            if (match)
            {
                attributes[i].swap(attributes.back());
                attributes.pop_back();
                std::swap(classification[i],classification.back());
                classification.resize(sample_size-1);
                --sample_size;
                return;
            }
        }
    }

    template<class attributes_iterator_type,class classifications_type>
    void learn(attributes_iterator_type attributes_from,
               classifications_type classification_)
    {
        attributes.push_back(attributes_from);
        classification.push_back(classification_);
        ++sample_size;
    }

    template<class attributes_iterator_type,class classifications_iterator_type>
    void learn(attributes_iterator_type attributes_from,
               attributes_iterator_type attributes_to,
               size_t attribute_dimension_,
               classifications_iterator_type classifications_from)
    {
        sample_size = attributes_to-attributes_from;
        attribute_dimension = attribute_dimension_;
        normalized_attributes<attribute_type> new_att(attributes_from,attributes_to,attribute_dimension);
        attributes.swap(new_att);
        std::vector<classification_type> new_class(classifications_from,classifications_from+sample_size);
        classification.swap(new_class);
    }

    template<class sample_iterator_type>
    double regression(sample_iterator_type predict_attributes_) const
    {
        std::vector<attribute_type> predict_attributes(predict_attributes_,predict_attributes_+attribute_dimension);
        attributes.normalize(predict_attributes.begin());

        std::map<double,size_t> neighbor_list;
        double least_distance = std::numeric_limits<double>::max();
        for (size_t i = 0;i < sample_size;++i)
        {
            const std::vector<attribute_type>& Si = attributes[i];
            double dis = 0.0;
            bool not_good = false;
            for (size_t index = 0;index < attribute_dimension;++index)
            {
                double value = Si[index] - predict_attributes[index];
                value *= value;
                dis += value;
                if (dis > least_distance)
                {
                    not_good = true;
                    break;
                }
            }
            if (not_good)
                continue;
            neighbor_list[dis] = i;
            if (neighbor_list.size() > neighbor_count)
            {
                neighbor_list.erase(--neighbor_list.end());
                least_distance = (--neighbor_list.end())->first;
            }
        }

        return weighted(neighbor_list,classification);
    }
    template<class sample_iterator_type>
    classification_type predict(sample_iterator_type predict_attributes_) const
    {
        return std::floor(regression(predict_attributes_)+0.5);
    }

};


template<class attribute_type,class classification_type,class weight_function_type>
class parzen_estimator : public nearest_neighbor<attribute_type,classification_type,weight_function_type>
{
    typedef nearest_neighbor<attribute_type,classification_type,weight_function_type> parent_type;
private:
    double window_size;
    size_t classification_dimension;
public:
    parzen_estimator(double window_size_ = 3):window_size(window_size_) {}
    template<class attributes_iterator_type,class classifications_iterator_type>
    void learn(attributes_iterator_type attributes_from,
               attributes_iterator_type attributes_to,
               size_t attribute_dimension_,
               classifications_iterator_type classifications_from)
    {
        classification_dimension = (*std::max_element(classifications_from,classifications_from+(size_t)(attributes_to-attributes_from))) + 1;
        nearest_neighbor<attribute_type,classification_type,weight_function_average>::
        learn(attributes_from,attributes_to,attribute_dimension_,classifications_from);
    }

    template<class sample_iterator_type>
    classification_type predict(sample_iterator_type predict_attributes_) const
    {
        return regression(predict_attributes_) >= 0.5 ? 1:0;
    }

    template<class sample_iterator_type>
    double regression(sample_iterator_type predict_attributes_) const
    {
        typename std::vector<attribute_type> predict_attributes(predict_attributes_,predict_attributes_+parent_type::attribute_dimension);
        parent_type::attributes.normalize(predict_attributes.begin());

        double windows_size2 = window_size*window_size*parent_type::attribute_dimension;
        std::map<double,size_t> neighbor_list;
        for (size_t i = 0;i < parent_type::sample_size;++i)
        {
            const std::vector<attribute_type>& Si = parent_type::attributes[i];
            double dis = 0.0;
            bool not_in = false;
            for (size_t index = 0;index < parent_type::attribute_dimension;++index)
            {
                double value = Si[index] - predict_attributes[index];
                value *= value;
                dis += value;
                if (dis > windows_size2)
                {
                    not_in = true;
                    break;
                }
            }
            if (not_in)
                continue;
            neighbor_list[dis] = i;
        }
        return weighted(neighbor_list,parent_type::classification);
    }

};


}// ml

}// image






#endif//ML_NON_PARAMETRIC_HPP
