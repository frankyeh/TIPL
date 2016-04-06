#ifndef ML_DECISION_TREE_HPP
#define ML_DECISION_TREE_HPP
#include <memory>


namespace image{

namespace ml{

template<class attribute_type,class classification_type>
class decision_tree
{

private:
    template<class attributes_iterator_type>
    std::pair<attribute_type,attribute_type> get_value_range(attributes_iterator_type attributes_from,
            attributes_iterator_type attributes_to,size_t attribute_index)
    {
        attribute_type max_value = attributes_from[0][attribute_index];
        attribute_type min_value = attributes_from[0][attribute_index];
        for (;attributes_from != attributes_to;++attributes_from)
        {
            if (attributes_from[0][attribute_index] > max_value)
                max_value = attributes_from[0][attribute_index];
            else
                if (attributes_from[0][attribute_index] < min_value)
                    min_value = attributes_from[0][attribute_index];
        }
        return std::make_pair(max_value,min_value);
    }
    template<class attributes_iterator_type,class classifications_iterator_type>
    double information_gain(attributes_iterator_type attributes_from,
                            attributes_iterator_type attributes_to,
                            classifications_iterator_type classifications_from)
    {
        size_t sample_size = attributes_to-attributes_from;
        size_t n_x0 = 0;
        size_t n_x1 = 0;
        size_t n_y0_x0 = 0;
        size_t n_y1_x0 = 0;
        size_t n_y0_x1 = 0;
        size_t n_y1_x1 = 0;
        for (size_t index = 0;index < sample_size;++index)
        {
            if (attributes_from[index][attribute_index] > param)// X=1
            {
                ++n_x1;
                if (classifications_from[index])
                    ++n_y1_x1;
                else
                    ++n_y0_x1;
            }
            else
                //X = 0
            {
                ++n_x0;
                if (classifications_from[index])
                    ++n_y1_x0;
                else
                    ++n_y0_x0;
            }
        }
        double p_x0 = ((double)n_x0)/((double)sample_size);
        double p_x1 = ((double)n_x1)/((double)sample_size);
        double p_y0_x0 = n_x0 == 0 ? 0.0: ((double)n_y0_x0)/((double)n_x0);
        double p_y1_x0 = n_x0 == 0 ? 0.0: ((double)n_y1_x0)/((double)n_x0);
        double p_y0_x1 = n_x1 == 0 ? 0.0: ((double)n_y0_x1)/((double)n_x1);
        double p_y1_x1 = n_x1 == 0 ? 0.0: ((double)n_y1_x1)/((double)n_x1);
        return -p_x0*((p_y0_x0 == 0 ? 0.0 : p_y0_x0*std::log(p_y0_x0))+
                      (p_y1_x0 == 0 ? 0.0 : p_y1_x0*std::log(p_y1_x0)))
               -p_x1*((p_y0_x1 == 0 ? 0.0 : p_y0_x1*std::log(p_y0_x1))+
                      (p_y1_x1 == 0 ? 0.0 : p_y1_x1*std::log(p_y1_x1)));
    }
private:
    std::auto_ptr<decision_tree<attribute_type,classification_type> > left_tree;
    std::auto_ptr<decision_tree<attribute_type,classification_type> > right_tree;
private:
    size_t attribute_index;
    double param;
    size_t minimum_sample;
    double termination_ratio;
    bool is_end_node;
    bool end_node_decision;
public:
    decision_tree(size_t minimum_sample_,double termination_ratio_):
            minimum_sample(minimum_sample_),termination_ratio(termination_ratio_),is_end_node(false) {}

    template<class attributes_iterator_type,class classifications_iterator_type>
    void learn(attributes_iterator_type attributes_from,
               attributes_iterator_type attributes_to,
               size_t attribute_dimension,
               classifications_iterator_type classifications_from)
    {
        size_t sample_size = attributes_to-attributes_from;
        classifications_iterator_type classifications_to = classifications_from + sample_size;
        // check for termination
        {
            size_t n_y0 = std::count(classifications_from,classifications_to,0);
            size_t n_y1 = sample_size - n_y0;
            if (n_y0 > ((double)sample_size)*termination_ratio ||
                    n_y1 > ((double)sample_size)*termination_ratio ||
                    sample_size < minimum_sample)
            {
                is_end_node = true;
                end_node_decision = n_y1 > n_y0;
                return;
            }
        }
        // decision the best decision
        {

            std::vector<std::pair<size_t,double> > decision_list;
            for (size_t index = 0;index < attribute_dimension;++index)
            {
                std::pair<attribute_type,attribute_type> range = get_value_range(attributes_from,attributes_to,index);
                double step = (range.first-range.second)/5.0;
                for (double value = range.second+step;value <= range.first-step;value += step)
                    decision_list.push_back(std::make_pair(index,value));
            }

            std::vector<double> ig(decision_list.size());
            for (size_t index = 0;index < decision_list.size();++index)
            {
                attribute_index = decision_list[index].first;
                param = decision_list[index].second;
                ig[index] = information_gain(attributes_from,attributes_to,classifications_from);
            }
            size_t best_decision_index = std::max_element(ig.begin(),ig.end())-ig.begin();

            attribute_index = decision_list[best_decision_index].first;
            param = decision_list[best_decision_index].second;
        }
        // split tree
        {
            std::vector<std::vector<attribute_type> > right_attributes;
            std::vector<std::vector<attribute_type> > left_attributes;
            std::vector<classification_type>		  right_classification;
            std::vector<classification_type>		  left_classification;
            right_attributes.reserve(sample_size);
            left_attributes.reserve(sample_size);
            right_classification.reserve(sample_size);
            left_classification.reserve(sample_size);

            for (size_t index = 0;index < sample_size;++index)
                if (attributes_from[index][attribute_index] > param)
                {
                    right_attributes.push_back(
                        std::vector<attribute_type>(
                            &(attributes_from[index][0]),&(attributes_from[index][0])+attribute_dimension));
                    right_classification.push_back(classifications_from[index]);
                }
                else
                {
                    left_attributes.push_back(
                        std::vector<attribute_type>(
                            &(attributes_from[index][0]),&(attributes_from[index][0])+attribute_dimension));
                    left_classification.push_back(classifications_from[index]);
                }

            right_tree.reset(new decision_tree(minimum_sample,termination_ratio));
            left_tree.reset(new decision_tree(minimum_sample,termination_ratio));
            right_tree->learn(right_attributes.begin(),right_attributes.end(),attribute_dimension,right_classification.begin());
            left_tree->learn(left_attributes.begin(),left_attributes.end(),attribute_dimension,left_classification.begin());
            is_end_node = false;
        }
    }

    template<class sample_iterator_type>
    classification_type predict(sample_iterator_type predict_attributes) const
    {
        if (is_end_node)
            return end_node_decision ? 1:0;
        return (predict_attributes[attribute_index] > param) ?
               right_tree->predict(predict_attributes):
               left_tree->predict(predict_attributes);
    }
};

}// ml

}// image


#endif