#ifndef ML_ADA_BOOST_HPP
#define ML_ADA_BOOST_HPP

#include <numeric>
#include <cmath>
#include <memory>

namespace image{

namespace ml{

template<typename attribute_type,typename classification_type>
class decision_stump
{
private:
    size_t which_attribute;
    attribute_type threshold;
    bool reverse;
public:
    template<typename attributes_iterator_type,typename sample_weighting_type,typename classifications_iterator_type>
    void learn(attributes_iterator_type attributes_from,
               attributes_iterator_type attributes_to,
               size_t attribute_dimension,
               sample_weighting_type Dt,
               classifications_iterator_type classifications_from)
    {
        size_t sample_size = attributes_to-attributes_from;
        which_attribute = 0;
        threshold = 0;
        reverse = false;

        double classification_number = 0.0;

        for (size_t index = 0;index < attribute_dimension;++index)
        {
            double cur_classification_number = 0.0;
            std::map<attribute_type,double,std::greater<attribute_type> > sorted_attribute;
            for (size_t i = 0;i < sample_size;++i)
                sorted_attribute[attributes_from[i][index]] += (classifications_from[i] ? Dt[i] : -Dt[i]);

            typename std::map<attribute_type,double,std::greater<attribute_type> >::const_iterator iter = sorted_attribute.begin();
            typename std::map<attribute_type,double,std::greater<attribute_type> >::const_iterator end = sorted_attribute.end();

            for (;iter != end;++iter)
            {
                cur_classification_number -= iter->second;
                if (std::abs(cur_classification_number) > classification_number)
                {
                    classification_number = std::abs(cur_classification_number);
                    which_attribute = index;
                    threshold = iter->first;
                    reverse = cur_classification_number > 0.0;
                }
            }
        }
    }

    template<typename attribute_iterator_type>
    classification_type predict(attribute_iterator_type attributes) const
    {
        return ((attributes[which_attribute] >= threshold) ^ reverse) ? 1:0;
    }

    friend std::ostream& operator<<(std::ostream& out,const decision_stump& rhs)
    {
        out << "X(" << rhs.which_attribute << ") " << (rhs.reverse ? "< " : ">= ") << rhs.threshold;
        return out;
    }
};


template<typename method_type>
class ada_boost
{
private:
    std::vector<method_type*> methods;
    std::vector<double> alphas;
    double sum_alpha;
    double alpha_threshold;
    size_t max_iteration;
    void clear(void)
    {
        for (size_t index = 0;index < methods.size();++index)
            delete methods[index];
        methods.clear();
        alphas.clear();

    }
public:
    ada_boost(double alpha_threshold_ = 0.01):alpha_threshold(alpha_threshold_),max_iteration(100) {}
    ada_boost(size_t max_iteration_):alpha_threshold(0.0),max_iteration(max_iteration_) {}

    const method_type& get(size_t index) const
    {
        return *(methods[index]);
    }
    template<typename attributes_iterator_type,typename classifications_iterator_type>
    void learn(attributes_iterator_type attributes_from,
               attributes_iterator_type attributes_to,
               size_t attribute_dimension,
               classifications_iterator_type classifications_from)
    {
        clear();
        size_t sample_size = attributes_to-attributes_from;
        std::vector<double> Dt(sample_size);
        std::fill(Dt.begin(),Dt.end(),1.0/((double)sample_size));
        for (size_t t = 0;t < max_iteration;++t)
        {
            std::auto_ptr<method_type> ht(new method_type);
            ht->learn(attributes_from,attributes_to,attribute_dimension,Dt.begin(),classifications_from);
            std::vector<unsigned char> correct_predict(sample_size);
            // get the classification result 1 for correc predict, 0 otherwise
            {
                for (size_t index = 0;index < sample_size;++index)
                    correct_predict[index] = (ht->predict(&(attributes_from[index][0])) == classifications_from[index]) ? 1:0;
            }
            //calculate £`t
            double et;
            {
                double wrong = 0.0;
                for (size_t index = 0;index < sample_size;++index)
                    if (!correct_predict[index])
                        wrong += Dt[index];
                et = wrong;
            }
            if (et == 0.5)
                break;
            if (et == 0.0 || et == 1.0)
            {
                alphas.push_back(1.0);
                methods.push_back(ht.release());
                break;
            }
            // calculate alpha
            double alpha = 0.5*std::log((1.0-et)/et);
            if (std::abs(alpha) < alpha_threshold)
                break;
            double z = 2.0*std::sqrt(et*(1.0-et));
            for (size_t index = 0;index < sample_size;++index)
                Dt[index] *= (correct_predict[index]) ? std::exp(-alpha)/z : std::exp(alpha)/z;
            alphas.push_back(alpha);
            methods.push_back(ht.release());
        }
        sum_alpha = std::accumulate(alphas.begin(),alphas.end(),0.0);
    }

    ~ada_boost(void)
    {
        clear();
    }

    template<typename sample_iterator_type>
    unsigned char predict(sample_iterator_type predict_attributes)  const
    {
        double vote_result = 0.0;
        for (size_t index = 0;index < methods.size();++index)
            vote_result += methods[index]->predict(predict_attributes) ? alphas[index] : -alphas[index];
        return (vote_result >= 0.0) ? 1 : 0;
    }

    template<typename sample_iterator_type>
    double regression(sample_iterator_type predict_attributes)  const
    {
        double vote_result = 0.0;
        for (size_t index = 0;index < methods.size();++index)
            vote_result += methods[index]->predict(predict_attributes) ? alphas[index] : -alphas[index];
        vote_result /= sum_alpha;
        return (vote_result+1.0)/2.0;
    }
};


}// ml

}// image

#endif//ML_ADA_BOOST_HPP
