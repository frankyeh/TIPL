#ifndef ML_NB_HPP
#define ML_NB_HPP
#include <vector>
#include <cmath>


namespace image{

namespace ml{

template<class classification_type>
class prior_estimator;

template<>
class prior_estimator<unsigned char>
{
private:
    std::vector<double> p_table;
public:
    template<class classifications_iterator_type>
    void estimate(classifications_iterator_type classifications_from,
                  classifications_iterator_type classifications_to)
    {
        estimate(classifications_from,classifications_to,
                 *std::max_element(classifications_from,classifications_to)+1);
    }
    template<class classifications_iterator_type>
    void estimate(classifications_iterator_type classifications_from,
                  classifications_iterator_type classifications_to,
                  unsigned char value_space)
    {
        std::vector<size_t> y_count(value_space);
        size_t sample_size = classifications_to-classifications_from;
        for (size_t index = 0;index < sample_size;++index)
            ++y_count[(unsigned char)classifications_from[index]];
        p_table.clear();
        p_table.resize(value_space);
        for (size_t index = 0;index < value_space;++index)
            p_table[index] = ((double)y_count[index])/((double)sample_size);
    }
    template<class classifications_iterator_type,class weighting_iterator_type>
    void estimate(classifications_iterator_type classifications_from,
                  classifications_iterator_type classifications_to,
                  const weighting_iterator_type weighting_from)
    {
        estimate(classifications_from,classifications_to,weighting_from,
                 *std::max_element(classifications_from,classifications_to)+1);
    }
    template<class classifications_iterator_type,class weighting_iterator_type>
    void estimate(classifications_iterator_type classifications_from,
                  classifications_iterator_type classifications_to,
                  const weighting_iterator_type weighting_from,
                  unsigned char value_space)
    {
        typename std::vector<class std::iterator_traits<weighting_iterator_type>::value_type> y_count(value_space);
        size_t sample_size = classifications_to-classifications_from;
        for (size_t index = 0;index < sample_size;++index)
            y_count[(unsigned char)classifications_from[index]] += weighting_from[index];
        p_table.clear();
        p_table.resize(value_space);
        for (size_t index = 0;index < value_space;++index)
            p_table[index] = ((double)y_count[index])/((double)sample_size);
    }
    //return Pr(Y)
    double operator()(unsigned char classification) const
    {
        return classification < p_table.size() ? p_table[classification] : 0;
    }
};

// Modelizing Pr(Attribute|Classification)
template<class attribute_type,class classification_type>
class likelihood_estimator;


// discrete features
template<>
class likelihood_estimator<unsigned char,unsigned char>
{
private:
    std::vector<double> cp_table; // stores P(L= l|X = x), X is the features, L is the labeling
public:
    template<class attributes_iterator_type,class classifications_iterator_type>
    void estimate(attributes_iterator_type attributes_from,
                  attributes_iterator_type attributes_to,
                  classifications_iterator_type classifications_from)
    {
        estimate(attributes_from,attributes_to,classifications_from,
                 *std::max_element(classifications_from,classifications_from+(attributes_to-attributes_from))+1);
    }
    template<class attributes_iterator_type,class classifications_iterator_type>
    void estimate(attributes_iterator_type attributes_from,
                  attributes_iterator_type attributes_to,
                  classifications_iterator_type classifications_from,
                  unsigned char value_space)
    {
        std::vector<size_t> table(256*value_space);//# of X|Y
        std::vector<size_t> y_count(value_space);
        size_t sample_size = attributes_to-attributes_from;
        for (size_t index = 0;index < sample_size;++index)
        {
            unsigned char x = attributes_from[index];
            unsigned char y = classifications_from[index];
            size_t store_index = y;
            store_index <<= 8;
            store_index += x;
            ++table[store_index];
            ++y_count[y];
        }
        cp_table.clear();
        cp_table.resize(256*value_space);
        for (size_t y = 0,index = 0;y < value_space;++y)
        {
            if (y_count[y] == 0)
            {
                index += 256;
                continue;
            }
            double y_sum = (double)y_count[y];
            for (size_t x = 0;x < 256;++x,++index)
                cp_table[index] = ((double)table[index])/y_sum;
        }
    }
    //return Pr(att|classification)
    double operator()(unsigned char att,unsigned char classification) const
    {
        size_t index = classification;
        index <<= 8;
        index += att;
        return cp_table[index];
    }
};



// continuous features
// assume Gaussian distribution
template<>
class likelihood_estimator<double,unsigned char>
{
private:
    std::vector<double> mean,variance;
    std::vector<double> constant;
public:
    template<class attributes_iterator_type,class classifications_iterator_type>
    void estimate(attributes_iterator_type attributes_from,
                  attributes_iterator_type attributes_to,
                  classifications_iterator_type classifications_from)
    {
        estimate(attributes_from,attributes_to,classifications_from,
                 *std::max_element(classifications_from,classifications_from+(attributes_to-attributes_from))+1);
    }
    template<class attributes_iterator_type,class classifications_iterator_type>
    void estimate(attributes_iterator_type attributes_from,
                  attributes_iterator_type attributes_to,
                  classifications_iterator_type classifications_from,
                  unsigned char value_space)
    {
        mean.resize(value_space);
        variance.resize(value_space);
        constant.resize(value_space);
        size_t sample_size = attributes_to-attributes_from;
        for (size_t y = 0;y < value_space;++y)
        {
            double sum = 0.0;
            double sum2 = 0.0;
            size_t count = 0;
            for (size_t i = 0;i < sample_size;++i)
            {
                if (classifications_from[i] != y)
                    continue;
                double value = attributes_from[i];
                sum += value;
                sum2 += value*value;
                ++count;
            }
            mean[y] = sum/((double)count);
            variance[y] = sum2/((double)count) - mean[y]*mean[y];
            constant[y] = std::sqrt(2*3.14159265358979323846264338328f*variance[y]);
        }

    }
    //return Pr(X|Y)
    double operator()(double att,unsigned char classification) const
    {
        if (classification >= mean.size())
            return 0;
        double dis = att;
        dis -= mean[classification];
        dis *= dis;
        dis /= -2.0*variance[classification];
        dis = std::exp(dis);
        dis /= constant[classification];
        return dis;
    }
};




// Pr(X1,X2,X3,...,Xn|Y)=Pr(X1|Y)Pr(X2|Y)...Pr(Xn|Y)
template<class attribute_type>
class likelihood_estimator<std::vector<attribute_type>,unsigned char>
{
private:
    template<class iterator_type,class attribute_type_>
    struct attribute_selector
    {
        typedef typename std::iterator_traits<iterator_type>::value_type attribut_iterator_type;
        iterator_type iter;
        size_t att_id;
        attribute_selector(iterator_type iter_,size_t att_id_):iter(iter_),att_id(att_id_) {}

        attribute_type_ operator[](size_t index)
        {
            return iter[index][att_id];
        }

        size_t operator-(const attribute_selector& rhs)
        {
            return iter-rhs.iter;
        }
    };


    std::vector<likelihood_estimator<attribute_type,unsigned char> > le_list;
public:
    template<class attributes_iterator_type,class classifications_iterator_type>
    void estimate(attributes_iterator_type attributes_from,
                  attributes_iterator_type attributes_to,
                  size_t attribute_dimension,
                  classifications_iterator_type classifications_from)
    {
        estimate(attributes_from,attributes_to,attribute_dimension,classifications_from,
                 *std::max_element(classifications_from,classifications_from)+1);
    }
    template<class attributes_iterator_type,class classifications_iterator_type>
    void estimate(attributes_iterator_type attributes_from,
                  attributes_iterator_type attributes_to,
                  size_t attribute_dimension,
                  classifications_iterator_type classifications_from,
                  unsigned char value_space)
    {
        le_list.clear();
        le_list.resize(attribute_dimension);
        for (size_t index = 0;index < le_list.size();++index)
            le_list[index].estimate(
                attribute_selector<attributes_iterator_type,attribute_type>(attributes_from,index),
                attribute_selector<attributes_iterator_type,attribute_type>(attributes_to,index),
                classifications_from,value_space);
    }
    //return Pr(att|classification)
    template<class attribute_iterator_type>
    double operator()(attribute_iterator_type att,unsigned char classification) const
    {
        double product = 1.0;
        for (size_t index = 0;index < le_list.size();++index)
            product *= le_list[index](att[index],classification);
        return product;
    }
};



// Modelizing Pr(Classification|Attribute)
template<class attribute_type,class classification_type>
class posterior_estimator;

template<class attribute_type>
class posterior_estimator<std::vector<attribute_type>,unsigned char>
{
    likelihood_estimator<std::vector<attribute_type>,unsigned char> likelihoods;
    prior_estimator<unsigned char> prior;
public:
    template<class attributes_iterator_type,class classifications_iterator_type>
    void estimate(attributes_iterator_type attributes_from,
                  attributes_iterator_type attributes_to,
                  size_t attribute_dimension,
                  classifications_iterator_type classifications_from)
    {
        estimate(attributes_from,attributes_to,attribute_dimension,classifications_from,
                 *std::max_element(classifications_from,classifications_from)+1);
    }
    template<class attributes_iterator_type,class classifications_iterator_type>
    void estimate(attributes_iterator_type attributes_from,
                  attributes_iterator_type attributes_to,
                  size_t attribute_dimension,
                  classifications_iterator_type classifications_from,
                  unsigned char value_space)
    {
        likelihoods.estimate(attributes_from,attributes_to,attribute_dimension,classifications_from,value_space);
        prior.estimate(classifications_from,classifications_from+(attributes_to-attributes_from),value_space);
    }
    //return Pr(att|classification)
    template<class attribute_iterator_type>
    double operator()(attribute_iterator_type attributes,unsigned char classification) const
    {
        return likelihoods(attributes,classification)*prior(classification);
    }
};



template<class attribute_type,class classification_type>
class naive_bayes
{
private:
    //std::vector<likelihood_estimator<attribute_type,classification_type> > likelihoods;
    posterior_estimator<std::vector<attribute_type>,classification_type> posterior;
    classification_type classification_dimension;
private:

public:
    template<class attributes_iterator_type,class classifications_iterator_type>
    void learn(attributes_iterator_type attributes_from,
               attributes_iterator_type attributes_to,
               size_t attribute_dimension,
               classifications_iterator_type classifications_from)
    {
        classification_dimension = (*std::max_element(classifications_from,classifications_from+(attributes_to-attributes_from))) + 1;
        posterior.estimate(attributes_from,attributes_to,attribute_dimension,classifications_from,classification_dimension);
    }

    template<class sample_iterator_type>
    double estimate_posterior(sample_iterator_type attributes,classification_type classification) const
    {
        return posterior(attributes,classification);
    }

    template<class sample_iterator_type>
    classification_type predict(sample_iterator_type attributes) const
    {
        std::vector<double> posterior(classification_dimension);
        for (size_t index = 0;index < classification_dimension;++index)
            posterior[index] = estimate_posterior(attributes,index);
        return std::max_element(posterior.begin(),posterior.end())-posterior.begin();
    }
};

}// ml

}// image


#endif///ML_NB_HPP
