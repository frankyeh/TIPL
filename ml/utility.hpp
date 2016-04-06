#ifndef ML_UTILITY_HPP
#define ML_UTILITY_HPP
#include <vector>
#include <cmath>
#include <set>


namespace image
{

namespace ml
{

template<class classification_type>
struct evaluate_error_imp
{
    template<class training_model,class attirubte_iterator_type,class classification_input_type>
    double operator()(const training_model& model,
                      attirubte_iterator_type data_from,
                      attirubte_iterator_type data_to,
                      classification_input_type classification) const
    {
        size_t wrong = 0;
        double sample_size = data_to - data_from;
        for(; data_from != data_to; ++data_from, ++classification)
            if(model.predict(&(data_from[0][0])) != (*classification))
                ++wrong;
        return ((double)wrong) / sample_size;
    }
};

// continous
template<>
struct evaluate_error_imp<float>
{
    template<class training_model,class attirubte_iterator_type,class classification_type>
    double operator()(const training_model& model,
                      attirubte_iterator_type data_from,
                      attirubte_iterator_type data_to,
                      classification_type classification) const
    {
        double sum_error = 0.0;
        double sample_size = data_to - data_from;
        for(; data_from != data_to; ++data_from, ++classification)
        {
            double value = model.regression(&(data_from[0][0]));
            value -= *classification;
            sum_error += value * value;
        }
        sum_error /= sample_size;
        return sum_error;
    }
};


// continous
template<>
struct evaluate_error_imp<double>
{
    template<class training_model,class attirubte_iterator_type,class classification_type>
    double operator()(const training_model& model,
                      attirubte_iterator_type data_from,
                      attirubte_iterator_type data_to,
                      classification_type classification) const
    {
        return evaluate_error_imp<float>()(model, data_from, data_to, classification);
    }
};


template<class training_model,class attirubte_iterator_type,class classification_type>
double evaluate_error(const training_model& model,
                      attirubte_iterator_type data_from,
                      attirubte_iterator_type data_to,
                      classification_type classification)
{
    return evaluate_error_imp<class std::iterator_traits<classification_type>::value_type>()(model, data_from, data_to, classification);
}

template<class feature_type_,class class_type_>
class training_data
{
public:
    typedef feature_type_ feature_type;
    typedef class_type_ class_type;
    std::vector<std::vector<feature_type> > features;
    std::vector<class_type> classification;
public:
	template<class training_model>
	void train(training_model& model) const
	{
        model.learn(features.begin(), features.end(), features.front().size(), classification.begin());
	}
	template<class training_model>
    void train(training_model& model, size_t from, size_t to) const
	{
        model.learn(features.begin() + from, features.begin() + to, features.front().size(), classification.begin() + from);
	}
	template<class model_type>
	double evaluate_error(const model_type& model) const
	{
        return image::ml::evaluate_error(model, features.begin(), features.end(), classification.begin());
	}
	template<class model_type>
    double evaluate_error(const model_type& model, size_t from, size_t to) const
	{
        return image::ml::evaluate_error(model, features.begin() + from, features.begin() + to, classification.begin() + from);
	}

};

template<class feature_type_,class class_type_>
class subsampled_data
{
public:
    typedef feature_type_ feature_type;
    typedef class_type_ class_type;
	std::vector<class std::vector<feature_type>::const_iterator> features;
    std::vector<class_type> classification;
	size_t feature_count;
public:
	template<class training_model>
	void train(training_model& model) const
	{
        model.learn(features.begin(), features.end(), feature_count, classification.begin());
	}
	template<class training_model>
    void train(training_model& model, size_t from, size_t to) const
	{
        model.learn(features.begin() + from, features.begin() + to, feature_count, classification.begin() + from);
	}

    void subsample(const training_data<feature_type, class_type>& data, size_t n)
	{
		// randomize all the data
        std::vector<size_t> new_sequence(data.features.size());
        for(size_t index = 0; index < new_sequence.size(); ++index)
            new_sequence[index] = index;
        std::random_shuffle(new_sequence.begin(), new_sequence.end());

		features.resize(n);
		classification.resize(n);
        for(size_t index = 0; index < n; ++index)
        {
            features[index] = data.features[new_sequence[index]].begin();
            classification[index] = data.classification[new_sequence[index]];
        }
		feature_count = data.features.front().size();
	}
    void subsample(const subsampled_data<feature_type, class_type>& data, size_t n)
	{
		feature_count = data.feature_count;
		// randomize all the data
        std::vector<size_t> new_sequence(data.features.size());
        for(size_t index = 0; index < new_sequence.size(); ++index)
            new_sequence[index] = index;
        std::random_shuffle(new_sequence.begin(), new_sequence.end());

		features.resize(n);
		classification.resize(n);
        for(size_t index = 0; index < n; ++index)
        {
            features[index] = data.features[new_sequence[index]];
            classification[index] = data.classification[new_sequence[index]];
        }
	}

	template<class model_type>
	double evaluate_error(const model_type& model) const
	{
        return image::ml::evaluate_error(model, features.begin(), features.end(), classification.begin());
	}
	template<class model_type>
    double evaluate_error(const model_type& model, size_t from, size_t to) const
	{
        return image::ml::evaluate_error(model, features.begin() + from, features.begin() + to, classification.begin() + from);
	}
};


/** perform cross validation
    return training and test error
*/

template<class training_data_type,class training_model>
double cross_validation(const training_data_type& data, training_model& model, int fold = 3, int run = 5)
{
    size_t sample_size = data.features.size();
    size_t training_size = sample_size - data.features.size() / fold;
    subsampled_data<class training_data_type::feature_type,class training_data_type::class_type> sub_data;
    double sum_test_error = 0.0;
    for(size_t iteration = 0; iteration < run; ++iteration)
    {
        sub_data.subsample(data, sample_size);
        sub_data.train(model, 0, training_size);
        sum_test_error += sub_data.evaluate_error(model, training_size, sample_size);
    }
    return 1.0 - sum_test_error / ((double)run);
}

template<class training_data_type,class training_model>
double leave_n_out_cross_validation(const training_data_type& data, training_model& model, size_t n)
{
    size_t sample_size = data.features.size();
    subsampled_data<class training_data_type::feature_type,class training_data_type::class_type> sub_data;
    sub_data.subsample(data, sample_size);
    sub_data.train(model);
    double sum_error = 0.0;
    size_t total_trial = 0;
    for(size_t index = 0; index < sample_size; index += n)
    {
        if(index + n > sample_size)
            break;
        for(size_t i = 0; i < n; ++i)
            model.unlearn(sub_data.features[i + index]);
        sum_error += sub_data.evaluate_error(model, index, index + n);
        ++total_trial;
        for(size_t i = 0; i < n; ++i)
            model.learn(sub_data.features[i + index], sub_data.classification[i + index]);
    }
    return 1.0 - sum_error / ((double)total_trial);
}



/** the input parameters for the classifier*/
struct classifier_parameters
{
    double param1;
    double param2;
    double param3;
    double param4;
    classifier_parameters(double p1 = 0.0, double p2 = 0.0, double p3 = 0.0, double p4 = 0.0):
        param1(p1), param2(p2), param3(p3), param4(p4) {}
};


template<class classifier_type,class attribute_type,class classification_type>
class multiple_class_classifier
{
private:
    std::vector<classifier_type*> classifiers;
    std::vector<classification_type> classifier_labels;
    void clear(void)
    {
        for(size_t index = 0; index < classifiers.size(); ++index)
            delete classifiers[index];
    }
    classifier_parameters params;
public:
    multiple_class_classifier(const classifier_parameters& param_): params(param_) {}
    multiple_class_classifier(void) {}
    template<class attributes_iterator_type,class classifications_iterator_type>
    void learn(attributes_iterator_type attributes_from,
               attributes_iterator_type attributes_to,
               size_t attribute_dimension_,
               classifications_iterator_type classifications_from)
    {
        clear();
        size_t sample_size = attributes_to - attributes_from;
        {
            std::set<classification_type> classifications(classifications_from, classifications_from + sample_size);
            classifier_labels.swap(std::vector<classification_type>(classifications.begin(), classifications.end()));
        }
        classifiers.resize(classifier_labels.size());
        for(size_t index = 0; index < classifier_labels.size(); ++index)
        {
            classifiers[index] = new classifier_type(params);
            std::vector<classification_type> new_classifications(sample_size);
            size_t cur_class = classifier_labels[index];
            for(size_t j = 0; j < sample_size; ++j)
                new_classifications[j] = (classifications_from[j] == cur_class) ? 1 : 0;
            classifiers[index]->learn(attributes_from, attributes_to, attribute_dimension_, new_classifications.begin());
        }
    }
    template<class sample_iterator_type>
    classification_type predict(sample_iterator_type attributes) const
    {
        classification_type best_label = 0;
        double best_label_prob = 0.0;
        for(size_t index = 0; index < classifier_labels.size(); ++index)
        {
            double value = classifiers[index]->regression(attributes);
            if(value > best_label_prob)
            {
                best_label_prob = value;
                best_label = classifier_labels[index];
            }
        }
        return best_label;
    }
    template<class sample_iterator_type>
    classification_type regression(sample_iterator_type attributes) const
    {
        double sum = 0.0;
        double values = 0.0;
        for(size_t index = 0; index < classifier_labels.size(); ++index)
        {
            values += classifiers[index]->regression(attributes) * ((double)classifier_labels[index]);
            sum += classifier_labels[index];
        }
        return values / sum;
    }
};

template<class attribute_type>
class normalized_attributes
{
private:
    std::vector<std::vector<attribute_type> > attributes;
    std::vector<double> X_std;
public:
    normalized_attributes(void) {}
    template<class attribute_input_iterator>
    normalized_attributes(attribute_input_iterator attributes_from,
                          attribute_input_iterator attributes_to,
                          size_t attribute_dimension):
        attributes(attributes_to - attributes_from),
        X_std(attribute_dimension)
    {
        size_t sample_size = attributes_to - attributes_from;
        for(size_t index = 0; index < sample_size; ++index)
            attributes[index] = std::vector<attribute_type>(&(attributes_from[index][0]), &(attributes_from[index][0]) + attribute_dimension);

        for(size_t index = 0; index < attribute_dimension; ++index)
        {
            double sum_x2 = 0.0;
            double sum_x = 0.0;
            for(size_t i = 0; i < sample_size; ++i)
            {
                double value = attributes[i][index];
                sum_x += value;
                sum_x2 += value * value;
            }
            sum_x /= ((double)sample_size);
            X_std[index] = std::sqrt(sum_x2 / ((double)sample_size) - sum_x * sum_x);
            if(X_std[index] == 0)
                X_std[index] = 1.0;
        }
        for(size_t j = 0; j < attribute_dimension; ++j)
        {
            if(X_std[j] == 0.0)
                continue;
            double x_std = X_std[j];
            for(size_t i = 0; i < sample_size; ++i)
                attributes[i][j] /= x_std;
        }
    }

    template<class attribute_input_iterator>
    normalized_attributes(attribute_input_iterator attributes_from,
                          attribute_input_iterator attributes_to):
        attributes(attributes_to - attributes_from),
        X_std(1)
    {
        size_t sample_size = attributes_to - attributes_from;
        for(size_t index = 0; index < sample_size; ++index)
            attributes[index].push_back(attributes_from[index]);

        {
            double sum_x2 = 0.0;
            double sum_x = 0.0;
            for(size_t i = 0; i < sample_size; ++i)
            {
                double value = attributes[i][0];
                sum_x += value;
                sum_x2 += value * value;
            }
            sum_x /= ((double)sample_size);
            X_std[0] = std::sqrt(sum_x2 / ((double)sample_size) - sum_x * sum_x);
            if(X_std[0] == 0)
                X_std[0] = 1.0;
        }
        for(size_t j = 0; j < attributes[0].size(); ++j)
        {
            if(X_std[j] == 0.0)
                continue;
            double x_std = X_std[j];
            for(size_t i = 0; i < sample_size; ++i)
                attributes[i][j] /= x_std;
        }
    }

    void swap(normalized_attributes& rhs)
    {
        attributes.swap(rhs.attributes);
        X_std.swap(rhs.X_std);
    }
    template<class attribute_input_iterator>
    void normalize(attribute_input_iterator att)  const
    {
        for(size_t index = 0; index < X_std.size(); ++index)
            att[index] /= X_std[index];
    }

    std::vector<attribute_type>& operator[](size_t index)
    {
        return attributes[index];
    }
    const std::vector<attribute_type>& operator[](size_t index) const
    {
        return attributes[index];
    }
    std::vector<attribute_type>& back(void)
    {
        return attributes.back();
    }
    const std::vector<attribute_type>& back(void) const
    {
        return attributes.back();
    }

    template<class attribute_input_iterator>
    void push_back(attribute_input_iterator att)
    {
        attributes.push_back(std::vector<attribute_type>());
        attributes.back().swap(std::vector<attribute_type>(att, att + X_std.size()));
        normalize(attributes.back().begin());
    }

    void pop_back(void)
    {
        attributes.pop_back();
    }

    size_t size(void) const
    {
        return attributes.size();
    }
    size_t attribute_dimension(void) const
    {
        return attributes.front().size();
    }
};

}// ml

}// image


#endif//ML_UTILITY_HPP
