#ifndef ML_HMC_HPP
#define ML_HMC_HPP

namespace image{

namespace ml{



template<class attribute_type,class classification_type>
class hidden_markov_chain
{
private:
    likelihood_estimator<classification_type,classification_type> transition; // Pr(Xi|Xi-1)
    likelihood_estimator<attribute_type,classification_type> state; // Pr(Oi|Xi)
    prior_estimator<classification_type> prior; //Pr(X1)
    classification_type hidden_value_space;
private:
    // for online algorithm
    std::vector<std::vector<double> > online_observed_le,online_a;
    std::vector<classification_type> online_classification;
public:
    template<class hidden_iterator_type>
    void learn_prior(hidden_iterator_type hidden_from,
                     hidden_iterator_type hidden_to)
    {
        hidden_value_space = *std::max_element(hidden_from,hidden_to) + 1;
        prior.estimate(hidden_from,hidden_to,hidden_value_space);
    }
    template<class hidden_iterator_type>
    void learn_transition(hidden_iterator_type hidden_from,
                          hidden_iterator_type hidden_to,
                          hidden_iterator_type hidden_parent_from)
    {
        transition.estimate(hidden_from,hidden_to,hidden_parent_from,hidden_value_space);
    }

    template<class observed_iterator_type,class hidden_iterator_type>
    void learn_state(observed_iterator_type observed_from,
                     observed_iterator_type observed_to,
                     size_t observation_dimension,
                     hidden_iterator_type hidden_from)
    {
        state.estimate(observed_from,observed_to,observation_dimension,hidden_from,hidden_value_space);
    }

    template<class observed_iterator_type,class hidden_iterator_type>
    void predict(observed_iterator_type observed_from,
                 observed_iterator_type observed_to,
                 hidden_iterator_type hidden_from)  const
    {
        size_t hidden_dimension = observed_to-observed_from;
        std::vector<std::vector<double> > observed_le(hidden_dimension);
        for (size_t t = 0;t < hidden_dimension;++t)
        {
            observed_le[t].resize(hidden_value_space);
            for (size_t i = 0;i < hidden_value_space;++i)
                observed_le[t][i] = state(observed_from[t],i);
        }
        std::vector<std::vector<double> > a(hidden_dimension);
        for (size_t t = 0;t < hidden_dimension;++t)
        {
            a[t].resize(hidden_value_space);
            for (size_t i = 0;i < hidden_value_space;++i)
            {
                double sum = 0;
                if (t == 0)
                    sum = prior(i);
                else
                    for (size_t j = 0;j < hidden_value_space;++j)
                        sum += a[t-1][j]*transition(i,j); // a(t-1,j)Pr(Xt=i|Xt-1=j)
                a[t][i] = sum*observed_le[t][i];
            }
        }

        std::vector<double> b_previous(hidden_value_space),b_current(hidden_value_space);
        for (int t = hidden_dimension-1;t >= 0;--t)
        {
            if (t == hidden_dimension-1)
                std::fill(b_current.begin(),b_current.end(),1);
            else
                for (size_t i = 0;i < hidden_value_space;++i)
                {
                    double sum = 0;
                    for (size_t j = 0;j < hidden_value_space;++j)
                        sum += observed_le[t+1][j]*transition(j,i)*b_previous[j];
                    b_current[i] = sum;
                    a[t][i] *= sum;
                }
            b_current.swap(b_previous);
            // x* = argmax P(X1,X2,...Xn|O)
            hidden_from[t] = std::max_element(a[t].begin(),a[t].end())-a[t].begin();
        }
    }
public:
    template<class observed_iterator_type>
    void online_add_observation(observed_iterator_type observed_from,bool first_observation,size_t lag)
    {
        if (first_observation)
        {
            online_observed_le.clear();
            online_a.clear();
            online_classification.clear();
        }
        // Update Pr(O|X)
        {
            std::vector<double> new_le(hidden_value_space);
            for (size_t i = 0;i < hidden_value_space;++i)
                new_le[i] = state(*observed_from,i);
            online_observed_le.push_back(std::vector<double>());
            online_observed_le.back().swap(new_le);
        }
        // Update a(i)
        {
            std::vector<double> new_a(hidden_value_space);
            for (size_t i = 0;i < hidden_value_space;++i)
            {
                double sum = 0;
                if (first_observation)
                    sum = prior(i);
                else
                    for (size_t j = 0;j < hidden_value_space;++j)
                        sum += online_a.back()[j]*transition(i,j);
                new_a[i] = sum*online_observed_le.back()[i];
            }
            online_a.push_back(std::vector<double>());
            online_a.back().swap(new_a);
        }

        online_classification.push_back(0);
        int max_t = online_classification.size()-1;
        int min_t = std::max((int)0,(int)max_t-(int)lag);
        std::vector<double> b_previous(hidden_value_space),b_current(hidden_value_space),result(hidden_value_space);
        for (int t = max_t;t >= min_t;--t)
        {

            if (t == max_t)
            {
                std::fill(b_current.begin(),b_current.end(),1);
                std::copy(online_a[t].begin(),online_a[t].end(),result.begin());
            }
            else
                for (size_t i = 0;i < hidden_value_space;++i)
                {
                    double sum = 0;
                    for (size_t j = 0;j < hidden_value_space;++j)
                        sum += online_observed_le[t+1][j]*transition(j,i)*b_previous[j];
                    b_current[i] = sum;
                    result[i] = online_a[t][i] * sum;
                }
            b_current.swap(b_previous);
            // x* = argmax P(X1,X2,...Xn|O)
            online_classification[t] = std::max_element(result.begin(),result.end())-result.begin();
        }
    }
    classification_type online_predict(size_t index) const
    {
        return online_classification[index];
    }
};

}// ml

}// image



#endif//ML_HMC_HPP
