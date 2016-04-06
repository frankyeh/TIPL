#ifndef ML_EM_HPP
#define ML_EM_HPP
#include "image/numerical/matrix.hpp"


namespace image{

namespace ml{

class multivariate_gaussian
{
    std::vector<double> mean;
    std::vector<double> covariance;
private:
    std::vector<double> iV,id;
    unsigned int dim;
    double constant;
    void assign_covariance(const std::vector<double>& co)
    {
        std::vector<double> covariance_matrix(dim*dim);
        for (unsigned int i = 0,index = 0;i < dim;++i)
            for (unsigned int k = 0;k <= i;++k,++index)
            {
                covariance_matrix[dim*i+k] = co[index];
                covariance_matrix[dim*k+i] = co[index];
            }
        covariance.swap(covariance_matrix);
    }
    void precompute_parameters(void)
    {
        iV.resize(covariance.size());
        id.resize(dim);
        image::mat::eigen_decomposition_sym(covariance.begin(),iV.begin(),id.begin(),image::dyndim(dim,dim));
        constant = 1.0;
        unsigned int dimension = 0;
        for (unsigned int i = 0;i < id.size();++i)
        {
            if (id[0] + id[i] == id[0] || id[i] < 0)
                id[i] = 0;
            else
            {
                constant *= id[i];
                id[i] = 1.0/id[i];
                ++dimension;
            }
        }
        constant = std::exp(-0.5*std::log(constant)-0.918938533204673*((double)dimension));
    }
public:
    multivariate_gaussian(void) {}

    template<class attributes_iterator_type>
    void estimate(attributes_iterator_type attributes_from,
                  attributes_iterator_type attributes_to,
                  unsigned int attribute_dimension)
    {
        unsigned int sample_size = attributes_to-attributes_from;
        dim = attribute_dimension;
        
        std::vector<double> new_mean(attribute_dimension);
        for (unsigned int j = 0;j < sample_size;++j)
            for (unsigned int i = 0;i < attribute_dimension;++i)
                new_mean[i] += attributes_from[j][i];
        for (unsigned int i = 0;i < attribute_dimension;++i)
			new_mean[i] /= (double)sample_size;
        std::vector<double> new_covariance((attribute_dimension+1)*attribute_dimension/2);
        for (unsigned int j = 0;j < sample_size;++j)
        {
            std::vector<double> dx(attribute_dimension);
            for (unsigned int i = 0;i < attribute_dimension;++i)
                dx[i] = new_mean[i]-attributes_from[j][i];
            for (unsigned int i = 0,index = 0;i < attribute_dimension;++i)
                for (unsigned int k = 0;k <= i;++k,++index)
                    new_covariance[index] += dx[i]*dx[k];

        }
        for(unsigned int j = 0;j < new_covariance.size();++j)
			new_covariance[j] /= (double)sample_size;
        assign_covariance(new_covariance);
        mean.swap(new_mean);
        precompute_parameters();
    }

    template<class attributes_iterator_type,class weighting_iterator_type>
    void estimate(attributes_iterator_type attributes_from,
                  attributes_iterator_type attributes_to,
                  unsigned int attribute_dimension,
                  weighting_iterator_type w)
    {
        unsigned int sample_size = attributes_to-attributes_from;
        dim = attribute_dimension;
        double sum_w = std::accumulate(w,w+sample_size,0.0);

        std::vector<double> new_mean(attribute_dimension);
        for (unsigned int j = 0;j < sample_size;++j)
            for (unsigned int i = 0;i < attribute_dimension;++i)
                new_mean[i] += attributes_from[j][i]*w[j];
        for (unsigned int i = 0;i < attribute_dimension;++i)
			new_mean[i] /= sum_w;

        std::vector<double> new_covariance((attribute_dimension+1)*attribute_dimension/2);
        for (unsigned int j = 0;j < sample_size;++j)
        {
            std::vector<double> dx(attribute_dimension);
            for (unsigned int i = 0;i < attribute_dimension;++i)
                dx[i] = new_mean[i]-attributes_from[j][i];
            for (unsigned int i = 0,index = 0;i < attribute_dimension;++i)
                for (unsigned int k = 0;k <= i;++k,++index)
                    new_covariance[index] += w[j]*dx[i]*dx[k];

        }
         for(unsigned int j = 0;j < new_covariance.size();++j)
			new_covariance[j] /= sum_w;
        assign_covariance(new_covariance);
        mean.swap(new_mean);
        precompute_parameters();
    }
    template<class attributes_iterator_type>
    double operator()(attributes_iterator_type attributes)
    {
        std::vector<double> dx(mean);
        for (unsigned int i = 0;i < dx.size();++i)
            dx[i] -= attributes[i];
        std::vector<double> idx(dx.size());
        image::mat::vector_product(iV.begin(),dx.begin(),idx.begin(),image::dyndim(dim,dim));
        double sum = 0.0;
        for (unsigned int i = 0;i < id.size();++i)
            sum += idx[i]*idx[i]*id[i];
        return std::exp(sum*-0.5)*constant;
    }
    const std::vector<double>& get_mean(void) const
    {
        return mean;
    }
    const std::vector<double>& get_covariance(void) const
    {
        return covariance;
    }
};


template<class attribute_type,class classification_type>
struct expectation_maximization
{
    unsigned int k;
    std::vector<multivariate_gaussian> model;
public:
    expectation_maximization(unsigned int k_):k(k_) {}
    unsigned int get_k(void) const
    {
        return model.size();
    }
    const std::vector<double>& get_mean(unsigned int index) const
    {
        return model[index].get_mean();
    }
    const std::vector<double>& get_covariance(unsigned int index) const
    {
        return model[index].get_covariance();
    }


    template<class attributes_iterator_type,class classifications_iterator_type>
    void operator()(attributes_iterator_type attributes,
                    attributes_iterator_type attributes_to,
                    unsigned int attribute_dimension,
                    classifications_iterator_type classifications_from)
    {
        unsigned int sample_size = attributes_to-attributes;
        image::dyndim variance_matrix_type(attribute_dimension,attribute_dimension);
        image::dyndim mean_matrix_type(attribute_dimension,1);
        // initial guess
        model.resize(k);
        for (unsigned int index = 0;index < k;++index)
            model[index].estimate(attributes+(sample_size/k)*index,
                                  attributes+(sample_size/k)*(index+1),attribute_dimension);

        unsigned int change_count = 0;
        std::vector<classification_type> classification(sample_size);
        std::vector<std::vector<double> > T(k);
        std::vector<double> prior(k);
        std::fill(prior.begin(),prior.end(),1.0/((double)k));
        for (unsigned int stable_iteration = 0,total_iteration = 0;stable_iteration <= total_iteration/20;++total_iteration)
        {
            // E-step

            for (unsigned int index = 0;index < k;++index)
            {
                std::vector<double> cur_T(sample_size);
				double prior_index = prior[index];
                for (unsigned int j = 0;j < sample_size;++j)
                    cur_T[j] = model[index](&attributes[j][0])*prior_index;
                T[index].swap(cur_T);
            }
            // normalize T
            for (unsigned int j = 0;j < sample_size;++j)
            {
                double sum = 0.0;
                for (unsigned int index = 0;index < k;++index)
                    sum += T[index][j];
                if (sum != 0.0)
                    for (unsigned int index = 0;index < k;++index)
                        T[index][j] /= sum;
            }
            // determine classification
            change_count = 0;
            for (unsigned int j = 0;j < sample_size;++j)
            {
                unsigned int best_cluster = 0;
                double best_T = T[0][j];
                for (unsigned int index = 1;index < k;++index)
                    if (T[index][j] > best_T)
                    {
                        best_cluster = index;
                        best_T = T[index][j];
                    }
                if (best_cluster != classification[j])
                {
                    classification[j] = best_cluster;
                    ++change_count;
                }
            }

            // normalize prior
            {
                for (unsigned int index = 0;index < k;++index)
                    prior[index] = std::accumulate(T[index].begin(),T[index].end(),0.0);
                double sum = std::accumulate(prior.begin(),prior.end(),0.0);
                
                for (unsigned int index = 0;index < k;++index)
                {
					if (prior[index] == 0.0)
                    {
                        prior[index] = 1.0/((double)k);
                        std::fill(T[index].begin(),T[index].end(),1.0);
                    }
					else
						prior[index] /= sum;
                }
            }

            // M-step
            for (unsigned int index = 0;index < k;++index)
                model[index].estimate(attributes,attributes_to,attribute_dimension,T[index].begin());

            if (!change_count)
                ++stable_iteration;
            else
                stable_iteration = 0;
        }
        std::copy(classification.begin(),classification.end(),classifications_from);
    }
};


}// ml

}// image





#endif//ML_EM_HPP
