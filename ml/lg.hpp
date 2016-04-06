#ifndef ML_LG_HPP
#define ML_LG_HPP


namespace image{

namespace ml{

template<class attribute_type,class classification_type>
class logistic_regression
{

private:
    size_t attribute_dimension;
    double learning_rate;
    double precision;
private:
    std::vector<double> w;

    template<class sample_iterator_type>
    void get_gradient_single(sample_iterator_type sample,classification_type classification,double* dw)
    {
        double dy = classification;
        dy -= estimate_posterior(sample,1);
        dw[0] += dy;
        ++dw;
        for (size_t index = 0;index < attribute_dimension;++index)
            dw[index] += sample[index]*dy;
    }
    template<class sample_iterator_type,class sample_weighted_type>
    void get_gradient_single(sample_iterator_type sample,sample_weighted_type Dt,classification_type classification,double* dw)
    {
        double dy = classification;
        dy -= estimate_posterior(sample,1);
        dy *= Dt;
        dw[0] += dy;
        ++dw;
        for (size_t index = 0;index < attribute_dimension;++index)
            dw[index] += sample[index]*dy;
    }
    template<class attributes_iterator_type,class classifications_iterator_type>
    void get_gradient(attributes_iterator_type attributes_from,
                      attributes_iterator_type attributes_to,
                      classifications_iterator_type classifications_from,double* dw)
    {
        size_t sample_size = attributes_to-attributes_from;
        for (;attributes_from != attributes_to;++attributes_from,++classifications_from)
            get_gradient_single(&((*attributes_from)[0]),*classifications_from,dw);
    }
    template<class attributes_iterator_type,class sample_weighting_type,class classifications_iterator_type>
    void get_gradient(attributes_iterator_type attributes_from,
                      attributes_iterator_type attributes_to,
                      sample_weighting_type Dt,
                      classifications_iterator_type classifications_from,double* dw)
    {
        size_t sample_size = attributes_to-attributes_from;
        for (;attributes_from != attributes_to;++attributes_from,++Dt,++classifications_from)
            get_gradient_single(&((*attributes_from)[0]),*Dt,*classifications_from,dw);
    }

public:
    logistic_regression(const classifier_parameters& params_):learning_rate(params_.param1),
            precision(params_.param2) {}

    logistic_regression(double learning_rate_ = 0.1,double precision_ = 0.001)
            :learning_rate(learning_rate_),
            precision(precision_) {}

    template<class sample_iterator_type>
    double estimate_posterior(sample_iterator_type sample,classification_type classification) const
    {
        double inner_product = w[0];
        std::vector<double>::const_iterator w_ = w.begin()+1;
        for (size_t index = 0;index < attribute_dimension;++index)
            inner_product += w_[index]*((double)sample[index]);
        double exp_wx = std::exp((classification) ? -inner_product : inner_product);
        return 1.0/(1.0+exp_wx);
    }

    template<class attributes_iterator_type,class classifications_iterator_type>
    void learn(attributes_iterator_type attributes_from,
               attributes_iterator_type attributes_to,
               size_t attribute_dimension_,
               classifications_iterator_type classifications_from)
    {
        attribute_dimension = attribute_dimension_;
        w.clear();
        w.resize(attribute_dimension+1);

        std::vector<double> dw(w);
        for (size_t index = 0;index < 500;++index)
        {
            std::fill(dw.begin(),dw.end(),0.0);
            get_gradient(attributes_from,attributes_to,classifications_from,&dw[0]);
			double rate = learning_rate/(double)(attributes_to-attributes_from);
            double product_ww = 0.0;
			for(size_t i = 0;i < dw.size();++i)
			{
				dw[i] *= rate;
				w[i] += dw[i];
				product_ww += dw[i]*dw[i];
			}
			
            if (std::sqrt(product_ww) < precision)
                break;
        }
    }


    template<class attributes_iterator_type,class sample_weighting_type,class classifications_iterator_type>
    void learn(attributes_iterator_type attributes_from,
               attributes_iterator_type attributes_to,
               size_t attribute_dimension_,
               sample_weighting_type Dt,
               classifications_iterator_type classifications_from)
    {
        attribute_dimension = attribute_dimension_;
        w.clear();
        w.resize(attribute_dimension+1);

        std::vector<double> dw(w);
        for (size_t index = 0;index < 500;++index)
        {
            std::fill(dw.begin(),dw.end(),0.0);
            get_gradient(attributes_from,attributes_to,Dt,classifications_from,&dw[0]);
            
			double rate = learning_rate/(double)(attributes_to-attributes_from);
            double product_ww = 0.0;
			for(size_t i = 0;i < dw.size();++i)
			{
				dw[i] *= rate;
				w[i] += dw[i];
				product_ww += dw[i]*dw[i];
			}
			
            if (std::sqrt(product_ww) < precision)
                break;
        }
    }


    template<class sample_iterator_type>
    classification_type predict(sample_iterator_type attributes)  const
    {
        return (estimate_posterior(attributes,1) >= 0.5 )? 1:0;
    }

    template<class sample_iterator_type>
    double regression(sample_iterator_type attributes) const
    {
        return estimate_posterior(attributes,1);
    }

};

}// ml

}// image


#endif//ML_LG_HPP
