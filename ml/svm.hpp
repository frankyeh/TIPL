#ifndef ML_SVM_HPP
#define ML_SVM_HPP
#include <memory>
#include <vector>
#define LIBSVM_VERSION 317

namespace image{

namespace ml{


extern int libsvm_version;

struct svm_node
{
    int index;
    double value;
};

struct svm_problem
{
    int l;
    double *y;
    struct svm_node **x;
};

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

struct svm_parameter
{
    int svm_type;
    int kernel_type;
    int degree;	/* for poly */
    double gamma;	/* for poly/rbf/sigmoid */
    double coef0;	/* for poly/sigmoid */

    /* these are for training only */
    double cache_size; /* in MB */
    double eps;	/* stopping criteria */
    double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
    int nr_weight;		/* for C_SVC */
    int *weight_label;	/* for C_SVC */
    double* weight;		/* for C_SVC */
    double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
    double p;	/* for EPSILON_SVR */
    int shrinking;	/* use the shrinking heuristics */
    int probability; /* do probability estimates */
};

//
// svm_model
//
struct svm_model
{
    struct svm_parameter param;	/* parameter */
    int nr_class;		/* number of classes, = 2 in regression/one class svm */
    int l;			/* total #SV */
    struct svm_node **SV;		/* SVs (SV[l]) */
    double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
    double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
    double *probA;		/* pariwise probability information */
    double *probB;
    int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

    /* for classification only */

    int *label;		/* label of each class (label[k]) */
    int *nSV;		/* number of SVs for each class (nSV[k]) */
                /* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
    /* XXX */
    int free_sv;		/* 1 if svm_model is created by svm_load_model*/
                /* 0 if svm_model is created by svm_train */
};

struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

int svm_save_model(const char *model_file_name, const struct svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);
int svm_get_nr_sv(const struct svm_model *model);
double svm_get_svr_probability(const struct svm_model *model);

double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
double svm_predict(const struct svm_model *model, const struct svm_node *x);
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

void svm_free_model_content(struct svm_model *model_ptr);
void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
void svm_destroy_param(struct svm_parameter *param);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
int svm_check_probability_model(const struct svm_model *model);

void svm_set_print_string_function(void (*print_func)(const char *));



template<typename attribute_type,typename classification_type>
class svm
{
    unsigned int attribute_dimension;
	std::vector<double> y_buf;
    std::vector<std::vector<svm_node> > data;
    std::vector<svm_node*> x_buf;
	svm_parameter param;
	svm_problem prob;
	svm_model* model;
public:
	svm(void):model(0)
    {
        // default values
        param.svm_type = C_SVC;
        param.kernel_type = RBF;
        param.degree = 3;
        param.gamma = 0;	// 1/num_features
        param.coef0 = 0;
        param.nu = 0.5;
        param.cache_size = 100;
        param.C = 100;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        param.probability = 0;
        param.nr_weight = 0;
        param.weight_label = NULL;
        param.weight = NULL;
    }
	~svm(void)
	{
        svm_free_and_destroy_model(&model);
	}
public:
    template<typename attributes_iterator_type,typename classifications_iterator_type>
    void learn(attributes_iterator_type attributes_from,
               attributes_iterator_type attributes_to,
               size_t attribute_dimension_,
               classifications_iterator_type classifications_from)
    {
        // put data in prob
        attribute_dimension = attribute_dimension_;
        {
			prob.l = attributes_to-attributes_from;
            y_buf.resize(prob.l);
			x_buf.resize(prob.l);
			data.resize(prob.l);
            for (size_t index = 0;index < prob.l;++index)
            {
                data[index].resize(attribute_dimension+1); // libsvm has -1 index at the end
                for(unsigned int j = 0;j < attribute_dimension;++j)
                {
                    data[index][j].index = j+1;
                    data[index][j].value = attributes_from[index][j];
                }
                data[index].back().index = -1;// libsvm has -1 index at the end
                x_buf[index] = &*(data[index].begin());
                y_buf[index] = classifications_from[index];
			}
			prob.y = &*y_buf.begin();
            prob.x = &*x_buf.begin();
        }
        param.gamma = 1.0/(double)attribute_dimension;
		if(model)
            svm_free_and_destroy_model(&model);
		model = svm_train(&prob,&param);
    }
    template<typename sample_iterator_type>
    classification_type predict(sample_iterator_type predict_attributes_) const
    {
        std::vector<svm_node> x(attribute_dimension+1);
        for(unsigned int j = 0;j < attribute_dimension;++j)
        {
            x[j].index = j+1;
            x[j].value = predict_attributes_[j];
        }
        x.back().index = -1;// libsvm has -1 index at the end
        return svm_predict(model,&x[0]);
    }
};


}// ml

}// image


#endif//ML_SYM_HPP
