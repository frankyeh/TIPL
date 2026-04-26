#ifndef optimization_hpp
#define optimization_hpp

#include <limits>
#include <vector>
#include <map>
#include "numerical.hpp"
#include "matrix.hpp"
namespace tipl
{

namespace optimization
{


// calculate fun(x+ei)
template<typename iter_type1,typename tol_type,typename iter_type2,typename function_type>
void estimate_change(iter_type1 x_beg,iter_type1 x_end,tol_type tol,iter_type2 fun_ei,function_type&& fun)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    par_for(x_end-x_beg,[&](unsigned int i)
    {
        if(tol[i] == 0)
            return;
        std::vector<param_type> x(x_beg,x_end);
        x[i] += tol[i];
        fun_ei[i] = fun(x);
    });
}
// calculate fun(x+ei)
template<typename storage_type,typename tol_storage_type,typename fun_type,typename function_type>
void estimate_change(const storage_type& x,const tol_storage_type& tol,fun_type& fun_ei,function_type&& fun)
{
    estimate_change(x.begin(),x.end(),tol.begin(),fun_ei.begin(),fun);
}

template<typename iter_type1,typename tol_type,typename value_type,typename iter_type2,typename iter_type3>
void gradient(iter_type1 x_beg,iter_type1 x_end,
              tol_type tol,
              value_type fun_x,
              iter_type2 fun_x_ei,
              iter_type3 g_beg)
{
    unsigned int size = x_end-x_beg;
    std::copy_n(fun_x_ei,size,g_beg);
    tipl::minus_constant(g_beg,g_beg+size,fun_x);
    for(unsigned int i = 0;i < size;++i)
        if(tol[i] == 0)
            g_beg[i] = 0;
        else
            g_beg[i] /= tol[i];
}
template<typename storage_type,typename tol_storage_type,typename value_type,typename storage_type2,typename storage_type3>
void gradient(const storage_type& x,const tol_storage_type& tol,value_type fun_x,const storage_type2& fun_x_ei,storage_type3& g)
{
    gradient(x.begin(),x.end(),tol.begin(),fun_x,fun_x_ei.begin(),g.begin());
}

template<typename iter_type1,typename iter_type2,typename g_type,typename value_type,typename function_type>
bool armijo_line_search(iter_type1 x_beg,iter_type1 x_end,
                        iter_type2 x_upper,iter_type2 x_lower,
                        g_type g_beg,
                        value_type& fun_x,
                        function_type&& fun)
{
    using param_type = typename std::iterator_traits<iter_type1>::value_type;
    unsigned int size = x_end-x_beg;
    std::vector<std::vector<param_type> > new_x_list;
    std::vector<value_type> cost;
    cost.push_back(fun_x);
    new_x_list.push_back(std::vector<param_type>(x_beg,x_end));
    for(double step = 0.01;step <= 10.0;step *= 2)
    {
        cost.push_back(std::numeric_limits<value_type>::max());
        std::vector<param_type> new_x(new_x_list[0]);
        tipl::vec::aypx(g_beg,g_beg+size,-step,new_x.begin());
        for(unsigned int j = 0;j < size;++j)
            new_x[j] = std::min<double>(std::max<double>(new_x[j],x_lower[j]),x_upper[j]);
        new_x_list.push_back(std::move(new_x));
    }

    par_for(cost.size(),[&](unsigned int index)
    {
        if(index == 0)
            return;
        cost[index] = fun(new_x_list[index]);
    });

    // find the step that has lowest cost
    unsigned int final = uint32_t(std::min_element(cost.begin(),cost.end())-cost.begin());
    if(final == 0)
        return false; // no new value
    fun_x = cost[final];
    std::copy(new_x_list[final].begin(),new_x_list[final].end(),x_beg);
    return true;
}

template<typename tol_type,typename iter_type>
double calculate_resolution(tol_type& tols,iter_type x_upper,iter_type x_lower,double precision = 0.001)
{
    for(unsigned int i = 0;i < tols.size();++i)
        tols[i] = (x_upper[i]-x_lower[i])*precision;
    return tipl::norm2(tols.begin(),tols.end());
}


template<typename iter_type1,typename iter_type2,typename function_type,typename terminated_class>
void gradient_descent(
                iter_type1 x_beg,iter_type1 x_end,
                iter_type2 x_upper,iter_type2 x_lower,
                function_type&& fun,
                double& fun_x,
                terminated_class& terminated,double precision = 0.001,int max_iteration = 30)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    typedef double value_type;
    unsigned int size = x_end-x_beg;
    std::vector<param_type> tols(size);
    double tol_length = calculate_resolution(tols,x_upper,x_lower,precision);
    for(unsigned int iter = 0;iter < max_iteration && !terminated();++iter)
    {
        std::vector<value_type> fun_x_ei(size);
        std::vector<param_type> g(size);
        estimate_change(x_beg,x_end,tols.data(),fun_x_ei.data(),fun);
        gradient(x_beg,x_end,tols.data(),fun_x,fun_x_ei.data(),g.data());

        tipl::multiply(g,tols); // scale the unit to parameter unit
        double length = tipl::norm2(g.data(),g.data()+size);
        if(length == 0.0)
            break;
        tipl::multiply_constant(g,tol_length/length);
        if(!armijo_line_search(x_beg,x_end,x_upper,x_lower,g.data(),fun_x,fun))
            break;
    }
}

template<typename iter_type1, typename iter_type2, typename function_type, typename teminated_class>
void line_search(iter_type1 x_beg, iter_type1 x_end,
                 iter_type2 x_upper, iter_type2 x_lower,
                 const std::vector<int>& search_strategy, // 0 = Linear, 1 = Gaussian/Cubic
                 function_type&& fun,
                 double& optimal_value,
                 int grid_samples,
                 teminated_class&& is_terminated)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    auto size = x_end - x_beg;
    const double ftol = 0.05;
    const int search_count = 2;
    std::vector<std::vector<param_type> > xi(size, std::vector<param_type>(size, 0.0));
    for(unsigned int i = 0; i < size; ++i)
        xi[i][i] = 1.0;

    std::vector<param_type> p(x_beg, x_end);
    double fret = optimal_value;

    for(int iter = 0; iter < search_count && !is_terminated(); ++iter)
    {
        double fp = fret;
        std::vector<double> dim_steps(size, 0.0);

        // =========================================================
        // STAGE 1: Identify Active Dimensions and Calculate Bounds
        // =========================================================
        std::vector<int> active_dims;
        std::vector<double> dim_cur_upper(size, 0.0);
        std::vector<double> dim_cur_lower(size, 0.0);

        for(unsigned int i = 0; i < size; ++i)
        {
            double bound_pos = 1e10, bound_neg = -1e10;
            bool is_movable = false;

            for(unsigned int j = 0; j < size; ++j)
            {
                if(std::abs(xi[i][j]) > 1e-8 && x_upper[j] != x_lower[j])
                {
                    is_movable = true;
                    double step_to_upper = (x_upper[j] - p[j]) / xi[i][j];
                    double step_to_lower = (x_lower[j] - p[j]) / xi[i][j];
                    if (step_to_upper < step_to_lower) std::swap(step_to_upper, step_to_lower);
                    bound_pos = std::min(bound_pos, step_to_upper);
                    bound_neg = std::max(bound_neg, step_to_lower);
                }
            }

            // Skip entirely if this direction has zero mobility
            if (!is_movable) continue;

            if (bound_pos < 0.0) bound_pos = 0.0;
            if (bound_neg > 0.0) bound_neg = 0.0;
            if (bound_pos > 1e9) bound_pos = 1.0;
            if (bound_neg < -1e9) bound_neg = -1.0;

            double search_scale = 0.5 / (iter + 1.0);
            double cur_upper = bound_pos * search_scale;
            double cur_lower = bound_neg * search_scale;

            if (cur_upper - cur_lower > 1e-8) {
                active_dims.push_back(i);
                dim_cur_upper[i] = cur_upper;
                dim_cur_lower[i] = cur_lower;
            }
        }

        // =========================================================
        // STAGE 2: Massive Flattened Parallel Grid Evaluation
        // =========================================================
        int total_grid_tasks = active_dims.size() * grid_samples;

        std::vector<double> all_sample_costs(size * grid_samples, fp);
        std::vector<double> all_sample_steps(size * grid_samples, 0.0);

        if (total_grid_tasks > 0)
        {
            tipl::par_for(total_grid_tasks, [&](int task_idx) {
                if (is_terminated()) return;

                int active_idx = task_idx / grid_samples;
                int s = task_idx % grid_samples;
                int i = active_dims[active_idx];
                int linear_idx = i * grid_samples + s;

                int strategy = (i < search_strategy.size()) ? search_strategy[i] : 0;
                double cur_upper = dim_cur_upper[i];
                double cur_lower = dim_cur_lower[i];
                double trial_step = 0.0;

                if (strategy == 0) {
                    double step_size = (cur_upper - cur_lower) / (grid_samples - 1);
                    trial_step = cur_lower + s * step_size;
                } else {
                    double u = (double)(s - (grid_samples - 1) / 2.0) / ((grid_samples - 1) / 2.0);
                    double v = u * u * u;
                    trial_step = (v > 0.0) ? (v * cur_upper) : (v * -cur_lower);
                }

                all_sample_steps[linear_idx] = trial_step;

                if (std::abs(trial_step) < 1e-8) {
                    all_sample_costs[linear_idx] = fp;
                } else {
                    std::vector<param_type> temp = p;
                    for(unsigned int j = 0; j < size; ++j) {
                        temp[j] = std::min<param_type>(x_upper[j], std::max<param_type>(x_lower[j], p[j] + trial_step * xi[i][j]));
                    }
                    all_sample_costs[linear_idx] = fun(temp);
                }
            });
        }

        // =========================================================
        // STAGE 3: Select Best Points
        // =========================================================
        for (int active_idx = 0; active_idx < active_dims.size(); ++active_idx)
        {
            int i = active_dims[active_idx];
            double best_f = fp;
            int s_min = (grid_samples - 1) / 2;

            for (int s = 0; s < grid_samples; ++s) {
                int linear_idx = i * grid_samples + s;
                if (all_sample_costs[linear_idx] < best_f) {
                    best_f = all_sample_costs[linear_idx];
                    s_min = s;
                }
            }
            dim_steps[i] = all_sample_steps[i * grid_samples + s_min];
        }

        // =========================================================
        // STAGE 4: Resultant Vector Search
        // =========================================================
        std::vector<param_type> new_dir(size, 0.0);
        bool has_movement = false;
        for(unsigned int i = 0; i < size; ++i) {
            if (std::abs(dim_steps[i]) > 1e-8) has_movement = true;
            for(unsigned int j = 0; j < size; ++j) {
                new_dir[j] += dim_steps[i] * xi[i][j];
            }
        }

        if(has_movement)
        {
            double bound_pos = 1e10, bound_neg = -1e10;
            bool is_movable = false;

            for(unsigned int j = 0; j < size; ++j) {
                if(std::abs(new_dir[j]) > 1e-8 && x_upper[j] != x_lower[j]) {
                    is_movable = true;
                    double step_to_upper = (x_upper[j] - p[j]) / new_dir[j];
                    double step_to_lower = (x_lower[j] - p[j]) / new_dir[j];
                    if (step_to_upper < step_to_lower) std::swap(step_to_upper, step_to_lower);
                    bound_pos = std::min(bound_pos, step_to_upper);
                    bound_neg = std::max(bound_neg, step_to_lower);
                }
            }

            if (is_movable)
            {
                if (bound_pos < 0.0) bound_pos = 0.0;
                if (bound_neg > 0.0) bound_neg = 0.0;
                if (bound_pos > 1e9) bound_pos = 1.0;
                if (bound_neg < -1e9) bound_neg = -1.0;

                double step_size = (bound_pos - bound_neg) / (grid_samples - 1);

                if (step_size > 1e-8) {
                    std::vector<double> res_sample_costs(grid_samples, fp);
                    std::vector<double> res_sample_steps(grid_samples, 0.0);

                    tipl::par_for(grid_samples, [&](int s) {
                        if (is_terminated()) return;
                        double trial_step = bound_neg + s * step_size;
                        res_sample_steps[s] = trial_step;

                        if(std::abs(trial_step) > 1e-8) {
                            std::vector<param_type> temp = p;
                            for(unsigned int j = 0; j < size; ++j) {
                                temp[j] = std::min<param_type>(x_upper[j], std::max<param_type>(x_lower[j], p[j] + trial_step * new_dir[j]));
                            }
                            res_sample_costs[s] = fun(temp);
                        }
                    });

                    double best_f = fp;
                    int s_min = (grid_samples - 1) / 2;

                    for (int s = 0; s < grid_samples; ++s) {
                        if (res_sample_costs[s] < best_f) {
                            best_f = res_sample_costs[s];
                            s_min = s;
                        }
                    }

                    double step_min = res_sample_steps[s_min];
                    fret = best_f;

                    for(unsigned int j = 0; j < size; ++j) {
                        p[j] = std::min<param_type>(x_upper[j], std::max<param_type>(x_lower[j], p[j] + step_min * new_dir[j]));
                    }
                }
            }
        }

        if(2.0 * (fp - fret) <= ftol * (std::abs(fp) + std::abs(fret)) + 1e-10)
            break;
    }

    std::copy(p.begin(), p.end(), x_beg);
    optimal_value = fret;
}

/*
template<typename param_type,typename value_type>
struct BFGS
{
    unsigned int dimension;
	unsigned int dim2;
    BFGS(unsigned int dim):dimension(dim),dim2(dim*dim) {}
    template<typename function_type,typename gradient_function_type>
    value_type minimize(const function_type& f,
						const gradient_function_type& g,
						param_type& xk,
						value_type radius,
						value_type tol = 0.001)
    {
        param_type g_k = g(xk);
        param_type p = -g_k;
        std::vector<value_type> invB(dim2),B1(dim2),B2(dim2),B2syn(dim2);
        math::matrix_identity(invB.begin(),math::shape(dimension,dimension));
        value_type end_gradient = tol*tol*(g_k*g_k);
		// parameter for back tracking
		value_type line_search_rate = 0.5;
		value_type c1 = 0.0001;
        radius /= std::sqrt(p*p);
        for(unsigned int iter = 0;iter < 100;++iter)
        {
            // back_tracking
            value_type f_x0 = f(xk);
			value_type dir_g_x0 = p*g_k;
			value_type alpha_k = radius;
	        do//back tracking
            {
				param_type x_alpha_dir = p;
				x_alpha_dir *= alpha_k;
				x_alpha_dir += xk;
				// the Armijo rule
				if (f(x_alpha_dir) <= f_x0 + c1*alpha_k*dir_g_x0)
					break;
                alpha_k *= line_search_rate;
            }
            while (alpha_k > 0.0);
			// set Sk = alphak*p;
            param_type s_k = p;s_k *= alpha_k; 
            
			// update Xk <- Xk + s_k
			param_type x_k_1 = xk;x_k_1 += s_k; 
            
			// Yk = g(Xk+1) - g(Xk)
			param_type g_k_1 = g(x_k_1);
            param_type y_k = g_k_1;y_k -= g_k;  
            
			value_type s_k_y_k = s_k*y_k;
			
			if(s_k_y_k == 0.0) // y_k = 0  or alpha too small
				break;

            param_type invB_y_k;
			
			// invB*Yk
            math::matrix_vector_product(invB.begin(),y_k.begin(),invB_y_k.begin(),math::shape(dimension,dimension));

			// B1 = Sk*Skt
            math::vector_op_gen(s_k.begin(),s_k.begin()+dimension,s_k.begin(),B1.begin());

			// B2 = B-1YkSkt
            math::vector_op_gen(invB_y_k.begin(),invB_y_k.begin()+dimension,s_k.begin(),B2.begin());

            math::matrix_transpose(B2.begin(),B2.begin(),math::shape(dimension,dimension));
			
            double tmp = (s_k_y_k+y_k*invB_y_k)/s_k_y_k;
            for (unsigned int index = 0;index < invB.size();++index)
                invB[index] += (tmp*B1[index]-(B2[index]+B2syn[index]))/s_k_y_k;

            param_type p_k_1;
            math::matrix_vector_product(invB.begin(),g_k_1.begin(),p_k_1.begin(),math::shape(dimension,dimension));

            p = -p_k_1;
            xk = x_k_1;
            g_k = g_k_1;
            if (g_k*g_k < end_gradient)
                break;
        }
        return f(xk);
    }

};
*/



}

}

#endif//optimization_hpp
