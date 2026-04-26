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
    for(double step = 0.015625;step <= 10.0;step *= 2)
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

template<typename iter_type1, typename iter_type2, typename function_type, typename terminated_class>
void lbfgs(iter_type1 x_beg, iter_type1 x_end,
           iter_type2 x_upper, iter_type2 x_lower,
           function_type&& fun,
           double& fun_x,
           terminated_class& terminated,
           double precision = 0.001,
           int max_iteration = 100,
           int m = 10) // L-BFGS Memory/History size
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    typedef double value_type;
    unsigned int size = x_end - x_beg;

    std::vector<param_type> tols(size);
    double tol_length = calculate_resolution(tols, x_upper, x_lower, precision);

    std::vector<std::vector<param_type>> S, Y;
    std::vector<double> RHO;

    std::vector<param_type> g(size), g_old(size), x_old(size);
    std::vector<value_type> fun_x_ei(size);

    estimate_change(x_beg, x_end, tols.begin(), fun_x_ei.begin(), fun);
    gradient(x_beg, x_end, tols.begin(), fun_x, fun_x_ei.begin(), g.begin());

    for(unsigned int iter = 0; iter < max_iteration && !terminated(); ++iter)
    {
        std::vector<param_type> q = g;
        std::vector<double> alpha(S.size());

        // Backward pass
        for(int i = (int)S.size() - 1; i >= 0; --i)
        {
            double sq = 0.0;
            for(unsigned int j = 0; j < size; ++j)
                sq += S[i][j] * q[j];

            alpha[i] = RHO[i] * sq;

            for(unsigned int j = 0; j < size; ++j)
                q[j] -= alpha[i] * Y[i][j];
        }

        // Central Preconditioner (H0 scaling with Diagonal Matrix)
        if(!S.empty())
        {
            double yty = 0.0;
            for(unsigned int j = 0; j < size; ++j)
                yty += Y.back()[j] * Y.back()[j];

            if(yty > 1e-14)
            {
                double gamma = (1.0 / RHO.back()) / yty;
                for(unsigned int j = 0; j < size; ++j)
                {
                    // SCALE FIX: Apply gamma AND the specific dimension's range
                    double range = x_upper[j] - x_lower[j];
                    q[j] *= gamma * range;
                }
            }
        }
        else
        {
            // Iteration 0: Scale by range and normalize safely to prevent blowout
            for(unsigned int j = 0; j < size; ++j)
            {
                double range = x_upper[j] - x_lower[j];
                q[j] *= range;
            }

            double length = tipl::norm2(q.begin(), q.end());
            if(length > 0.0)
                tipl::multiply_constant(q, tol_length / length);
        }

        // Forward pass
        for(size_t i = 0; i < S.size(); ++i)
        {
            double yq = 0.0;
            for(unsigned int j = 0; j < size; ++j)
                yq += Y[i][j] * q[j];

            double beta = RHO[i] * yq;

            for(unsigned int j = 0; j < size; ++j)
                q[j] += S[i][j] * (alpha[i] - beta);
        }

        x_old.assign(x_beg, x_end);
        g_old = g;

        bool moved = armijo_line_search(x_beg, x_end, x_upper, x_lower, q.begin(), fun_x, fun);

        if(!moved)
        {
            if(!S.empty())
            {
                S.clear();
                Y.clear();
                RHO.clear();
                continue;
            }
            else
                break;
        }

        estimate_change(x_beg, x_end, tols.begin(), fun_x_ei.begin(), fun);
        gradient(x_beg, x_end, tols.begin(), fun_x, fun_x_ei.begin(), g.begin());

        std::vector<param_type> s_k(size), y_k(size);
        double sty = 0.0;

        for(size_t i = 0; i < size; ++i)
        {
            s_k[i] = x_beg[i] - x_old[i];
            y_k[i] = g[i] - g_old[i];
            sty += s_k[i] * y_k[i];
        }

        if(sty > 1e-14)
        {
            if(S.size() == m)
            {
                S.erase(S.begin());
                Y.erase(Y.begin());
                RHO.erase(RHO.begin());
            }
            S.push_back(std::move(s_k));
            Y.push_back(std::move(y_k));
            RHO.push_back(1.0 / sty);
        }
    }
}

template<typename iter_type1, typename iter_type2, typename function_type, typename terminated_class>
void line_search(iter_type1 x_beg, iter_type1 x_end,
                 iter_type2 x_upper, iter_type2 x_lower,
                 const std::vector<int>& search_strategy, // 0 = Linear, 1 = Gaussian/Cubic
                 function_type&& fun,
                 double& optimal_value,
                 int grid_samples,
                 terminated_class&& is_terminated)
{
    using param_type = typename std::iterator_traits<iter_type1>::value_type;
    auto size = x_end - x_beg;
    const double ftol = 0.05;
    const int search_count = 2;

    std::vector<std::vector<param_type> > xi(size, std::vector<param_type>(size, 0.0));
    for(unsigned int i = 0; i < size; ++i)
        xi[i][i] = 1.0;

    std::vector<param_type> p(x_beg, x_end);
    std::mutex update_mtx; // Mutex to safely update x_beg and optimal_value concurrently

    for(int iter = 0; iter < search_count; ++iter)
    {
        if(is_terminated()) return;

        double fp = optimal_value;
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

            if (!is_movable) continue;

            if (bound_pos < 0.0) bound_pos = 0.0;
            if (bound_neg > 0.0) bound_neg = 0.0;
            if (bound_pos > 1e9) bound_pos = 1.0;
            if (bound_neg < -1e9) bound_neg = -1.0;

            double search_scale = 0.5 / (iter + 1.0);
            double cur_upper = bound_pos * search_scale;
            double cur_lower = bound_neg * search_scale;

            if (cur_upper - cur_lower > 1e-8)
            {
                active_dims.push_back(i);
                dim_cur_upper[i] = cur_upper;
                dim_cur_lower[i] = cur_lower;
            }
        }

        if(is_terminated()) return;

        // =========================================================
        // STAGE 2: Massive Flattened Parallel Grid Evaluation
        // =========================================================
        int total_grid_tasks = active_dims.size() * grid_samples;

        std::vector<double> all_sample_costs(size * grid_samples, fp);
        std::vector<double> all_sample_steps(size * grid_samples, 0.0);

        if (total_grid_tasks > 0)
        {
            tipl::par_for(total_grid_tasks, [&](int task_idx)
            {
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

                if (std::abs(trial_step) < 1e-8)
                {
                    all_sample_costs[linear_idx] = fp;
                }
                else
                {
                    std::vector<param_type> temp = p;
                    for(unsigned int j = 0; j < size; ++j)
                        temp[j] = std::min<param_type>(x_upper[j], std::max<param_type>(x_lower[j], p[j] + trial_step * xi[i][j]));

                    double cost = fun(temp);
                    all_sample_costs[linear_idx] = cost;

                    if(cost < optimal_value)
                    {
                        std::lock_guard<std::mutex> lock(update_mtx);
                        if(cost < optimal_value)
                        {
                            optimal_value = cost;
                            std::copy(temp.begin(), temp.end(), x_beg);
                        }
                    }
                }
            });
        }

        if(is_terminated()) return;

        // =========================================================
        // STAGE 3: Select Best Points for Resultant Vector
        // =========================================================
        for (int active_idx = 0; active_idx < active_dims.size(); ++active_idx)
        {
            int i = active_dims[active_idx];
            double best_f = fp;
            int s_min = (grid_samples - 1) / 2;

            for (int s = 0; s < grid_samples; ++s)
            {
                int linear_idx = i * grid_samples + s;
                if (all_sample_costs[linear_idx] < best_f)
                {
                    best_f = all_sample_costs[linear_idx];
                    s_min = s;
                }
            }
            dim_steps[i] = all_sample_steps[i * grid_samples + s_min];
        }

        if(is_terminated()) return;

        // =========================================================
        // STAGE 4: Resultant Vector Search
        // =========================================================
        std::vector<param_type> new_dir(size, 0.0);
        bool has_movement = false;

        for(unsigned int i = 0; i < size; ++i)
        {
            if (std::abs(dim_steps[i]) > 1e-8) has_movement = true;
            for(unsigned int j = 0; j < size; ++j)
                new_dir[j] += dim_steps[i] * xi[i][j];
        }

        if(has_movement)
        {
            double bound_pos = 1e10, bound_neg = -1e10;
            bool is_movable = false;

            for(unsigned int j = 0; j < size; ++j)
            {
                if(std::abs(new_dir[j]) > 1e-8 && x_upper[j] != x_lower[j])
                {
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

                if (step_size > 1e-8)
                {
                    tipl::par_for(grid_samples, [&](int s)
                    {
                        if (is_terminated()) return;

                        double trial_step = bound_neg + s * step_size;

                        if(std::abs(trial_step) > 1e-8)
                        {
                            std::vector<param_type> temp = p;
                            for(unsigned int j = 0; j < size; ++j)
                                temp[j] = std::min<param_type>(x_upper[j], std::max<param_type>(x_lower[j], p[j] + trial_step * new_dir[j]));

                            double cost = fun(temp);

                            if(cost < optimal_value)
                            {
                                std::lock_guard<std::mutex> lock(update_mtx);
                                if(cost < optimal_value)
                                {
                                    optimal_value = cost;
                                    std::copy(temp.begin(), temp.end(), x_beg);
                                }
                            }
                        }
                    });
                }
            }
        }

        if(is_terminated()) return;

        // Sync `p` with whatever the absolute best position found so far is
        std::copy(x_beg, x_end, p.begin());

        // Convergence check using the strictly continuously updated optimal_value
        if(2.0 * (fp - optimal_value) <= ftol * (std::abs(fp) + std::abs(optimal_value)) + 1e-10)
            break;
    }
}

}

}

#endif//optimization_hpp
