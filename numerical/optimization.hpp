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

template<typename image_type,typename iter_type1,typename function_type>
void plot_fun_2d(
                image_type& I,
                iter_type1 x_beg,iter_type1 x_end,
                iter_type1 x_upper,iter_type1 x_lower,
                function_type& fun,
                unsigned int dim1,unsigned int dim2,unsigned int sample_frequency = 100)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    I.resize(tipl::shape<2>(sample_frequency,sample_frequency));
    for(tipl::pixel_index<2> index(I.shape());index < I.size();++index)
    {
        std::vector<param_type> x(x_beg,x_end);
        x[dim1] = (x_upper[dim1]-x_lower[dim1])*index[0]/(float)sample_frequency+x_lower[dim1];
        x[dim2] = (x_upper[dim2]-x_lower[dim2])*index[1]/(float)sample_frequency+x_lower[dim2];
        I[index.index()] = fun(x.begin());
    }
}

// calculate fun(x+ei)
template<typename iter_type1,typename tol_type,typename iter_type2,typename function_type>
void estimate_change_mt(iter_type1 x_beg,iter_type1 x_end,tol_type tol,iter_type2 fun_ei,function_type& fun)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    par_for(x_end-x_beg,[&](unsigned int i,int thread)
    {
        if(tol[i] == 0)
            return;
        std::vector<param_type> x(x_beg,x_end);
        x[i] += tol[i];
        fun_ei[i] = fun(x,thread);
    });
}
// calculate fun(x+ei)
template<typename storage_type,typename tol_storage_type,typename fun_type,typename function_type>
void estimate_change_mt(const storage_type& x,const tol_storage_type& tol,fun_type& fun_ei,function_type& fun)
{
    estimate_change_mt(x.begin(),x.end(),tol.begin(),fun_ei.begin(),fun);
}

template<typename iter_type1,typename tol_type,typename iter_type2,typename function_type>
void estimate_change(iter_type1 x_beg,iter_type1 x_end,tol_type tol,iter_type2 fun_ei,function_type& fun)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    size_t size = x_end-x_beg;
    for(unsigned int i = 0; i < size;++i)
    {
        if(tol[i] == 0)
            continue;
        std::vector<param_type> x(x_beg,x_end);
        x[i] += tol[i];
        fun_ei[i] = fun(x);
    }
}
// calculate fun(x+ei)
template<typename storage_type,typename tol_storage_type,typename fun_type,typename function_type>
void estimate_change(const storage_type& x,const tol_storage_type& tol,fun_type& fun_ei,function_type& fun)
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
    std::copy(fun_x_ei,fun_x_ei+size,g_beg);
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

template<typename iter_type1,typename tol_type,typename value_type,typename iter_type2,typename iter_type3,typename function_type>
void hessian_mt(iter_type1 x_beg,iter_type1 x_end,
             tol_type tol,
             value_type fun_x,
             iter_type2 fun_x_ei,
             iter_type3 h_iter,
             function_type& fun)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    unsigned int size = x_end-x_beg;
    tipl::par_for(size*size,[&](size_t index)
    {
        size_t i = index%size;
        size_t j = index/size;
        if(j < i)
            return;
        auto tol2 =  tol[i]*tol[j];
        if(tol2 == 0)
            h_iter[index] = (i == j ? 1.0:0.0);
        else
        {
            std::vector<param_type> new_x(x_beg,x_end);
            new_x[i] += tol[i];
            new_x[j] += tol[j];
            // h = fun(x+ei+ej)+fun(x)-fun(ei)-fun(ej)/tol(i)/tol(j);
            h_iter[index] = (fun(new_x)-fun_x_ei[i]-fun_x_ei[j]+fun_x)/tol2;
        }
        if(j != i)
            h_iter[i*size+j] = h_iter[index];
    });
}

template<typename storage_type,typename tol_storage_type,typename value_type,typename storage_type2,typename storage_type3,typename function_type>
void hessian_mt(const storage_type& x,const tol_storage_type& tol,value_type fun_x,const storage_type2& fun_x_ei,storage_type3& h,function_type& fun)
{
    hessian_mt(x.begin(),x.end(),tol.begin(),fun_x,fun_x_ei.begin(),h.begin(),fun);
}


template<typename param_type,typename g_type,typename value_type,typename function_type>
bool armijo_line_search_1d(param_type& x,
                        param_type upper,param_type lower,
                        g_type g,
                        value_type& fun_x,
                        function_type& fun,double precision)
{
    bool has_new_x = false;
    param_type old_x = x;
    for(double step = 0.1;step <= 100.0;step *= 2)
    {
        param_type new_x = old_x;
        new_x -= g*step;
        new_x = std::min(std::max(new_x,lower),upper);
        value_type new_fun_x = fun(new_x);
        if(fun_x-new_fun_x > 0)
        {
            fun_x = new_fun_x;
            x = new_x;
            has_new_x = true;
        }
        else
            break;
    }
    return has_new_x;
}

template<typename iter_type1,typename iter_type2,typename g_type,typename value_type,typename function_type>
bool armijo_line_search(iter_type1 x_beg,iter_type1 x_end,
                        iter_type2 x_upper,iter_type2 x_lower,
                        g_type g_beg,
                        value_type& fun_x,
                        function_type& fun)
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

template<typename iter_type1,typename function_type,typename terminated_class>
void quasi_newtons_minimize_mt(
                iter_type1 x_beg,iter_type1 x_end,
                iter_type1 x_upper,iter_type1 x_lower,
                function_type& fun,
                typename function_type::value_type& fun_x,
                terminated_class&& is_terminated,double precision = 0.001)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    typedef typename function_type::value_type value_type;
    const int line_search_count = 10;
    unsigned int size = x_end-x_beg;
    std::vector<param_type> tols(size);
    double tol_length = calculate_resolution(tols,x_upper,x_lower,precision);
    for(unsigned int iter = 0;iter < 500 && !is_terminated();++iter)
    {
        std::vector<value_type> fun_x_ei(size);
        std::vector<param_type> g(size),h(size*size),p(size);
        estimate_change_mt(x_beg,x_end,tols.begin(),fun_x_ei.begin(),fun);
        gradient(x_beg,x_end,tols.begin(),fun_x,fun_x_ei.begin(),g.begin());
        hessian_mt(x_beg,x_end,tols.begin(),fun_x,fun_x_ei.begin(),h.begin(),fun);

        std::vector<unsigned int> pivot(size);
        if(!tipl::mat::lu_decomposition(h.begin(),pivot.begin(),tipl::shape<2>(size,size)) ||
           !tipl::mat::lu_solve(h.begin(),pivot.begin(),g.begin(),p.begin(),tipl::shape<2>(size,size)))
            return;

        std::vector<param_type> cost(line_search_count);
        std::vector<std::vector<param_type> > new_xs(line_search_count);
        {
            float L = -0.005f;
            for(int j = 0;j < line_search_count;++j,L *= 2.0f)
            {
                std::vector<param_type> new_x(x_beg,x_end);
                tipl::vec::aypx(p.begin(),p.end(),L,new_x.begin());
                for(size_t i = 0;i < new_x.size();++i)
                    new_x[i] = std::min<param_type>(x_upper[i],std::max<param_type>(x_lower[i],new_x[i]));
                new_xs[j].swap(new_x);
            }
        }

        par_for(line_search_count,[&](unsigned int i,int thread)
        {
            cost[i] = fun(new_xs[i],thread);
        });

        size_t min_index = size_t(std::min_element(cost.begin(),cost.end())-cost.begin());
        if(cost[min_index] >= fun_x)
            return;
        fun_x = cost[min_index];
        std::copy(new_xs[min_index].begin(),new_xs[min_index].end(),x_beg);
        if(tipl::vec::norm2(p.begin(),p.end()) < tol_length)
            return;
    }
}

template<typename param_type,typename function_type,typename value_type,typename terminated_class>
void graient_descent_1d(param_type& x,param_type upper,param_type lower,
                     function_type& fun,value_type& fun_x,terminated_class& terminated,double precision = 0.001)
{
    param_type tol = (upper-lower)*precision;
    if(tol == 0)
        return;
    for(unsigned int iter = 0;iter < 1000 && !terminated;++iter)
    {
        param_type g = (fun(x+tol)-fun_x);
        if(g == 0.0)
            return;
        g *= tol/std::fabs(g);
        if(!armijo_line_search_1d(x,upper,lower,g,fun_x,fun,precision))
            return;
    }
}

template<typename iter_type1,typename iter_type2,typename function_type,typename teminated_class>
void random_search(iter_type1 x_beg,iter_type1 x_end,
                     iter_type2 x_upper,iter_type2 x_lower,
                     function_type& fun,
                     double& optimal_value,
                     teminated_class& terminated,
                     int random_search_count)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    std::default_random_engine gen;
    std::uniform_int_distribution<int> un(0,x_end-x_beg-1);
    tipl::par_for(std::thread::hardware_concurrency(),[&](int)
    {
        for(int j = 0;j < random_search_count && !terminated;)
        {
            int cur_dim = un(gen);
            if(x_upper[cur_dim] == x_lower[cur_dim])
                continue;
            ++j;
            float sd = std::max<float>(std::fabs(x_upper[cur_dim]-x_beg[cur_dim]),std::fabs(x_lower[cur_dim]-x_beg[cur_dim]))/2.0f;
            std::normal_distribution<double> distribution(x_beg[cur_dim],sd);
            std::vector<param_type> param(x_beg,x_end);
            param[cur_dim] = distribution(gen);
            double current_value = fun(param);
            if(current_value < optimal_value)
            {
                optimal_value = current_value;
                x_beg[cur_dim] = param[cur_dim];
            }
        }
    });
}

template<typename iter_type1,typename iter_type2,typename function_type,typename teminated_class>
void line_search_mt(iter_type1 x_beg,iter_type1 x_end,
                     iter_type2 x_upper,iter_type2 x_lower,
                     function_type& fun,
                     typename function_type::value_type& optimal_value,
                     teminated_class&& is_terminated)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    float dis[8] = {0.05f,0.10f,0.20f,0.40f,
                     -0.05f,-0.10f,-0.20f,-0.40f};
    std::vector<param_type> range(x_end-x_beg);
    for(size_t i = 0;i < range.size();++i)
        range[i] = x_upper[i]-x_lower[i];
    float ratio = 1.0f;
    for(int iter = 0;iter < 10;++iter,ratio*=0.95f)
    {
        for(int cur_dim = 0;cur_dim < range.size() && !is_terminated();++cur_dim)
        {
            if(x_upper[cur_dim] == x_lower[cur_dim])
                continue;
            std::mutex m;
            param_type best_x = x_beg[cur_dim];
            tipl::par_for(8,[&](int seg,int id)
            {
                auto new_x = x_beg[cur_dim]+range[cur_dim]*dis[seg]*ratio;
                if(new_x < x_lower[cur_dim] ||
                   new_x > x_upper[cur_dim])
                    return;
                std::vector<param_type> param(x_beg,x_end);
                param[cur_dim] = new_x;
                double current_value = fun(param,id);
                std::lock_guard<std::mutex> lock(m);
                if(current_value < optimal_value)
                {
                    optimal_value = current_value;
                    best_x = param[cur_dim];
                }
            });
            x_beg[cur_dim] = best_x;
        }
    }
}


template<typename iter_type1,typename iter_type2,typename function_type,typename terminated_class>
void gradient_descent(
                iter_type1 x_beg,iter_type1 x_end,
                iter_type2 x_upper,iter_type2 x_lower,
                function_type& fun,
                typename function_type::value_type& fun_x,
                terminated_class& terminated,double precision = 0.001,int max_iteration = 30)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    typedef typename function_type::value_type value_type;
    unsigned int size = x_end-x_beg;
    std::vector<param_type> tols(size);
    double tol_length = calculate_resolution(tols,x_upper,x_lower,precision);
    for(unsigned int iter = 0;iter < max_iteration && !terminated;++iter)
    {
        std::vector<value_type> fun_x_ei(size);
        std::vector<param_type> g(size);
        estimate_change(x_beg,x_end,tols.begin(),fun_x_ei.begin(),fun);
        gradient(x_beg,x_end,tols.begin(),fun_x,fun_x_ei.begin(),g.begin());

        tipl::multiply(g,tols); // scale the unit to parameter unit
        double length = tipl::norm2(g.begin(),g.end());
        if(length == 0.0)
            break;
        tipl::multiply_constant(g,tol_length/length);
        if(!armijo_line_search(x_beg,x_end,x_upper,x_lower,g.begin(),fun_x,fun))
            break;
    }
}

template<typename iter_type1,typename iter_type2,typename function_type,typename terminated_class>
void conjugate_descent(
                iter_type1 x_beg,iter_type1 x_end,
                iter_type2 x_upper,iter_type2 x_lower,
                function_type& fun,
                typename function_type::value_type& fun_x,
                terminated_class& terminated,double precision = 0.001)
{
    typedef typename std::iterator_traits<iter_type1>::value_type param_type;
    typedef typename function_type::value_type value_type;
    unsigned int size = x_end-x_beg;
    std::vector<param_type> tols(size);
    double tol_length = calculate_resolution(tols,x_upper,x_lower,precision);

    std::vector<param_type> g(size),d(size),y(size);
    for(unsigned int iter = 0;iter < 1000 && !terminated;++iter)
    {
        std::vector<value_type> fun_x_ei(size);
        estimate_change(x_beg,x_end,tols.begin(),fun_x_ei.begin(),fun);
        gradient(x_beg,x_end,tols.begin(),fun_x,fun_x_ei.begin(),g.begin());
        if(iter == 0)
            d = g;
        else
        {
            tipl::minus(y.begin(),y.end(),g.begin());      // y = g_k-g_k_1
            double dt_yk = tipl::vec::dot(d.begin(),d.end(),y.begin());
            double y2 = tipl::vec::dot(y.begin(),y.end(),y.begin());
            tipl::vec::axpy(y.begin(),y.end(),-2.0*y2/dt_yk,d.begin()); // y = yk-(2|y|^2/dt_yk)dk
            double beta = tipl::vec::dot(y.begin(),y.end(),g.begin())/dt_yk;
            tipl::multiply_constant(d.begin(),d.end(),-beta);
            tipl::add(d,g);
        }
        y.swap(g);
        g = d;

        tipl::multiply(g,tols); // scale the unit to parameter unit
        double length = tipl::norm2(g.begin(),g.end());
        tipl::multiply_constant(g,tol_length/length);
        if(!armijo_line_search(x_beg,x_end,x_upper,x_lower,g.begin(),fun_x,fun,precision))
            return;
    }
}


template<typename value_type,typename value_type2,typename value_type3,typename function_type>
bool rand_search(value_type& x,value_type2 x_upper,value_type2 x_lower,
                 value_type3& fun_x,function_type& fun,double variance)
{
    value_type new_x(x);
    {
        float seed1 = (float)std::rand()+1.0;
        float seed2 = (float)std::rand()+1.0;
        seed1 /= (float)RAND_MAX+1.0;
        seed2 /= (float)RAND_MAX+1.0;
        seed1 *= 6.28318530718;
        seed2 = std::sqrt(std::max<float>(0.0,-2.0*std::log(seed2)));
        float r1 = seed2*std::cos(seed1);
        new_x += (x_upper-x_lower)*r1/variance;
        new_x = std::min(std::max(new_x,x_lower),x_upper);
    }
    value_type new_fun_x(fun(new_x));
    if(new_fun_x < fun_x)
    {
        fun_x = new_fun_x;
        x = new_x;
        return true;
    }
    return false;
}

template<typename value_type,typename value_type2,typename value_type3,typename function_type>
bool rand_search2(value_type& x,value_type2 x_upper,value_type2 x_lower,
                         value_type3& fun_x,function_type& fun)
{
    value_type new_x;
    value_type new_fun_x(fun(new_x = std::min(std::max((x_upper-x_lower)*((float)std::rand()/(float)RAND_MAX) + x_lower,x_lower),x_upper)));
    if (new_fun_x < fun_x)
    {
        fun_x = new_fun_x;
        x = new_x;
        return true;
    }
    return false;
}

template<typename value_type,typename value_type2,typename value_type3,typename function_type>
void linear_search2(value_type& x,value_type2& x_upper,value_type2& x_lower,
                         value_type3& fun_x,function_type& fun,int count)
{
    value_type2 dis = (x_upper-x_lower)/count;
    std::deque<value_type3> x_list;
    std::deque<value_type3> value_list;
    x_list.push_back(x);
    value_list.push_back(fun_x);

    value_type2 d = dis;
    for(value_type new_x = x-dis;new_x > x_lower;new_x -= d,d *= 2.0)
    {
        x_list.push_front(new_x);
        value_list.push_front(fun(new_x));
    }
    d = dis;
    for(value_type new_x = x+dis;new_x < x_upper;new_x += d,d *= 2.0)
    {
        x_list.push_back(new_x);
        value_list.push_back(fun(new_x));
    }
    unsigned int min_index = std::min_element(value_list.begin(),value_list.end())-value_list.begin();
    x = x_list[min_index];
    fun_x = value_list[min_index];
    if(min_index+1 < x_list.size())
        x_upper = x_list[min_index+1];
    if(min_index > 0)
        x_lower = x_list[min_index-1];
}



template<typename value_type,typename value_type2,typename value_type3,typename function_type>
bool simulated_annealing(value_type& x,value_type2 x_upper,value_type2 x_lower,
                         value_type3& fun_x,function_type& fun,double T)
{
    value_type new_x;
    value_type new_fun_x(fun(new_x = std::min(std::max((x_upper-x_lower)*((float)std::rand()/(float)RAND_MAX) + x_lower,x_lower),x_upper)));
    if (new_fun_x < fun_x || std::rand() <= std::exp((fun_x-new_fun_x)/T)*(float)RAND_MAX)
    {
        fun_x = new_fun_x;
        x = new_x;
        return true;
    }
    return false;
}


template<typename eval_fun_type,typename value_type,typename termination_type,typename tol_type>
void brent_method(eval_fun_type& f,value_type b/*max*/,value_type a/*min*/,value_type& arg_min,
                        termination_type& terminated,tol_type tol)
{
    const unsigned int max_iteration = 100;
    value_type bx = arg_min;
    std::map<value_type,value_type> record;
    const value_type gold_ratio=0.3819660;
    const value_type ZEPS=std::numeric_limits<double>::epsilon()*1.0e-3;
    value_type d=0.0,e=0.0;
    value_type etemp = f(bx);
    value_type tol1,tol2,xm;

    record[bx] = etemp;

    std::pair<value_type,value_type> x(bx,etemp),w(bx,etemp),v(bx,etemp),u(0.0,0.0);

    for (unsigned int iter=0;iter< max_iteration && !terminated;iter++)
    {
        xm=(a+b)/2.0;
        tol2=2.0*(tol1=tol*std::abs(x.first)+ZEPS);
        if (std::abs(x.first-xm) <= (tol2-0.5*(b-a)))
            return;
        if (std::abs(e) > tol1)
        {
            value_type r=(x.first-w.first)*(x.second-v.second);
            value_type q=(x.first-v.first)*(x.second-w.second);
            value_type p=(x.first-v.first)*q-(x.first-w.first)*r;
            q=2.0*(q-r);
            if (q > 0.0)
                p = -p;
            if (q < 0.0)
                q = -q;
            etemp=e;
            e=d;
            if (std::abs(p) >= std::abs(0.5*q*etemp) || p <= q*(a-x.first) || p >= q*(b-x.first))
                d=gold_ratio*(e=(x.first >= xm ? a-x.first : b-x.first));
            else
            {
                d=p/q;
                u.first=x.first+d;
                if (u.first-a < tol2 || b-u.first < tol2)
                    d=(tol1 >= 0 ? xm-x.first:x.first-xm);
            }
        }
        else
            d=gold_ratio*(e=(x.first >= xm ? a-x.first : b-x.first));
        u.first=(std::abs(d) >= tol1 ? x.first + d : (x.first + ((d >= 0) ? tol1:-tol1)));

        typename std::map<value_type,value_type>::const_iterator past_result = record.find(u.first);
        if (past_result != record.end())
            u.second=past_result->second;
        else
        {
            u.second=f(u.first);
            record[u.first] = u.second;
        }
        if (u.second <= x.second)
        {
            if (u.first >= x.first)
                a=x.first;
            else
                b=x.first;
            v = w;
            w = x;
            x = u;
        }
        else
        {
            if (u.first < x.first)
                a=u.first;
            else
                b=u.first;
            if (u.second <= w.second || w.first == x.first)
            {
                v = w;
                w = u;
            }
            else
                if (u.second <= v.second || v.first == x.first || v.first == w.first)
                    v = u;
        }
        arg_min = x.first;

    }
}

struct brent_method_object{
    template<typename eval_fun_type,typename value_type,typename termination_type>
    void operator()(eval_fun_type& f,value_type b/*max*/,value_type a/*min*/,value_type& arg_min,
                            termination_type& terminated,value_type tol)
    {
        brent_method(f,b,a,arg_min,terminated,tol);
    }
};




template<typename value_type,typename eval_fun_type,typename termination_type>
value_type enhanced_brent(eval_fun_type& f,value_type cur_max,value_type cur_min,value_type& out_arg_min,
                          termination_type& terminated,value_type tol)
{
    const unsigned int max_iteration = 100;
    value_type arg_min = out_arg_min;
    if(arg_min < cur_min || arg_min > cur_max)
        arg_min = (cur_min+cur_max)/2.0;
    for(unsigned int iter = 0;iter < max_iteration && !terminated;++iter)
    {
        std::deque<value_type> values;
        std::deque<value_type> params;
        value_type interval = (cur_max-cur_min)/10.0;
        for(value_type x = arg_min;x > cur_min;x -= interval)
        {
            values.push_front(f(x));
            params.push_front(x);
        }
        for(value_type x = arg_min+interval;x < cur_max;x += interval)
        {
            values.push_back(f(x));
            params.push_back(x);
        }
        values.push_front(f(cur_min));
        params.push_front(cur_min);
        values.push_back(f(cur_max));
        params.push_back(cur_max);
        std::vector<unsigned char> greater(values.size()-1);
        for(int i=0;i < greater.size();++i)
            greater[i] = values[i] > values[i+1];
        unsigned char change_sign = 0;
        for(int i=1;i < greater.size();++i)
            if(greater[i-1] != greater[i])
                change_sign++;

        int min_index = std::min_element(values.begin(),values.end())-values.begin();

        cur_min = params[std::max<int>(0,min_index-2)];
        cur_max = params[std::min<int>(params.size()-1,min_index+2)];
        arg_min = params[min_index];
        if(change_sign <= 2) // monotonic or u-shape then use brent method
            break;
    }

    brent_method(f,cur_max,cur_min,arg_min,terminated,tol);
    out_arg_min = arg_min;
}
struct enhanced_brent_object{
    template<typename value_type,typename eval_fun_type,typename termination_type>
    void operator()(eval_fun_type& f,value_type cur_max,value_type cur_min,value_type& out_arg_min,
                              termination_type& terminated,value_type tol)
    {
        enhanced_brent(f,cur_max,cur_min,out_arg_min,terminated,tol);
    }
};


template<typename eval_fun_type,typename param_type>
struct powell_fasade
{
    eval_fun_type& eval_fun;
    param_type& param;
    unsigned int current_dim;
public:
    powell_fasade(eval_fun_type& eval_fun_,param_type& param_,unsigned int current_dim_):
            eval_fun(eval_fun_),param(param_),current_dim(current_dim_) {}

    template<typename input_param_type>
    float operator()(input_param_type next_param)
    {
        param_type temp(param);
        temp[current_dim] = next_param;
        return eval_fun(temp);
    }
};

template<typename optimization_method,typename eval_fun_type,typename param_type,typename teminated_class>
void powell_method(optimization_method optimize,
                         eval_fun_type& fun,
                         param_type& upper,param_type& lower,param_type& arg_min,
                         teminated_class& terminated,float tol = 0.01)
{
    // estimate the acceptable error level
    const unsigned int max_iteration = 100;
    std::vector<class param_type::value_type> eplson(arg_min.size());
    for (unsigned int j = 0; j < arg_min.size();++j)
        eplson[j] = tol*0.05*std::fabs(upper[j] - lower[j]);

    bool improved = true;
    for (unsigned int i = 0; i < max_iteration && improved && !terminated;++i)
    {
        improved = false;
        for (unsigned int j = 0; j < arg_min.size() && !terminated;++j)
        {
            if (lower[j] >= upper[j])
                continue;
            powell_fasade<eval_fun_type,param_type> search_fun(fun,arg_min,j);
            typename param_type::value_type old_value = arg_min[j];
            optimize(search_fun,upper[j],lower[j],arg_min[j],terminated,tol);
            if (!improved && std::abs(arg_min[j] - old_value) > eplson[j])
                improved = true;
        }
    }
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
