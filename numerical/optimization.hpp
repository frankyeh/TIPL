#ifndef optimization_hpp
#define optimization_hpp

#include <limits>
#include <map>

namespace image
{

namespace optimization
{
template<typename param_type,typename value_type,unsigned int max_iteration = 100>
struct brent_method
{
    param_type min;
    param_type max;
    bool ended;
public:
    brent_method(void):ended(false) {}
    template<typename eval_fun_type,typename termination_type>
    value_type minimize(eval_fun_type& f,value_type& arg_min,termination_type& terminated,value_type tol)
    {
        value_type bx = arg_min;
        value_type a = min;
        value_type b = max;
        struct assign
        {
            void operator()(std::pair<value_type,value_type>& lhs,const std::pair<value_type,value_type>& rhs)
            {
                lhs.first = rhs.first;
                lhs.second = rhs.second;
            }
        };
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
            {
                goto end;
            }
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
                        d=tol1 >= 0 ? xm-x.first:x.first-xm;
                }
            }
            else
                d=gold_ratio*(e=(x.first >= xm ? a-x.first : b-x.first));
            u.first=(std::abs(d) >= tol1 ? x.first + d : (x.first + (d >= 0) ? tol1:-tol1));

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
                assign()(v,w);
                assign()(w,x);
                assign()(x,u);
            }
            else
            {
                if (u.first < x.first)
                    a=u.first;
                else
                    b=u.first;
                if (u.second <= w.second || w.first == x.first)
                {
                    assign()(v,w);
                    assign()(w,u);
                }
                else
                    if (u.second <= v.second || v.first == x.first || v.first == w.first)
                        assign()(v,u);
            }
        }
end:
        arg_min = x.first;
        ended = true;
        return x.second;
    }
};


template<typename param_type,typename value_type,unsigned int max_iteration = 100>
struct enhanced_brent{
    param_type min;
    param_type max;
    bool ended;
public:
    enhanced_brent(void):ended(false) {}
    template<typename eval_fun_type,typename termination_type>
    value_type minimize(eval_fun_type& f,value_type& out_arg_min,termination_type& terminated,value_type tol)
    {
        param_type cur_min = min;
        param_type cur_max = max;
        param_type arg_min = out_arg_min;
        if(arg_min < min && arg_min > max)
            arg_min = (max+min)/2.0;
        for(unsigned int iter = 0;iter < max_iteration && !terminated;++iter)
        {
            std::deque<value_type> values;
            std::deque<param_type> params;
            param_type interval = (cur_max-cur_min)/10.0;
            for(param_type x = arg_min;x > cur_min;x -= interval)
            {
                values.push_front(f(x));
                params.push_front(x);
            }
            for(param_type x = arg_min+interval;x < cur_max;x += interval)
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
            if(change_sign <= 2) // monotonic or u-shape then use breant method
                break;
        }

        float result = 0.0;
        brent_method<param_type,value_type,max_iteration> brent;
        brent.min = cur_min;
        brent.max = cur_max;
        result = brent.minimize(f,arg_min,terminated,tol);
        ended = true;
        out_arg_min = arg_min;
        return result;
    }

};

/**

    param_type::dimension
	param_type::operator[]

	eval_fun_type::operator()(parameter_type)

*/
template<typename method_type,typename param_type_,typename value_type_,unsigned int max_iteration = 100>
struct powell_method
{
public:
    typedef param_type_ param_type;
    typedef value_type_ value_type;
    std::vector<method_type> search_methods;
    bool ended;
public:
    powell_method(unsigned int dimension):search_methods(dimension),ended(false) {}

    template<typename eval_fun_type,typename value_type>
    struct powell_fasade
    {
        eval_fun_type& eval_fun;
        param_type param;
        unsigned int current_dim;
public:
        powell_fasade(eval_fun_type& eval_fun_,param_type param_,unsigned int current_dim_):
                eval_fun(eval_fun_),param(param_),current_dim(current_dim_) {}

        template<typename input_param_type>
        value_type operator()(input_param_type next_param)
        {
            param[current_dim] = next_param;
            return eval_fun(param);
        }
    };



    template<typename eval_fun_type,typename teminated_class>
    value_type minimize(eval_fun_type& fun,param_type& arg_min,teminated_class& terminated,value_type tol = 0.01)
    {
        // estimate the acceptable error level
        std::vector<value_type> eplson(search_methods.size());
        for (unsigned int j = 0; j < search_methods.size();++j)
            eplson[j] = tol*0.05*(search_methods[j].max - search_methods[j].min);

        value_type min_value = 0;
        bool improved = true;
        powell_fasade<eval_fun_type,value_type> search_fun(fun,arg_min,0);
        for (unsigned int i = 0; i < max_iteration && improved && !terminated;++i)
        {
            improved = false;
            for (unsigned int j = 0; j < search_methods.size() && !terminated;++j)
            {
                search_fun.current_dim = j;
                search_fun.param[j] = arg_min[j];
                if (search_methods[j].min >= search_methods[j].max)
                    continue;
                value_type next_value = search_methods[j].minimize(search_fun,search_fun.param[j],terminated,tol);
                if (!improved && next_value != min_value && std::abs(arg_min[j] - search_fun.param[j]) > eplson[j])
                    improved = true;
                arg_min[j] = search_fun.param[j];
                min_value = next_value;
            }
        }
        ended = true;
        return min_value;
    }

};
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
        math::matrix_identity(invB.begin(),math::dyndim(dimension,dimension));
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
            math::matrix_vector_product(invB.begin(),y_k.begin(),invB_y_k.begin(),math::dyndim(dimension,dimension));

			// B1 = Sk*Skt
            math::vector_op_gen(s_k.begin(),s_k.begin()+dimension,s_k.begin(),B1.begin());

			// B2 = B-1YkSkt
            math::vector_op_gen(invB_y_k.begin(),invB_y_k.begin()+dimension,s_k.begin(),B2.begin());

            math::matrix_transpose(B2.begin(),B2.begin(),math::dyndim(dimension,dimension));
			
            double tmp = (s_k_y_k+y_k*invB_y_k)/s_k_y_k;
            for (unsigned int index = 0;index < invB.size();++index)
                invB[index] += (tmp*B1[index]-(B2[index]+B2syn[index]))/s_k_y_k;

            param_type p_k_1;
            math::matrix_vector_product(invB.begin(),g_k_1.begin(),p_k_1.begin(),math::dyndim(dimension,dimension));

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
template<typename coordinate_type,typename gradient_function_type>
void conjugate_descent(coordinate_type& x0,const gradient_function_type& g,double precision = 0.0001)
{
    coordinate_type x = x0;
    coordinate_type g_k = g(x);
    coordinate_type d_k = -g_k;
    coordinate_type y,alpha_d_k,g_k_1;
    double alpha = 0;
    do
    {
        y = x;
        y -= g_k;
        alpha = 1.0/(1.0-(d_k*g(y))/(d_k*g_k));

        alpha_d_k = d_k;
        alpha_d_k *= alpha;
        x += alpha_d_k;

        g_k_1 = g(x);

        d_k *= (g_k_1*g_k_1)/(g_k*g_k);

        d_k -= g_k_1;
        g_k = g_k_1;
    }
    while (std::abs(alpha) > precision);
    x0 = x;
}

template<typename coordinate_type,typename gradient_function_type,typename function_type,typename value_type>
void gradient_descent(coordinate_type& x,const function_type& f,const gradient_function_type& g,value_type radius,value_type precision = 0.001)
{
    coordinate_type g_k = -g(x);
    coordinate_type dx;
	
	value_type line_search_rate = 0.5;
	value_type c1 = 0.0001;
	value_type dir_g_x0 = g_k * g_k;
	
	precision *= precision;
	precision *= dir_g_x0;
    radius /= std::sqrt(dir_g_x0); 
	for (unsigned int index = 0;index < 100;++index)
    {
        value_type f_x0 = f(x);
		value_type alpha_k = radius;
		do//back tracking
            {
				coordinate_type x_alpha_dir = g_k;
				x_alpha_dir *= alpha_k;
				x_alpha_dir += x;
				// condition 1
				// the Armijo rule
				if (f(x_alpha_dir) <= f_x0 + c1*alpha_k*dir_g_x0)
					break;
                alpha_k *= line_search_rate;
            }
            while (alpha_k > 0.0);
		dx = g_k;
        dx *= alpha_k;
		coordinate_type next_x(x);
		next_x += dx;
		if(next_x == x)
			break;
        x = next_x;
		g_k = -g(x);
		dir_g_x0 = g_k * g_k;
        if (dir_g_x0 < precision || alpha_k == 0.0)
            break;
    }
}



}

}

#endif//optimization_hpp
