#ifndef REG_HPP
#define REG_HPP
#include "image/reg/linear.hpp"
#include "image/reg/bfnorm.hpp"
namespace image{

namespace reg{

enum reg_cost_type{corr,mutual_info};

template<class value_type>
struct normalization{
    image::geometry<3> from_geo,to_geo;
    image::vector<3> from_vs,to_vs;
private:
    image::affine_transform<value_type> arg;
    image::transformation_matrix<value_type> T;
    image::transformation_matrix<value_type> iT;
    bool has_T;

private:
    std::shared_ptr<image::reg::bfnorm_mapping<value_type,3> > bnorm_data;
    int prog;



public:
    normalization(void):prog(0),has_T(false){}
public:
    void update_affine(void)
    {
        has_T = true;
        T = image::transformation_matrix<value_type>(arg,from_geo,from_vs,to_geo,to_vs);
        iT = T;
        iT.inverse();
    }
    const image::transformation_matrix<value_type>& get_T(void) const{return T;}
    const image::transformation_matrix<value_type>& get_iT(void) const{return iT;}
    const image::affine_transform<value_type> get_arg(void) const{return arg;}
    image::affine_transform<value_type>& get_arg(void){return arg;}
    template<class rhs_type>
    void set_arg(const rhs_type& rhs){arg = rhs;update_affine();}
public:
    template<class image_type,class vector_type,class terminate_type>
    void run_reg(const image_type& from,
                 const vector_type& from_vs_,
                 const image_type& to,
                 const vector_type& to_vs_,
                 int factor,
                 reg_cost_type cost_function,
                 image::reg::reg_type reg_type,
                 terminate_type& terminated,
                 int thread_count = std::thread::hardware_concurrency())
    {
        has_T = false;
        from_geo = from.geometry();
        to_geo = to.geometry();
        from_vs = from_vs_;
        to_vs = to_vs_;
        bnorm_data.reset(new image::reg::bfnorm_mapping<value_type,3>(to.geometry(),
                                                                  image::geometry<3>(7*factor,9*factor,7*factor)));
        prog = 0;
        if(cost_function == mutual_info)
        {
            image::reg::linear(from,from_vs,to,to_vs,arg,reg_type,image::reg::mutual_information(),terminated);
            image::reg::linear(from,from_vs,to,to_vs,arg,reg_type,image::reg::mutual_information(),terminated);
        }
        else
        {
            image::reg::linear(from,from_vs,to,to_vs,arg,reg_type,image::reg::mt_correlation<image::basic_image<float,3>,
                           image::transformation_matrix<double> >(0),terminated);
            prog = 1;
            image::reg::linear(from,from_vs,to,to_vs,arg,reg_type,image::reg::mt_correlation<image::basic_image<float,3>,
                           image::transformation_matrix<double> >(0),terminated);
        }
        prog = 2;
        update_affine();
        if(terminated)
            return;

        //std::cout << T.data[0] << " " << T.data[1] << " " << T.data[2] << " " << T.data[9] << std::endl;
        //std::cout << T.data[3] << " " << T.data[4] << " " << T.data[5] << " " << T.data[10] << std::endl;
        //std::cout << T.data[6] << " " << T.data[7] << " " << T.data[8] << " " << T.data[11] << std::endl;

        if(!factor || reg_type == image::reg::rigid_body)
        {
            prog = 3;
            return;
        }
        image::basic_image<typename image_type::value_type,image_type::dimension> new_from(to.geometry());
        image::resample(from,new_from,iT,image::linear);
        image::reg::bfnorm(*bnorm_data.get(),new_from,to,thread_count,terminated);
        prog = 3;
    }

    int get_prog(void)const{return prog;}
    template<class vtype,class vtype2>
    void operator()(const vtype& index,vtype2& out)
    {
        if(!has_T)
            update_affine();
        vtype2 pos;
        T(index,pos);// from -> new_from
        if(prog > 0)
        {
            pos += 0.5;
            (*bnorm_data.get())(image::vector<3,int>(pos[0],pos[1],pos[2]),out);
        }
        else
            out = pos;
    }
    template<class vtype>
    void operator()(vtype& pos)
    {
        vtype out(pos);
        (*this)(out,pos);
    }
};


}
}

#endif//REG_HPP
