//---------------------------------------------------------------------------
#ifndef MEAN_FILTER_HPP
#define MEAN_FILTER_HPP
#include "filter_model.hpp"
//---------------------------------------------------------------------------
namespace tipl
{

namespace filter
{


template<typename value_type,size_t dimension>
class mean_filter_imp;

template<typename value_type>
struct mean_filter_imp<value_type,1>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<typename image_type>
    void operator()(image_type& src)
    {
        std::vector<manip_type> dest(src.begin(),src.end());
        add_weight<1>(dest,src,1);
        add_weight<1>(dest,src,-1);
        divide_constant(dest.begin(),dest.end(),3);
        std::copy(dest.begin(),dest.end(),src.begin());
    }
};

template<typename value_type>
class mean_filter_imp<value_type,2>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<typename image_type>
    void operator()(image_type& src)
    {
        std::vector<manip_type> dest(src.begin(),src.end());
        int w = src.width();
        add_weight<1>(dest,src,-1-w);
        add_weight<1>(dest,src,-w);
        add_weight<1>(dest,src,1-w);
        add_weight<1>(dest,src,-1);
        add_weight<1>(dest,src,1);
        add_weight<1>(dest,src,-1+w);
        add_weight<1>(dest,src,w);
        add_weight<1>(dest,src,1+w);
        divide_constant(dest.begin(),dest.end(),9);
        std::copy(dest.begin(),dest.end(),src.begin());
    }
};


template<typename value_type>
class mean_filter_imp<value_type,3>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<typename image_type>
    void operator()(image_type& src)
    {
        std::vector<manip_type> dest(src.begin(),src.end());
        int w = src.width();
        int64_t wh = src.plane_size();

        add_weight<1>(dest,src,-1-w-wh);
        add_weight<1>(dest,src,-w-wh);
        add_weight<1>(dest,src,1-w-wh);
        add_weight<1>(dest,src,-1-wh);
        add_weight<1>(dest,src,-wh);
        add_weight<1>(dest,src,1-wh);
        add_weight<1>(dest,src,-1+w-wh);
        add_weight<1>(dest,src,w-wh);
        add_weight<1>(dest,src,1+w-wh);

        add_weight<1>(dest,src,-1-w);
        add_weight<1>(dest,src,-w);
        add_weight<1>(dest,src,1-w);
        add_weight<1>(dest,src,-1);
        add_weight<1>(dest,src,1);
        add_weight<1>(dest,src,-1+w);
        add_weight<1>(dest,src,w);
        add_weight<1>(dest,src,1+w);

        add_weight<1>(dest,src,-1-w+wh);
        add_weight<1>(dest,src,-w+wh);
        add_weight<1>(dest,src,1-w+wh);
        add_weight<1>(dest,src,-1+wh);
        add_weight<1>(dest,src,wh);
        add_weight<1>(dest,src,1+wh);
        add_weight<1>(dest,src,-1+w+wh);
        add_weight<1>(dest,src,w+wh);
        add_weight<1>(dest,src,1+w+wh);

        divide_constant(dest.begin(),dest.end(),27);

        std::copy(dest.begin(),dest.end(),src.begin());
    }
};

template<typename image_type>
image_type& mean(image_type& src)
{
    mean_filter_imp<typename image_type::value_type,image_type::dimension>()(src);
    return src;
}


}

}
#endif
