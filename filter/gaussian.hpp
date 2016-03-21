//---------------------------------------------------------------------------
#ifndef GAUSSIAN_HPP
#define GAUSSIAN_HPP
#include "filter_model.hpp"
//---------------------------------------------------------------------------
namespace image
{

namespace filter
{


template<typename value_type,size_t dimension>
class gaussian_filter_imp;

template<typename value_type>
struct gaussian_filter_imp<value_type,1>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<typename image_type>
    void operator()(image_type& src)
    {
        std::vector<manip_type> dest(src.size());
        add_weight<1>(dest,src,1);
        add_weight<1>(dest,src,-1);
        add_weight<2>(dest,src,0);
		divide_constant(dest.begin(),dest.end(),4);
		std::copy(dest.begin(),dest.end(),src.begin());
    }
};


template<typename value_type>
class gaussian_filter_imp<value_type,2>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<typename image_type>
    void operator()(image_type& src)
    {
        std::vector<manip_type> dest(src.size());
        int w = src.width();
        add_weight<1>(dest,src,-1-w);
        add_weight<1>(dest,src,-1+w);
        add_weight<1>(dest,src,1-w);
        add_weight<1>(dest,src,1+w);
        add_weight<1>(dest,src,-2);
        add_weight<1>(dest,src,2);
        add_weight<1>(dest,src,-w-w);
        add_weight<1>(dest,src,w+w);

        add_weight<2>(dest,src,-1);
        add_weight<2>(dest,src,1);
        add_weight<2>(dest,src,-w);
        add_weight<2>(dest,src,+w);
        add_weight<4>(dest,src,0);

        divide_constant(dest.begin(),dest.end(),20);

        std::copy(dest.begin(),dest.end(),src.begin());
    }
};


template<typename value_type>
class gaussian_filter_imp<value_type,3>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<typename image_type>
    void operator()(image_type& src)
    {
        std::vector<manip_type> dest(src.size());
        int w = src.width();
        int wh = src.width()*src.height();
        add_weight<1>(dest,src,-1-w);
        add_weight<1>(dest,src,-1+w);
        add_weight<1>(dest,src,1-w);
        add_weight<1>(dest,src,1+w);
        add_weight<1>(dest,src,-1-wh);
        add_weight<1>(dest,src,-1+wh);
        add_weight<1>(dest,src,1-wh);
        add_weight<1>(dest,src,1+wh);
        add_weight<1>(dest,src,-w-wh);
        add_weight<1>(dest,src,-w+wh);
        add_weight<1>(dest,src,w-wh);
        add_weight<1>(dest,src,w+wh);
        add_weight<1>(dest,src,-2);
        add_weight<1>(dest,src,2);
        add_weight<1>(dest,src,-w-w);
        add_weight<1>(dest,src,w+w);
        add_weight<1>(dest,src,-wh-wh);
        add_weight<1>(dest,src,wh+wh);

        add_weight<2>(dest,src,-1);
        add_weight<2>(dest,src,1);
        add_weight<2>(dest,src,-w);
        add_weight<2>(dest,src,+w);
        add_weight<2>(dest,src,-wh);
        add_weight<2>(dest,src,+wh);
        add_weight<4>(dest,src,0);

        divide_constant(dest.begin(),dest.end(),34);

        std::copy(dest.begin(),dest.end(),src.begin());
    }
};

template<typename image_type>
void gaussian(image_type& src)
{
    gaussian_filter_imp<image_type::value_type,image_type::dimension>()(src);
}


}

}
#endif
