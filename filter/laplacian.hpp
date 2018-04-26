#ifndef LAPLACIAN_FILTER_HPP
#define LAPLACIAN_FILTER_HPP
#include <cmath>
#include "filter_model.hpp"
namespace tipl{


namespace filter{




template<class value_type,size_t dimension>
class laplacian_filter_imp;

/**
kernel
1 -2 1
*/

template<class value_type>
struct laplacian_filter_imp<value_type,1>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<class image_type>
    void operator()(image_type& src)
    {
        std::vector<manip_type> dest(src.size());
        add_weight<1>(dest,src,1);
        add_weight<1>(dest,src,-1);
        minus_weight<2>(dest,src,0);
		std::copy(dest.begin(),dest.end(),src.begin());
    }
};

/**
kernel
0 1 0
1 -4 1
0 1 0
*/


template<class value_type>
class laplacian_filter_imp<value_type,2>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<class image_type>
    void operator()(image_type& src)
    {
        std::vector<manip_type> dest(src.size());
        int w = src.width();
        add_weight<1>(dest,src,-1);
        add_weight<1>(dest,src,1);
        add_weight<1>(dest,src,-w);
        add_weight<1>(dest,src,w);
        minus_weight<4>(dest,src,0);
        std::copy(dest.begin(),dest.end(),src.begin());
    }
};

/**
kernel
0 0 0	0 1 0	0 0 0
0 1 0	1 -6 1  0 1 0
0 0 0	0 1 0	0 0 0
*/

template<class value_type>
class laplacian_filter_imp<value_type,3>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<class image_type>
    void operator()(image_type& src)
    {
        std::vector<manip_type> dest(src.size());
        int w = src.width();
        int wh = src.width()*src.height();
        add_weight<1>(dest,src,1);
        add_weight<1>(dest,src,-1);
        add_weight<1>(dest,src,w);
        add_weight<1>(dest,src,-w);
        add_weight<1>(dest,src,wh);
        add_weight<1>(dest,src,-wh);
        minus_weight<6>(dest,src,0);

        std::copy(dest.begin(),dest.end(),src.begin());
    }
};

template<class pixel_type,size_t dimension>
void laplacian(image<pixel_type,dimension>& src)
{
    laplacian_filter_imp<pixel_type,dimension>()(src);
}


}
}
#endif//
