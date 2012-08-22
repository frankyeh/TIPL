//---------------------------------------------------------------------------
#ifndef SOBEL_FILTER_HPP
#define SOBEL_FILTER_HPP
#include "filter_model.hpp"
//---------------------------------------------------------------------------
namespace image
{

namespace filter
{


template<typename value_type>
value_type sobel_filter_abs(value_type value)
{
	return value >= 0 ? value : -value;
}

template<typename value_type,size_t dimension>
class sobel_filter_imp;

template<typename value_type>
class sobel_filter_imp<value_type,2>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<typename image_type>
    void operator()(image_type& src)
    {
        std::vector<manip_type> gx(src.size());
        int w = src.width();

		add_weight<2>(gx,src,1);
        add_weight<1>(gx,src,1-w);
        add_weight<1>(gx,src,1+w);
        minus_weight<2>(gx,src,-1);
        minus_weight<1>(gx,src,-1-w);
        minus_weight<1>(gx,src,-1+w);

        std::vector<manip_type> gy(src.size());

		add_weight<2>(gy,src,w);
        add_weight<1>(gy,src,w-1);
        add_weight<1>(gy,src,w+1);
        minus_weight<2>(gy,src,-w);
        minus_weight<1>(gy,src,-w-1);
        minus_weight<1>(gy,src,-w+1);

		for(size_t index = 0;index < src.size();++index)
			src[index] = sobel_filter_abs(gx[index]) + sobel_filter_abs(gy[index]);
    }
};


template<typename value_type>
class sobel_filter_imp<value_type,3>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<typename image_type>
    void operator()(image_type& src)
    {
        int w = src.width();
        int wh = src.geometry().plane_size();

		std::vector<manip_type> gx(src.size());

		add_weight<2>(gx,src,1);
        add_weight<1>(gx,src,1-w);
        add_weight<1>(gx,src,1+w);
        add_weight<1>(gx,src,1-wh);
        add_weight<1>(gx,src,1+wh);
        minus_weight<2>(gx,src,-1);
        minus_weight<1>(gx,src,-1-w);
        minus_weight<1>(gx,src,-1+w);
		minus_weight<1>(gx,src,-1-wh);
        minus_weight<1>(gx,src,-1+wh);

        std::vector<manip_type> gy(src.size());

		add_weight<2>(gy,src,w);
        add_weight<1>(gy,src,w-1);
        add_weight<1>(gy,src,w+1);
        add_weight<1>(gy,src,w+wh);
        add_weight<1>(gy,src,w-wh);
        minus_weight<2>(gy,src,-w);
        minus_weight<1>(gy,src,-w-1);
        minus_weight<1>(gy,src,-w+1);
		minus_weight<1>(gy,src,-w-wh);
        minus_weight<1>(gy,src,-w+wh);

        std::vector<manip_type> gz(src.size());

		add_weight<2>(gy,src,wh);
        add_weight<1>(gy,src,wh-1);
        add_weight<1>(gy,src,wh+1);
        add_weight<1>(gy,src,wh+w);
        add_weight<1>(gy,src,wh-w);
        minus_weight<2>(gy,src,-wh);
        minus_weight<1>(gy,src,-wh-1);
        minus_weight<1>(gy,src,-wh+1);
		minus_weight<1>(gy,src,-wh-w);
        minus_weight<1>(gy,src,-wh+w);

		for(size_t index = 0;index < src.size();++index)
			src[index] = sobel_filter_abs(gx[index]) + sobel_filter_abs(gy[index]) + sobel_filter_abs(gz[index]);

    }
};

template<typename image_type>
void sobel(image_type& src)
{
    sobel_filter_imp<typename image_type::value_type,image_type::dimension>()(src);
}




}

}
#endif
