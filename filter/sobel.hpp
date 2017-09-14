//---------------------------------------------------------------------------
#ifndef SOBEL_FILTER_HPP
#define SOBEL_FILTER_HPP
#include "filter_model.hpp"
//---------------------------------------------------------------------------
namespace image
{

namespace filter
{

template<class value_type>
struct sobel_filter_abs_sum;

template<class value_type>
struct sobel_filter_abs_sum{

    typedef typename pixel_manip<value_type>::type manip_type;
    value_type operator()(const manip_type& a,const manip_type& b)
    {
        manip_type d(0);
        d = a > 0 ? a:-a;
        d += b > 0 ? b: -b;
        return d;
    }
    value_type operator()(const manip_type& a,const manip_type& b,const manip_type& c)
    {
        manip_type d(0);
        d = a > 0 ? a:-a;
        d += b > 0 ? b: -b;
        d += c > 0 ? c: -c;
        return d;
    }
};

template<>
struct sobel_filter_abs_sum<image::rgb_color>{

    typedef pixel_manip<image::rgb_color>::type manip_type;
    image::rgb_color operator()(const manip_type& a,const manip_type& b)
    {
        manip_type d;
        d.r = a.r > 0 ? a.r:-a.r;
        d.g = a.g > 0 ? a.g:-a.g;
        d.b = a.b > 0 ? a.b:-a.b;

        d.r += b.r > 0 ? b.r: -b.r;
        d.g += b.g > 0 ? b.g: -b.g;
        d.b += b.b > 0 ? b.b: -b.b;
        return d.to_rgb();
    }
    image::rgb_color operator()(const manip_type& a,const manip_type& b,const manip_type& c)
    {
        manip_type d;
        d.r = a.r > 0 ? a.r:-a.r;
        d.g = a.g > 0 ? a.g:-a.g;
        d.b = a.b > 0 ? a.b:-a.b;

        d.r += b.r > 0 ? b.r: -b.r;
        d.g += b.g > 0 ? b.g: -b.g;
        d.b += b.b > 0 ? b.b: -b.b;

        d.r += c.r > 0 ? c.r: -c.r;
        d.g += c.g > 0 ? c.g: -c.g;
        d.b += c.b > 0 ? c.b: -c.b;
        return d.to_rgb();
    }
};


template<class value_type,size_t dimension>
struct sobel_filter_imp;

template<class value_type>
struct sobel_filter_imp<value_type,2>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<class image_type>
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
        sobel_filter_abs_sum<value_type> sum;
        for(size_t index = 0;index < src.size();++index)
            src[index] = sum(gx[index],gy[index]);
    }
};


template<class value_type>
struct sobel_filter_imp<value_type,3>
{
    typedef typename pixel_manip<value_type>::type manip_type;
public:
    template<class image_type>
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
        sobel_filter_abs_sum<value_type> sum;
        for(size_t index = 0;index < src.size();++index)
            src[index] = sum(gx[index],gy[index],gz[index]);
    }
};

template<class image_type>
void sobel(image_type& src)
{
    sobel_filter_imp<typename image_type::value_type,image_type::dimension>()(src);
}




}

}
#endif
