#ifndef CANNY_EDGE_HPP_INCLUDED
#define CANNY_EDGE_HPP_INCLUDED


#include "filter_model.hpp"
//---------------------------------------------------------------------------
namespace tipl
{

namespace filter
{

template<class value_type,size_t dimension>
class canny_edge_filter_imp;

template<class value_type>
class canny_edge_filter_imp<value_type,2>
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

        for(size_t index = 0;index < src.size();++index)
        {
            float fx = gx[index];
            float fy = gy[index];
            src[index] = std::sqrt(fx*fx+fy*fy);
        }

        int i1,i2;
        float abs_x,abs_y,fx,fy;
        for(int index = 0;index < src.size();++index)
        {
            fx = gx[index];
            fy = gy[index];
            abs_x = (fx >= 0) ? fx : -fx;
            abs_y = (fy >= 0) ? fy : -fy;

            //edge at x direction
            if(abs_x > abs_y * 2.41421356)
            {
                i1 = index - 1;
                i2 = index + 1;
            }
            else
            //edge at y direction
            if(abs_y > abs_x * 2.41421356)
            {
                i1 = index - w;
                i2 = index + w;
            }
            else
            if((fx > 0 && fy > 0) || (fx < 0 && fy < 0))
            {
                i1 = index - 1 - w;
                i2 = index + 1 + w;
            }
            else
            {
                i1 = index + 1 - w;
                i2 = index - 1 + w;
            }

            if(i1 < 0 || i2 >= src.size())
                continue;
            // perform non-maximum elimination
            if(src[index] < src[i1] || src[index] < src[i2])
                src[index] = 0;

        }

    }
};

template<class value_type>
class canny_edge_filter_imp<value_type,3>
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

        for(size_t index = 0;index < src.size();++index)
        {
            float fx = gx[index];
            float fy = gy[index];
            float fz = gz[index];
            src[index] = std::sqrt(fx*fx+fy*fy+fz*fz);
        }

        int shift,i1,i2;
        float fx,fy,fz,max_value;
        for(int index = 0;index < src.size();++index)
        {
            if(src[index] == 0)
                continue;
            max_value = src[index];
            fx = gx[index]/max_value;
            fy = gy[index]/max_value;
            fz = gz[index]/max_value;

            shift = 0;
            if(fx >= 0.577)
                shift += 1;
            else
            if(fx < -0.577)
                shift -= 1;

            if(fy >= 0.577)
                shift += w;
            else
            if(fy < -0.577)
                shift -= w;

            if(fz >= 0.577)
                shift += wh;
            else
            if(fz < -0.577)
                shift -= wh;

            i1 = index - shift;
            i2 = index + shift;

            if(i1 < 0 || i2 >= src.size())
                continue;
            // perform non-maximum elimination
            if(src[index] < src[i1] || src[index] < src[i2])
                src[index] = 0;

        }

    }
};

template<class pixel_type,size_t dimension>
void canny_edge(image<pixel_type,dimension>& src)
{
    canny_edge_filter_imp<pixel_type,dimension>()(src);
}




}

}

#endif // CANNY_EDGE_HPP_INCLUDED
