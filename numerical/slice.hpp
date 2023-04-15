
#ifndef IMAGE_SLICE_HPP
#define IMAGE_SLICE_HPP

namespace tipl
{

template<typename T,typename U>
inline void space2slice(unsigned char dim,const T& p,U& v)
{
    v[0] = p[dim == 0 ? 1:0];
    v[1] = p[dim == 2 ? 1:2];
}

template<typename T,typename U>
inline void space2slice(unsigned char dim,const T& p,U& x,U& y,U& slice_index)
{
    x = p[dim == 0 ? 1:0];
    y = p[dim == 2 ? 1:2];
    slice_index = p[dim];
}

template<typename T,typename U,typename V>
void slice2space(unsigned char dim_index,T x,T y,U slice_index,V& p)
{
    switch(dim_index)
    {
        case 2:
             p[0] = x;p[1] = y;p[2] = slice_index;break;
        case 1:
             p[0] = x;p[1] = slice_index;p[2] = y;break;
        case 0:
             p[0] = slice_index;p[1] = x;p[2] = y;break;
    }
}

template<typename RangeType1,typename RangeType2,typename ResultType>
void get_slice_positions(unsigned char dim,double pos,
                         const RangeType1& range_min,
                         const RangeType2& range_max,
                         ResultType& points)
{
    double x_min,x_max,y_min,y_max;
    if(dim == 1)
    {
    x_min = (double)range_min[0]-0.5;
    x_max = (double)range_max[0]-0.5;
    y_min = (double)range_min[2]-0.5;
    y_max = (double)range_max[2]-0.5;
    }
    else
    {
    x_min = (double)range_min[(dim+1) >= 3 ? dim-2 : dim+1]-0.5;
    x_max = (double)range_max[(dim+1) >= 3 ? dim-2 : dim+1]-0.5;
    y_min = (double)range_min[(dim+2) >= 3 ? dim-1 : dim+2]-0.5;
    y_max = (double)range_max[(dim+2) >= 3 ? dim-1 : dim+2]-0.5;
    }

    double z_pos = pos+(double)range_min[dim];
    tipl::slice2space(dim,x_min,y_min,z_pos,points[0]);
    tipl::slice2space(dim,x_max,y_min,z_pos,points[1]);
    tipl::slice2space(dim,x_min,y_max,z_pos,points[2]);
    tipl::slice2space(dim,x_max,y_max,z_pos,points[3]);
}



template<typename dim_type,typename pos_type,typename GeoType,typename ResultType>
void get_slice_positions(dim_type dim,pos_type pos,const GeoType& geo,ResultType& points)
{
    double x_min,x_max,y_min,y_max;
    if(dim == 1)
    {
    x_min = (double)-0.5;
    x_max = (double)geo[0]-0.5;
    y_min = (double)-0.5;
    y_max = (double)geo[2]-0.5;
    }
    else
    {
    x_min = (double)0-0.5;
    x_max = (double)geo[(dim+1) >= 3 ? dim-2 : dim+1]-0.5;
    y_min = (double)0-0.5;
    y_max = (double)geo[(dim+2) >= 3 ? dim-1 : dim+2]-0.5;
    }

    tipl::slice2space(dim,x_min,y_min,pos,points[0]);
    tipl::slice2space(dim,x_max,y_min,pos,points[1]);
    tipl::slice2space(dim,x_min,y_max,pos,points[2]);
    tipl::slice2space(dim,x_max,y_max,pos,points[3]);
}







};

#endif//
