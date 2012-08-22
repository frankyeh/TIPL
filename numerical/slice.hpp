
#ifndef IMAGE_SLICE_HPP
#define IMAGE_SLICE_HPP

namespace image
{

template<typename value_type1,typename value_type2>
void space2slice(unsigned char dim_index,
        value_type1 px,value_type1 py,value_type1 pz,
        value_type2& x,value_type2& y,value_type2& slice_index)
{
    switch(dim_index)
    {
        case 2:
             x = px;y = py;slice_index = pz;return;
        case 1:
             x = px;y = pz;slice_index = py;return;
        case 0:
             x = py;y = pz;slice_index = px;return;
    }
}


template<typename value_type1,typename slice_type,typename value_type2>
void slice2space(unsigned char dim_index,
        value_type1 x,value_type1 y,slice_type slice_index,
        value_type2& px,value_type2& py,value_type2& pz)
{
    switch(dim_index)
    {
        case 2:
             px = x;py = y;pz = slice_index;break;
        case 1:
             px = x;py = slice_index;pz = y;break;
        case 0:
             px = slice_index;py = x;pz = y;break;
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
    image::slice2space(dim,x_min,y_min,z_pos,points[0][0],points[0][1],points[0][2]);
    image::slice2space(dim,x_max,y_min,z_pos,points[1][0],points[1][1],points[1][2]);
    image::slice2space(dim,x_min,y_max,z_pos,points[2][0],points[2][1],points[2][2]);
    image::slice2space(dim,x_max,y_max,z_pos,points[3][0],points[3][1],points[3][2]);
}



template<typename GeoType,typename ResultType>
void get_slice_positions(unsigned char dim,double pos,const GeoType& geo,ResultType& points)
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

    image::slice2space(dim,x_min,y_min,pos,points[0][0],points[0][1],points[0][2]);
    image::slice2space(dim,x_max,y_min,pos,points[1][0],points[1][1],points[1][2]);
    image::slice2space(dim,x_min,y_max,pos,points[2][0],points[2][1],points[2][2]);
    image::slice2space(dim,x_max,y_max,pos,points[3][0],points[3][1],points[3][2]);
}







};

#endif//
