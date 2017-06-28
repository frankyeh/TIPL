//---------------------------------------------------------------------------
#ifndef BASIC_OP_HPP
#define BASIC_OP_HPP
#include "image/utility/pixel_index.hpp"
#include "image/utility/basic_image.hpp"
#include "image/utility/multi_thread.hpp"

namespace image
{

template<class iterator_type1,class iterator_type2,class int_type>
inline void copy_ptr(iterator_type1 iter1,iterator_type2 iter2,int_type size)
{
    for(iterator_type1 end = iter1+size; iter1 != end; ++iter1,++iter2)
        *iter2 = typename std::iterator_traits<iterator_type2>::value_type(*iter1);
}

template<class iterator_type1,class iterator_type2,class fun_type>
inline void for_each(iterator_type1 iter1,iterator_type1 end,iterator_type2 iter2,fun_type fun)
{
    for(; iter1 != end; ++iter1,++iter2)
        fun(*iter1,*iter2);
}
/*
example image::binary(classification,label,std::bind2nd (std::not_equal_to<unsigned char>(), background_index));
*/

template<class ImageType,class LabelImageType,class fun_type>
inline void binary(const ImageType& image,LabelImageType& out,fun_type fun)
{
    out.resize(image.geometry());
    typename ImageType::const_iterator iter = image.begin();
    typename ImageType::const_iterator end = image.end();
    typename LabelImageType::iterator out_iter = out.begin();
    for(; iter!=end; ++iter,++out_iter)
        if(fun(*iter))
            *out_iter = 1;
        else
            *out_iter = 0;
}

template<class ImageType,class fun_type>
inline void binary(ImageType& image,fun_type fun)
{
    typename ImageType::iterator iter = image.begin();
    typename ImageType::iterator end = image.end();
    for(; iter!=end; ++iter)
        if(fun(*iter))
            *iter = 1;
        else
            *iter = 0;
}

//---------------------------------------------------------------------------
template<class ImageType,class LabelImageType>
void threshold(const ImageType& image,LabelImageType& out,typename ImageType::value_type threshold_value,typename LabelImageType::value_type foreground = 255,typename LabelImageType::value_type background = 0)
{
    out.resize(image.geometry());
    typename ImageType::const_iterator iter = image.begin();
    typename ImageType::const_iterator end = image.end();
    typename LabelImageType::iterator out_iter = out.begin();
    for(; iter!=end; ++iter,++out_iter)
        if(*iter >= threshold_value)
            *out_iter = foreground;
        else
            *out_iter = background;
}

//--------------------------------------------------------------------------
template<class PixelType,class DimensionType,class storage_type>
void crop(const basic_image<PixelType,2,storage_type>& from_image,
          basic_image<PixelType,2,storage_type>& to_image,
          const DimensionType& from,
          const DimensionType& to)
{
    if (to[0] <= from[0] || to[1] <= from[1])
        return;
    image::geometry<2> geo(to[0]-from[0],to[1]-from[1]);
    to_image.resize(geo);
    unsigned int size = geo.size();
    unsigned int from_index = from[1]*from_image.width() + from[0];
    unsigned int shift = from_image.width()-geo.width();
    for (unsigned int to_index = 0,step_index = geo.width();
            to_index < size; from_index += shift,step_index += geo.width())
        for (; to_index != step_index; ++to_index,++from_index)
            to_image[to_index] = from_image[from_index];

}
//--------------------------------------------------------------------------
template<class PixelType,class DimensionType,class storage_type>
void crop(const basic_image<PixelType,3,storage_type>& from_image,
          basic_image<PixelType,3,storage_type>& to_image,
          const DimensionType& from,
          const DimensionType& to)
{
    if (to[0] <= from[0] || to[1] <= from[1] ||
            to[2] <= from[2])
        return;
    image::geometry<3> geo(to[0]-from[0],to[1]-from[1],to[2]-from[2]);
    to_image.resize(geo);
    unsigned int from_index = (from[2]*from_image.height()+from[1])*from_image.width()+from[0];
    unsigned int y_shift = from_image.width()-geo.width();
    unsigned int z_shift = from_image.width()*from_image.height()-geo.height()*from_image.width();
    for (unsigned int z = from[2],to_index = 0; z < to[2]; ++z,from_index += z_shift)
        for (unsigned int y = from[1]; y < to[1]; ++y,from_index += y_shift)
            for (unsigned int x = from[0]; x < to[0]; ++x,++to_index,++from_index)
                to_image[to_index] = from_image[from_index];
}
//---------------------------------------------------------------------------
template<class ImageType,class DimensionType>
void crop(ImageType& in_image,
          const DimensionType& from,
          const DimensionType& to)
{
    ImageType out_image;
    crop(in_image,out_image,from,to);
    in_image.swap(out_image);
}
//--------------------------------------------------------------------------
template<class image_type,class PosType,class pixel_type>
void fill_rect(image_type& I,PosType from,PosType to,pixel_type value)
{
    int line_pos = from[0] + from[1]*I.width();
    int line_width = to[0]-from[0];
    for(int y = from[1];y < to[1];++y)
    {
        std::fill(I.begin()+line_pos,I.begin()+line_pos+line_width,value);
        line_pos += I.width();
    }
}

//--------------------------------------------------------------------------
template<class pixel_type1,class storage_type1,
         typename pixel_type2,class storage_type2,class PosType>
void draw(const basic_image<pixel_type1,2,storage_type1>& from_image,
          basic_image<pixel_type2,2,storage_type2>& to_image,
          PosType pos)
{
    typedef basic_image<pixel_type1,2,storage_type1> from_image_type;
    typedef basic_image<pixel_type2,2,storage_type2> to_image_type;
    int x_shift,y_shift;
    if (pos[0] < 0)
    {
        x_shift = -pos[0];
        pos[0] = 0;
    }
    else
        x_shift = 0;
    if (pos[1] < 0)
    {
        y_shift = -pos[1];
        pos[1] = 0;
    }
    else
        y_shift = 0;

    int x_width = std::min((int)to_image.width() - (int)pos[0],(int)from_image.width()-x_shift);
    if (x_width <= 0)
        return;
    int y_height = std::min((int)to_image.height() - (int)pos[1],(int)from_image.height()-y_shift);
    if (y_height <= 0)
        return;
    typename from_image_type::const_iterator iter = from_image.begin() + y_shift*from_image.width()+x_shift;
    typename from_image_type::const_iterator end = iter + (y_height-1)*from_image.width();
    typename to_image_type::iterator out = to_image.begin() + pos[1]*to_image.width()+pos[0];
    for (; iter != end; iter += from_image.width(),out += to_image.width())
        std::copy(iter,iter+x_width,out);
    std::copy(iter,iter+x_width,out);

}
//--------------------------------------------------------------------------
template<class pixel_type1,class storage_type1,
         typename pixel_type2,class storage_type2,class PosType>
void draw(const basic_image<pixel_type1,3,storage_type1>& from_image,
          basic_image<pixel_type2,3,storage_type2>& to_image,
          PosType pos)
{
    typedef basic_image<pixel_type1,3,storage_type1> from_image_type;
    typedef basic_image<pixel_type2,3,storage_type2> to_image_type;
    int x_shift,y_shift,z_shift;
    if (pos[0] < 0)
    {
        x_shift = -pos[0];
        pos[0] = 0;
    }
    else
        x_shift = 0;
    if (pos[1] < 0)
    {
        y_shift = -pos[1];
        pos[1] = 0;
    }
    else
        y_shift = 0;
    if (pos[2] < 0)
    {
        z_shift = -pos[2];
        pos[2] = 0;
    }
    else
        z_shift = 0;

    int x_width = std::min((int)to_image.width() - (int)pos[0],(int)from_image.width()-x_shift);
    if (x_width <= 0)
        return;
    int y_height = std::min((int)to_image.height() - (int)pos[1],(int)from_image.height()-y_shift);
    if (y_height <= 0)
        return;
    int z_depth = std::min((int)to_image.depth() - (int)pos[2],(int)from_image.depth()-z_shift);
    if (z_depth <= 0)
        return;
    for (unsigned int z = 0; z < z_depth; ++z)
    {
        typename from_image_type::const_iterator iter = from_image.begin() +
                ((z_shift+z)*from_image.height() + y_shift)*from_image.width()+x_shift;
        typename from_image_type::const_iterator end = iter + (y_height-1)*from_image.width();
        typename to_image_type::iterator out = to_image.begin() +
                ((pos[2]+z)*to_image.height() + pos[1])*to_image.width()+pos[0];
        for (; iter != end; iter += from_image.width(),out += to_image.width())
            std::copy(iter,iter+x_width,out);
        std::copy(iter,iter+x_width,out);
    }
}
//---------------------------------------------------------------------------
template<class PixelType,class PosType>
void move(basic_image<PixelType,2>& src,PosType pos)
{
    basic_image<PixelType,2> dest(geometry<2>(src.width() + std::abs(pos[0]),src.height() + std::abs(pos[1])));
    draw(src,dest,pos);
    dest.swap(src);
}
//---------------------------------------------------------------------------
template<class PixelType,class PosType>
void move(basic_image<PixelType,3>& src,PosType pos)
{
    basic_image<PixelType,3> dest(
        geometry<3>(src.width() + std::abs(pos[0]),
                    src.height() + std::abs(pos[1]),
                    src.depth() + std::abs(pos[2])));
    draw(src,dest,pos);
    dest.swap(src);
}
//---------------------------------------------------------------------------
template<class ImageType,class DimensionType>
void bounding_box(const ImageType& I,
          DimensionType& range_min,
          DimensionType& range_max,
          typename ImageType::value_type background = 0)
{
    //get_border(image,range_min,range_max);
    for (unsigned int di = 0; di < ImageType::dimension; ++di)
    {
        range_min[di] = I.geometry()[di]-1;
        range_max[di] = 0;
    }
    for (pixel_index<ImageType::dimension> iter(I.geometry());iter < I.size();++iter)
    {
        if (I[iter.index()] == background)
            continue;
        for (unsigned int di = 0; di < ImageType::dimension; ++di)
        {
            if (iter[di] < range_min[di])
                range_min[di] = iter[di];
            if (iter[di] > range_max[di])
                range_max[di] = iter[di];
        }
    }

    for (unsigned int di = 0; di < ImageType::dimension; ++di)
        range_max[di] = range_max[di] + 1;

}

// ---------------------------------------------------------------------------
template<typename point_type>
void bounding_box_mt(const std::vector<point_type>& points,point_type& max_value,point_type& min_value)
{
    if(points.empty())
        return;
    std::vector<point_type> max_values(std::thread::hardware_concurrency()),
                            min_values(std::thread::hardware_concurrency());
    for(int i = 0;i < max_values.size();++i)
    {
        max_values[i] = points[0];
        min_values[i] = points[0];
    }
    unsigned char dim = points[0].size();
    image::par_for2(points.size(),[&](unsigned int index,unsigned int id)
    {
        for (unsigned char d = 0; d < dim; ++d)
            if (points[index][d] > max_values[id][d])
                max_values[id][d] = points[index][d];
            else if (points[index][d] < min_values[id][d])
                min_values[id][d] = points[index][d];
    });
    max_value = max_values[0];
    min_value = min_values[0];

    for(int i = 0;i < max_value.size();++i)
    {
        for (unsigned char d = 0; d < dim; ++d)
            if (max_values[i][d] > max_value[d])
                max_value[d] = max_values[i][d];
            else if (max_values[i][d] < min_value[d])
                min_value[d] = max_values[i][d];
    }
}

template<class ImageType>
void trim(ImageType& image,class ImageType::value_type background = 0)
{
    image::geometry<ImageType::dimension> range_min,range_max;
    bounding_box(image,range_min,range_max,background);
    if (range_min[0] < range_max[0])
        crop(image,range_min,range_max);
}


/** get axis orientation from rotation matrix
    dim_order[3] = {2,1,0} means the iteration goes from z->y->x
    flip ={ 1,0,0} mean the first dimension need to be flipped
*/
//---------------------------------------------------------------------------
template<class iterator_type,class dim_order_type,class flip_type>
void get_orientation(int dim,iterator_type rotation_matrix,dim_order_type dim_order,flip_type flipped)
{
    iterator_type vec = rotation_matrix;
    for (int index = 0; index < dim; ++index,vec += dim)
    {
        dim_order[index] = 0;
        flipped[index] = vec[0] < 0;
        for(int j = 1; j < dim; ++j)
            if(std::abs(vec[j]) > std::abs(vec[dim_order[index]]))
            {
                dim_order[index] = j;
                flipped[index] = vec[j] < 0;
            }
    }
}
//---------------------------------------------------------------------------
template<class iterator_type,class dim_order_type,class flip_type>
void get_inverse_orientation(int dim,iterator_type rotation_matrix,dim_order_type dim_order,flip_type flipped)
{
    iterator_type vec = rotation_matrix;
    for (int index = 0; index < dim; ++index,++vec)
    {
        dim_order[index] = 0;
        flipped[index] = vec[0] < 0;
        for(int j = 1; j < dim; ++j)
            if(std::abs(vec[j*dim]) > std::abs(vec[dim_order[index]*dim]))
            {
                dim_order[index] = j;
                flipped[index] = (vec[j*dim] < 0);
            }
    }
}
//---------------------------------------------------------------------------
template<class image_type1,class image_type2>
void reorder(const image_type1& volume,image_type2& volume_out,int origin[],int shift[],int index_dim)
{
    unsigned int index = 0;
    unsigned int base_index = 0;
    while(index < volume.size())
    {
        if(index_dim == 2)
        {
            int y_index = base_index + origin[1];
            for (int y = 0; y < volume.height(); ++y)
            {
                int x_index = y_index + origin[0];
                for (int x = 0; x < volume.width(); ++x,++index)
                {
                    volume_out[x_index] = volume[index];
                    x_index += shift[0];
                }
                y_index += shift[1];
            }
            base_index += unsigned int(volume_out.plane_size());
        }
        if(index_dim == 3)
        {
            int z_index = base_index + origin[2];
            for (int z = 0; z < volume.geometry()[2]; ++z)
            {
                int y_index = z_index + origin[1];
                for (int y = 0; y < volume.height(); ++y)
                {
                    int x_index = y_index + origin[0];
                    for (int x = 0; x < volume.width(); ++x,++index)
                    {
                        volume_out[x_index] = volume[index];
                        x_index += shift[0];
                    }
                    y_index += shift[1];
                }
                z_index += shift[2];
            }
            base_index += unsigned int(volume_out.plane_size()*volume.depth());
        }
    }
}
template<class iterator_type,class dim_order_type>
void reorient_vector(iterator_type spatial_resolution,dim_order_type dim_order)
{
    float sr[3];
    std::copy(spatial_resolution,spatial_resolution+3,sr);
    for(unsigned int index = 0;index < 3;++index)
        spatial_resolution[dim_order[index]] = sr[index];
}
template<class iterator_type2,class dim_order_type,class flip_type>
void reorient_matrix(iterator_type2 orientation_matrix,dim_order_type dim_order,flip_type flip)
{
    float orientation_matrix_[9];
    std::copy(orientation_matrix,orientation_matrix+9,orientation_matrix_);
    for(unsigned int index = 0,ptr = 0;index < 3;++index,ptr += 3)
        if(flip[index])
        {
            orientation_matrix_[ptr] = -orientation_matrix_[ptr];
            orientation_matrix_[ptr+1] = -orientation_matrix_[ptr+1];
            orientation_matrix_[ptr+2] = -orientation_matrix_[ptr+2];
        }
    for(unsigned int index = 0;index < 3;++index)
    std::copy(orientation_matrix_+index*3,
              orientation_matrix_+index*3+3,
              orientation_matrix+dim_order[index]*3);
}
//---------------------------------------------------------------------------
template<class geo_type,class dim_order_type,class flip_type,class origin_type,class shift_type>
bool reorder_shift_index(const geo_type& geo,
                         dim_order_type dim_order,
                         flip_type flip,
                         geo_type& new_geo,
                         origin_type origin_index,
                         shift_type shift_index)
{

    bool need_update = false;
    // get the dimension mapping
    for (unsigned char index = 0; index < geo_type::dimension; ++index)
    {
        new_geo[dim_order[index]] = geo[index];
        if (dim_order[index] != index)
            need_update = true;
    }

    std::vector<int> shift_vector(geo_type::dimension);
    shift_vector[0] = 1;
    for(unsigned char dim = 1;dim < geo_type::dimension;++dim)
        shift_vector[dim] = shift_vector[dim-1]*new_geo[dim-1];

    for (unsigned int index = 0; index < geo_type::dimension;++index)
    {
        if (flip[index])
        {
            origin_index[index] = shift_vector[dim_order[index]]*(new_geo[dim_order[index]]-1);
            shift_index[index] = -shift_vector[dim_order[index]];
            need_update = true;
        }
        else
        {
            origin_index[index] = 0;
            shift_index[index] = shift_vector[dim_order[index]];
        }
    }
    return need_update;
}


/**
   dim_order[3] = {2,1,0} flip = {1,0,0}
   output (-z,y,x) <- input(x,y,z);
*/

template<class image_type1,class image_type2,class dim_order_type,class flip_type>
void reorder(const image_type1& volume,image_type2& volume_out,dim_order_type dim_order,flip_type flip)
{
    image::geometry<image_type1::dimension> new_geo;
    int origin[image_type1::dimension];
    int shift[image_type1::dimension];
    if (!reorder_shift_index(volume.geometry(),dim_order,flip,new_geo,origin,shift))
    {
        volume_out = volume;
        return;
    }
    volume_out.resize(new_geo);
    reorder(volume,volume_out,origin,shift,image_type1::dimension);
}
//---------------------------------------------------------------------------
template<class image_type,class dim_order_type,class flip_type>
void reorder(image_type& volume,dim_order_type dim_order,flip_type flip)
{
    image_type volume_out;
    reorder(volume,volume_out,dim_order,flip);
    volume.swap(volume_out);
}

//---------------------------------------------------------------------------
template<class iterator_type>
void flip_block(iterator_type beg,iterator_type end,unsigned int block_size)
{
    while (beg < end)
    {
        iterator_type from = beg;
        beg += block_size;
        iterator_type to = beg-1;
        while (from < to)
        {
            std::swap(*from,*to);
            ++from;
            --to;
        }
    }
}
//---------------------------------------------------------------------------
template<class iterator_type>
void flip_block_line(iterator_type beg,iterator_type end,unsigned int block_size,unsigned int line_length)
{
    if(line_length == 1)
        flip_block(beg,end,block_size);
    else
        while (beg < end)
        {
            iterator_type from = beg;
            beg += block_size;
            iterator_type to = beg-line_length;
            while (from < to)
            {
                for(unsigned int index = 0; index < line_length; ++index)
                    std::swap(*(from+index),*(to+index));
                from += line_length;
                to -= line_length;
            }
        }
}
//---------------------------------------------------------------------------
template<class ImageType>
void flip_x(ImageType& image)
{
    flip_block(image.begin(),image.end(),image.width());
}
//---------------------------------------------------------------------------
template<class ImageType>
void flip_y(ImageType& image)
{
    flip_block_line(image.begin(),image.end(),image.height() * image.width(),image.width());
}
//---------------------------------------------------------------------------
template<class ImageType>
void flip_z(ImageType& image)
{
    flip_block_line(image.begin(),image.end(),image.geometry().plane_size() * image.depth(),image.geometry().plane_size());
}
//---------------------------------------------------------------------------
template<class ImageType>
void flip_xy(ImageType& I)
{
    flip_block(I.begin(),I.end(),I.height() * I.width());
}

template<class ImageType>
void swap_xy(ImageType& I)
{
    typedef typename ImageType::value_type value_type;
    image::geometry<ImageType::dimension> new_geo(I.geometry());
    std::swap(new_geo[0],new_geo[1]);
    image::basic_image<value_type,ImageType::dimension> new_volume(new_geo);
    int origin[2] = {0,0};
    int shift[2];
    shift[0] = new_geo.width();
    shift[1] = 1;
    reorder(I,new_volume,origin,shift,2);

    I.resize(new_geo);
    std::copy(new_volume.begin(),new_volume.end(),I.begin());
}
//---------------------------------------------------------------------------
template<class ImageType>
void swap_xz(ImageType& I)
{
    typedef typename ImageType::value_type value_type;
    image::geometry<ImageType::dimension> new_geo(I.geometry());
    std::swap(new_geo[0],new_geo[2]);
    image::basic_image<value_type,ImageType::dimension> new_volume(new_geo);

    int origin[3] = {0,0,0};
    int shift[3];
    shift[0] = new_geo.plane_size();
    shift[1] = new_geo.width();
    shift[2] = 1;
    reorder(I,new_volume,origin,shift,3);

    I.resize(new_geo);
    std::copy(new_volume.begin(),new_volume.end(),I.begin());
}
//---------------------------------------------------------------------------
template<class ImageType>
void swap_yz(ImageType& I)
{
    typedef typename ImageType::value_type value_type;
    image::geometry<ImageType::dimension> new_geo(I.geometry());
    std::swap(new_geo[1],new_geo[2]);
    image::basic_image<value_type,ImageType::dimension> new_volume(new_geo);

    int origin[3] = {0,0,0};
    int shift[3];
    shift[0] = 1;
    shift[1] = new_geo.plane_size();
    shift[2] = new_geo.width();
    reorder(I,new_volume,origin,shift,3);

    I.resize(new_geo);
    std::copy(new_volume.begin(),new_volume.end(),I.begin());
}
//---------------------------------------------------------------------------
template<class ImageType>
void flip(ImageType& image,unsigned char dim)
{
    switch(dim)
    {
    case 0:
        flip_x(image);
    break;
    case 1:
        flip_y(image);
    break;
    case 2:
        flip_z(image);
    break;
    case 3:
        swap_xy(image);
    break;
    case 4:
        swap_yz(image);
    break;
    case 5:
        swap_xz(image);
    break;
    }
}
//---------------------------------------------------------------------------
template<class ImageType,class value_type>
void negate(ImageType& image,value_type maximum)
{
    typename ImageType::iterator iter = image.begin();
    typename ImageType::iterator end = image.end();
    for (; iter != end; ++iter)
        *iter = maximum - *iter;
}
//---------------------------------------------------------------------------
template<class ImageType>
void negate(ImageType& image)
{
    negate(image,*std::max_element(image.begin(),image.end()));
}

template<class ImageType1,class ImageType2,class PixelType2>
void paint(const ImageType1& image1,ImageType2& image2,PixelType2 paint_value)
{
    typename ImageType1::const_iterator iter1 = image1.begin();
    typename ImageType2::iterator iter2 = image2.begin();
    typename ImageType1::const_iterator end = image1.end();
    for (; iter1 != end; ++iter1,++iter2)
        if (*iter1)
            *iter2 = paint_value;
}

/*
template<class PixelType1,class PixelType2,class LocationType>
void draw(const image::basic_image<PixelType1,2>& src,
          image::basic_image<PixelType2,2>& des,LocationType place)
{
    int x_src = 0;
    int x_des = place[0];
    int y_src = 0;
    int y_des = place[1];
    if (x_des < 0)
    {
        x_src = -x_des;
        x_des = 0;
    }
    if (y_des < 0)
    {
        x_src = -y_des;
        y_des = 0;
    }
    int draw_width = src.width() - x_src;
    int draw_height = src.height() - y_src;
    if (x_des + draw_width > des.width())
        draw_width = des.width() - x_des;
    if (y_des + draw_height > des.height())
        draw_height = des.height() - y_des;
    const PixelType1* src_iter = src.begin()+y_src*src.width()+x_src;
    const PixelType1* src_end = src_iter + draw_height*src.width();
    const PixelType2* des_iter = des.begin()+y_des*des.width()+x_des;
    for(; src_iter != src_end; src_iter += src.width(),des_iter += des.width())
        std::copy(src_iter,src_iter+draw_width,des_iter);
}
*/

template<class PixelType1,class PixelType2,class LocationType,class DetermineType>
void draw_if(const image::basic_image<PixelType1,2>& src,
             image::basic_image<PixelType2,2>& des,LocationType place,DetermineType pred_background)
{
    int x_src = 0;
    int x_des = place[0];
    int y_src = 0;
    int y_des = place[1];
    if (x_des < 0)
    {
        x_src = -x_des;
        x_des = 0;
    }
    if (y_des < 0)
    {
        x_src = -y_des;
        y_des = 0;
    }
    int draw_width = src.width() - x_src;
    int draw_height = src.height() - y_src;
    if (x_des + draw_width > des.width())
        draw_width = des.width() - x_des;
    if (y_des + draw_height > des.height())
        draw_height = des.height() - y_des;
    const PixelType1* src_iter = src.begin()+y_src*src.width()+x_src;
    const PixelType1* src_end = src_iter + draw_height*src.width();
    PixelType2* des_iter = des.begin()+y_des*des.width()+x_des;
    for(; src_iter != src_end; des_iter += des.width())
    {
        const PixelType1* from = src_iter;
        const PixelType1* to = src_iter+src.width();
        PixelType2* des = des_iter;
        for(; from != to; ++from,++des)
            if(!pred_background(*from))
                *des = *from;
        src_iter = to;
    }
}

template<class PixelType1,class OutImageType>
void project(const image::basic_image<PixelType1,2>& src,OutImageType& result,unsigned int dim)
{
    if(dim == 0) // project x
    {
        result.resize(src.height());
        for(unsigned int y = 0,index = 0; y < src.height(); ++y,index += src.width())
            result[y] = std::accumulate(src.begin()+index,src.begin()+index+src.width(),(typename OutImageType::value_type)0);
    }
    else//project y
    {
        result.clear();
        result.resize(src.width());
        for(pixel_index<2> index(src.geometry());index < src.size();++index)
            result[index.x()] += src[index.index()];
    }
}

template<class ImageType>
float variance(const ImageType& I)
{
    float sum = 0;
    float sum_square = 0;
    typename ImageType::const_iterator iter = I.begin();
    typename ImageType::const_iterator end = I.end();
    for(; iter != end; ++iter)
    {
        float value = *iter;
        sum += value;
        sum_square += value*value;
    }
    sum_square /= I.size();
    sum /= I.size();
    return sum_square-sum*sum;
}

template<class ImageType>
void histogram(const ImageType& src,std::vector<unsigned int>& hist,
               typename ImageType::value_type min_value,
               typename ImageType::value_type max_value,unsigned int resolution_count = 256)
{
    if(min_value >= max_value)
        return;
    float range = max_value;
    range -= min_value;
    if(range + 1.0 == range)
        range = 1.0;
    hist.clear();
    range = (float)resolution_count/range;
    hist.resize(resolution_count);
    typename ImageType::const_iterator iter = src.begin();
    typename ImageType::const_iterator end = src.end();
    for(; iter != end; ++iter)
    {
        float value = *iter;
        value -= min_value;
        value *= range;
        int index = std::floor(value);
        if(index < 0)
            index = 0;
        if(index >= hist.size())
            index = hist.size()-1;
        ++hist[index];
    }
}
template<class image_type1,class image_type2>
void hist_norm(const image_type1& I1,image_type2& I2,unsigned int bin_count)
{
    typename image_type1::value_type min_v = *std::min_element(I1.begin(),I1.end());
    typename image_type1::value_type max_v = *std::max_element(I1.begin(),I1.end());

    std::vector<unsigned int> hist;
    image::histogram(I1,hist,min_v,max_v,bin_count);

    for(unsigned int i = 1;i < hist.size();++i)
        hist[i] += hist[i-1];
    if(I2.size() != I1.size())
        I2.resize(I1.geometry());
    float range = max_v-min_v;
    if(range == 0)
        range = 1;
    float r = (hist.size()+1)/range;
    for(unsigned int i = 1;i < I1.size();++i)
    {
        int rank = std::floor((float)(I1[i]-min_v)*r);
        if(rank <= 0)
            I2[i] = min_v;
        else
        {
            --rank;
            if(rank >= hist.size())
                rank = hist.size()-1;
            I2[i] = range*(float)hist[rank]/(float)hist.back()+min_v;
        }
    }
}
template<class image_type>
void hist_norm(image_type& I1,unsigned int bin_count)
{
    hist_norm(I1,I1,bin_count);
}

template<class type>
void change_endian(type& value)
{
    type data = value;
    unsigned char* temp = (unsigned char*)&value;
    unsigned char* pdata = ((unsigned char*)&data)+sizeof(type)-1;
    for (char i = 0; i < sizeof(type); ++i,--pdata)
        temp[i] = *pdata;
}
inline void change_endian(unsigned short& data)
{
    unsigned char* h = (unsigned char*)&data;
    std::swap(*h,*(h+1));
}

inline void change_endian(short& data)
{
    unsigned char* h = (unsigned char*)&data;
    std::swap(*h,*(h+1));
}


inline void change_endian(unsigned int& data)
{
    unsigned char* h = (unsigned char*)&data;
    std::swap(*h,*(h+3));
    std::swap(*(h+1),*(h+2));
}

inline void change_endian(int& data)
{
    unsigned char* h = (unsigned char*)&data;
    std::swap(*h,*(h+3));
    std::swap(*(h+1),*(h+2));
}

inline void change_endian(float& data)
{
    change_endian(*(int*)&data);
}

template<class datatype>
inline void change_endian(datatype* data,int count)
{
    for (int index = 0; index < count; ++index)
        change_endian(data[index]);
}

}
#endif
