//---------------------------------------------------------------------------
#ifndef BASIC_OP_HPP
#define BASIC_OP_HPP
#include "image/utility/basic_image.hpp"
#include "numerical.hpp"

namespace image
{

template<typename iterator_type1,typename iterator_type2,typename fun_type>
inline void for_each(iterator_type1 iter1,iterator_type1 end,iterator_type2 iter2,fun_type fun)
{
    for(; iter1 != end; ++iter1,++iter2)
        fun(*iter1,*iter2);
}
/*
example image::binary(classification,label,std::bind2nd (std::not_equal_to<unsigned char>(), background_index));
*/

template<typename ImageType,typename LabelImageType,typename fun_type>
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

template<typename ImageType,typename fun_type>
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
template<typename ImageType,typename LabelImageType>
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
template<typename PixelType,typename DimensionType>
void crop(const basic_image<PixelType,2>& from_image,
          basic_image<PixelType,2>& to_image,
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
template<typename PixelType,typename DimensionType>
void crop(const basic_image<PixelType,3>& from_image,
          basic_image<PixelType,3>& to_image,
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
template<typename ImageType,typename DimensionType>
void crop(ImageType& in_image,
          const DimensionType& from,
          const DimensionType& to)
{
    ImageType out_image;
    crop(in_image,out_image,from,to);
    in_image.swap(out_image);
}
//--------------------------------------------------------------------------
template<typename pixel_type1,typename storage_type1,
         typename pixel_type2,typename storage_type2,typename PosType>
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
template<typename pixel_type1,typename storage_type1,
         typename pixel_type2,typename storage_type2,typename PosType>
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
template<typename PixelType,typename PosType>
void move(basic_image<PixelType,2>& src,PosType pos)
{
    basic_image<PixelType,2> dest(geometry<2>(src.width() + std::abs(pos[0]),src.height() + std::abs(pos[1])));
    draw(src,dest,pos);
    dest.swap(src);
}
//---------------------------------------------------------------------------
template<typename PixelType,typename PosType>
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
template<typename ImageType,typename DimensionType>
void range(ImageType& image,
          DimensionType& range_min,
          DimensionType& range_max,
          typename ImageType::value_type background = 0)
{
    //get_border(image,range_min,range_max);
    for (unsigned int di = 0; di < ImageType::dimension; ++di)
    {
        range_min[di] = image.geometry()[di]-1;
        range_max[di] = 0;
    }
    for (pixel_index<ImageType::dimension> iter; iter.valid(image.geometry()); iter.next(image.geometry()))
    {
        if (image[iter.index()] == background)
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

template<typename ImageType,typename DimensionType>
void trim(ImageType& image,
          DimensionType& range_min,
          DimensionType& range_max,
          typename ImageType::value_type background = 0)
{
    range(image,range_min,range_max,background);
    if (range_min[0] < range_max[0])
        crop(image,range_min,range_max);
}

/** get axis orientation from rotation matrix
    dim_order[3] = {2,1,0} means the iteration goes from z->y->x
    flip ={ 1,0,0} mean the first dimension need to be flipped
*/
//---------------------------------------------------------------------------
template<typename iterator_type,typename dim_order_type,typename flip_type>
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
template<typename iterator_type,typename dim_order_type,typename flip_type>
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
template<typename value_type,typename dim_order_type,typename flip_type>
void reorder(const image::basic_image<value_type,2>& volume,
             image::basic_image<value_type,2>& volume_out,
             dim_order_type dim_order,flip_type flip)
{
    image::geometry<2> old_geometry(volume.geometry());
    image::geometry<2> new_geometry;
    bool need_reorder = false;
    // get the dimension mapping
    for (unsigned int index = 0; index < 2; ++index)
    {
        new_geometry[dim_order[index]] = old_geometry[index];
        if (dim_order[index] != index)
            need_reorder = true;
    }
    int shift_vector[2];
    int xyz_origin_index[2];
    int xyz_shift_index[2];

    shift_vector[0] = 1;
    shift_vector[1] = new_geometry.width();

    for (unsigned int index = 0; index < 2; ++index)
    {
        if (flip[index])
        {
            xyz_origin_index[index] = shift_vector[dim_order[index]]*(new_geometry[dim_order[index]]-1);
            xyz_shift_index[index] = -shift_vector[dim_order[index]];
            need_reorder = true;
        }
        else
        {
            xyz_origin_index[index] = 0;
            xyz_shift_index[index] = shift_vector[dim_order[index]];
        }
    }
    if (!need_reorder)
    {
        volume_out = volume;
        return;
    }
    image::basic_image<value_type,2> new_volume(new_geometry);

    int y_index = xyz_origin_index[1];
        for (unsigned int y = 0,index = 0; y < old_geometry.height(); ++y)
        {
            int x_index = y_index + xyz_origin_index[0];
            for (unsigned int x = 0; x < old_geometry.width(); ++x,++index)
            {
                new_volume[x_index] = volume[index];
                x_index += xyz_shift_index[0];
            }
            y_index += xyz_shift_index[1];
        }

    new_volume.swap(volume_out);
}


/**
   dim_order[3] = {2,1,0} flip = {1,0,0}
   output (-z,y,x) <- input(x,y,z);
*/
template<typename value_type,typename dim_order_type,typename flip_type>
void reorder(const image::basic_image<value_type,3>& volume,
             image::basic_image<value_type,3>& volume_out,
             dim_order_type dim_order,flip_type flip)
{
    image::geometry<3> old_geometry(volume.geometry());
    image::geometry<3> new_geometry;
    bool need_reorder = false;
    // get the dimension mapping
    for (unsigned int index = 0; index < 3; ++index)
    {
        new_geometry[dim_order[index]] = old_geometry[index];
        if (dim_order[index] != index)
            need_reorder = true;
    }
    int shift_vector[3];
    int xyz_origin_index[3];
    int xyz_shift_index[3];

    shift_vector[0] = 1;
    shift_vector[1] = new_geometry.width();
    shift_vector[2] = new_geometry.plane_size();

    for (unsigned int index = 0; index < 3; ++index)
    {
        if (flip[index])
        {
            xyz_origin_index[index] = shift_vector[dim_order[index]]*(new_geometry[dim_order[index]]-1);
            xyz_shift_index[index] = -shift_vector[dim_order[index]];
            need_reorder = true;
        }
        else
        {
            xyz_origin_index[index] = 0;
            xyz_shift_index[index] = shift_vector[dim_order[index]];
        }
    }
    if (!need_reorder)
    {
        volume_out = volume;
        return;
    }
    image::basic_image<value_type,3> new_volume(new_geometry);
    int z_index = xyz_origin_index[2];
    for (unsigned int z = 0,index = 0; z < old_geometry.depth(); ++z)
    {
        int y_index = z_index + xyz_origin_index[1];
        for (unsigned int y = 0; y < old_geometry.height(); ++y)
        {
            int x_index = y_index + xyz_origin_index[0];
            for (unsigned int x = 0; x < old_geometry.width(); ++x,++index)
            {
                new_volume[x_index] = volume[index];
                x_index += xyz_shift_index[0];
            }
            y_index += xyz_shift_index[1];
        }
        z_index += xyz_shift_index[2];
    }
    new_volume.swap(volume_out);
}
//---------------------------------------------------------------------------
template<typename image_type,typename dim_order_type,typename flip_type>
void reorder(image_type& volume,dim_order_type dim_order,flip_type flip)
{
    image_type volume_out;
    reorder(volume,volume_out,dim_order,flip);
    volume.swap(volume_out);
}

//---------------------------------------------------------------------------
template<typename iterator_type>
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
template<typename iterator_type>
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
template<typename ImageType>
void flip_x(ImageType& image)
{
    flip_block(image.begin(),image.end(),image.width());
}
//---------------------------------------------------------------------------
template<typename ImageType>
void flip_y(ImageType& image)
{
    flip_block_line(image.begin(),image.end(),image.height() * image.width(),image.width());
}
//---------------------------------------------------------------------------
template<typename ImageType>
void flip_xy(ImageType& image)
{
    flip_block(image.begin(),image.end(),image.height() * image.width());
}
//---------------------------------------------------------------------------
template<typename ImageType>
void flip_z(ImageType& image)
{
    flip_block_line(image.begin(),image.end(),image.geometry().plane_size() * image.depth(),image.geometry().plane_size());
}
//---------------------------------------------------------------------------
template<typename ImageType>
void negate(ImageType& image,typename ImageType::value_type maximum)
{
    typename ImageType::iterator iter = image.begin();
    typename ImageType::iterator end = image.end();
    for (; iter != end; ++iter)
        *iter = maximum - *iter;
}
//---------------------------------------------------------------------------
template<typename ImageType>
void negate(ImageType& image)
{
    negate(image,*std::max_element(image.begin(),image.end()));
}

template<typename ImageType1,typename ImageType2,typename PixelType2>
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
template<typename PixelType1,typename PixelType2,typename LocationType>
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

template<typename PixelType1,typename PixelType2,typename LocationType,typename DetermineType>
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

template<typename PixelType1,typename OutImageType>
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
        for(pixel_index<2> index; index.valid(src.geometry()); index.next(src.geometry()))
            result[index.x()] += src[index.index()];
    }
}

template<typename iterator_type>
std::pair<typename std::iterator_traits<iterator_type>::value_type,typename std::iterator_traits<iterator_type>::value_type>
min_max_value(iterator_type iter,iterator_type end)
{
    if(iter == end)
        return std::make_pair(0,0);
    typename std::iterator_traits<iterator_type>::value_type min_value = *iter;
    typename std::iterator_traits<iterator_type>::value_type max_value = *iter;
    typename std::iterator_traits<iterator_type>::value_type value;
    for(++iter; iter != end; ++iter)
    {
        value = *iter;
        if(value > max_value)
            max_value = value;
        else if(value < min_value)
            min_value = value;
    }
    return std::make_pair(min_value,max_value);
}

template<typename ImageType>
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

template<typename ImageType>
void histogram(const ImageType& src,std::vector<unsigned int>& hist,
               typename ImageType::value_type min_value,
               typename ImageType::value_type max_value,unsigned int resolution_count = 256)
{
    if(min_value >= max_value)
        return;
    float range = max_value;
    range -= min_value;
    hist.clear();
    range = ((float)resolution_count-0.01)/range;
    hist.resize(resolution_count);
    typename ImageType::const_iterator iter = src.begin();
    typename ImageType::const_iterator end = src.end();
    for(; iter != end; ++iter)
    {
        float value = *iter;
        value -= min_value;
        value *= range;
        ++hist[std::floor(value)];
    }
}

template<typename type>
void change_endian(type& value)
{
    type data = value;
    unsigned char temp[sizeof(type)];
    unsigned char* pdata = ((unsigned char*)&data)+sizeof(type)-1;
    for (char i = 0; i < sizeof(type); ++i,--pdata)
        temp[i] = *pdata;
    value = *(type*)temp;
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

template<typename datatype>
void change_endian(datatype* data,unsigned int count)
{
    for (unsigned int index = 0; index < count; ++index)
        change_endian(data[index]);
}

}
#endif
