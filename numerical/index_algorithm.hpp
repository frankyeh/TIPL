#ifndef INDEX_ALGORITHM_HPP
#define INDEX_ALGORITHM_HPP
#include <vector>
#include <deque>
#include <map>
#include <algorithm>
#include "image/utility/pixel_index.hpp"
#include "image/utility/basic_image.hpp"

namespace image
{
/**
    connected neighbors
    0 1 0
    1 x 1
    0 1 0
*/
inline void get_connected_neighbors(const pixel_index<2>& index,const geometry<2>& geo,
                                        std::vector<pixel_index<2> >& iterations)
{
    iterations.clear();
    iterations.reserve(4);
    if (index.x() >= 1)
        iterations.push_back(pixel_index<2>(index.x()-1,index.y(),index.index()-1,geo));

    if (index.x()+1 < geo.width())
        iterations.push_back(pixel_index<2>(index.x()+1,index.y(),index.index()+1,geo));

    if (index.y() >= 1)
        iterations.push_back(pixel_index<2>(index.x(),index.y()-1,index.index()-geo.width(),geo));

    if (index.y()+1 < geo.height())
        iterations.push_back(pixel_index<2>(index.x(),index.y()+1,index.index()+geo.width(),geo));
}

/**
    connected neighbors
    0 1 0
    1 x 1
    0 1 0
*/
inline void get_connected_neighbors(const pixel_index<3>& index,const geometry<3>& geo,
                                        std::vector<pixel_index<3> >& iterations)
{
    iterations.clear();
    iterations.reserve(6);

    if (index.x() >= 1)
        iterations.push_back(pixel_index<3>(index.x()-1,index.y(),index.z(),index.index()-1,geo));

    if (index.x()+1 < geo.width())
        iterations.push_back(pixel_index<3>(index.x()+1,index.y(),index.z(),index.index()+1,geo));

    if (index.y() >= 1)
        iterations.push_back(pixel_index<3>(index.x(),index.y()-1,index.z(),index.index()-geo.width(),geo));

    if (index.y()+1 < geo.height())
        iterations.push_back(pixel_index<3>(index.x(),index.y()+1,index.z(),index.index()+geo.width(),geo));

    if (index.z() >= 1)
        iterations.push_back(pixel_index<3>(index.x(),index.y(),index.z()-1,index.index()-geo.plane_size(),geo));

    if (index.z()+1 < geo.depth())
        iterations.push_back(pixel_index<3>(index.x(),index.y(),index.z()+1,index.index()+geo.plane_size(),geo));
}

/**
    1 1 1
    1 x 1
    1 1 1
*/
inline void get_neighbors(const pixel_index<2>& index,const geometry<2>& geo,
                                        std::vector<pixel_index<2> >& iterations)
{
    iterations.clear();
    iterations.reserve(8);
    bool has_left = index.x() >= 1;
    bool has_right = index.x()+1 < geo.width();
    int x_left,x_right;
    if(has_left)
        x_left = index.x()-1;
    if(has_right)
        x_right = index.x()+1;
    if (index.y() >= 1)
    {
        int y_top = index.y()-1;
        int base_index = index.index()-geo.width();
        if (has_left)
            iterations.push_back(pixel_index<2>(x_left,y_top,base_index-1,geo));

        iterations.push_back(pixel_index<2>(index.x()  ,y_top,base_index,geo));
        if (has_right)
            iterations.push_back(pixel_index<2>(x_right,y_top,base_index+1,geo));
    }
    {
        if (has_left)
            iterations.push_back(pixel_index<2>(x_left,index.y(),index.index()-1,geo));

        //iterations.push_back(pixel_index<2>(index.x()  ,index.y(),index.index()));
        if (has_right)
            iterations.push_back(pixel_index<2>(x_right,index.y(),index.index()+1,geo));
    }
    if (index.y()+1 < geo.height())
    {
        int y_bottom = index.y()+1;
        int base_index = index.index()+geo.width();
        if (has_left)
            iterations.push_back(pixel_index<2>(x_left,y_bottom,base_index-1,geo));

        iterations.push_back(pixel_index<2>(index.x()  ,y_bottom,base_index,geo));
        if (has_right)
            iterations.push_back(pixel_index<2>(x_right,y_bottom,base_index+1,geo));
    }

}


inline void get_neighbors(const pixel_index<3>& index,const geometry<3>& geo,
                                        std::vector<pixel_index<3> >& iterations)
{
    iterations.clear();
    iterations.reserve(26);
    int z_offset = geo.plane_size();
    int y_offset = geo.width();
    bool has_left = index.x() >= 1;
    bool has_right = index.x()+1 < geo.width();
    bool has_top = index.y() >= 1;
    bool has_bottom = index.y()+1 < geo.height();
    int x_left,x_right,y_top,y_bottom;
    if(has_left)
        x_left = index.x()-1;
    if(has_right)
        x_right = index.x()+1;
    if(has_top)
        y_top = index.y()-1;
    if(has_bottom)
        y_bottom = index.y()+1;
    if (index.z() >= 1)
    {
        int z =  index.z()-1;
        int base_index = index.index()-z_offset;
        if (has_top)
        {
            int base_index2 = base_index - y_offset;
            if (has_left)
                iterations.push_back(pixel_index<3>(x_left,y_top,z,base_index2-1,geo));

            iterations.push_back(pixel_index<3>(index.x()  ,y_top,z,base_index2,geo));

            if (has_right)
                iterations.push_back(pixel_index<3>(x_right,y_top,z,base_index2+1,geo));
        }
        {
            if (has_left)
                iterations.push_back(pixel_index<3>(x_left,index.y(),z,base_index-1,geo));

            iterations.push_back(pixel_index<3>(index.x()  ,index.y(),z,base_index,geo));

            if (has_right)
                iterations.push_back(pixel_index<3>(x_right,index.y(),z,base_index+1,geo));
        }
        if (has_bottom)
        {
            int base_index2 = base_index + y_offset;
            if (has_left)
                iterations.push_back(pixel_index<3>(x_left,y_bottom,z,base_index2-1,geo));

            iterations.push_back(pixel_index<3>(index.x()  ,y_bottom,z,base_index2,geo));

            if (has_right)
                iterations.push_back(pixel_index<3>(x_right,y_bottom,z,base_index2+1,geo));
        }
    }

    {
        if (has_top)
        {
            int base_index2 = index.index() - y_offset;
            if (has_left)
                iterations.push_back(pixel_index<3>(x_left,y_top,index.z(),base_index2-1,geo));

            iterations.push_back(pixel_index<3>(index.x()  ,y_top,index.z(),base_index2,geo));

            if (has_right)
                iterations.push_back(pixel_index<3>(x_right,y_top,index.z(),base_index2+1,geo));
        }
        {
            if (has_left)
                iterations.push_back(pixel_index<3>(x_left,index.y(),index.z(),index.index()-1,geo));

            //iterations.push_back(pixel_index<3>(index.x()  ,index.y(),index.z(),index.index()  ));

            if (has_right)
                iterations.push_back(pixel_index<3>(x_right,index.y(),index.z(),index.index()+1,geo));
        }
        if (has_bottom)
        {
            int base_index2 = index.index() + y_offset;
            if (has_left)
                iterations.push_back(pixel_index<3>(x_left,y_bottom,index.z(),base_index2-1,geo));

            iterations.push_back(pixel_index<3>(index.x()  ,y_bottom,index.z(),base_index2,geo));

            if (has_right)
                iterations.push_back(pixel_index<3>(x_right,y_bottom,index.z(),base_index2+1,geo));
        }

    }
    if (index.z()+1 < geo.depth())
    {
        int z = index.z()+1;
        int base_index = index.index()+z_offset;
        if (has_top)
        {
            int base_index2 = base_index - y_offset;
            if (has_left)
                iterations.push_back(pixel_index<3>(x_left,y_top,z,base_index2-1,geo));

            iterations.push_back(pixel_index<3>(index.x()  ,y_top,z,base_index2,geo));

            if (has_right)
                iterations.push_back(pixel_index<3>(x_right,y_top,z,base_index2+1,geo));
        }
        {
            if (has_left)
                iterations.push_back(pixel_index<3>(x_left,index.y(),z,base_index-1,geo));

            iterations.push_back(pixel_index<3>(index.x()  ,index.y(),z,base_index,geo));

            if (has_right)
                iterations.push_back(pixel_index<3>(x_right,index.y(),z,base_index+1,geo));
        }
        if (has_bottom)
        {
            int base_index2 = base_index + y_offset;
            if (has_left)
                iterations.push_back(pixel_index<3>(x_left,y_bottom,z,base_index2-1,geo));

            iterations.push_back(pixel_index<3>(index.x()  ,y_bottom,z,base_index2,geo));

            if (has_right)
                iterations.push_back(pixel_index<3>(x_right,y_bottom,z,base_index2+1,geo));
        }
    }

}


template<int Dim>
inline void get_neighbors(const pixel_index<Dim>& index,const geometry<Dim>& geo,int range,std::vector<pixel_index<Dim> >& iterations)
{
    iterations.clear();
    iterations.reserve(9);
    throw;
}

inline void get_neighbors(const pixel_index<2>& index,const geometry<2>& geo,int range,std::vector<pixel_index<2> >& iterations)
{
    iterations.clear();
    iterations.reserve(9);
    int fx = (index.x() > range) ? index.x() - range:0;
    int fy = (index.y() > range) ? index.y() - range:0;
    int tx = std::min<int>(index.x() + range,geo.width()-1);
    int ty = std::min<int>(index.y() + range,geo.height()-1);
    int y_index = fy*geo.width()+fx;
    int radius2 = range*range;
    for (int y = fy;y <= ty;++y,y_index += geo.width())
    {
        int x_index = y_index;
        int dy = (int)index.y()-y;
        int dy2 = dy*dy;
        for (int x = fx;x <= tx;++x,++x_index)
        {
            int dx = (int)index.x()-x;
            int dx2 = dx*dx;
            if(dx2+dy2 <= radius2)
                iterations.push_back(pixel_index<2>(x,y,x_index,geo));
        }
    }
}

inline void get_neighbors(const pixel_index<3>& index,const geometry<3>& geo,int range,std::vector<pixel_index<3> >& iterations)
{
    iterations.clear();
    iterations.reserve(26);
    int wh = geo.plane_size();
    int fx = (index.x() > range) ? index.x() - range:0;
    int fy = (index.y() > range) ? index.y() - range:0;
    int fz = (index.z() > range) ? index.z() - range:0;
    int tx = std::min<int>(index.x() + range,geo.width()-1);
    int ty = std::min<int>(index.y() + range,geo.height()-1);
    int tz = std::min<int>(index.z() + range,geo.depth()-1);
    int z_index = (fz*geo.height()+fy)*geo.width()+fx;
    int radius2 = range*range;
    for (int z = fz;z <= tz;++z,z_index += wh)
    {
        int y_index = z_index;
        int dz = (int)index.z()-z;
        int dz2 = dz*dz;
        for (int y = fy;y <= ty;++y,y_index += geo.width())
        {
            int x_index = y_index;
            int dy = (int)index.y()-y;
            int dyz2 = dy*dy+dz2;
            for (int x = fx;x <= tx;++x,++x_index)
            {
                int dx = (int)index.x()-x;
                if(dx*dx+dyz2 <= radius2)
                    iterations.push_back(pixel_index<3>(x,y,z,x_index,geo));
            }
        }
    }
}


template<int dim>
class neighbor_index_shift;

template<>
class neighbor_index_shift<2>
{
public:
    std::vector<int> index_shift;
public:
    neighbor_index_shift(const geometry<2>& geo)
    {
        int w = geo.width();
            for (int y = -1;y <= 1; ++y)
            {
                int yw = y*w;
                for (int x = -1;x <= 1; ++x)
                    index_shift.push_back(x + yw);
            }
    }
    neighbor_index_shift(const geometry<2>& geo,int radius)
    {
        int w = geo.width();
            for (int y = -radius;y <= radius; ++y)
            {
                int yw = y*w;
                for (int x = -radius;x <= radius; ++x)
                    if(x*x + y*y < radius*radius)
                        index_shift.push_back(x + yw);
            }
    }
};


template<>
class neighbor_index_shift<3>
{
public:
    std::vector<int> index_shift;
public:
    neighbor_index_shift(const geometry<3>& geo)
    {
        int wh = geo.plane_size();
        int w = geo.width();
        for (int z = -1;z <= 1; ++z)
        {
            int zwh = z*wh;
            for (int y = -1;y <= 1; ++y)
            {
                int yw = y*w;
                for (int x = -1;x <= 1; ++x)
                    index_shift.push_back(x + yw + zwh);
            }
        }
    }
    neighbor_index_shift(const geometry<3>& geo,int radius)
    {
        int wh = geo.plane_size();
        int w = geo.width();
        for (int z = -radius;z <= radius; ++z)
        {
            int zwh = z*wh;
            for (int y = -radius;y <= radius; ++y)
            {
                int yw = y*w;
                for (int x = -radius;x <= radius; ++x)
                    if(x*x + y*y + z*z < radius*radius)
                        index_shift.push_back(x + yw + zwh);
            }
        }
    }
};


template<int dim>
class neighbor_index_shift_narrow;

template<>
class neighbor_index_shift_narrow<2>
{
public:
    std::vector<int> index_shift;
public:
    neighbor_index_shift_narrow(const geometry<2>& geo)
    {
        index_shift.push_back(-geo.width());
        index_shift.push_back(-1);
        index_shift.push_back(0);
        index_shift.push_back(1);
        index_shift.push_back(geo.width());
    }
};


template<>
class neighbor_index_shift_narrow<3>
{
public:
    std::vector<int> index_shift;
public:
    neighbor_index_shift_narrow(const geometry<3>& geo)
    {
        index_shift.push_back(-geo.plane_size());
        index_shift.push_back(-geo.width());
        index_shift.push_back(-1);
        index_shift.push_back(0);
        index_shift.push_back(1);
        index_shift.push_back(geo.width());
        index_shift.push_back(geo.plane_size());
    }
};



}
#endif
