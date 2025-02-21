#ifndef INDEX_ALGORITHM_HPP
#define INDEX_ALGORITHM_HPP
#include <vector>
#include <deque>
#include <map>
#include <algorithm>
#include "../utility/pixel_index.hpp"
#include "../utility/basic_image.hpp"

namespace tipl
{
/**
    connected neighbors
    0 1 0
    1 x 1
    0 1 0
*/
template<typename T>
__INLINE__ void for_each_connected_neighbors(const pixel_index<2>& index,const shape<2>& geo,T&& fun)
{
    if (index.x() >= 1)
        fun(pixel_index<2>(index.x()-1,index.y(),index.index()-1,geo));

    if (index.x()+1 < geo.width())
        fun(pixel_index<2>(index.x()+1,index.y(),index.index()+1,geo));

    if (index.y() >= 1)
        fun(pixel_index<2>(index.x(),index.y()-1,index.index()-int(geo.width()),geo));

    if (index.y()+1 < geo.height())
        fun(pixel_index<2>(index.x(),index.y()+1,index.index()+int(geo.width()),geo));
}

/**
    connected neighbors
    0 1 0
    1 x 1
    0 1 0
*/
template<typename T>
__INLINE__ void for_each_connected_neighbors(const pixel_index<3>& index,const shape<3>& geo,T&& fun)
{
    if (index.x() >= 1)
        fun(pixel_index<3>(index.x()-1,index.y(),index.z(),index.index()-1,geo));

    if (index.x()+1 < geo.width())
        fun(pixel_index<3>(index.x()+1,index.y(),index.z(),index.index()+1,geo));

    if (index.y() >= 1)
        fun(pixel_index<3>(index.x(),index.y()-1,index.z(),index.index()-geo.width(),geo));

    if (index.y()+1 < geo.height())
        fun(pixel_index<3>(index.x(),index.y()+1,index.z(),index.index()+geo.width(),geo));

    if (index.z() >= 1)
        fun(pixel_index<3>(index.x(),index.y(),index.z()-1,index.index()-geo.plane_size(),geo));

    if (index.z()+1 < geo.depth())
        fun(pixel_index<3>(index.x(),index.y(),index.z()+1,index.index()+geo.plane_size(),geo));
}

/**
    1 1 1
    1 x 1
    1 1 1
*/
template<typename T>
__INLINE__ void for_each_neighbors(const pixel_index<2>& index,const shape<2>& geo,T&& fun)
{
    bool has_left = index.x() >= 1;
    bool has_right = index.x()+1 < geo.width();
    int x_left(0),x_right(0);
    if(has_left)
        x_left = index.x()-1;
    if(has_right)
        x_right = index.x()+1;
    if (index.y() >= 1)
    {
        int y_top = index.y()-1;
        int base_index = index.index()-geo.width();
        if (has_left)
            fun(pixel_index<2>(x_left,y_top,base_index-1,geo));

        fun(pixel_index<2>(index.x()  ,y_top,base_index,geo));
        if (has_right)
            fun(pixel_index<2>(x_right,y_top,base_index+1,geo));
    }
    {
        if (has_left)
            fun(pixel_index<2>(x_left,index.y(),index.index()-1,geo));

        //fun(pixel_index<2>(index.x()  ,index.y(),index.index()));
        if (has_right)
            fun(pixel_index<2>(x_right,index.y(),index.index()+1,geo));
    }
    if (index.y()+1 < geo.height())
    {
        int y_bottom = index.y()+1;
        int base_index = index.index()+geo.width();
        if (has_left)
            fun(pixel_index<2>(x_left,y_bottom,base_index-1,geo));

        fun(pixel_index<2>(index.x()  ,y_bottom,base_index,geo));
        if (has_right)
            fun(pixel_index<2>(x_right,y_bottom,base_index+1,geo));
    }

}

template<typename T>
__INLINE__ void for_each_neighbors(const pixel_index<3>& index,const shape<3>& geo,T&& fun)
{
    auto z_offset = geo.plane_size();
    auto y_offset = geo.width();
    bool has_left = index.x() >= 1;
    bool has_right = index.x()+1 < geo.width();
    bool has_top = index.y() >= 1;
    bool has_bottom = index.y()+1 < geo.height();
    int x_left(0),x_right(0),y_top(0),y_bottom(0);
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
        size_t base_index = index.index()-z_offset;
        if (has_top)
        {
            size_t base_index2 = base_index - y_offset;
            if (has_left)
                fun(pixel_index<3>(x_left,y_top,z,base_index2-1,geo));

            fun(pixel_index<3>(index.x()  ,y_top,z,base_index2,geo));

            if (has_right)
                fun(pixel_index<3>(x_right,y_top,z,base_index2+1,geo));
        }
        {
            if (has_left)
                fun(pixel_index<3>(x_left,index.y(),z,base_index-1,geo));

            fun(pixel_index<3>(index.x()  ,index.y(),z,base_index,geo));

            if (has_right)
                fun(pixel_index<3>(x_right,index.y(),z,base_index+1,geo));
        }
        if (has_bottom)
        {
            size_t base_index2 = base_index + y_offset;
            if (has_left)
                fun(pixel_index<3>(x_left,y_bottom,z,base_index2-1,geo));

            fun(pixel_index<3>(index.x()  ,y_bottom,z,base_index2,geo));

            if (has_right)
                fun(pixel_index<3>(x_right,y_bottom,z,base_index2+1,geo));
        }
    }

    {
        if (has_top)
        {
            size_t base_index2 = index.index() - y_offset;
            if (has_left)
                fun(pixel_index<3>(x_left,y_top,index.z(),base_index2-1,geo));

            fun(pixel_index<3>(index.x()  ,y_top,index.z(),base_index2,geo));

            if (has_right)
                fun(pixel_index<3>(x_right,y_top,index.z(),base_index2+1,geo));
        }
        {
            if (has_left)
                fun(pixel_index<3>(x_left,index.y(),index.z(),index.index()-1,geo));

            //fun(pixel_index<3>(index.x()  ,index.y(),index.z(),index.index()  ));

            if (has_right)
                fun(pixel_index<3>(x_right,index.y(),index.z(),index.index()+1,geo));
        }
        if (has_bottom)
        {
            size_t base_index2 = index.index() + y_offset;
            if (has_left)
                fun(pixel_index<3>(x_left,y_bottom,index.z(),base_index2-1,geo));

            fun(pixel_index<3>(index.x()  ,y_bottom,index.z(),base_index2,geo));

            if (has_right)
                fun(pixel_index<3>(x_right,y_bottom,index.z(),base_index2+1,geo));
        }

    }
    if (index.z()+1 < geo.depth())
    {
        int z = index.z()+1;
        size_t base_index = index.index()+z_offset;
        if (has_top)
        {
            size_t base_index2 = base_index - y_offset;
            if (has_left)
                fun(pixel_index<3>(x_left,y_top,z,base_index2-1,geo));

            fun(pixel_index<3>(index.x()  ,y_top,z,base_index2,geo));

            if (has_right)
                fun(pixel_index<3>(x_right,y_top,z,base_index2+1,geo));
        }
        {
            if (has_left)
                fun(pixel_index<3>(x_left,index.y(),z,base_index-1,geo));

            fun(pixel_index<3>(index.x()  ,index.y(),z,base_index,geo));

            if (has_right)
                fun(pixel_index<3>(x_right,index.y(),z,base_index+1,geo));
        }
        if (has_bottom)
        {
            size_t base_index2 = base_index + y_offset;
            if (has_left)
                fun(pixel_index<3>(x_left,y_bottom,z,base_index2-1,geo));

            fun(pixel_index<3>(index.x()  ,y_bottom,z,base_index2,geo));

            if (has_right)
                fun(pixel_index<3>(x_right,y_bottom,z,base_index2+1,geo));
        }
    }

}

template<typename T>
__INLINE__ void for_each_neighbors(const pixel_index<2>& index,const shape<2>& geo,int range,T&& fun)
{
    int fx = (index.x() > range) ? index.x() - range:0;
    int fy = (index.y() > range) ? index.y() - range:0;
    int tx = std::min<int>(index.x() + range,int(geo.width())-1);
    int ty = std::min<int>(index.y() + range,int(geo.height())-1);
    int y_index = fy*int(geo.width())+fx;
    int radius2 = range*range;
    for (int y = fy;y <= ty;++y,y_index += geo.width())
    {
        int x_index = y_index;
        int dy = int(index.y())-y;
        int dy2 = dy*dy;
        for (int x = fx;x <= tx;++x,++x_index)
        {
            int dx = int(index.x())-x;
            int dx2 = dx*dx;
            if(dx2+dy2 <= radius2)
                fun(pixel_index<2>(x,y,x_index,geo));
        }
    }
}

template<typename T>
__INLINE__ void for_each_neighbors(const pixel_index<3>& index,const shape<3>& geo,int range,T&& fun)
{
    size_t wh = geo.plane_size();
    size_t fx = (index.x() > range) ? index.x() - range:0;
    size_t fy = (index.y() > range) ? index.y() - range:0;
    size_t fz = (index.z() > range) ? index.z() - range:0;
    size_t tx = std::min<int>(index.x() + range,int(geo.width())-1);
    size_t ty = std::min<int>(index.y() + range,int(geo.height())-1);
    size_t tz = std::min<int>(index.z() + range,int(geo.depth())-1);
    size_t z_index = size_t((fz*size_t(geo.height())+fy)*size_t(geo.width())+fx);
    size_t radius2 = range*range;
    for (size_t z = fz;z <= tz;++z,z_index += wh)
    {
        size_t y_index = z_index;
        size_t dz = size_t(index.z())-z;
        size_t dz2 = dz*dz;
        for (size_t y = fy;y <= ty;++y,y_index += geo.width())
        {
            size_t x_index = y_index;
            size_t dy = int(index.y())-y;
            size_t dyz2 = dy*dy+dz2;
            for (size_t x = fx;x <= tx;++x,++x_index)
            {
                size_t dx = size_t(index.x())-x;
                if(dx*dx+dyz2 <= radius2)
                    fun(pixel_index<3>(x,y,z,x_index,geo));
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
    std::vector<int64_t> index_shift;
public:
    neighbor_index_shift(const shape<2>& geo)
    {
        int64_t w = int64_t(geo.width());
            for (int64_t y = -1;y <= 1; ++y)
            {
                int64_t yw = y*w;
                for (int64_t x = -1;x <= 1; ++x)
                    index_shift.push_back(x + yw);
            }
    }
    neighbor_index_shift(const shape<2>& geo,int radius)
    {
        int64_t w = int64_t(geo.width());
            for (int64_t y = -radius;y <= radius; ++y)
            {
                int64_t yw = y*w;
                for (int64_t x = -radius;x <= radius; ++x)
                    if(x*x + y*y <= radius*radius)
                        index_shift.push_back(x + yw);
            }
    }
};


template<>
class neighbor_index_shift<3>
{
public:
    std::vector<int64_t> index_shift;
public:
    neighbor_index_shift(const shape<3>& geo)
    {
        int64_t wh = int64_t(geo.plane_size());
        int64_t w = int64_t(geo.width());
        for (int64_t z = -1;z <= 1; ++z)
        {
            int64_t zwh = z*wh;
            for (int64_t y = -1;y <= 1; ++y)
            {
                int64_t yw = y*w;
                for (int64_t x = -1;x <= 1; ++x)
                    index_shift.push_back(x + yw + zwh);
            }
        }
    }
    neighbor_index_shift(const shape<3>& geo,int radius)
    {
        int64_t wh = int64_t(geo.plane_size());
        int64_t w = int64_t(geo.width());
        for (int64_t z = -radius;z <= radius; ++z)
        {
            int64_t zwh = z*wh;
            for (int64_t y = -radius;y <= radius; ++y)
            {
                int64_t yw = y*w;
                for (int64_t x = -radius;x <= radius; ++x)
                    if(x*x + y*y + z*z <= radius*radius)
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
    std::vector<int64_t> index_shift;
public:
    neighbor_index_shift_narrow(const shape<2>& geo)
    {
        index_shift.push_back(-int64_t(geo.width()));
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
    std::vector<int64_t> index_shift;
public:
    neighbor_index_shift_narrow(const shape<3>& geo)
    {
        index_shift.push_back(-int64_t(geo.plane_size()));
        index_shift.push_back(-int64_t(geo.width()));
        index_shift.push_back(-1);
        index_shift.push_back(0);
        index_shift.push_back(1);
        index_shift.push_back(geo.width());
        index_shift.push_back(geo.plane_size());
    }
};



}
#endif
