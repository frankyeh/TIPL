#include <vector>
#include <limits>
#include <memory>

namespace tipl
{

namespace segmentation
{

namespace imp
{

// The subrutine for fast marching
template<class pass_time_type>
float fast_marching_estimateT(const pass_time_type& T,float g,const geometry<2>& geo,const pixel_index<2>& index)
{
    float Tx,Ty;
    {
        float Tx1 = (index.x()) ? T[index.index()-1] : std::numeric_limits<float>::max();
        float Tx2 = (index.x() + 1 < geo.height()) ? T[index.index()+1] : std::numeric_limits<float>::max();
        Tx = std::min(Tx1,Tx2);
    }
    {
        float Ty1 = (index.y()) ? T[index.index()-geo.width()] : std::numeric_limits<float>::max();
        float Ty2 = (index.y() + 1 < geo.height()) ? T[index.index()+geo.width()] : std::numeric_limits<float>::max();
        Ty = std::min(Ty1,Ty2);
    }
    // sort Tx,Ty,Tz
    if(Tx > Ty)
        std::swap(Tx,Ty);
    if(Ty == std::numeric_limits<float>::max())
        return Tx+g;
    float Td;
    if((Td = Ty - Tx) > g)
        return Tx+g;
    // T = [Tx+Ty + (2g^2-(Tx-Tx)^2)^1/2]/2
    return 0.5*(Tx+Ty+sqrt(2.0*g*g-Td*Td));
}

// The subrutine for fast marching
template<class pass_time_type>
float fast_marching_estimateT(const pass_time_type& T,float g,const geometry<3>& geo,const pixel_index<3>& index)
{
    float Tx,Ty,Tz;
    {
        float Tx1 = (index.x()) ? T[index.index()-1] : std::numeric_limits<float>::max();
        float Tx2 = (index.x() + 1 < geo.height()) ? T[index.index()+1] : std::numeric_limits<float>::max();
        Tx = std::min(Tx1,Tx2);
    }
    {
        float Ty1 = (index.y()) ? T[index.index()-geo.width()] : std::numeric_limits<float>::max();
        float Ty2 = (index.y() + 1 < geo.height()) ? T[index.index()+geo.width()] : std::numeric_limits<float>::max();
        Ty = std::min(Ty1,Ty2);
    }
    {
        float Tz1 = (index.z()) ? T[index.index()-geo.plane_size()] : std::numeric_limits<float>::max();
        float Tz2 = (index.z() + 1 < geo.depth()) ? T[index.index()+geo.plane_size()] : std::numeric_limits<float>::max();
        Tz = std::min(Tz1,Tz2);
    }
    // sort Tx,Ty,Tz
    if(Tx > Ty)
        std::swap(Tx,Ty);
    if(Tx > Tz)
        std::swap(Tx,Tz);
    if(Ty > Tz)
        std::swap(Ty,Tz);
    if(Ty == std::numeric_limits<float>::max())
        return Tx+g;
    if(Tz == std::numeric_limits<float>::max())
    {
        float Td;
        if((Td = Ty - Tx) > g)
            return Tx+g;
        return 0.5*(Tx+Ty+sqrt(2.0*g*g-Td*Td));
    }
    float Tsum = Tx+Ty+Tz;
    float b2_4ac = Tsum*Tsum-3*(Tx*Tx+Ty*Ty+Tz*Tz-g*g);
    if(b2_4ac <= 0)
        return Tx+g;
    b2_4ac = std::sqrt(b2_4ac);
    return (Tsum + b2_4ac)/3.0;
}

}

/**
   Fast marching method
   Referece: J.A. Sethian, "A fast marching level set method for monotonically advancing fronts", PNAS, 93, pp.1591-1595, 1996.
*/

template<class ImageType,class TimeType,class IndexType>
void fast_marching(const ImageType& gradient_image,TimeType& pass_time,IndexType seed)
{
    typedef std::pair<float,pixel_index<ImageType::dimension> > narrow_band_point;

    std::vector<narrow_band_point*> narrow_band;
    std::vector<pixel_index<ImageType::dimension> > neighbor_points;
    narrow_band.push_back(new narrow_band_point(0.001,seed));

    float infinity_time = std::numeric_limits<float>::max();
    pass_time.resize(gradient_image.geometry());
    std::fill(pass_time.begin(),pass_time.end(),infinity_time);
    pass_time[seed.index()] = 0;

    while(!narrow_band.empty())
    {
        std::shared_ptr<narrow_band_point> active_point(narrow_band.front());
        std::pop_heap(narrow_band.begin(),narrow_band.end(),[&](const narrow_band_point* p1,const narrow_band_point* p2)
        {
            return p1->first > p2->first;
        });
        narrow_band.pop_back();
        get_connected_neighbors(active_point->second,gradient_image.geometry(),neighbor_points);
        for(size_t index = 0; index < neighbor_points.size(); ++index)
        {
            size_t cur_index = neighbor_points[index].index();
            if(pass_time[cur_index] != infinity_time)
                continue;
            float cur_T = imp::fast_marching_estimateT(pass_time,gradient_image[cur_index],gradient_image.geometry(),neighbor_points[index]);
            pass_time[cur_index] = cur_T;
            narrow_band.push_back(new narrow_band_point(cur_T,neighbor_points[index]));
            std::push_heap(narrow_band.begin(),narrow_band.end(),[&](const narrow_band_point* p1,const narrow_band_point* p2)
            {
                return p1->first > p2->first;
            });
        }
    }
}




}
}
