#ifndef DIF_HPP
#define DIF_HPP
#include "../mt.hpp"
#include "../utility/pixel_index.hpp"
#include "interpolation.hpp"
namespace tipl
{

template<typename MappingType>
void make_identity(MappingType& s)
{
    tipl::par_for(tipl::begin_index(s.shape()),tipl::end_index(s.shape()),
                [&](const tipl::pixel_index<MappingType::dimension>& index)
    {
        s[index.index()] = index;
    });
}
//---------------------------------------------------------------------------
template<typename DisType>
void displacement_to_mapping(DisType& s)
{
    tipl::par_for(tipl::begin_index(s.shape()),tipl::end_index(s.shape()),
                [&](const tipl::pixel_index<DisType::dimension>& index)
    {
        s[index.index()] += index;
    });
}
//---------------------------------------------------------------------------
template<typename MappingType>
void mapping_to_displacement(MappingType& s)
{
    tipl::par_for(tipl::begin_index(s.shape()),tipl::end_index(s.shape()),
                [&](const tipl::pixel_index<MappingType::dimension>& index)
    {
        s[index.index()] -= index;
    });
}
//---------------------------------------------------------------------------
template<typename T,typename U>
void displacement_to_mapping(const T& dis,U& mapping)
{
    mapping = dis;
    tipl::par_for(tipl::begin_index(mapping.shape()),tipl::end_index(mapping.shape()),
                            [&](const tipl::pixel_index<T::dimension>& index)
    {
        typename U::value_type vtor(index);
        mapping[index.index()] += vtor;
    });
}
//---------------------------------------------------------------------------
template<typename DisType,typename MappingType,typename transform_type>
void displacement_to_mapping(const DisType& dis,MappingType& mapping,const transform_type& T)
{
    mapping = dis;
    tipl::par_for(tipl::begin_index(mapping.shape()),tipl::end_index(mapping.shape()),
                            [&](const tipl::pixel_index<MappingType::dimension>& index)
    {
        typename MappingType::value_type vtor(index);
        vtor += mapping[index.index()];
        T(vtor,mapping[index.index()]);
    });
}

//---------------------------------------------------------------------------
template<tipl::interpolation Type = linear,typename DisType,typename MappingType,typename transform_type>
void inv_displacement_to_mapping(const DisType& inv_dis,MappingType& inv_mapping,const transform_type& T)
{
    auto iT = T;
    iT.inverse();
    tipl::par_for(tipl::begin_index(inv_mapping.shape()),tipl::end_index(inv_mapping.shape()),
        [&](const tipl::pixel_index<3>& index)
    {
        tipl::vector<3> p;
        iT(index,p);
        p += tipl::estimate<Type>(inv_dis,p);
        inv_mapping[index.index()] = p;
    });
}

//---------------------------------------------------------------------------
template<tipl::interpolation Type = linear,typename ImageType,typename MappingType,typename OutImageType>
void compose_mapping(const ImageType& src,const MappingType& mapping,OutImageType& dest)
{
    dest.clear();
    dest.resize(mapping.shape());
    tipl::par_for(dest.size(),[&](unsigned int index)
    {
        estimate<Type>(src,mapping[index],dest[index]);
    });
}
//---------------------------------------------------------------------------
template<tipl::interpolation Type = linear,typename T,typename U,typename V>
void compose_displacement(const T& from,const U& dis,V& to)
{
    to.clear();
    to.resize(from.shape());
    tipl::par_for(tipl::begin_index(from.shape()),tipl::end_index(from.shape()),
        [&](const tipl::pixel_index<U::dimension>& index)
    {
        if(dis[index.index()] == typename U::value_type())
            to[index.index()] = from[index.index()];
        else
        {
            typename U::value_type v(index);
            v += dis[index.index()];
            tipl::estimate<Type>(from,v,to[index.index()]);
        }
    });
}
//---------------------------------------------------------------------------
template<tipl::interpolation Type = linear,typename ImageType,typename ComposeImageType,typename OutImageType,typename transform_type>
void compose_displacement_with_affine(const ImageType& src,OutImageType& dest,
                          const transform_type& transform,
                          const ComposeImageType& displace)
{
    dest.clear();
    dest.resize(displace.shape());
    tipl::par_for(tipl::begin_index(displace.shape()),tipl::end_index(displace.shape()),
        [&](const tipl::pixel_index<OutImageType::dimension>& index)
    {
        typename ComposeImageType::value_type vtor(index);
        vtor += displace[index.index()];
        tipl::vector<OutImageType::dimension> pos;
        transform(vtor,pos);
        tipl::estimate<Type>(src,pos,dest[index.index()]);
    });
}

//---------------------------------------------------------------------------
template<typename ComposeImageType>
void invert_displacement_imp(const ComposeImageType& v0,ComposeImageType& v1)
{
    ComposeImageType vv;
    for(uint8_t i = 0;i < 4;++i)
    {
        tipl::compose_displacement(v0,v1,vv);
        for(size_t index = 0;index < v1.size();++index)
            v1[index] = -vv[index];
    }
}
template<typename ComposeImageType>
void invert_displacement(const ComposeImageType& v0,ComposeImageType& v1)
{
    v1.resize(v0.shape());
    for(size_t i = 1;i <= 7;++i)
    {
        float ratio = float(i)/8.0f;
        ComposeImageType v0_reduced(v0.shape());
        for(size_t j = 0;j < v0.size();++j)
            v0_reduced[j] = v0[j]*ratio;
        invert_displacement_imp(v0_reduced,v1);
    }
    invert_displacement_imp(v0,v1);
}
//---------------------------------------------------------------------------
template<typename ComposeImageType>
void invert_displacement(ComposeImageType& v)
{
    ComposeImageType v0;
    invert_displacement(v,v0);
    v.swap(v0);
}

//---------------------------------------------------------------------------
template<typename ComposeImageType>
void invert_mapping(const ComposeImageType& s0,ComposeImageType& s1,uint8_t iterations)
{
    ComposeImageType v0(s0);
    mapping_to_displacement(v0);
    invert_displacement(v0,s1,iterations);
    displacement_to_mapping(s1);
}
//---------------------------------------------------------------------------
template<typename T,typename U,typename V>
__INLINE__ void accumulate_displacement_imp(T& dis,U& new_dis,V& mapping,
                                       const tipl::pixel_index<3>& index)
{
    tipl::vector<3> d = new_dis[index.index()];
    if(d != tipl::vector<3>())
    {
        if(tipl::estimate<tipl::interpolation::linear>(mapping,tipl::vector<3>(index)+d,dis[index.index()]))
            dis[index.index()] -= index;
    }
}
//---------------------------------------------------------------------------
template<typename T>
void accumulate_displacement(T& dis,const T& new_dis)
{
    T mapping;
    displacement_to_mapping(dis,mapping);
    tipl::par_for(tipl::begin_index(dis.shape()),tipl::end_index(dis.shape()),
        [&](const tipl::pixel_index<3>& index)
    {
        accumulate_displacement_imp(dis,new_dis,mapping,index);
    });
}
//---------------------------------------------------------------------------
// v = vx compose vy
// use vy(x) = v(x)-vx(x+vy(x))
template<typename ComposeImageType>
void decompose_displacement(const ComposeImageType& v,const ComposeImageType& vx,
                            ComposeImageType& vy)
{
    ComposeImageType vtemp(vx);
    vy.resize(v.shape());
    for (int index = 0;index < vy.size();++index)
        vy[index] = v[index]-vtemp[index];
    for(int i = 0;i < 15;++i)
    {
        tipl::compose_displacement(vx,vy,vtemp);
        for (int index = 0;index < vy.size();++index)
            vy[index] = v[index]-vtemp[index];
    }
}
//---------------------------------------------------------------------------
template<typename VectorType,typename DetType>
void jacobian_determinant(const image<3,VectorType>& src,DetType& dest)
{
    typedef typename DetType::value_type value_type;
    shape<3> geo(src.shape());
    dest.resize(geo);
    int w = src.width();
    int wh = src.plane_size();
    for (tipl::pixel_index<3> index(geo); index < geo.size();++index)
    {
        if (geo.is_edge(index))
        {
            dest[index.index()] = 1;
            continue;
        }
        const VectorType& v1_0 = src[index.index()+1];
        const VectorType& v1_1 = src[index.index()-1];
        const VectorType& v2_0 = src[index.index()+w];
        const VectorType& v2_1 = src[index.index()-w];
        const VectorType& v3_0 = src[index.index()+wh];
        const VectorType& v3_1 = src[index.index()-wh];

        value_type d2_0 = v2_0[0] - v2_1[0];
        value_type d2_1 = v2_0[1] - v2_1[1];
        value_type d2_2 = v2_0[2] - v2_1[2];

        value_type d3_0 = v3_0[0] - v3_1[0];
        value_type d3_1 = v3_0[1] - v3_1[1];
        value_type d3_2 = v3_0[2] - v3_1[2];

        dest[index.index()] = (v1_0[0] - v1_1[0])*(d2_1*d3_2-d2_2*d3_1)+
                                       (v1_0[1] - v1_1[1])*(d2_2*d3_0-d2_0*d3_2)+
                                       (v1_0[2] - v1_1[2])*(d2_0*d3_1-d2_1*d3_0);
    }
}
template<typename VectorType>
double jacobian_determinant_dis_at(const image<3,VectorType>& src,const tipl::pixel_index<3>& index)
{
    unsigned int w = src.width();
    unsigned int wh = src.plane_size();

    const VectorType& v1_0 = src[index.index()+1];
    const VectorType& v1_1 = src[index.index()-1];
    const VectorType& v2_0 = src[index.index()+w];
    const VectorType& v2_1 = src[index.index()-w];
    const VectorType& v3_0 = src[index.index()+wh];
    const VectorType& v3_1 = src[index.index()-wh];

    double d2_0 = v2_0[0] - v2_1[0];
    double d2_1 = v2_0[1] - v2_1[1]+1.0;
    double d2_2 = v2_0[2] - v2_1[2];

    double d3_0 = v3_0[0] - v3_1[0];
    double d3_1 = v3_0[1] - v3_1[1];
    double d3_2 = v3_0[2] - v3_1[2]+1.0;

    return (v1_0[0] - v1_1[0]+1.0)*(d2_1*d3_2-d2_2*d3_1)+
                                   (v1_0[1] - v1_1[1])*(d2_2*d3_0-d2_0*d3_2)+
                                   (v1_0[2] - v1_1[2])*(d2_0*d3_1-d2_1*d3_0);
}
template<typename VectorType,typename out_type>
void jacobian_dis_at(const image<3,VectorType>& src,const tipl::pixel_index<3>& index,out_type* J)
{
    unsigned int w = src.width();
    unsigned int wh = src.plane_size();

    VectorType vx = src[index.index()+1];
    vx -= src[index.index()-1];
    VectorType vy = src[index.index()+w];
    vy -= src[index.index()-w];
    VectorType vz = src[index.index()+wh];
    vz -= src[index.index()-wh];

    J[0] = vx[0]*0.5+1.0;
    J[1] = vx[1]*0.5;
    J[2] = vx[2]*0.5;

    J[3] = vy[0]*0.5;
    J[4] = vy[1]*0.5+1.0;
    J[5] = vy[2]*0.5;

    J[6] = vz[0]*0.5;
    J[7] = vz[1]*0.5;
    J[8] = vz[2]*0.5+1.0;
}
template<typename VectorType,typename DetType>
void jacobian_determinant_dis(const image<3,VectorType>& src,DetType& dest)
{
    shape<3> geo(src.shape());
    dest.resize(geo);
    for (tipl::pixel_index<3> index(geo); index < geo.size();++index)
    {
        if (geo.is_edge(index))
        {
            dest[index.index()] = 1;
            continue;
        }
        dest[index.index()] = jacobian_determinant_dis_at(src,index);
    }
}

//---------------------------------------------------------------------------
template<typename VectorType,typename PixelType>
void jacobian_determinant(const image<2,VectorType>& src,image<2,PixelType>& dest)
{
    shape<2> geo(src.shape());
    dest.resize(geo);
    int w = src.width();
    for (tipl::pixel_index<2> index(geo); index < geo.size();++index)
    {
        if (geo.is_edge(index))
        {
            dest[index.index()] = 1;
            continue;
        }
        const VectorType& v1_0 = src[index.index()+1];
        const VectorType& v1_1 = src[index.index()-1];
        const VectorType& v2_0 = src[index.index()+w];
        const VectorType& v2_1 = src[index.index()-w];
        dest[index.index()] = (v1_0[0] - v1_1[0])*(v2_0[1] - v2_1[1])-(v1_0[1] - v1_1[1])*(v2_0[0] - v2_1[0]);
    }
}

template<typename VectorType>
double jacobian_determinant_dis_at(const image<2,VectorType>& src,const tipl::pixel_index<2>& index)
{
    unsigned int w = src.width();
    const VectorType& v1_0 = src[index.index()+1];
    const VectorType& v1_1 = src[index.index()];
    const VectorType& v2_0 = src[index.index()+w];
    const VectorType& v2_1 = src[index.index()];
    return (v1_0[0] - v1_1[0]+1.0)*(v2_0[1] - v2_1[1]+1.0)-(v1_0[1] - v1_1[1])*(v2_0[0] - v2_1[0]);
}

template<typename VectorType,typename PixelType>
void jacobian_determinant_dis(const image<2,VectorType>& src,image<2,PixelType>& dest)
{
    shape<2> geo(src.shape());
    dest.resize(geo);
    for (tipl::pixel_index<2> index(geo); index < geo.size();++index)
    {
        if (geo.is_edge(index))
        {
            dest[index.index()] = 1;
            continue;
        }
        dest[index.index()] = jacobian_determinant_dis_at(src,index);
    }
}

}
#endif // DIF_HPP
