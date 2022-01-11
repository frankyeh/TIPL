#ifndef DIF_HPP
#define DIF_HPP
#include "../utility/basic_image.hpp"
#include "interpolation.hpp"
namespace tipl
{

template<typename vtor_type,unsigned int dimension>
void make_identity(image<dimension,vtor_type>& s)
{
    for (tipl::pixel_index<dimension> index(s.shape()); index < s.size();++index)
        s[index.index()] = index;
}
//---------------------------------------------------------------------------
template<typename vtor_type,unsigned int dimension>
void displacement_to_mapping(image<dimension,vtor_type>& s)
{
    for (tipl::pixel_index<dimension> index(s.shape()); index < s.size();++index)
        s[index.index()] += index;
}
//---------------------------------------------------------------------------
template<typename vtor_type,unsigned int dimension>
void mapping_to_displacement(image<dimension,vtor_type>& s)
{
    for (tipl::pixel_index<dimension> index(s.shape()); index < s.size();++index)
        s[index.index()] -= index;
}
//---------------------------------------------------------------------------
template<typename DisType,typename MappingType,typename transform_type>
void displacement_to_mapping(const DisType& dis,MappingType& mapping,const transform_type& T)
{
    mapping = dis;
    mapping.for_each_mt([&](typename MappingType::value_type& value,
                            const tipl::pixel_index<MappingType::dimension>& index)
    {
        typename MappingType::value_type vtor(index);
        vtor += value;
        T(vtor,value);
    });
}

//---------------------------------------------------------------------------
template<typename DisType,typename MappingType,typename transform_type>
void inv_displacement_to_mapping(const DisType& inv_dis,MappingType& inv_mapping,const transform_type& T)
{
    auto iT = T;
    iT.inverse();
    inv_mapping.for_each_mt([&](tipl::vector<3,float>& v,const tipl::pixel_index<3>& pos)
    {
        tipl::vector<3> p(pos),d;
        iT(p);
        v = p;
        tipl::estimate(inv_dis,v,d,tipl::linear);
        v += d;
    });
}

//---------------------------------------------------------------------------
template<typename ImageType,typename MappingType,typename OutImageType>
void compose_mapping(const ImageType& src,const MappingType& mapping,OutImageType& dest,
                     interpolation_type type = interpolation_type::linear)
{
    dest.clear();
    dest.resize(mapping.shape());
    tipl::par_for(dest.size(),[&](unsigned int index)
    {
        estimate(src,mapping[index],dest[index],type);
    });
}
//---------------------------------------------------------------------------
template<typename ImageType,typename ComposeImageType,typename OutImageType>
void compose_displacement(const ImageType& src,const ComposeImageType& displace,OutImageType& dest,
                          interpolation_type type = interpolation_type::linear)
{
    dest.clear();
    dest.resize(src.shape());
    dest.for_each_mt([&](typename OutImageType::value_type& value,
                         tipl::pixel_index<ComposeImageType::dimension> index)
    {
        if(displace[index.index()] == typename ComposeImageType::value_type())
        {
            value = src[index.index()];
            return;
        }
        typename ComposeImageType::value_type vtor(index);
        vtor += displace[index.index()];
        tipl::estimate(src,vtor,value,type);
    });
}
//---------------------------------------------------------------------------
template<typename ImageType,typename ComposeImageType,typename OutImageType,typename transform_type>
void compose_displacement_with_affine(const ImageType& src,OutImageType& dest,
                          const transform_type& transform,
                          const ComposeImageType& displace,
                          interpolation_type type = interpolation_type::linear)
{
    dest.clear();
    dest.resize(displace.shape());
    dest.for_each_mt([&](typename OutImageType::value_type& value,tipl::pixel_index<OutImageType::dimension> index)
    {
        typename ComposeImageType::value_type vtor(index);
        vtor += displace[index.index()];
        tipl::vector<OutImageType::dimension> pos;
        transform(vtor,pos);
        tipl::estimate(src,pos,value,type);
    });
}

//---------------------------------------------------------------------------
template<typename ComposeImageType>
void invert_displacement(const ComposeImageType& v0,ComposeImageType& v1,uint8_t iterations = 16)
{
    ComposeImageType vv;
    v1.resize(v0.shape());
    for(size_t index = 0;index < v1.size();++index)
        v1[index] = -v0[index];
    for(uint8_t i = 0;i < iterations;++i)
    {
        tipl::compose_displacement(v0,v1,vv);
        for(size_t index = 0;index < v1.size();++index)
            v1[index] = -vv[index];
    }
}
//---------------------------------------------------------------------------
template<typename ComposeImageType>
void invert_displacement(ComposeImageType& v,uint8_t iterations = 16)
{
    ComposeImageType v0;
    invert_displacement(v,v0,iterations);
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
template<typename ComposeImageType>
void accumulate_displacement(const ComposeImageType& vin,
                             const ComposeImageType& vv,
                             ComposeImageType& vout)
{
    compose_displacement(vin,vv,vout);
    vout += vv;
}
//---------------------------------------------------------------------------
template<typename ComposeImageType>
void accumulate_displacement(ComposeImageType& v0,const ComposeImageType& vv)
{
    ComposeImageType nv;
    compose_displacement(v0,vv,nv);
    v0.swap(nv);
    v0 += vv;
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
