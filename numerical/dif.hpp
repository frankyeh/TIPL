#ifndef DIF_HPP
#define DIF_HPP
#include "tipl/utility/basic_image.hpp"
#include "tipl/numerical/interpolation.hpp"
namespace tipl
{

template<class vtor_type,unsigned int dimension>
void make_identity(image<vtor_type,dimension>& s)
{
    for (tipl::pixel_index<dimension> index(s.geometry()); index < s.size();++index)
        s[index.index()] = index;
}
//---------------------------------------------------------------------------
template<class vtor_type,unsigned int dimension>
void displacement_to_mapping(image<vtor_type,dimension>& s)
{
    for (tipl::pixel_index<dimension> index(s.geometry()); index < s.size();++index)
        s[index.index()] += index;
}
//---------------------------------------------------------------------------
template<class vtor_type,unsigned int dimension>
void mapping_to_displacement(image<vtor_type,dimension>& s)
{
    for (tipl::pixel_index<dimension> index(s.geometry()); index < s.size();++index)
        s[index.index()] -= index;
}
//---------------------------------------------------------------------------
template<class ImageType,class ComposeImageType,class OutImageType>
void compose_mapping(const ImageType& src,const ComposeImageType& compose,OutImageType& dest)
{
    dest.clear();
    dest.resize(compose.geometry());
    typename ComposeImageType::const_iterator iter = compose.begin();
    typename ComposeImageType::const_iterator end = compose.end();
    typename OutImageType::iterator out = dest.begin();
    for (; iter != end; ++iter,++out)
        estimate(src,*iter,*out);
}
//---------------------------------------------------------------------------
template<class ImageType,class ComposeImageType,class OutImageType>
void compose_displacement(const ImageType& src,const ComposeImageType& displace,OutImageType& dest,
                          interpolation_type type = interpolation_type::linear)
{
    tipl::geometry<ImageType::dimension> geo(src.geometry());
    dest.clear();
    dest.resize(geo);
    for(tipl::pixel_index<ImageType::dimension> index(geo);index.is_valid(geo);++index)
    {
        if(displace[index.index()] == typename ComposeImageType::value_type())
        {
            dest[index.index()] = src[index.index()];
            continue;
        }
        typename ComposeImageType::value_type vtor(index);
        vtor += displace[index.index()];
        tipl::estimate(src,vtor,dest[index.index()],type);
    }
}
//---------------------------------------------------------------------------
template<class ComposeImageType,class transform_type>
void displacement_to_mapping(ComposeImageType& mapping,
                          const transform_type& transform)
{
    mapping.for_each_mt([&](typename ComposeImageType::value_type& value,
                             tipl::pixel_index<ComposeImageType::dimension> index)
    {
        typename ComposeImageType::value_type vtor(index),pos;
        vtor += value;
        transform(vtor,value);
    });
}
//---------------------------------------------------------------------------
template<class ImageType,class ComposeImageType,class OutImageType,class transform_type>
void compose_displacement_with_affine(const ImageType& src,OutImageType& dest,
                          const transform_type& transform,
                          const ComposeImageType& displace,
                          interpolation_type type = interpolation_type::linear)
{
    tipl::geometry<ImageType::dimension> geo(displace.geometry());
    dest.clear();
    dest.resize(geo);
    dest.for_each_mt([&](typename OutImageType::value_type&,tipl::pixel_index<OutImageType::dimension> index)
    {
        typename ComposeImageType::value_type vtor(index);
        vtor += displace[index.index()];
        tipl::vector<OutImageType::dimension,double> pos;
        transform(vtor,pos);
        tipl::estimate(src,pos,dest[index.index()],type);
    });
}

//---------------------------------------------------------------------------
template<class ImageType,class ComposeImageType,class OutImageType>
void compose_displacement_with_jacobian(const ImageType& src,const ComposeImageType& displace,OutImageType& dest)
{
    tipl::geometry<ImageType::dimension> geo(src.geometry());
    dest.clear();
    dest.resize(geo);
    for(tipl::pixel_index<ImageType::dimension> index(geo);index.is_valid(geo);++index)
    {
        typename ComposeImageType::value_type vtor(index);
        vtor += displace[index.index()];
        tipl::estimate(src,vtor,dest[index.index()]);
    }
}
//---------------------------------------------------------------------------
template<class ComposeImageType>
void invert_displacement(const ComposeImageType& v0,ComposeImageType& v1,uint8_t iterations = 16)
{
    ComposeImageType vv;
    v1.resize(v0.geometry());
    for (int index = 0;index < v1.size();++index)
        v1[index] = -v0[index];
    for(uint8_t i = 0;i < iterations;++i)
    {
        tipl::compose_displacement(v0,v1,vv);
        for (int index = 0;index < v1.size();++index)
            v1[index] = -vv[index];
    }
}
//---------------------------------------------------------------------------
template<class ComposeImageType>
void invert_displacement(ComposeImageType& v,uint8_t iterations = 16)
{
    ComposeImageType v0;
    invert_displacement(v,v0,iterations);
    v.swap(v0);
}

//---------------------------------------------------------------------------
template<class ComposeImageType>
void invert_mapping(const ComposeImageType& s0,ComposeImageType& s1,uint8_t iterations)
{
    ComposeImageType v0(s0);
    mapping_to_displacement(v0);
    invert_displacement(v0,s1,iterations);
    displacement_to_mapping(s1);
}
//---------------------------------------------------------------------------
template<class ComposeImageType>
void accumulate_displacement(const ComposeImageType& vin,
                             const ComposeImageType& vv,
                             ComposeImageType& vout)
{
    compose_displacement(vin,vv,vout);
    vout += vv;
}
//---------------------------------------------------------------------------
template<class ComposeImageType>
void accumulate_displacement(ComposeImageType& v0,const ComposeImageType& vv)
{
    ComposeImageType nv;
    compose_displacement(v0,vv,nv);
    v0 = nv;
    v0 += vv;
}
//---------------------------------------------------------------------------
// v = vx compose vy
// use vy(x) = v(x)-vx(x+vy(x))
template<class ComposeImageType>
void decompose_displacement(const ComposeImageType& v,const ComposeImageType& vx,
                            ComposeImageType& vy)
{
    ComposeImageType vtemp(vx);
    vy.resize(v.geometry());
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
template<class VectorType,class DetType>
void jacobian_determinant(const image<VectorType,3>& src,DetType& dest)
{
    typedef typename DetType::value_type value_type;
    geometry<3> geo(src.geometry());
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
template<class VectorType>
double jacobian_determinant_dis_at(const image<VectorType,3>& src,const tipl::pixel_index<3>& index)
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
template<class VectorType,class out_type>
void jacobian_dis_at(const image<VectorType,3>& src,const tipl::pixel_index<3>& index,out_type* J)
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
template<class VectorType,class DetType>
void jacobian_determinant_dis(const image<VectorType,3>& src,DetType& dest)
{
    geometry<3> geo(src.geometry());
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
template<class VectorType,class PixelType>
void jacobian_determinant(const image<VectorType,2>& src,image<PixelType,2>& dest)
{
    geometry<2> geo(src.geometry());
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

template<class VectorType>
double jacobian_determinant_dis_at(const image<VectorType,2>& src,const tipl::pixel_index<2>& index)
{
    unsigned int w = src.width();
    const VectorType& v1_0 = src[index.index()+1];
    const VectorType& v1_1 = src[index.index()];
    const VectorType& v2_0 = src[index.index()+w];
    const VectorType& v2_1 = src[index.index()];
    return (v1_0[0] - v1_1[0]+1.0)*(v2_0[1] - v2_1[1]+1.0)-(v1_0[1] - v1_1[1])*(v2_0[0] - v2_1[0]);
}

template<class VectorType,class PixelType>
void jacobian_determinant_dis(const image<VectorType,2>& src,image<PixelType,2>& dest)
{
    geometry<2> geo(src.geometry());
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
