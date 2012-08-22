#ifndef DIF_HPP
#define DIF_HPP

namespace image
{

template<typename vtor_type,unsigned int dimension>
void make_identity(basic_image<vtor_type,dimension>& s)
{
    for (image::pixel_index<dimension> index; index.valid(s.geometry());index.next(s.geometry()))
        s[index.index()] = index;
}
//---------------------------------------------------------------------------
template<typename vtor_type,unsigned int dimension>
void displacement_to_mapping(basic_image<vtor_type,dimension>& s)
{
    for (image::pixel_index<dimension> index; index.valid(s.geometry());index.next(s.geometry()))
        s[index.index()] += index;
}
//---------------------------------------------------------------------------
template<typename vtor_type,unsigned int dimension>
void mapping_to_displacement(basic_image<vtor_type,dimension>& s)
{
    for (image::pixel_index<dimension> index; index.valid(s.geometry());index.next(s.geometry()))
        s[index.index()] -= index;
}
//---------------------------------------------------------------------------
template<typename ImageType,typename ComposeImageType,typename OutImageType>
void compose_mapping(const ImageType& src,const ComposeImageType& compose,OutImageType& dest)
{
    dest.clear();
    dest.resize(compose.geometry());
    typename ComposeImageType::const_iterator iter = compose.begin();
    typename ComposeImageType::const_iterator end = compose.end();
    typename OutImageType::iterator out = dest.begin();
    for (; iter != end; ++iter,++out)
        image::linear_estimate(src,*iter,*out);
}
//---------------------------------------------------------------------------
template<typename ImageType,typename ComposeImageType,typename OutImageType>
void compose_displacement(const ImageType& src,const ComposeImageType& displace,OutImageType& dest)
{
    image::geometry<ImageType::dimension> geo(src.geometry());
    dest.clear();
    dest.resize(geo);
    for(pixel_index<ImageType::dimension> index;index.valid(geo);index.next(geo))
    {
        typename ComposeImageType::value_type vtor(index);
        vtor += displace[index.index()];
        image::linear_estimate(src,vtor,dest[index.index()]);
    }
}
//---------------------------------------------------------------------------
template<typename ComposeImageType>
void invert_displacement(const ComposeImageType& v0,ComposeImageType& v1)
{
    ComposeImageType vv;
    v1.resize(v0.geometry());
    for (int index = 0;index < v1.size();++index)
        v1[index] = -v0[index];
    for(int i = 0;i < 15;++i)
    {
        image::compose_displacement(v0,v1,vv);
        for (int index = 0;index < v1.size();++index)
            v1[index] = -vv[index];
    }
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
void invert_mapping(const ComposeImageType& s0,ComposeImageType& s1)
{
    ComposeImageType v0(s0);
    mapping_to_displacement(v0);
    invert_displacement(v0,s1);
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
    v0 = nv;
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
    vy.resize(v.geometry());
    for (int index = 0;index < vy.size();++index)
        vy[index] = v[index]-vtemp[index];
    for(int i = 0;i < 15;++i)
    {
        image::compose_displacement(vx,vy,vtemp);
        for (int index = 0;index < vy.size();++index)
            vy[index] = v[index]-vtemp[index];
    }
}
//---------------------------------------------------------------------------
template<typename VectorType,typename DetType>
void jacobian_determinant(const basic_image<VectorType,3>& src,DetType& dest)
{
    typedef typename DetType::value_type value_type;
    geometry<3> geo(src.geometry());
    dest.resize(geo);
    int w = src.width();
    int wh = src.plane_size();
    for (image::pixel_index<3> index; index.valid(geo); index.next(geo))
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

        dest[index.index()] = std::fabs((v1_0[0] - v1_1[0])*(d2_1*d3_2-d2_2*d3_1)+
                                       (v1_0[1] - v1_1[1])*(d2_2*d3_0-d2_0*d3_2)+
                                       (v1_0[2] - v1_1[2])*(d2_0*d3_1-d2_1*d3_0));
    }
}
template<typename VectorType,typename DetType>
void jacobian_determinant_dis(const basic_image<VectorType,3>& src,DetType& dest)
{
    typedef typename DetType::value_type value_type;
    geometry<3> geo(src.geometry());
    dest.resize(geo);
    int w = src.width();
    int wh = src.plane_size();
    for (image::pixel_index<3> index; index.valid(geo); index.next(geo))
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
        value_type d2_1 = v2_0[1] - v2_1[1]+1.0;
        value_type d2_2 = v2_0[2] - v2_1[2];

        value_type d3_0 = v3_0[0] - v3_1[0];
        value_type d3_1 = v3_0[1] - v3_1[1];
        value_type d3_2 = v3_0[2] - v3_1[2]+1.0;

        dest[index.index()] = std::fabs((v1_0[0] - v1_1[0]+1.0)*(d2_1*d3_2-d2_2*d3_1)+
                                       (v1_0[1] - v1_1[1])*(d2_2*d3_0-d2_0*d3_2)+
                                       (v1_0[2] - v1_1[2])*(d2_0*d3_1-d2_1*d3_0));
    }
}
//---------------------------------------------------------------------------
/*
template<typename ImageType,typename DetType>
void jacobian_determine(const std::vector<ImageType>& src,DetType& dest)
{
    typedef typename DetType::value_type value_type;
    geometry<3> geo(src[0].geometry());
    dest.resize(geo);

    for (image::pixel_index<3> index; index.valid(geo); index.next(geo))
    {
        if (geo.is_edge(index))
        {
            dest[index.index()] = 0;
            continue;
        }

        unsigned int v1_0 = index.index();
        unsigned int v1_1 = index.index();
        unsigned int v2_0 = index.index();
        unsigned int v2_1 = index.index();
        unsigned int v3_0 = index.index();
        unsigned int v3_1 = index.index();
        const ImageType& s0 = src[0];
        const ImageType& s1 = src[1];
        const ImageType& s2 = src[2];


        value_type d2_0 = s0[v2_0] - s0[v2_1];
        value_type d2_1 = s1[v2_0] - s1[v2_1];
        value_type d2_2 = s2[v2_0] - s2[v2_1];

        value_type d3_0 = s0[v3_0] - s0[v3_1];
        value_type d3_1 = s1[v3_0] - s1[v3_1];
        value_type d3_2 = s2[v3_0] - s2[v3_1];

        dest[index.index()] = std::abs((s0[v1_0] - s0[v1_1])*(d2_1*d3_2-d2_2*d3_1)+
                                       (s1[v1_0] - s1[v1_1])*(d2_2*d3_0-d2_0*d3_2)+
                                       (s2[v1_0] - s2[v1_1])*(d2_0*d3_1-d2_1*d3_0));
    }
}
*/
//---------------------------------------------------------------------------
template<typename VectorType,typename PixelType>
void jacobian_determinant(const basic_image<VectorType,2>& src,basic_image<PixelType,2>& dest)
{
    geometry<2> geo(src.geometry());
    dest.resize(geo);
    int w = src.width();
    for (image::pixel_index<2> index; index.valid(geo); index.next(geo))
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
        dest[index.index()] = std::fabs((v1_0[0] - v1_1[0])*(v2_0[1] - v2_1[1])-(v1_0[1] - v1_1[1])*(v2_0[0] - v2_1[0]));
    }
}

template<typename VectorType,typename PixelType>
void jacobian_determinant_dis(const basic_image<VectorType,2>& src,basic_image<PixelType,2>& dest)
{
    geometry<2> geo(src.geometry());
    dest.resize(geo);
    int w = src.width();
    for (image::pixel_index<2> index; index.valid(geo); index.next(geo))
    {
        if (geo.is_edge(index))
        {
            dest[index.index()] = 1;
            continue;
        }
        const VectorType& v1_0 = src[index.index()+1];
        const VectorType& v1_1 = src[index.index()];
        const VectorType& v2_0 = src[index.index()+w];
        const VectorType& v2_1 = src[index.index()];
        dest[index.index()] = std::fabs((v1_0[0] - v1_1[0]+1.0)*(v2_0[1] - v2_1[1]+1.0)-(v1_0[1] - v1_1[1])*(v2_0[0] - v2_1[0]));
    }
}

}
#endif // DIF_HPP
