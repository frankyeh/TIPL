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
                [&](const auto& index)
    {
        s[index.index()] = index;
    });
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T>
__global__ void displacement_to_mapping_cuda_kernel(T dis)
{
    TIPL_FOR(index,dis.size())
        dis[index] += tipl::pixel_index<T::dimension>(index,dis.shape());
}

#endif
//---------------------------------------------------------------------------
template<typename T>
void displacement_to_mapping(T& dis)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(displacement_to_mapping_cuda_kernel,dis.size())
                (tipl::make_shared(dis));
        #endif
    }
    else
    tipl::par_for(tipl::begin_index(dis.shape()),tipl::end_index(dis.shape()),
                [&](const auto& index)
    {
        dis[index.index()] += index;
    });
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T,typename U>
__global__ void displacement_to_mapping_cuda_kernel2(T dis,U mapping)
{
    TIPL_FOR(index,dis.size())
        mapping[index] += tipl::pixel_index<T::dimension>(index,dis.shape());
}
#endif
//---------------------------------------------------------------------------
template<typename T,typename U>
void displacement_to_mapping(const T& dis,U& mapping)
{
    mapping = dis;
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(displacement_to_mapping_cuda_kernel2,dis.size())
                (tipl::make_shared(dis),tipl::make_shared(mapping));
        #endif
    }
    else
    tipl::par_for(tipl::begin_index(mapping.shape()),tipl::end_index(mapping.shape()),
                            [&](const auto& index)
    {
        mapping[index.index()] += index;
    });
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T>
__global__ void mapping_to_displacement_cuda_kernel(T mapping)
{
    TIPL_FOR(index,mapping.size())
        mapping[index] -= tipl::pixel_index<T::dimension>(index,mapping.shape());
}
#endif
//---------------------------------------------------------------------------
template<typename T>
void mapping_to_displacement(T& mapping)
{
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(mapping_to_displacement_cuda_kernel,mapping.size())
                (tipl::make_shared(mapping));
        #endif
    }
    else
    tipl::par_for(tipl::begin_index(mapping.shape()),tipl::end_index(mapping.shape()),
                [&](const auto& index)
    {
        mapping[index.index()] -= index;
    });
}

//---------------------------------------------------------------------------
template<typename DisType,typename MappingType,typename transform_type>
void displacement_to_mapping(const DisType& dis,MappingType& mapping,const transform_type& T)
{
    mapping = dis;
    tipl::par_for(tipl::begin_index(mapping.shape()),tipl::end_index(mapping.shape()),
                            [&](const auto& index)
    {
        typename MappingType::value_type vtor(index);
        vtor += mapping[index.index()];
        T(vtor,mapping[index.index()]);
    });
}

//---------------------------------------------------------------------------
template<tipl::interpolation itype = linear,typename DisType,typename MappingType,typename transform_type>
void inv_displacement_to_mapping(const DisType& inv_dis,MappingType& inv_mapping,const transform_type& T)
{
    auto iT = T;
    iT.inverse();
    tipl::par_for(tipl::begin_index(inv_mapping.shape()),tipl::end_index(inv_mapping.shape()),
        [&](const auto& index)
    {
        tipl::vector<DisType::dimension> p;
        iT(index,p);
        p += tipl::estimate<itype>(inv_dis,p);
        inv_mapping[index.index()] = p;
    });
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<tipl::interpolation itype,typename T1,typename T2,typename T3>
__global__ void compose_mapping_cuda_kernel(T1 from,T2 mapping,T3 to)
{
    TIPL_FOR(index,to.size())
        tipl::estimate<itype>(from,mapping[index],to[index]);
}

#endif
//---------------------------------------------------------------------------
template<tipl::interpolation itype = tipl::interpolation::linear,
         typename T1,typename T2,typename T3>
inline void compose_mapping(const T1& from,const T2& mapping,T3& to)

{
    to.clear();
    to.resize(mapping.shape());
    if constexpr(memory_location<T1>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(compose_mapping_cuda_kernel<itype>,mapping.size())
                (tipl::make_shared(from),tipl::make_shared(mapping),tipl::make_shared(to));
        #endif
    }
    else
    tipl::par_for(mapping.size(),[&](size_t index)
    {
        estimate<itype>(from,mapping[index],to[index]);
    });
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__

template<tipl::interpolation itype,typename T1,typename T2,typename T3>
__global__ void compose_displacement_cuda_kernel(T1 from,T2 dis,T3 to)
{
    TIPL_FOR(index,to.size())
    {
        if(dis[index] == typename T2::value_type())
            to[index] = from[index];
        else
        {
            typename T2::value_type v(tipl::pixel_index<T1::dimension>(index,to.shape()));
            v += dis[index];
            tipl::estimate<itype>(from,v,to[index]);
        }
    }
}
#endif
template<tipl::interpolation itype = linear,typename T,typename U,typename V>
void compose_displacement(const T& from,const U& dis,V& to)
{
    to.clear();
    to.resize(from.shape());
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(compose_displacement_cuda_kernel<itype>,to.size())
                (tipl::make_shared(from),tipl::make_shared(dis),tipl::make_shared(to));
        #endif
    }
    else
    {
        tipl::par_for(tipl::begin_index(from.shape()),tipl::end_index(from.shape()),
            [&](const auto& index)
        {
            if(dis[index.index()] == typename U::value_type())
                to[index.index()] = from[index.index()];
            else
            {
                typename U::value_type v(index);
                v += dis[index.index()];
                tipl::estimate<itype>(from,v,to[index.index()]);
            }
        });
    }
}
template<tipl::interpolation itype = linear,typename T,typename U>
inline auto compose_displacement(const T& from,const U& dis)
{
    typename T::buffer_type to;
    compose_displacement<itype>(from,dis,to);
    return to;
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
        [&](const auto& index)
    {
        typename ComposeImageType::value_type vtor(index);
        vtor += displace[index.index()];
        tipl::vector<OutImageType::dimension> pos;
        transform(vtor,pos);
        tipl::estimate<Type>(src,pos,dest[index.index()]);
    });
}
template<typename T,typename U,typename V>
__INLINE__ void invert_displacement_imp(const V& index,T& v1,U& mapping)
{
    auto& v = v1[index.index()];
    tipl::vector<T::dimension> vv;
    if(tipl::estimate<linear>(mapping,v+index,vv))
    {
        vv -= index;
        v -= vv;
    }
    else
        v *= 0.5f;
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T,typename U>
__global__ void invert_displacement_cuda_kernel(T v1,U mapping)
{
    TIPL_FOR(index,v1.size())
    {
        invert_displacement_imp(tipl::pixel_index<T::dimension>(index,v1.shape()),v1,mapping);
    }
}
#endif
//---------------------------------------------------------------------------
template<typename T>
void invert_displacement(const T& v0,T& v1,size_t count = 8)
{
    v1.resize(v0.shape());
    T mapping(v0);
    displacement_to_mapping(mapping);
    for(uint8_t i = 0;i < count;++i)
    {
        if constexpr(memory_location<T>::at == CUDA)
        {
            #ifdef __CUDACC__
            TIPL_RUN(invert_displacement_cuda_kernel,v1.size())
                    (tipl::make_shared(v1),tipl::make_shared(mapping));
            #endif
        }
        else
        tipl::par_for(tipl::begin_index(v1.shape()),tipl::end_index(v1.shape()),
            [&](const auto& index)
        {
            invert_displacement_imp(index,v1,mapping);
        });
    }
}
//---------------------------------------------------------------------------
template<typename T,typename U,typename V,typename W>
__INLINE__ void accumulate_displacement_imp(const W& index,const T& dis,U& new_dis,const V& mapping)
{
    auto d = new_dis[index.index()];
    if(d != decltype(d)() && tipl::estimate<tipl::interpolation::linear>(mapping,d+=index,new_dis[index.index()]))
        new_dis[index.index()] -= index;
    else
        new_dis[index.index()] = dis[index.index()];
}
//---------------------------------------------------------------------------
#ifdef __CUDACC__
template<typename T1,typename T2,typename T3>
__global__ void accumulate_displacement_cuda_kernel(T1 dis,T2 new_dis,T3 mapping)
{
    TIPL_FOR(index,new_dis.size())
    {
        accumulate_displacement_imp(tipl::pixel_index<T1::dimension>(index,new_dis.shape()),dis,new_dis,mapping);
    }
}

#endif
//---------------------------------------------------------------------------
template<typename T>
void accumulate_displacement(const T& dis,T& new_dis)
{
    T mapping;
    displacement_to_mapping(dis,mapping);
    if constexpr(memory_location<T>::at == CUDA)
    {
        #ifdef __CUDACC__
        TIPL_RUN(accumulate_displacement_cuda_kernel,dis.size())
                (tipl::make_shared(dis),tipl::make_shared(new_dis),tipl::make_shared(mapping));
        #endif
    }
    else
    tipl::par_for(tipl::begin_index(dis.shape()),tipl::end_index(dis.shape()),[&](const auto& index)
    {
        accumulate_displacement_imp(index,dis,new_dis,mapping);
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
    for (size_t index = 0;index < vy.size();++index)
        vy[index] = v[index]-vtemp[index];
    for(int i = 0;i < 15;++i)
    {
        tipl::compose_displacement(vx,vy,vtemp);
        for (size_t index = 0;index < vy.size();++index)
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
    auto w = src.width();
    auto wh = src.plane_size();
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
    auto w = src.width();
    auto wh = src.plane_size();

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
    auto w = src.width();
    auto wh = src.plane_size();

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
    auto w = src.width();
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
    auto w = src.width();
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
