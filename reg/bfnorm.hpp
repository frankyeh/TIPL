#ifndef BFNORM_HPP
#define BFNORM_HPP

#include <cmath>
#include <vector>
#include <future>
#include "image/numerical/matrix.hpp"
#include "image/numerical/numerical.hpp"
namespace image {

namespace reg {

template<class ImageType,class value_type>
value_type resample_d(const ImageType& vol,value_type& gradx,value_type& grady,value_type& gradz,value_type x,value_type y,value_type z)
{
    const value_type TINY = 5e-2;
    int xdim = vol.width();
    int ydim = vol.height();
    int zdim = vol.depth();
    {
        value_type xi,yi,zi;
        xi=x-1.0;
        yi=y-1.0;
        zi=z-1.0;
        if (zi>=-TINY && zi<zdim+TINY-1 &&
                yi>=-TINY && yi<ydim+TINY-1 &&
                xi>=-TINY && xi<xdim+TINY-1)
        {
            value_type k111,k112,k121,k122,k211,k212,k221,k222;
            value_type dx1, dx2, dy1, dy2, dz1, dz2;
            int off1, off2, offx, offy, offz, xcoord, ycoord, zcoord;

            xcoord = (int)floor(xi);
            dx1=xi-xcoord;
            dx2=1.0-dx1;
            ycoord = (int)floor(yi);
            dy1=yi-ycoord;
            dy2=1.0-dy1;
            zcoord = (int)floor(zi);
            dz1=zi-zcoord;
            dz2=1.0-dz1;

            xcoord = (xcoord < 0) ? ((offx=0),0) : ((xcoord>=xdim-1) ? ((offx=0),xdim-1) : ((offx=1   ),xcoord));
            ycoord = (ycoord < 0) ? ((offy=0),0) : ((ycoord>=ydim-1) ? ((offy=0),ydim-1) : ((offy=xdim),ycoord));
            zcoord = (zcoord < 0) ? ((offz=0),0) : ((zcoord>=zdim-1) ? ((offz=0),zdim-1) : ((offz=1   ),zcoord));

            off1 = xcoord  + xdim*ycoord;
            off2 = off1+offy;
            int off1_offx = off1+offx;
            int off2_offx = off2+offx;
            int z_offset = zcoord*vol.plane_size();
            int z_offset_off = z_offset + (offz == 1 ? vol.plane_size():0);

            const typename ImageType::value_type* vol_ptr = &*vol.begin();
            const typename ImageType::value_type* vol_z_offset = vol_ptr+z_offset;
            const typename ImageType::value_type* vol_z_offset_off = vol_ptr+z_offset_off;


            k222 = *(vol_z_offset+off1);
            k122 = *(vol_z_offset+off1_offx);
            k212 = *(vol_z_offset+off2);
            k112 = *(vol_z_offset+off2_offx);
            k221 = *(vol_z_offset_off+off1);
            k121 = *(vol_z_offset_off+off1_offx);
            k211 = *(vol_z_offset_off+off2);
            k111 = *(vol_z_offset_off+off2_offx);

            gradx = (((k111 - k211)*dy1 + (k121 - k221)*dy2))*dz1
                    + (((k112 - k212)*dy1 + (k122 - k222)*dy2))*dz2;

            k111 = (k111*dx1 + k211*dx2);
            k121 = (k121*dx1 + k221*dx2);
            k112 = (k112*dx1 + k212*dx2);
            k122 = (k122*dx1 + k222*dx2);

            grady = (k111 - k121)*dz1 + (k112 - k122)*dz2;

            k111 = k111*dy1 + k121*dy2;
            k112 = k112*dy1 + k122*dy2;

            gradz = k111 - k112;
            return k111*dz1 + k112*dz2;
        }
        else
        {
            gradx = 0.0;
            grady = 0.0;
            gradz = 0.0;
            return 0;
        }
    }
}


template<class value_type,int dim = 3>
class bfnorm_mapping {
public:
    image::geometry<dim> VGgeo;
    image::geometry<dim> k_base;
    std::vector<value_type> T;
    std::vector<std::vector<value_type> > bas,dbas;
public:
    bfnorm_mapping(const image::geometry<3>& geo_,const image::geometry<dim>& k_base_):VGgeo(geo_),k_base(k_base_)
    {
        //void initialize_basis_function(value_type stabilise) // bounding offset
        value_type stabilise = 8;
        bas.resize(dim);
        dbas.resize(dim);
        for(int d = 0; d < dim; ++d)
        {
            value_type pi_inv_mni_dim = 3.14159265358979323846/value_type(VGgeo[d]);
            bas[d].resize(VGgeo[d]*k_base[d]);
            dbas[d].resize(VGgeo[d]*k_base[d]);
            // C(:,1)=ones(size(n,1),1)/sqrt(N);
            std::fill(bas[d].begin(),bas[d].begin()+VGgeo[d],stabilise/std::sqrt((float)VGgeo[d]));
            std::fill(dbas[d].begin(),dbas[d].begin()+VGgeo[d],0.0);
            for(int i = 1,index = VGgeo[d]; i < k_base[d]; ++i)
                for(int n = 0; n < VGgeo[d]; ++n,++index)
                {
                    // C(:,k) = sqrt(2/N)*cos(pi*(2*n+1)*(k-1)/(2*N));
                    bas[d][index] = stabilise*std::sqrt(2.0/value_type(VGgeo[d]))*std::cos(pi_inv_mni_dim*(value_type)i*((value_type)n+0.5));
                    // C(:,k) = -2^(1/2)*(1/N)^(1/2)*sin(1/2*pi*(2*n*k-2*n+k-1)/N)*pi*(k-1)/N;
                    dbas[d][index] = -stabilise*std::sqrt(2.0/value_type(VGgeo[d]))*std::sin(pi_inv_mni_dim*(value_type)i*((value_type)n+0.5))*pi_inv_mni_dim*i;
                }
        }

        T.resize(3*k_base.size()+4);
        T[3*k_base.size()] = 1;
    }

    template<class rhs_type>
    void operator()(const image::pixel_index<3>& from,rhs_type& to) const
    {
        return (*this)(image::vector<3,int>(from[0],from[1],from[2]),to);
    }
    template<class rhs_type>
    void operator()(const image::vector<3,int>& from,rhs_type& to) const
    {
        to = from;
        if(!VGgeo.is_valid(from))
            return;
        int nx = k_base[0];
        int ny = k_base[1];
        int nz = k_base[2];
        int nyz =ny*nz;
        int nxyz = k_base.size();

        image::dyndim dyz_x(nyz,nx),dz_y(nz,ny),dx_1(nx,1),dy_1(ny,1);
        std::vector<value_type> bx_(nx),by_(ny),bz_(nz),temp_(nyz),temp2_(nz);
        value_type *bx = &bx_[0];
        value_type *by = &by_[0];
        value_type *bz = &bz_[0];
        value_type *temp = &temp_[0];
        value_type *temp2 = &temp2_[0];

        {
            for(int k = 0,index = from[0]; k < nx; ++k,index += VGgeo[0])
                bx[k] = bas[0][index];
            for(int k = 0,index = from[1]; k < ny; ++k,index += VGgeo[1])
                by[k] = bas[1][index];
            for(int k = 0,index = from[2]; k < nz; ++k,index += VGgeo[2])
                bz[k] = bas[2][index];
        }

        image::mat::product(T.begin(),bx,temp,dyz_x,dx_1);
        image::mat::product(temp,by,temp2,dz_y,dy_1);
        to[0] += image::vec::dot(bz,bz+nz,temp2);

        image::mat::product(T.begin()+nxyz,bx,temp,dyz_x,dx_1);
        image::mat::product(temp,by,temp2,dz_y,dy_1);
        to[1] += image::vec::dot(bz,bz+nz,temp2);

        image::mat::product(T.begin()+(nxyz << 1),bx,temp,dyz_x,dx_1);
        image::mat::product(temp,by,temp2,dz_y,dy_1);
        to[2] += image::vec::dot(bz,bz+nz,temp2);
    }
};

template<class value_type>
void fill_values(std::vector<value_type>& values,value_type step)
{
    value_type value = 0;
    for(unsigned int index = 0; index < values.size(); ++index,value += step)
        values[index] = value;
}
template<class parameter_type>
void bfnorm_mrqcof_zero_half(std::vector<parameter_type>& alpha,int m1)
{
    for (int x1=0; x1<m1; x1++)
    {
        for (int x2=0; x2<=x1; x2++)
            alpha[m1*x1+x2] = 0.0;
    }
}


template<class image_type,class value_type>
class bfnorm_slice_data{
    const image_type& VG;
    const image_type& VF;
    const std::vector<value_type>& T;
private:
    int nx,ny,nz;
    int nxy,nxyz,nx3,nxy3,nxyz3;
    std::vector<int> nxy_values,dim1_2_values,nx_values,ny_values,nz_values,dim1_1_values,dim1_0_values;
    int edgeskip[3],samp[3];
    image::geometry<3> dim1;
private:
    const std::vector<value_type>& B0;
    const std::vector<value_type>& B1;
    const std::vector<value_type>& B2;
    const std::vector<value_type>& dB0;
    const std::vector<value_type>& dB1;
    const std::vector<value_type>& dB2;
    const value_type *bz3[3], *by3[3], *bx3[3];
public:
    std::vector<value_type> alphaxy,alphax,betaxy,betax,Tz,Ty;
    std::vector<std::vector<std::vector<value_type> > > Jz,Jy;
    int s0[3];
public:
    std::vector<value_type> dif_alpha,dif_beta;
public:
    unsigned int thread_id,thread_count;
public:
    bfnorm_slice_data(const image_type& VG_,
                      const image_type& VF_,
                      const std::vector<value_type>& T_,
                      const std::vector<std::vector<value_type> >& base,
                      const std::vector<std::vector<value_type> >& dbase,
                      int nx_,int ny_,int nz_,value_type fwhm,
                      unsigned int thread_id_,unsigned int thread_count_):
        VG(VG_),VF(VF_),T(T_),dim1(VG_.geometry()),
                B0(base[0]),B1(base[1]),B2(base[2]),
                dB0(dbase[0]),dB1(dbase[1]),dB2(dbase[2]),
                nx(nx_),ny(ny_),nz(nz_),
                thread_id(thread_id_),thread_count(thread_count_)
    {
        nx3 = nx*3;
        nxy3 = nx*ny*3;
        nxy = nx*ny;
        nxyz = nxy*nz;
        nxyz3 = nxyz*3;
        bx3[0] = &dB0[0];
        bx3[1] =  &B0[0];
        bx3[2] =  &B0[0];
        by3[0] =  &B1[0];
        by3[1] = &dB1[0];
        by3[2] =  &B1[0];
        bz3[0] =  &B2[0];
        bz3[1] =  &B2[0];
        bz3[2] = &dB2[0];

        nxy_values.resize(nz);
        dim1_2_values.resize(nz);
        nx_values.resize(ny);
        ny_values.resize(4);
        nz_values.resize(4);
        dim1_1_values.resize(ny);
        dim1_0_values.resize(nx);
        fill_values(nx_values,nx);
        fill_values(ny_values,ny);
        fill_values(nz_values,nz);
        fill_values(nxy_values,nxy);
        fill_values(dim1_2_values,dim1[2]);
        fill_values(dim1_1_values,dim1[1]);
        fill_values(dim1_0_values,dim1[0]);

        /* Because of edge effects from the smoothing, ignore voxels that are too close */
        edgeskip[0] = std::floor(fwhm);
        edgeskip[0] = ((edgeskip[0]<1) ? 0 : edgeskip[0]);
        edgeskip[1] = std::floor(fwhm);
        edgeskip[1] = ((edgeskip[1]<1) ? 0 : edgeskip[1]);
        edgeskip[2] = std::floor(fwhm);
        edgeskip[2] = ((edgeskip[2]<1) ? 0 : edgeskip[2]);


        /* sample about every fwhm/2 */
        samp[0] = std::floor(fwhm/2.0);
        samp[0] = ((samp[0]<1) ? 1 : samp[0]);
        samp[1] = std::floor(fwhm/2.0);
        samp[1] = ((samp[1]<1) ? 1 : samp[1]);
        samp[2] = std::floor(fwhm/2.0);
        samp[2] = ((samp[2]<1) ? 1 : samp[2]);
    }

    void init()
    {
        alphaxy.clear();
        alphaxy.resize((nxy3 + 4)*(nxy3 + 4));
        alphax.clear();
        alphax.resize((nx3+ 4)*(nx3+ 4));
        betaxy.clear();
        betaxy.resize(nxy3 + 4);
        betax.clear();
        betax.resize(nx3+4);
        Tz.clear();
        Tz.resize( nxy3 );
        Ty.clear();
        Ty.resize( nx3 );
        Jz.clear();
        Jz.resize(3);
        Jy.clear();
        Jy.resize(3);
        for (int i1=0; i1<3; i1++)
        {
            Jz[i1].resize(3);
            Jy[i1].resize(3);
            for(int i2=0; i2<3; i2++)
            {
                Jz[i1][i2].resize(nxy);
                Jy[i1][i2].resize(nx);
            }
        }
        dif_alpha.clear();
        dif_alpha.resize((nxyz3+4)*(nxyz3+4));
        dif_beta.clear();
        dif_beta.resize(nxyz3+4);
    }
public:// calculation results to accumulate
    value_type ss,nsamp,ss_deriv[3];
    void accumulate(value_type& ss_,value_type& nsamp_,value_type* ss_deriv_)
    {
        ss_ += ss;
        nsamp_ += nsamp;
        ss_deriv_[0] += ss_deriv[0];
        ss_deriv_[1] += ss_deriv[1];
        ss_deriv_[2] += ss_deriv[2];
    }
public:
    void run(void)
    {
        ss = 0.0;
        nsamp = 0.0;
        std::fill(ss_deriv,ss_deriv+3,0.0);
        //started from slice 1
        for(s0[2]=thread_id+1; s0[2]<dim1[2]; s0[2]+=samp[2]*thread_count) /* For each plane of the template images */
        {
            bfnorm_mrqcof_zero_half(alphaxy,nxy3 + 4);
            std::fill(betaxy.begin(),betaxy.end(),0.0);
            /* build up the deformation field (and derivatives) from it's seperable form */
            {
                const value_type* ptr = &T[0];
                for(int i1=0; i1<3; i1++, ptr += nxyz)
                    for(int x1=0; x1<nxy; x1++)
                    {
                        /* intermediate step in computing nonlinear deformation field */
                        {
                            value_type tmp = 0.0;
                            for(int z1=0; z1<nz; z1++)
                                tmp  += ptr[x1+nxy_values[z1]] * B2[dim1_2_values[z1]+s0[2]];
                            Tz[nxy_values[i1] + x1] = tmp;
                        }
                        /* intermediate step in computing Jacobian of nonlinear deformation field */
                        for(int i2=0; i2<3; i2++)
                        {
                            value_type tmp = 0.0;
                            for(int z1=0; z1<nz; z1++)
                                tmp += ptr[x1+nxy_values[z1]] * bz3[i2][dim1_2_values[z1]+s0[2]];
                            Jz[i2][i1][x1] = tmp;
                        }
                    }
            }


            for(s0[1]=1; s0[1]<dim1[1]; s0[1]+=samp[1]) /* For each row of the template images plane */
            {
                /* build up the deformation field (and derivatives) from it's seperable form */
                {
                    const value_type* ptr=&Tz[0];
                    for(int i1=0; i1<3; i1++, ptr+=nxy)
                    {
                        for(int x1=0; x1<nx; x1++)
                        {
                            /* intermediate step in computing nonlinear deformation field */
                            {
                                value_type tmp = 0.0;
                                for(int y1=0; y1<ny; y1++)
                                    tmp  += ptr[x1+nx_values[y1]] *  B1[dim1_1_values[y1]+s0[1]];
                                Ty[nx_values[i1] + x1] = tmp;
                            }

                            /* intermediate step in computing Jacobian of nonlinear deformation field */
                            for(int i2=0; i2<3; i2++)
                            {
                                value_type tmp = 0.0;
                                for(int y1=0; y1<ny; y1++)
                                    tmp += Jz[i2][i1][x1+nx_values[y1]] * by3[i2][dim1_1_values[y1]+s0[1]];
                                Jy[i2][i1][x1] = tmp;
                            }
                        }
                    }
                }
                bfnorm_mrqcof_zero_half(alphax,nx3+4);
                std::fill(betax.begin(),betax.end(),0.0);

                for(s0[0]=1; s0[0]<dim1[0]; s0[0]+=samp[0]) /* For each pixel in the row */
                {
                    /* nonlinear deformation of the template space, followed by the affine transform */
                    const value_type* ptr = &Ty[0];
                    value_type J[3][3];
                    value_type trans[3];
                    for(int i1=0; i1<3; i1++, ptr += nx)
                    {
                        /* compute nonlinear deformation field */
                        {
                            value_type tmp = 0.0;
                            for(int x1=0; x1<nx; x1++)
                                tmp  += ptr[x1] * B0[dim1_0_values[x1]+s0[0]];
                            trans[i1] = tmp + s0[i1];
                        }
                        /* compute Jacobian of nonlinear deformation field */
                        for(int i2=0; i2<3; i2++)
                        {
                            value_type tmp = (i1 == i2) ? 1.0:0.0;
                            for(int x1=0; x1<nx; x1++)
                                tmp += Jy[i2][i1][x1] * bx3[i2][dim1_0_values[x1]+s0[0]];
                            J[i2][i1] = tmp;
                        }
                    }

                    value_type s2[3];
                    s2[0] = trans[0];
                    s2[1] = trans[1];
                    s2[2] = trans[2];

                    /* is the transformed position in range? */
                    if (	s2[0]>=1+edgeskip[0] && s2[0]< VF.width()-edgeskip[0] &&
                            s2[1]>=1+edgeskip[1] && s2[1]< VF.height()-edgeskip[1] &&
                            s2[2]>=1+edgeskip[2] && s2[2]< VF.depth()-edgeskip[2] )
                    {
                        std::vector<value_type> dvdt( nx3    + 4);
                        value_type f, df[3], dv, dvds0[3];
                        value_type wtf, wtg, wt;
                        value_type s0d[3];
                        s0d[0]=s0[0];
                        s0d[1]=s0[1];
                        s0d[2]=s0[2];
                        /* rate of change of voxel with respect to change in parameters */
                        f = resample_d(VF,df[0],df[1],df[2],s2[0],s2[1],s2[2]);

                        wtg = 1.0;
                        wtf = 1.0;

                        if (wtf && wtg) wt = sqrt(1.0 /(1.0/wtf + 1.0/wtg));
                        else wt = 0.0;

                        /* nonlinear transform the gradients to the same space as the template */
                        image::vector_rotation(df,dvds0,&(J[0][0]),image::vdim<3>());

                        dv = f;
                        {
                            value_type g, dg[3], tmp;
                            /* pointer to scales for each of the template images */
                            const value_type* scal = &T[nxyz3];

                            g = resample_d(VG,dg[0],dg[1],dg[2],s0d[0],s0d[1],s0d[2]);

                            /* linear combination of image and image modulated by constant
                               gradients in x, y and z */
                            dvdt[nx3] = wt*g;
                            dvdt[1+nx3] = dvdt[nx3]*s2[0];
                            dvdt[2+nx3] = dvdt[nx3]*s2[1];
                            dvdt[3+nx3] = dvdt[nx3]*s2[2];

                            tmp = scal[0] + s2[0]*scal[1] + s2[1]*scal[2] + s2[2]*scal[3];

                            dv       -= tmp*g;
                            dvds0[0] -= tmp*dg[0];
                            dvds0[1] -= tmp*dg[1];
                            dvds0[2] -= tmp*dg[2];
                        }

                        for(int i1=0; i1<3; i1++)
                        {
                            value_type tmp = -wt*df[i1];
                            for(int x1=0; x1<nx; x1++)
                                dvdt[i1*nx+x1] = tmp * B0[dim1_0_values[x1]+s0[0]];
                        }

                        /* cf Numerical Recipies "mrqcof.c" routine */
                        int m1 = nx3+4;
                        for(int x1=0; x1<m1; x1++)
                        {
                            for (int x2=0; x2<=x1; x2++)
                                alphax[m1*x1+x2] += dvdt[x1]*dvdt[x2];
                            betax[x1] += dvdt[x1]*dv*wt;
                        }

                        /* sum of squares */
                        wt          *= wt;
                        nsamp       += wt;
                        ss          += wt*dv*dv;
                        ss_deriv[0] += wt*dvds0[0]*dvds0[0];
                        ss_deriv[1] += wt*dvds0[1]*dvds0[1];
                        ss_deriv[2] += wt*dvds0[2]*dvds0[2];
                    }
                }

                int m1 = nxy3+4;
                int m2 = nx3+4;

                /* Kronecker tensor products */
                for(int y1=0; y1<ny; y1++)
                {
                    value_type wt1 = B1[dim1_1_values[y1]+s0[1]];

                    for(int i1=0; i1<3; i1++)	/* loop over deformations in x, y and z */
                    {
                        /* spatial-spatial covariances */
                        for(int i2=0; i2<=i1; i2++)	/* symmetric matrixes - so only work on half */
                        {
                            for(int y2=0; y2<=y1; y2++)
                            {
                                /* Kronecker tensor products with B1'*B1 */
                                value_type wt2 = wt1 * B1[dim1_1_values[y2]+s0[1]];

                                value_type* ptr1 = &alphaxy[nx*(m1*(ny_values[i1] + y1) + ny_values[i2] + y2)];
                                value_type* ptr2 = &alphax[nx*(m2*i1 + i2)];

                                for(int x1=0; x1<nx; x1++)
                                {
                                    image::vec::axpy(ptr1,ptr1+x1+1,wt2,ptr2);
                                    ptr1 += m1;
                                    ptr2 += m2;
                                }
                            }
                        }

                        /* spatial-intensity covariances */
                        value_type* ptr1 = &alphaxy[nx*(m1*ny_values[3] + ny_values[i1] + y1)];
                        value_type* ptr2 = &alphax[nx*(m2*3 + i1)];
                        for(int x1=0; x1<4; x1++)
                        {
                            image::vec::axpy(ptr1,ptr1+nx,wt1,ptr2);
                            ptr1 += m1;
                            ptr2 += m2;
                        }

                        /* spatial component of beta */
                        for(int x1=0; x1<nx; x1++)
                            betaxy[x1+nx*(ny_values[i1] + y1)] += wt1 * betax[x1 + nx_values[i1]];
                    }
                }
                value_type* ptr1 = &alphaxy[nx*((m1+1)*ny_values[3])];
                value_type* ptr2 = &alphax[nx*(m2*3 + 3)];
                for(int x1=0; x1<4; x1++)
                {
                    image::vec::add(ptr1,ptr1+x1+1,ptr2);
                    ptr1 += m1;
                    ptr2 += m2;
                    betaxy[nxy3 + x1] += betax[nx3 + x1];
                }
            }

            int m1 = nxyz3+4;
            int m2 = nxy3+4;

            /* Kronecker tensor products */
            for(int z1=0; z1<nz; z1++)
            {
                value_type wt1 = B2[dim1_2_values[z1]+s0[2]];

                for(int i1=0; i1<3; i1++)	/* loop over deformations in x, y and z */
                {
                    /* spatial-spatial covariances */
                    for(int i2=0; i2<=i1; i2++)	/* symmetric matrixes - so only work on half */
                    {
                        for(int z2=0; z2<=z1; z2++)
                        {
                            /* Kronecker tensor products with B2'*B2 */
                            value_type wt2 = wt1 * B2[dim1_2_values[z2]+s0[2]];

                            value_type* ptr1 = &dif_alpha[nxy*(m1*(nz_values[i1] + z1) + nz_values[i2] + z2)];
                            value_type* ptr2 = &alphaxy[nxy*(m2*i1 + i2)];
                            for(int y1=0; y1<nxy; y1++)
                            {
                                image::vec::axpy(ptr1,ptr1+y1+1,wt2,ptr2);
                                ptr1 += m1;
                                ptr2 += m2;
                            }
                        }
                    }
                    /* spatial-intensity covariances */
                    value_type* ptr1 = &dif_alpha[nxy*(m1*nz_values[3] + nz_values[i1] + z1)];
                    value_type* ptr2 = &alphaxy[nxy*(m2*3 + i1)];
                    for(int y1=0; y1<4; y1++)
                    {
                        image::vec::axpy(ptr1,ptr1+nxy,wt1,ptr2);
                        ptr1 += m1;
                        ptr2 += m2;
                    }
                    /* spatial component of beta */
                    for(int y1=0; y1<nxy; y1++)
                        dif_beta[y1 + nxy*(nz_values[i1] + z1)] += wt1 * betaxy[y1 + nxy_values[i1]];
                }
            }

            value_type* ptr1 = &dif_alpha[nxy*(m1+1)*nz_values[3]];
            value_type* ptr2 = &alphaxy[nxy*(m2*3 + 3)];
            for(int y1=0; y1<4; y1++)
            {
                image::vec::add(ptr1,ptr1+y1+1,ptr2);
                ptr1 += m1;
                ptr2 += m2;
                /* intensity component of beta */
                dif_beta[nxyz3 + y1] += betaxy[nxy3 + y1];
            }

        }
    }
};

template<class image_type,class value_type>
class bfnorm_mrqcof {
private:
    const image_type& VG;
    const image_type& VF;
    const std::vector<std::vector<value_type> >& base;
    const std::vector<std::vector<value_type> >& dbase;

private:
    int nx,ny,nz,nxy,nxyz,nx3,nxy3,nxyz3;

private:
    int samp[3];
    value_type fwhm,fwhm2;
private: // slice temporary data
    std::vector<std::shared_ptr<bfnorm_slice_data<image_type,value_type> > > data;
public:
    std::vector<value_type> IC0;
    std::vector<value_type> alpha,beta;
public:
    std::vector<value_type>& T;
public:

    bfnorm_mrqcof(const image_type& VG_,const image_type& VF_,bfnorm_mapping<value_type,3>& mapping,unsigned int thread_count):
        VG(VG_),VF(VF_),
        base(mapping.bas),dbase(mapping.dbas),T(mapping.T)
    {
        const value_type stabilise = 8,reg = 1.0;
        fwhm = 1.0;// sampling rate = every voxel
        fwhm2= 30;
        nx = mapping.k_base[0];
        ny = mapping.k_base[1];
        nz = mapping.k_base[2];
        nxy = nx*ny;
        nxyz = nxy*nz;
        nx3 = nx*3;
        nxy3 = nxy*3;
        nxyz3 = nxyz*3;

        /* sample about every fwhm/2 */
        samp[0] = std::floor(fwhm/2.0);
        samp[0] = ((samp[0]<1) ? 1 : samp[0]);
        samp[1] = std::floor(fwhm/2.0);
        samp[1] = ((samp[1]<1) ? 1 : samp[1]);
        samp[2] = std::floor(fwhm/2.0);
        samp[2] = ((samp[2]<1) ? 1 : samp[2]);

        alpha.resize((nxyz3+4)*(nxyz3+4)); // plhs[0]
        beta.resize(nxyz3+4);
        IC0.resize(3*mapping.k_base.size()+4);
        const int dim = image_type::dimension;

        {
            std::vector<std::vector<value_type> > kxyz(3);
            for(int d = 0; d < dim; ++d)
            {
                kxyz[d].resize(mapping.k_base[d]);
                for(int i = 0; i < kxyz[d].size(); ++i)
                {
                    kxyz[d][i] = 3.14159265358979323846*(value_type)i/mapping.VGgeo[d];
                    kxyz[d][i] *= kxyz[d][i];
                }
            }
            int ICO_ = mapping.k_base.size();
            value_type IC0_co = reg*std::pow(stabilise,6);
            for(image::pixel_index<dim> pos(mapping.k_base);pos < mapping.k_base.size();++pos)
            {
                int m = pos[0];
                int j = pos[1];
                int i = pos[2];
                int index = pos.index();
                IC0[index] = kxyz[2][i]*kxyz[2][i]+kxyz[1][j]*kxyz[1][j]+kxyz[0][m]*kxyz[0][m]+
                             2*kxyz[0][m]*kxyz[1][j]+2*kxyz[0][m]*kxyz[2][i]+2*kxyz[1][j]*kxyz[2][i];
                IC0[index] *= IC0_co;
                IC0[index+ICO_] = IC0[index];
                IC0[index+ICO_+ICO_] = IC0[index];
            }
        }
        try{
            for(unsigned int index = 0;index < thread_count;++index)
            {
                auto ptr = std::make_shared<bfnorm_slice_data<image_type,value_type> >
                                        (VG,VF,T,base,dbase,nx,ny,nz,fwhm,index,thread_count);
                ptr->init();
                data.push_back(ptr);
            }

        }
        catch(...)
        {
            thread_count = data.size();
            for(unsigned int index = 0;index < thread_count;++index)
                data[index]->thread_count = thread_count;
            if(thread_count == 0)
                throw std::bad_alloc();
        }

    }

    template<class terminated_type>
    void optimize(const terminated_type& terminated)
    {
        float prev_ss = std::numeric_limits<float>::max();
        for(int iteration = 0; iteration < 64 && !terminated; ++iteration)
        {
            // zero alpha and beta
            for(unsigned int index = 0;index < data.size();++index)
                data[index]->init();
            std::fill(alpha.begin(),alpha.end(),0.0);
            std::fill(beta.begin(),beta.end(),0.0);

            // calculate difference in alpha and beta
            {
                std::vector<std::shared_ptr<std::future<void> > > threads;
                for (unsigned int index = 1;index < data.size();++index)
                    threads.push_back(std::make_shared<std::future<void> >(std::async(std::launch::async,
                        [this,index](){data[index]->run();})));
                data[0]->run();
                for(int i = 0;i < threads.size();++i)
                    threads[i]->wait();
            }

            // accumulate alpha beta
            for(unsigned int index = 0;index < data.size();++index)
            {
                image::add(alpha,data[index]->dif_alpha);
                image::add(beta,data[index]->dif_beta);
            }


            value_type ss = 0.0,nsamp = 0.0,ss_deriv[3];
            std::fill(ss_deriv,ss_deriv+3,0.0);
            for(unsigned int index = 0;index < data.size();++index)
                data[index]->accumulate(ss,nsamp,ss_deriv);

            // update alpha
            int m1 = nxyz3+4;
            for(int i1=0; i1<3; i1++)
            {
                value_type *ptrz, *ptry, *ptrx;
                for(int i2=0; i2<=i1; i2++)
                {
                    ptrz = &alpha[nxyz*(m1*i1 + i2)];
                    for(int z1=0; z1<nz; z1++)
                        for(int z2=0; z2<=z1; z2++)
                        {
                            ptry = ptrz + nxy*(m1*z1 + z2);
                            for(int y1=0; y1<ny; y1++)
                                for (int y2=0; y2<=y1; y2++)
                                {
                                    ptrx = ptry + nx*(m1*y1 + y2);
                                    for(int x1=0; x1<nx; x1++)
                                        for(int x2=0; x2<x1; x2++)
                                            ptrx[m1*x2+x1] = ptrx[m1*x1+x2];
                                }
                            for(int x1=0; x1<nxy; x1++)
                                for (int x2=0; x2<x1; x2++)
                                    ptry[m1*x2+x1] = ptry[m1*x1+x2];
                        }
                    for(int x1=0; x1<nxyz; x1++)
                        for (int x2=0; x2<x1; x2++)
                            ptrz[m1*x2+x1] = ptrz[m1*x1+x2];
                }
            }
            for(int x1=0; x1<nxyz3+4; x1++)
                for (int x2=0; x2<x1; x2++)
                    alpha[m1*x2+x1] = alpha[m1*x1+x2];

            //

            value_type fw = ((1.0/std::sqrt(2.0*ss_deriv[0]/ss))*sqrt(8.0*std::log(2.0)) +
                             (1.0/std::sqrt(2.0*ss_deriv[1]/ss))*sqrt(8.0*std::log(2.0)) +
                             (1.0/std::sqrt(2.0*ss_deriv[2]/ss))*sqrt(8.0*std::log(2.0)))/3.0;


            if (fw<fwhm2)
                fwhm2 = fw;
            if (fwhm2<fwhm)
                fwhm2 = fwhm;

            ss /= (std::min(samp[0]/(fwhm2*1.0645),1.0) *
                   std::min(samp[1]/(fwhm2*1.0645),1.0) *
                   std::min(samp[2]/(fwhm2*1.0645),1.0)) * (nsamp - (nxyz3 + 4));

            //std::cout << "FWHM = " << fw << " Var = " << ss << std::endl;
            if(iteration > 10 && ss > prev_ss)
                return;

            prev_ss = ss;
            fwhm2 = std::min(fw,fwhm2);

            image::divide_constant(alpha.begin(),alpha.end(), ss);
            image::divide_constant(beta.begin(),beta.end(), ss);
            {
                // beta = beta + alpha*T;
                std::vector<value_type> alphaT(T.size());
                image::mat::vector_product(alpha.begin(),T.begin(),alphaT.begin(),image::dyndim(T.size(),T.size()));
                image::add(beta.begin(),beta.end(),alphaT.begin());
            }


            //Alpha + IC0*scal
            value_type pvar = std::numeric_limits<value_type>::max();
            if(ss > pvar)
            {
                value_type scal = pvar/ss;
                for(int i = 0,j = 0; i < alpha.size(); i+=T.size()+1,++j)
                    alpha[i] += IC0[j]*scal;
            }
            else
                for(int i = 0,j = 0; i < alpha.size(); i+=T.size()+1,++j)
                    alpha[i] += IC0[j];

            // solve T = (Alpha + IC0*scal)\(Alpha*T + Beta);

            /*
            for(unsigned int i = 0;i < 20;++i)
            {

                if(!image::mat::jacobi_solve(&*alpha.begin(),&*beta.begin(),&*T.begin(),image::dyndim(T.size(),T.size())))
                {
                    // use LL decomposition instead
                    std::vector<value_type> piv(T.size());
                    image::mat::ll_decomposition(&*alpha.begin(),&*piv.begin(),image::dyndim(T.size(),T.size()));
                    image::mat::ll_solve(&*alpha.begin(),&*piv.begin(),&*beta.begin(),&*T.begin(),image::dyndim(T.size(),T.size()));
                    break;
                }
            }*/

            // solve T = (Alpha + IC0*scal)\(Alpha*T + Beta);
            // alpha is a diagonal dominant matrix, which can use Jacobi method to solve
            //image::mat::jacobi_solve(&*alpha.begin(),&*beta.begin(),&*T.begin(),image::dyndim(T.size(),T.size()));
            unsigned int size = T.size();
            for(unsigned int iter = 0;iter < 40;++iter)
            {
                const value_type* A_row = &*(alpha.end() - size);
                // going bacward because because alpha values is incremental
                for(int i = size-1;i >= 0;--i,A_row -= size)
                {
                    value_type new_T_value = beta[i];
                    value_type scale = 0.0;
                    for(unsigned int j = 0;j < size;++j)
                        if(j != i)
                            new_T_value -= A_row[j]*T[j];
                        else
                            scale = A_row[j];
                    if(scale == 0.0)
                        return;
                    // stablize using weighted jacobi method
                    T[i] = new_T_value/scale/1.5 + T[i]/3;
                }
            }
        }
    }
};

template<class ImageType,class value_type,class terminator_type>
void bfnorm(bfnorm_mapping<value_type>& mapping,
            const ImageType& VG,
            const ImageType& VFF,unsigned int thread_count,terminator_type& terminated)
{
    bfnorm_mrqcof<ImageType,value_type> bf_optimize(VG,VFF,mapping,thread_count);
    // image::reg::bfnorm(VG,VFF,*mni.get(),terminated);
    bf_optimize.optimize(terminated);
}



template<class value_type,class from_type,class matrix_type>
void bfnorm_get_jacobian(const bfnorm_mapping<value_type>& mapping,const from_type& from,matrix_type Jbet)
{
    int nx = mapping.k_base[0];
    int ny = mapping.k_base[1];
    int nz = mapping.k_base[2];
    int nyz =ny*nz;
    int nxyz = mapping.k_base.size();

    const std::vector<value_type>& T = mapping.T;

    std::vector<value_type> bx_(nx),by_(ny),bz_(nz),dbx_(nx),dby_(ny),dbz_(nz),temp_(nyz),temp2_(nz);
    value_type *bx = &bx_[0];
    value_type *by = &by_[0];
    value_type *bz = &bz_[0];
    value_type *dbx = &dbx_[0];
    value_type *dby = &dby_[0];
    value_type *dbz = &dbz_[0];
    value_type *temp = &temp_[0];
    value_type *temp2 = &temp2_[0];

    for(unsigned int k = 0,index = from[0]; k < nx; ++k,index += mapping.VGgeo[0])
    {
        bx[k] = mapping.bas[0][index];
        dbx[k] = mapping.dbas[0][index];
    }
    for(unsigned int k = 0,index = from[1]; k < ny; ++k,index += mapping.VGgeo[1])
    {
        by[k] = mapping.bas[1][index];
        dby[k] = mapping.dbas[1][index];
    }
    for(unsigned int k = 0,index = from[2]; k < nz; ++k,index += mapping.VGgeo[2])
    {
        bz[k] = mapping.bas[2][index];
        dbz[k] = mapping.dbas[2][index];
    }
    image::dyndim dyz_x(nyz,nx),dz_y(nz,ny),dx_1(nx,1),dy_1(ny,1);


    // f(x)/dx
    image::mat::product(T.begin(),dbx,temp,dyz_x,dx_1);
    image::mat::product(temp,by,temp2,dz_y,dy_1);
    Jbet[0] = 1 + image::vec::dot(bz,bz+nz,temp2);
    // f(x)/dy
    image::mat::product(T.begin(),bx,temp,dyz_x,dx_1);
    image::mat::product(temp,dby,temp2,dz_y,dy_1);
    Jbet[1] = image::vec::dot(bz,bz+nz,temp2);
    // f(x)/dz
    image::mat::product(T.begin(),bx,temp,dyz_x,dx_1);
    image::mat::product(temp,by,temp2,dz_y,dy_1);
    Jbet[2] = image::vec::dot(dbz,dbz+nz,temp2);

    // f(y)/dx
    image::mat::product(T.begin()+nxyz,dbx,temp,dyz_x,dx_1);
    image::mat::product(temp,by,temp2,dz_y,dy_1);
    Jbet[3] = image::vec::dot(bz,bz+nz,temp2);
    // f(y)/dy
    image::mat::product(T.begin()+nxyz,bx,temp,dyz_x,dx_1);
    image::mat::product(temp,dby,temp2,dz_y,dy_1);
    Jbet[4] = 1 + image::vec::dot(bz,bz+nz,temp2);
    // f(y)/dz
    image::mat::product(T.begin()+nxyz,bx,temp,dyz_x,dx_1);
    image::mat::product(temp,by,temp2,dz_y,dy_1);
    Jbet[5] = image::vec::dot(dbz,dbz+nz,temp2);

    // f(z)/dx
    image::mat::product(T.begin()+(nxyz << 1),dbx,temp,dyz_x,dx_1);
    image::mat::product(temp,by,temp2,dz_y,dy_1);
    Jbet[6] = image::vec::dot(bz,bz+nz,temp2);
    // f(z)/dy
    image::mat::product(T.begin()+(nxyz << 1),bx,temp,dyz_x,dx_1);
    image::mat::product(temp,dby,temp2,dz_y,dy_1);
    Jbet[7] = image::vec::dot(bz,bz+nz,temp2);
    // f(z)/dz
    image::mat::product(T.begin()+(nxyz << 1),bx,temp,dyz_x,dx_1);
    image::mat::product(temp,by,temp2,dz_y,dy_1);
    Jbet[8] = 1 + image::vec::dot(dbz,dbz+nz,temp2);

    //image::mat::product(affine_rotation,Jbet,M,math::dim<3,3>(),math::dim<3,3>());
}

}// reg
}// image

#endif
