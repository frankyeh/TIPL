#ifndef BFNORM_HPP
#define BFNORM_HPP

#include <cmath>
#include <vector>
#include <future>
#include "../numerical/matrix.hpp"
#include "../numerical/numerical.hpp"
namespace tipl {

namespace reg {

template<typename ImageType,typename value_type>
value_type resample_d(const ImageType& vol,value_type& gradx,value_type& grady,value_type& gradz,value_type x,value_type y,value_type z)
{
    const value_type TINY = 5e-2f;
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


template<typename value_type,int dim = 3>
class bfnorm_mapping {
public:
    tipl::shape<dim> VGgeo;
    tipl::shape<dim> k_base;
    std::vector<value_type> T;
    std::vector<std::vector<value_type> > bas,dbas;
public:
    bfnorm_mapping(const tipl::shape<3>& geo_,const tipl::shape<dim>& k_base_):VGgeo(geo_),k_base(k_base_)
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
            std::fill(bas[d].begin(),bas[d].begin()+VGgeo[d],stabilise/std::sqrt((value_type)VGgeo[d]));
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

    template<typename rhs_type>
    void get_displacement(const tipl::pixel_index<3>& from,rhs_type& to) const
    {
        return get_displacement(tipl::vector<3,int>(from[0],from[1],from[2]),to);
    }
    template<typename rhs_type>
    void operator()(const tipl::pixel_index<3>& from,rhs_type& to) const
    {
        return (*this)(tipl::vector<3,int>(from[0],from[1],from[2]),to);
    }

    template<typename rhs_type>
    void get_displacement(const tipl::vector<3,int>& from,rhs_type& to) const
    {
        if(!VGgeo.is_valid(from))
        {
            to = rhs_type();
            return;
        }
        int nx = k_base[0];
        int ny = k_base[1];
        int nz = k_base[2];
        int nyz =ny*nz;
        int nxyz = k_base.size();

        tipl::shape<2> dyz_x(nyz,nx),dz_y(nz,ny),dx_1(nx,1),dy_1(ny,1);
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

        tipl::mat::product(T.begin(),bx,temp,dyz_x,dx_1);
        tipl::mat::product(temp,by,temp2,dz_y,dy_1);
        to[0] = tipl::vec::dot(bz,bz+nz,temp2);

        tipl::mat::product(T.begin()+nxyz,bx,temp,dyz_x,dx_1);
        tipl::mat::product(temp,by,temp2,dz_y,dy_1);
        to[1] = tipl::vec::dot(bz,bz+nz,temp2);

        tipl::mat::product(T.begin()+(nxyz << 1),bx,temp,dyz_x,dx_1);
        tipl::mat::product(temp,by,temp2,dz_y,dy_1);
        to[2] = tipl::vec::dot(bz,bz+nz,temp2);
    }
    template<typename rhs_type>
    void operator()(const tipl::vector<3,int>& from,rhs_type& to) const
    {
        get_displacement(from,to);
        to += from;
    }
};

template<typename value_type>
void fill_values(std::vector<value_type>& values,value_type step)
{
    value_type value = 0;
    for(unsigned int index = 0; index < values.size(); ++index,value += step)
        values[index] = value;
}
template<typename parameter_type>
void bfnorm_mrqcof_zero_half(std::vector<parameter_type>& alpha,int m1)
{
    for (int x1=0; x1<m1; x1++)
    {
        for (int x2=0; x2<=x1; x2++)
            alpha[m1*x1+x2] = 0.0;
    }
}

template<typename image_type,typename value_type>
class bfnorm_mrqcof {
private:
    const image_type& VG;
    const image_type& VF;
    const std::vector<std::vector<value_type> >& base;
    const std::vector<std::vector<value_type> >& dbase;
private:
    int nx,ny,nz;
    int nxy,nxyz,nx3,nxy3,nxyz3;
    std::vector<int> nxy_values,dim1_2_values,nx_values,ny_values,nz_values,dim1_1_values,dim1_0_values;
    int edgeskip[3];
    tipl::shape<3> dim1;
private:
    const std::vector<value_type>& B0;
    const std::vector<value_type>& B1;
    const std::vector<value_type>& B2;
    const std::vector<value_type>& dB0;
    const std::vector<value_type>& dB1;
    const std::vector<value_type>& dB2;
    const value_type *bz3[3], *by3[3], *bx3[3];
public:
    int samp[3];

public:
    value_type fwhm,fwhm2;
public:
    std::vector<value_type> IC0;
    std::vector<value_type> alpha,beta;
public:
    std::vector<value_type>& T;
public:

    bfnorm_mrqcof(const image_type& VG_,const image_type& VF_,
                  bfnorm_mapping<value_type,3>& mapping):
        VG(VG_),VF(VF_),
        base(mapping.bas),dbase(mapping.dbas),T(mapping.T),
        dim1(VG_.shape()),
        B0(mapping.bas[0]),B1(mapping.bas[1]),B2(mapping.bas[2]),
        dB0(mapping.dbas[0]),dB1(mapping.dbas[1]),dB2(mapping.dbas[2])

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
            for(tipl::pixel_index<dim> pos(mapping.k_base);pos < mapping.k_base.size();++pos)
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
        fill_values(dim1_2_values,int(dim1[2]));
        fill_values(dim1_1_values,int(dim1[1]));
        fill_values(dim1_0_values,int(dim1[0]));

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

public:
    template<typename terminated_type>
    void optimize(const terminated_type& terminated,unsigned int thread_count)
    {
        value_type prev_ss = std::numeric_limits<value_type>::max();
        for(int iteration = 0; iteration < 64 && !terminated; ++iteration)
        {
            // zero alpha and beta


            std::fill(alpha.begin(),alpha.end(),0.0);
            std::fill(beta.begin(),beta.end(),0.0);

            //started from slice 1
            // /* For each plane of the template images */
            std::vector<int> s0_list;
            for(int i=1; i<dim1[2]; i+=samp[2])
                s0_list.push_back(i);

            std::mutex alpha_beta_lock,ss_lock;
            value_type ss_ = 0.0,nsamp_ = 0.0,ss_deriv_[3] = {0.0,0.0,0.0};

            tipl::par_for(s0_list.size(),[&](int i)
            {
                value_type ss = 0.0,nsamp = 0.0,ss_deriv[3] = {0.0,0.0,0.0};
                int s0[3] = {0,0,0};
                s0[2] = s0_list[i];

                std::vector<value_type> Tz( nxy3 ),Ty( nx3 );
                std::vector<std::vector<std::vector<value_type> > > Jz(3),Jy(3);
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
                std::vector<value_type> betaxy(nxy3 + 4),alphaxy((nxy3 + 4)*(nxy3 + 4));
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
                    std::vector<value_type> betax(nx3+4),alphax((nx3+ 4)*(nx3+ 4));
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
                            tipl::vector_rotation(df,dvds0,&(J[0][0]),tipl::vdim<3>());

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

                            for(int i1=0,i1_nx=0; i1<3; i1++,i1_nx+=nx)
                            {
                                value_type tmp = -wt*df[i1];
                                for(int x1=0; x1<nx; x1++)
                                    dvdt[i1_nx+x1] = tmp * B0[dim1_0_values[x1]+s0[0]];
                            }

                            /* cf Numerical Recipies "mrqcof.c" routine */
                            int m1 = nx3+4;
                            value_type dv_wt = dv*wt;
                            for(int x1=0,m1_x1 = 0; x1<m1; x1++,m1_x1 += m1)
                            {
                                for (int x2=0; x2<=x1; x2++)
                                    alphax[m1_x1+x2] += dvdt[x1]*dvdt[x2];
                                betax[x1] += dvdt[x1]*dv_wt;
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
                                        tipl::vec::axpy(ptr1,ptr1+x1+1,wt2,ptr2);
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
                                tipl::vec::axpy(ptr1,ptr1+nx,wt1,ptr2);
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
                        tipl::vec::add(ptr1,ptr1+x1+1,ptr2);
                        ptr1 += m1;
                        ptr2 += m2;
                        betaxy[nxy3 + x1] += betax[nx3 + x1];
                    }
                }

                int m1 = nxyz3+4;
                int m2 = nxy3+4;

                /* Kronecker tensor products */
                std::lock_guard<std::mutex> lock(alpha_beta_lock);

                tipl::par_for(nz,[&](int z1)
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

                                value_type* ptr1 = &alpha[nxy*(m1*(nz_values[i1] + z1) + nz_values[i2] + z2)];
                                const value_type* ptr2 = &alphaxy[nxy*(m2*i1 + i2)];
                                for(int y1=0; y1<nxy; y1++)
                                {
                                    tipl::vec::axpy(ptr1,ptr1+y1+1,wt2,ptr2);
                                    ptr1 += m1;
                                    ptr2 += m2;
                                }
                            }
                        }
                        /* spatial-intensity covariances */
                        value_type* ptr1 = &alpha[nxy*(m1*nz_values[3] + nz_values[i1] + z1)];
                        const value_type* ptr2 = &alphaxy[nxy*(m2*3 + i1)];
                        for(int y1=0; y1<4; y1++)
                        {
                            tipl::vec::axpy(ptr1,ptr1+nxy,wt1,ptr2);
                            ptr1 += m1;
                            ptr2 += m2;
                        }
                        /* spatial component of beta */
                        value_type* ptr3 = &beta[nxy*(nz_values[i1] + z1)];
                        tipl::vec::axpy(ptr3,ptr3+nxy,wt1,&betaxy[nxy_values[i1]]);
                    }
                },thread_count);

                value_type* ptr1 = &alpha[nxy*(m1+1)*nz_values[3]];
                const value_type* ptr2 = &alphaxy[nxy*(m2*3 + 3)];
                for(int y1=0; y1<4; y1++)
                {
                    tipl::vec::add(ptr1,ptr1+y1+1,ptr2);
                    ptr1 += m1;
                    ptr2 += m2;
                    /* intensity component of beta */
                    beta[nxyz3 + y1] += betaxy[nxy3 + y1];
                }

                std::lock_guard<std::mutex> lock2(ss_lock);
                nsamp_       += nsamp;
                ss_          += ss;
                ss_deriv_[0] += ss_deriv[0];
                ss_deriv_[1] += ss_deriv[1];
                ss_deriv_[2] += ss_deriv[2];
            },thread_count);


            // update alpha
            int m1 = nxyz3+4;
            for(int i1=0; i1<3; i1++)
            {
                tipl::par_for(i1+1,[&](int i2)
                {
                    value_type *ptrz, *ptry, *ptrx;
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
                },thread_count);
            }

            tipl::par_for(nxyz3+4,[&](int x1)
            {
                for (int x2=0; x2<x1; x2++)
                    alpha[m1*x2+x1] = alpha[m1*x1+x2];
            },thread_count);
            //

            value_type fw = ((1.0/std::sqrt(2.0*ss_deriv_[0]/ss_))*sqrt(8.0*std::log(2.0)) +
                             (1.0/std::sqrt(2.0*ss_deriv_[1]/ss_))*sqrt(8.0*std::log(2.0)) +
                             (1.0/std::sqrt(2.0*ss_deriv_[2]/ss_))*sqrt(8.0*std::log(2.0)))/3.0;


            if (fw<fwhm2)
                fwhm2 = fw;
            if (fwhm2<fwhm)
                fwhm2 = fwhm;

            ss_ /= (std::min(samp[0]/(fwhm2*1.0645),1.0) *
                   std::min(samp[1]/(fwhm2*1.0645),1.0) *
                   std::min(samp[2]/(fwhm2*1.0645),1.0)) * (nsamp_ - (nxyz3 + 4));

            //std::cout << "FWHM = " << fw << " Var = " << ss_ << std::endl;
            if(iteration > 10 && ss_ > prev_ss)
                return;

            prev_ss = ss_;
            fwhm2 = std::min(fw,fwhm2);


            tipl::divide_constant(alpha, ss_);
            tipl::divide_constant(beta, ss_);

            tipl::par_for(beta.size(),[&](int i)
            {
                unsigned int pos = i*T.size();
                beta[i] += tipl::vec::dot(alpha.begin()+pos,alpha.begin()+pos+T.size(),T.begin());
            },thread_count);


            //Alpha + IC0*scal
            value_type pvar = std::numeric_limits<value_type>::max();
            if(ss_ > pvar)
            {
                value_type scal = pvar/ss_;
                for(int i = 0,j = 0; i < alpha.size(); i+=T.size()+1,++j)
                    alpha[i] += IC0[j]*scal;
            }
            else
                for(int i = 0,j = 0; i < alpha.size(); i+=T.size()+1,++j)
                    alpha[i] += IC0[j];

            // solve T = (Alpha + IC0*scal)\(Alpha*T + Beta);

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

template<typename ImageType,typename value_type,typename terminator_type>
void bfnorm(bfnorm_mapping<value_type>& mapping,
            const ImageType& VG,
            const ImageType& VFF,terminator_type& terminated,unsigned int thread_count)
{
    bfnorm_mrqcof<ImageType,value_type> bf_optimize(VG,VFF,mapping);
    // tipl::reg::bfnorm(VG,VFF,*mni.get(),terminated);
    bf_optimize.optimize(terminated,thread_count);
}



template<typename value_type,typename from_type,typename matrix_type>
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
    tipl::shape<2> dyz_x(nyz,nx),dz_y(nz,ny),dx_1(nx,1),dy_1(ny,1);


    // f(x)/dx
    tipl::mat::product(T.begin(),dbx,temp,dyz_x,dx_1);
    tipl::mat::product(temp,by,temp2,dz_y,dy_1);
    Jbet[0] = 1 + tipl::vec::dot(bz,bz+nz,temp2);
    // f(x)/dy
    tipl::mat::product(T.begin(),bx,temp,dyz_x,dx_1);
    tipl::mat::product(temp,dby,temp2,dz_y,dy_1);
    Jbet[1] = tipl::vec::dot(bz,bz+nz,temp2);
    // f(x)/dz
    tipl::mat::product(T.begin(),bx,temp,dyz_x,dx_1);
    tipl::mat::product(temp,by,temp2,dz_y,dy_1);
    Jbet[2] = tipl::vec::dot(dbz,dbz+nz,temp2);

    // f(y)/dx
    tipl::mat::product(T.begin()+nxyz,dbx,temp,dyz_x,dx_1);
    tipl::mat::product(temp,by,temp2,dz_y,dy_1);
    Jbet[3] = tipl::vec::dot(bz,bz+nz,temp2);
    // f(y)/dy
    tipl::mat::product(T.begin()+nxyz,bx,temp,dyz_x,dx_1);
    tipl::mat::product(temp,dby,temp2,dz_y,dy_1);
    Jbet[4] = 1 + tipl::vec::dot(bz,bz+nz,temp2);
    // f(y)/dz
    tipl::mat::product(T.begin()+nxyz,bx,temp,dyz_x,dx_1);
    tipl::mat::product(temp,by,temp2,dz_y,dy_1);
    Jbet[5] = tipl::vec::dot(dbz,dbz+nz,temp2);

    // f(z)/dx
    tipl::mat::product(T.begin()+(nxyz << 1),dbx,temp,dyz_x,dx_1);
    tipl::mat::product(temp,by,temp2,dz_y,dy_1);
    Jbet[6] = tipl::vec::dot(bz,bz+nz,temp2);
    // f(z)/dy
    tipl::mat::product(T.begin()+(nxyz << 1),bx,temp,dyz_x,dx_1);
    tipl::mat::product(temp,dby,temp2,dz_y,dy_1);
    Jbet[7] = tipl::vec::dot(bz,bz+nz,temp2);
    // f(z)/dz
    tipl::mat::product(T.begin()+(nxyz << 1),bx,temp,dyz_x,dx_1);
    tipl::mat::product(temp,by,temp2,dz_y,dy_1);
    Jbet[8] = 1 + tipl::vec::dot(dbz,dbz+nz,temp2);

    //tipl::mat::product(affine_rotation,Jbet,M,math::dim<3,3>(),math::dim<3,3>());
}

}// reg
}// image

#endif
