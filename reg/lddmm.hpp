#ifndef LDDMM_HPP_INCLUDED
#define LDDMM_HPP_INCLUDED
#include "image/numerical/basic_op.hpp"
#include "image/numerical/fft.hpp"
#include "image/numerical/dif.hpp"
#include "image/numerical/resampling.hpp"

#include "image/io/nifti.hpp"
#include "image/filter/gaussian.hpp"

#include <iostream>
#include <limits>

namespace image
{

namespace reg
{

//------------------------------------------------------------------------------------
template<class pixel_type,class vtor_type,unsigned int dimension>
void fast_lddmm(const basic_image<pixel_type,dimension>& I0,
                const basic_image<pixel_type,dimension>& I1,
                basic_image<pixel_type,dimension>& J0, // the deformed I0 images at different time frame
                basic_image<pixel_type,dimension>& J1, // the deformed I0 images at different time frame
                basic_image<vtor_type,dimension>& fs0,
                basic_image<vtor_type,dimension>& fs1,
                float dt = 0.2,float alpha = 0.02)
{
    geometry<dimension> geo = I0.geometry();
    if(I0.geometry() != I1.geometry())
        throw std::runtime_error("The image size of I0 and I1 is not consistent.");
    if(image::fft_round_up_geometry(geo) != geo)
        throw std::runtime_error("The geometry must be rounded up to 2 to the power of n");
    J0 = I0;
    J1 = I1;
    float sigma = *std::max_element(I0.begin(),I0.end())/10.0;
    image::fftn<dimension> fft(geo);
    basic_image<pixel_type,dimension> K(geo);
    image::basic_image<float,dimension> jdet(geo),dJ(geo);
    basic_image<vtor_type,dimension> v(geo),v2(geo),dv(geo),dv2(geo),dvimg(geo),s0(geo),s1(geo);
    unsigned int res = 0;
    float total_I = std::accumulate(I0.begin(),I0.end(),0.0) + std::accumulate(I1.begin(),I1.end(),0.0);
    float last_total_e = std::numeric_limits<float>::max();
    float total_e = std::numeric_limits<float>::max();

    for (image::pixel_index<dimension> index(geo); index < geo.size(); ++index)
    {
        s0[index.index()] = index;
        s1[index.index()] = index;
    }
    fs0.resize(geo);
    fs1.resize(geo);
    std::fill(fs0.begin(),fs0.end(),vtor_type());
    std::fill(fs1.begin(),fs1.end(),vtor_type());

    bool update_K = true;
    bool swi = true;
    for(unsigned int k = 0; k < 1000 && res < 20; ++k)
    {
        if(update_K)
        {
            vector<dimension,float> bandwidth = K.geometry();
            for(pixel_index<dimension> index(K.geometry());index < K.size();++index)
            {
                float Ak = 0;
                for(unsigned int dim = 0; dim < dimension; ++dim)
                    Ak += (1-std::cos(2.0*3.1415926*((float)index[dim])/bandwidth[dim]))*bandwidth[dim]*bandwidth[dim];
                Ak = 1.0 + Ak*alpha;
                K[index.index()] = 1.0 / Ak / Ak;
            }
            update_K = false;
        }

        if(swi)
            image::gradient(J0,dv);
        else
            image::gradient(J1,dv);
        //image::jacobian_determine(s1,jdet);
        image::minus(J0.begin(),J0.end(),J1.begin());

        image::multiply(dv.begin(),dv.end(),J0.begin());


        fft.apply(dv,dvimg);
        image::multiply(dv.begin(),dv.end(),K.begin());
        image::multiply(dvimg.begin(),dvimg.end(),K.begin());
        fft.apply_inverse(dv,dvimg);
        image::multiply_constant(dv.begin(),dv.end(),0.7*dt/sigma/sigma/((float)(1 << dimension))/dv.size());

        if(swi)
            image::add(v.begin(),v.end(),dv.begin());
        else
            image::add(v2.begin(),v2.end(),dv.begin());

        float next_sum_dif = 0.0;
        for(unsigned int index = 0; index < J0.size(); ++index)
            next_sum_dif += std::abs(J0[index]);

        std::cout << "dif=" << (next_sum_dif*100.0/total_I) << "% iteration=" << k << " dt = " << dt << " alpha=" << alpha << std::endl;

        if(total_e < next_sum_dif)
        {
            ++res;
            dt *= 0.5;
            for(unsigned int d =0; d < dimension; ++d)
                alpha *= 0.5;
            for (image::pixel_index<dimension> index(geo); index < geo.size(); ++index)
            {
                if(last_total_e > total_e)
                {
                    fs0[index.index()] += s0[index.index()]-vtor_type(index);
                    fs1[index.index()] += s1[index.index()]-vtor_type(index);
                }
                s0[index.index()] = index;
                s1[index.index()] = index;
            }
            std::fill(v.begin(),v.end(),vtor_type());
            std::fill(v2.begin(),v2.end(),vtor_type());
            last_total_e = total_e;
            update_K = true;
        }

        basic_image<vtor_type,dimension> s0_next(s0),s1_next(s1);
        /* Calculate for j = 0 to j = N ? 1 the mapping using Eq. (18).
        */
        if(swi)
            for (image::pixel_index<dimension> index(geo); index < geo.size(); ++index)
                image::estimate(s0,vtor_type(index)-v[index.index()],s0_next[index.index()]);
        else
            for (image::pixel_index<dimension> index(geo); index < geo.size(); ++index)
                image::estimate(s1,vtor_type(index)+v2[index.index()],s1_next[index.index()]);

        s0 = s0_next;
        s1 = s1_next;

        image::add(s0_next.begin(),s0_next.end(),fs0.begin());
        image::add(s1_next.begin(),s1_next.end(),fs1.begin());

        image::compose_mapping(I0,s0_next,J0);
        image::compose_mapping(I1,s1_next,J1);

        total_e = next_sum_dif;
        swi = !swi;

        /*
        image::geometry<dimension> geo_(geo);
        geo_[0] *= 2;
        image::basic_image<pixel_type,dimension> JOut(geo_);
        image::draw(J0,JOut,image::pixel_index<dimension>());
        image::pixel_index<dimension> shift;
        shift[0] = geo[0];
        image::draw(J1,JOut,shift);
        */

        //image::io::nifti nifti_file;
        //nifti_file << JO;
        //nifti_file.save_to_file("result_lddmm_J0.nii");
        //nifti_file << J1;
        //nifti_file.save_to_file("result_lddmm_J1.nii");
    }
}


/**

This LDDMM implementation follows the following reference

Faisal Beg, Michael Miller, Alain Trouve, and Laurent Younes.
Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms.
International Journal of Computer Vision,
Volume 61, Issue 2; February 2005.

*/
template<class pixel_type,class vtor_type,unsigned int dimension>
void lddmm(const basic_image<pixel_type,dimension>& I0,
           const basic_image<pixel_type,dimension>& I1,
           std::vector<basic_image<pixel_type,dimension> >& J0, // the deformed I0 images at different time frame
           std::vector<basic_image<pixel_type,dimension> >& J1, // the deformed I1 images at different time frame
           std::vector<basic_image<vtor_type,dimension> >& s0,// the deformation metric of I0 at different time frame
           std::vector<basic_image<vtor_type,dimension> >& s1,// the deformation metric of I1 at different time frame
           unsigned int T = 20,float dt = 0.2,float gamma = 1.0)
{

    geometry<dimension> geo = I0.geometry();
    if(I0.geometry() != I1.geometry())
        throw std::runtime_error("The image size of I0 and I1 is not consistent.");
    if(image::fft_round_up_geometry(geo) != geo)
        throw std::runtime_error("The geometry must be rounded up to 2 to the power of n");
    J0.resize(T);
    J1.resize(T);
    s0.resize(T);
    s1.resize(T);

    std::vector<basic_image<vtor_type,dimension> > v(T);   // the velocity function
    std::vector<basic_image<vtor_type,dimension> > alpha_dis(T);   // the displacement

    // initialize mapping J0,J1, s0, s1
    for(unsigned int j = 0; j < T; ++j)
    {
        J0[j] = I0;
        J1[j] = I1;
        s0[j].resize(geo);
        s1[j].resize(geo);
        v[j].resize(geo);
        alpha_dis[j].resize(geo);
        for (image::pixel_index<dimension> index(geo); index < geo.size(); ++index)
        {
            s0[j][index.index()] = index;
            s1[j][index.index()] = index;
        }
    }

    float sigma = *std::max_element(I0.begin(),I0.end())/10.0;
    float e = 0.99; // the velocity update coefficent in  vk+1 = vk - e(Ev)
    dt /= sigma*sigma;
    dt /= (1 << dimension); //compensate the jacobian determinant |Ds|


    // calculate the invert(LL*) operator
    image::fftn<dimension> fft(geo);
    basic_image<pixel_type,dimension> K(geo);
    float alpha = 0.02;
    //float gamma = 1.0;
    {
        vector<dimension,float> bandwidth = K.geometry();
        for(pixel_index<dimension> index(K.geometry());index < K.size();++index)
        {
            float Ak = 0;
            for(unsigned int dim = 0; dim < dimension; ++dim)
                Ak += (1-std::cos(2.0f*3.1415926f*((float)index[dim])/bandwidth[dim]))*bandwidth[dim]*bandwidth[dim];
            Ak = gamma + Ak*alpha;
            K[index.index()] = 1.0f / Ak / Ak;
        }
    }


    float total_e = std::numeric_limits<float>::max();
    image::basic_image<float,dimension> jdet(geo),dJ(geo),dif(geo);
    basic_image<vtor_type,dimension> dv(geo),dvimg(geo);

    for(unsigned int k = 0; k < 200; ++k)
    {
        if(k %10 == 9)// reparameterize
        {
            std::vector<float> v_length(T);
            for (image::pixel_index<dimension> index(geo); index < geo.size(); ++index)
            {
                for(unsigned int j =0; j < T; ++j)
                    v_length[j] = v[j][index.index()].length();
                float length = std::accumulate(v_length.begin(),v_length.end(),(float)0);
                length /= (float)T;
                if(length == 0)
                    continue;
                for(unsigned int j =0; j < T; ++j)
                    if(v_length[j] > 0.0)
                        v[j][index.index()] *= length/v_length[j];
            }
        }


        // Calculate new estimate of velocity vk+1 = vk ? e vk E.
        for(unsigned int j =0; j < T; ++j)
        {
            // calculate the gradient of J0
            image::gradient(J0[j],dv);

            // calculate |Ds1|
            image::jacobian_determinant(s1[j],jdet);

            // calculate J0-J1
            std::copy(J0[j].begin(),J0[j].end(),dJ.begin());
            image::minus(dJ.begin(),dJ.end(),J1[j].begin());
            if(j == T-1)
                dif = dJ;

            // calculate |Ds1|(J0-J1)
            image::multiply(jdet.begin(),jdet.end(),dJ.begin());

            // calculate (2/sigma^2)*|Ds1|(J0-J1)
            image::multiply_constant(jdet.begin(),jdet.end(),dt);

            // now dV = (2/sigma^2)|Ds|gJ0(J0-J1)
            image::multiply(dv.begin(),dv.end(),jdet.begin());

            // update v
            image::multiply_constant(v[j].begin(),v[j].end(),e);

            // apply K() operator

            fft.apply(dv,dvimg);
            image::multiply(dv.begin(),dv.end(),K.begin());
            image::multiply(dvimg.begin(),dvimg.end(),K.begin());
            fft.apply_inverse(dv,dvimg);
            image::divide_constant(dv.begin(),dv.end(),dv.size());

            // vk+1 = vk - eD(E)
            image::add(v[j].begin(),v[j].end(),dv.begin());

        }

        // calculate α using Eq. (20) α = δt * vt ( y − α / 2);
        // note that vt here is already scaled with δt
        alpha_dis = v;
        for(unsigned int j = 0;j < alpha_dis.size();++j)
        {
            basic_image<vtor_type,dimension>& vj = v[j];
            basic_image<vtor_type,dimension>& alpha_j = alpha_dis[j];
            for (image::pixel_index<dimension> index(geo); index < geo.size(); ++index)
            {
                for(unsigned char i = 0;i < 5;++i)
                    image::estimate(vj,vtor_type(index)-alpha_j[index.index()]/2,
                                           alpha_j[index.index()]);
            }
        }

        //Calculate for j = N ? 1 to j = 0 the mapping £pk+1t j ,T (y) using Eq. (19).
        for(int j = T-2; j >= 0; --j)
        {
            basic_image<vtor_type,dimension>& alpha_j = alpha_dis[j];
            // £pj(y) = £pj+1(y + α).
            for (image::pixel_index<dimension> index(geo); index < geo.size(); ++index)
                image::estimate(s1[j+1],vtor_type(index)+alpha_j[index.index()],s1[j][index.index()]);
        }

        // Calculate for j = 0 to j = N ? 1 the mapping £pk+1t j ,0 (y) using Eq. (18).
        for(int j = 1; j < T; ++j)
        {
            basic_image<vtor_type,dimension>& alpha_j = alpha_dis[j];
            // £pj(y) = £pj-1(y - α).
            for (image::pixel_index<dimension> index(geo); index < geo.size(); ++index)
                image::estimate(s0[j-1],vtor_type(index)-alpha_j[index.index()],s0[j][index.index()]);
        }
        // Calculate for j = 0 to j = N ? 1 the image J0j= I0 ? £pk+1 j,0
        for(int j = 0; j < T; ++j)
            image::compose_mapping(I0,s0[j],J0[j]);
        // Calculate for j = N - 1 to j = 0 the image J1j= I1 ? £pk+1 j,t
        for(int j = T - 1; j >= 0; --j)
            image::compose_mapping(I1,s1[j],J1[j]);

        float next_sum_dif = 0.0;
        for(unsigned int index = 0; index < dif.size(); ++index)
            next_sum_dif += std::abs(dif[index]);

        std::cout << next_sum_dif << "..." << std::flush;

        if(total_e < next_sum_dif)
            break;
        total_e = next_sum_dif;
    }
}


template<class pixel_type,class vtor_type,unsigned int dimension>
void lddmm(const basic_image<pixel_type,dimension>& I0,
           const basic_image<pixel_type,dimension>& I1,
           basic_image<vtor_type,dimension>& mapping,
           unsigned int T = 20,float dt = 0.2,float gamma = 1.0)
{
    std::vector<basic_image<pixel_type,dimension> > J0;// the deformed I0 images at different time frame
    std::vector<basic_image<pixel_type,dimension> > J1;// the deformed I1 images at different time frame
    std::vector<basic_image<vtor_type,dimension> > s0;// the deformation metric of I0 at different time frame
    std::vector<basic_image<vtor_type,dimension> > s1;// the deformation metric of I1 at different time frame
    lddmm(I0,I1,J0,J1,s0,s1,T,dt,gamma);
    mapping = s0.back();
}


}

}

#endif // LDDMM_HPP_INCLUDED
