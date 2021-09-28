#ifndef ANISOTROPIC_DIFFUSION
#define ANISOTROPIC_DIFFUSION
#include "../utility/basic_image.hpp"
#include "../numerical/numerical.hpp"

namespace tipl
{


namespace filter
{

/**

    conductance_parameter
    delta_t: the time step
             stable time step for this image must be smaller than
               minSpacing / 2.0^(ImageDimension+1);

    iteration: diffusion iteration
*/

template<class image_type>
void anisotropic_diffusion(image_type& src,float conductance_parameter = 1.0,int iteration = 5)
{

    conductance_parameter *= conductance_parameter;
    conductance_parameter *= 1.38629436; // 2.0*ln(2)   the ln(2) is used to change the base from ln to ln2
    std::vector<int> gx(src.size()),gx2(src.size()),total_gx(src.size());
    for (int iter = 0;iter != iteration;++iter)
    {
        unsigned int shift = 1;
        for (unsigned char index = 0;index < image_type::dimension;++index)
        {
            // gx = gradien(I), the gradient at the current dimension
            tipl::gradient(src.begin(),src.end(),gx.begin(),shift,shift);
            gx2 = gx;

            //   gx*gx
            tipl::square(gx2.begin(),gx2.end());

            float K = (float)std::accumulate(gx2.begin(),gx2.end(),(float)0);
            K /= (float)src.size();
            K *= conductance_parameter;

            // gx2 <= gx*gx/K
            tipl::divide_constant(gx2.begin(),gx2.end(),K);

            // add scaling to avoid trancation error
            // will be scaled back later
            tipl::minus_constant(gx2.begin(),gx2.end(),8);

            // px <- pow(2,-gx*gx/K)*gx
            tipl::divide_pow(gx.begin(),gx.end(),gx2.begin());

            // exp(gx1*gx1/K)*gx1-exp(gx2*gx2/K)*gx2
            tipl::gradient(gx.begin(),gx.end(),gx2.begin(),shift,0);

            // accumulate the diffusion magnitude
            // skip the boundary
            tipl::add(total_gx.begin(),total_gx.end(),gx2.begin());

            // proceed to next dimensiom
            shift *= src.shape()[index];
        }
        // perform I <= I + total_gx * delta_t
        // delta_t = 1.0/(1 << dimension);
        // scale back the multiplication of 8
        tipl::divide_pow_constant(total_gx.begin(),total_gx.end(),image_type::dimension+8);
        tipl::add(src.begin(),src.end(),total_gx.begin());
    }
}


template<class pixel_type,size_t dimension>
void anisotropic_diffusion_inv(tipl::image<pixel_type,dimension>& src,
                           float conductance_parameter = 1.0,
                           size_t iteration = 5)
{
    conductance_parameter *= conductance_parameter;
    std::vector<int> gx(src.size()),gx2(src.size()),total_gx(src.size());
    for (size_t iter = 0;iter != iteration;++iter)
    {
        size_t shift = 1;
        for (size_t index = 0;index < dimension;++index)
        {
            // gx = gradien(I), the gradient at the current dimension
            tipl::gradient(src.begin(),src.end(),gx.begin(),shift,shift);
            gx2 = gx;

            //   gx*gx
            tipl::square(gx2.begin(),gx2.end());

            float K = (float)std::accumulate(gx2.begin(),gx2.end(),(float)0);
            K /= (float)src.size();
            K *= conductance_parameter;

            // gx2 <- K+(gx*gx);
            tipl::add_constant(gx2.begin(),gx2.end(),K);

            // gx <- gx*K/(K+(gx*gx))
            tipl::multiply_constant(gx.begin(),gx.end(),K);
            tipl::divide(gx.begin(),gx.end(),gx2.begin());

            // gx2 <- gradient(gx);
            tipl::gradient(gx.begin(),gx.end(),gx2.begin(),shift,0);

            // accumulate the diffusion magnitude
            tipl::add(total_gx.begin(),total_gx.end(),gx2.begin());

            // proceed to next dimensiom
            shift *= src.shape()[index];
        }
        // perform I <= I + total_gx * delta_t
        // delta_t = 1.0/(1 << dimension);
        // scale back the multiplication of 8
        tipl::divide_pow_constant(total_gx.begin(),total_gx.end(),dimension);
        tipl::add(src.begin(),src.end(),total_gx.begin());
    }
}



}


}


#endif
