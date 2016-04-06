#ifndef TRANSFORMATION_HPP_INCLUDED
#define TRANSFORMATION_HPP_INCLUDED
#include "image/numerical/matrix.hpp"
namespace image{

template<unsigned int dim>
struct vdim {};

template<class input_iter1,class input_iter2,class output_iter>
void vector_transformation(input_iter1 vec_in,output_iter vec_out,input_iter2 trans,vdim<2>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    vec_out[0] = ((value_type)vec_in[0])*trans[0] +
                 ((value_type)vec_in[1])*trans[1] +
                 ((value_type)trans[2]);
    vec_out[1] = ((value_type)vec_in[0])*trans[3] +
                 ((value_type)vec_in[1])*trans[4] +
                 ((value_type)trans[5]);
}

template<class input_iter1,class input_iter2,class output_iter>
void vector_transformation(input_iter1 vec_in,output_iter vec_out,input_iter2 trans,vdim<3>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    vec_out[0] = ((value_type)vec_in[0])*trans[0] +
                 ((value_type)vec_in[1])*trans[1] +
                 ((value_type)vec_in[2])*trans[2] +
                 ((value_type)trans[3]);
    vec_out[1] = ((value_type)vec_in[0])*trans[4] +
                 ((value_type)vec_in[1])*trans[5] +
                 ((value_type)vec_in[2])*trans[6] +
                 ((value_type)trans[7]);
    vec_out[2] = ((value_type)vec_in[0])*trans[8] +
                 ((value_type)vec_in[1])*trans[9] +
                 ((value_type)vec_in[2])*trans[10] +
                 ((value_type)trans[11]);
}

template<class input_iter1,class input_iter2,class input_iter3,class output_iter>
void vector_transformation(input_iter1 vec_in,output_iter vec_out,input_iter2 rotation,input_iter3 shift,vdim<2>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    vec_out[0] = ((value_type)vec_in[0])*rotation[0] +
                 ((value_type)vec_in[1])*rotation[1] +
                 ((value_type)shift[0]);
    vec_out[1] = ((value_type)vec_in[0])*rotation[2] +
                 ((value_type)vec_in[1])*rotation[3] +
                 ((value_type)shift[1]);
}


template<class input_iter1,class input_iter2,class input_iter3,class output_iter>
void vector_transformation(input_iter1 vec_in,output_iter vec_out,input_iter2 rotation,input_iter3 shift,vdim<3>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;

    vec_out[0] = ((value_type)vec_in[0])*rotation[0] +
                 ((value_type)vec_in[1])*rotation[1] +
                 ((value_type)vec_in[2])*rotation[2] +
                 ((value_type)shift[0]);
    vec_out[1] = ((value_type)vec_in[0])*rotation[3] +
                 ((value_type)vec_in[1])*rotation[4] +
                 ((value_type)vec_in[2])*rotation[5] +
                 ((value_type)shift[1]);
    vec_out[2] = ((value_type)vec_in[0])*rotation[6] +
                 ((value_type)vec_in[1])*rotation[7] +
                 ((value_type)vec_in[2])*rotation[8] +
                 ((value_type)shift[2]);
}

template<class input_iter1,class input_iter2,class output_iter>
void vector_rotation(input_iter1 vec_in,output_iter vec_out,input_iter2 rotation,vdim<2>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    vec_out[0] = ((value_type)vec_in[0])*rotation[0] +
                 ((value_type)vec_in[1])*rotation[1];
    vec_out[1] = ((value_type)vec_in[0])*rotation[2] +
                 ((value_type)vec_in[1])*rotation[3];
}

template<class input_iter1,class input_iter2,class output_iter>
void vector_rotation(input_iter1 vec_in,output_iter vec_out,input_iter2 rotation,vdim<3>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    vec_out[0] = ((value_type)vec_in[0])*rotation[0] +
                 ((value_type)vec_in[1])*rotation[1] +
                 ((value_type)vec_in[2])*rotation[2];
    vec_out[1] = ((value_type)vec_in[0])*rotation[3] +
                 ((value_type)vec_in[1])*rotation[4] +
                 ((value_type)vec_in[2])*rotation[5];
    vec_out[2] = ((value_type)vec_in[0])*rotation[6] +
                 ((value_type)vec_in[1])*rotation[7] +
                 ((value_type)vec_in[2])*rotation[8];
}


/** Perform C= AB
    A,B,C are 2-by-2 matrices
*/
template<class input_iterator1,class input_iterator2,class output_iterator>
void matrix_product(input_iterator1 A,input_iterator2 B,output_iterator C,vdim<2>)
{
    C[0] = A[0] * B[0] + A[1] * B[2];
    C[1] = A[0] * B[1] + A[1] * B[3];

    C[2] = A[2] * B[0] + A[3] * B[2];
    C[3] = A[2] * B[1] + A[3] * B[3];

}

/** Perform C= AB
    A,B,C are 3-by-3 matrices
*/
template<class input_iterator1,class input_iterator2,class output_iterator>
void matrix_product(input_iterator1 A,input_iterator2 B,output_iterator C,vdim<3>)
{
    C[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
    C[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
    C[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];

    C[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
    C[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
    C[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];

    C[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
    C[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
    C[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}

template<class angle_type,class output_iter>
void rotation_matrix(angle_type theta,output_iter m,vdim<2>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    value_type cos_theta = std::cos(theta[0]);
    value_type sin_theta = std::sin(theta[0]);
    m[0] = cos_theta;
    m[1] = -sin_theta;
    m[2] = sin_theta;
    m[3] = cos_theta;
}

/*
 example

    float angle[3] = {1,2,3};// SPM use [1 -2 3]
    float result[9];
    image::rotation_matrix(angle,result,image::vdim<3>());
    std::copy(result,result+9,std::ostream_iterator<float>(std::cout," "));
    return 0;

 */
//a clockwise/left-handed rotation with Euler angles
template<class angle_type,class output_type>
void rotation_matrix(angle_type theta,output_type m,vdim<3>)
{
    typedef typename std::iterator_traits<angle_type>::value_type value_type;
    value_type sin_x = std::sin(theta[0]);
    value_type cos_x = std::cos(theta[0]);
    value_type sin_y = std::sin(theta[1]);
    value_type cos_y = std::cos(theta[1]);
    value_type sin_z = std::sin(theta[2]);
    value_type cos_z = std::cos(theta[2]);
    value_type cos_x_sin_z = cos_x*sin_z;
    value_type cos_x_cos_z = cos_x*cos_z;
    value_type sin_x_sin_z = sin_x*sin_z;
    value_type sin_x_cos_z = sin_x*cos_z;
    m[0] = cos_y*cos_z;
    m[1] = cos_y*sin_z;
    m[2] = -sin_y;
    m[3] = -cos_x_sin_z+ sin_x_cos_z*sin_y;
    m[4] = cos_x_cos_z+ sin_x_sin_z*sin_y;
    m[5] = sin_x*cos_y;
    m[6] = sin_x_sin_z+cos_x_cos_z*sin_y;
    m[7] = -sin_x_cos_z+cos_x_sin_z*sin_y;
    m[8] = cos_x*cos_y;
    /* Euler angle
    m[0] = cos_x_cos_z - cos_y*sin_x_sin_z;
    m[1] = sin_x_cos_z + cos_y*cos_x_sin_z;
    m[2] = sin_z*sin_y;
    m[3] = -cos_x_sin_z - cos_y*sin_x_cos_z;
    m[4] = -sin_x_sin_z + cos_y*cos_x_cos_z;
    m[5] = cos_z*sin_y;
    m[6] = sin_y*sin_x;
    m[7] = -sin_y*cos_x;
    m[8] = cos_y;
     */
}
/*
 Scaling*Rotate
 */
template<class angle_type,class scale_type,class output_type>
void rotation_scaling_matrix(angle_type theta,scale_type s,output_type m,vdim<2>)
{
    rotation_matrix(theta,m,vdim<2>());
    m[0] *= s[0];
    m[1] *= s[0];
    m[2] *= s[1];
    m[3] *= s[1];
}
/*
 Scaling*Rx*Ry*Rz
 */
template<class angle_type,class scale_type,class output_type>
void rotation_scaling_matrix(angle_type theta,scale_type s,output_type m,vdim<3>)
{
    rotation_matrix(theta,m,vdim<3>());
    m[0] *= s[0];
    m[1] *= s[0];
    m[2] *= s[0];
    m[3] *= s[1];
    m[4] *= s[1];
    m[5] *= s[1];
    m[6] *= s[2];
    m[7] *= s[2];
    m[8] *= s[2];
}

/*
 Affine*Scaling*R

Affine   = [1   	a[0]    0;
            0   	1 	    0;
            0   	0   	1];
 */
template<class angle_type,class scale_type,class affine_type,class output_type>
void rotation_scaling_affine_matrix(angle_type theta,scale_type s,affine_type a,output_type m,vdim<2>)
{
    rotation_scaling_matrix(theta,s,m,vdim<2>());
    m[0] += m[2]*a[0];
    m[1] += m[3]*a[0];
}

/*
 Affine*Scaling*R1*R2*R3

Affine   = [1   	a[0]    a[1]   0;
            0   	1 	a[2]   0;
            0   	0   	1      0;
            0    	0    	0      1];
 */
template<class angle_type,class scale_type,class affine_type,class output_type>
void rotation_scaling_affine_matrix(angle_type theta,scale_type s,affine_type a,output_type m,vdim<3>)
{
    rotation_scaling_matrix(theta,s,m,vdim<3>());
    m[0] += m[3]*a[0]+m[6]*a[1];
    m[1] += m[4]*a[0]+m[7]*a[1];
    m[2] += m[5]*a[0]+m[8]*a[1];

    m[3] += m[6]*a[2];
    m[4] += m[7]*a[2];
    m[5] += m[8]*a[2];
}

template<class output_iter>
void rotation_x_matrix(typename std::iterator_traits<output_iter>::value_type theta,output_iter m/*a 3x3 matrix*/)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    value_type cos_theta = std::cos(theta);
    value_type sin_theta = std::sin(theta);
    m[0] = 1.0;
    m[1] = 0.0;
    m[2] = 0.0;
    m[3] = 0.0;
    m[4] = cos_theta;
    m[5] = sin_theta;
    m[7] = -sin_theta;
    m[6] = 0.0;
    m[8] = cos_theta;
}

template<class output_iter>
void rotation_y_matrix(typename std::iterator_traits<output_iter>::value_type theta,output_iter m/*a 3x3 matrix*/)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    value_type cos_theta = std::cos(theta);
    value_type sin_theta = std::sin(theta);
    m[0] = cos_theta;
    m[1] = 0.0;
    m[2] = -sin_theta;
    m[3] = 0.0;
    m[4] = 1.0;
    m[5] = 0.0;
    m[6] = sin_theta;
    m[7] = 0.0;
    m[8] = cos_theta;
}

template<class output_iter>
void rotation_z_matrix(typename std::iterator_traits<output_iter>::value_type theta,output_iter m/*a 3x3 matrix*/)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    value_type cos_theta = std::cos(theta);
    value_type sin_theta = std::sin(theta);
    m[0] = cos_theta;
    m[1] = sin_theta;
    m[2] = 0.0;
    m[3] = -sin_theta;
    m[4] = cos_theta;
    m[5] = 0.0;
    m[6] = 0.0;
    m[7] = 0.0;
    m[8] = 1.0;
}

/**
    rotate from u to v
    R : left roration matrix
*/

template<class input_iter1,class input_iter2,class output_iter>
void rotation_vector_matrix(output_iter r,input_iter1 u,input_iter2 v)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;

    value_type value = u[0]*v[0]+u[1]*v[1]+u[2]*v[2]+ 1.0;
    if (value == 0.0)
    {
        r[0] = r[4] = r[8] = -1.0;
        r[1] = r[2] = r[3] = r[5] = r[6] = r[7] = 0.0;
        return;
    }
    value_type uv[3];
    uv[0] = u[0] + v[0];
    uv[1] = u[1] + v[1];
    uv[2] = u[2] + v[2];
    //R(u->v) = (u+v)*(u+v)T/(uT*v+1) - I
    r[0] = uv[0]*uv[0]/value-1;
    r[1] = uv[1]*uv[0]/value;
    r[2] = uv[2]*uv[0]/value;
    r[3] = uv[0]*uv[1]/value;
    r[4] = uv[1]*uv[1]/value-1;
    r[5] = uv[2]*uv[1]/value;
    r[6] = uv[0]*uv[2]/value;
    r[7] = uv[1]*uv[2]/value;
    r[8] = uv[2]*uv[2]/value-1;
}


template<class input_iter,class output_iter>
void rotation_matrix(input_iter uv/*a 3d unit vector as the axis*/,
                     typename std::iterator_traits<input_iter>::value_type theta,output_iter m/*a 3x3 matrix*/,vdim<3>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    value_type cos_theta = std::cos(theta);
    value_type sin_theta = std::sin(theta);
    value_type cos_theta_1 = (1-cos_theta);
    value_type zs = uv[2]*sin_theta;
    value_type ys = uv[1]*sin_theta;
    value_type xs = uv[0]*sin_theta;
    m[0] = uv[0]*uv[0]*cos_theta_1+cos_theta;
    m[1] = uv[1]*uv[0]*cos_theta_1-zs;
    m[2] = uv[2]*uv[0]*cos_theta_1+ys;
    m[3] = uv[0]*uv[1]*cos_theta_1+zs;
    m[4] = uv[1]*uv[1]*cos_theta_1+cos_theta;
    m[5] = uv[2]*uv[1]*cos_theta_1-xs;
    m[6] = uv[0]*uv[2]*cos_theta_1-ys;
    m[7] = uv[1]*uv[2]*cos_theta_1+xs;
    m[8] = uv[2]*uv[2]*cos_theta_1+cos_theta;
}

template<class input_iter,class output_iter>
void scaling_matrix(input_iter scaling,output_iter m,vdim<2>)
{
    m[0] = scaling[0];
    m[1] = 0.0;
    m[2] = 0.0;
    m[3] = scaling[1];
}

template<class input_iter,class output_iter>
void scaling_matrix(input_iter scaling,output_iter m,vdim<3>)
{
    m[0] = scaling[0];
    m[1] = 0.0;
    m[2] = 0.0;
    m[3] = 0.0;
    m[4] = scaling[1];
    m[5] = 0.0;
    m[6] = 0.0;
    m[7] = 0.0;
    m[8] = scaling[2];
}



template<class input_scaling_iter,class angle_type,class output_iter>
void rotation_angle_to_rotation_matrix(input_scaling_iter scaling,angle_type rotation,output_iter m,vdim<2>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    value_type S[4],R[4];
    scaling_matrix(scaling,S,vdim<2>());
    rotation_matrix(rotation[0],R,vdim<2>());
    matrix_product(R,S,m,vdim<2>());
}

// the rotation is the Euler angles, which has Z-X-Z configuration
template<class input_scaling_iter,class input_rotation_iter,class output_iter>
void rotation_angle_to_rotation_matrix(input_scaling_iter scaling,input_rotation_iter rotation,output_iter m,vdim<3>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    value_type S[9],R[9],M[9];
    scaling_matrix(scaling,S,vdim<3>());
    rotation_z_matrix(rotation[0],R);
    matrix_product(R,S,M,vdim<3>());
    rotation_x_matrix(rotation[1],R);
    matrix_product(R,M,S,vdim<3>());
    rotation_z_matrix(rotation[2],R);
    matrix_product(R,S,m,vdim<3>());
}

// the rotation is the Euler angles, which has Z-X-Z configuration
/*
template<class input_rotation_iter,class output_iter>
void rotation_angle_to_rotation_matrix(input_rotation_iter rotation,output_iter m)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    value_type S[9],R[9],M[9];
    rotation_z_matrix(rotation[0],M);
    rotation_x_matrix(rotation[1],R);
    matrix_product(R,M,S,vdim<3>());
    rotation_z_matrix(rotation[2],R);
    matrix_product(R,S,m,vdim<3>());
}
*/

// Output Euler angle from rotation matrix
template<class input_rotation_iter,class output_iter>
void rotation_matrix_to_rotation_angle(input_rotation_iter rotation_matrix,output_iter rotation_angle,vdim<3>)
{
    rotation_angle[0] = std::atan2(rotation_matrix[6],rotation_matrix[7]); //Z
    rotation_angle[1] = std::acos(rotation_matrix[8]);//X
    rotation_angle[2] = -std::atan2(rotation_matrix[2],rotation_matrix[5]);//Z
}


template<class input_rotation_iter,class input_shift_iter,class output_iter>
void create_affine_transformation_matrix(input_rotation_iter rotation_scaling,input_shift_iter shift,output_iter m,vdim<3>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    std::copy(rotation_scaling,rotation_scaling+3,m);
    std::copy(rotation_scaling+3,rotation_scaling+6,m+4);
    std::copy(rotation_scaling+6,rotation_scaling+9,m+8);
    m[3] = shift[0];
    m[7] = shift[1];
    m[11] = shift[2];
    m[12] = m[13] = m[14] = 0;
    m[15] = 1;

}

template<class input_scaling_iter,class input_rotation_iter,class input_shift_iter,class output_iter>
void create_affine_transformation_matrix(input_scaling_iter scaling,input_rotation_iter rotation,input_shift_iter shift,output_iter m,vdim<3>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    value_type M[9];
    sr_matrix(scaling,rotation,M);
    create_transformation_matrixation_matrix(M,shift,m,vdim<3>());
}

template<class value_type_ = float>
class affine_transform
{
public:
    typedef value_type_ value_type;
    static const unsigned int dimension = 3;
    static const unsigned int affine_dim = 3;
    static const unsigned int total_size = 12;
    union
    {
        struct
        {
            value_type translocation[3];
            value_type rotation[3];
            value_type scaling[3];
            value_type affine[3];
        };
        value_type data[12];
    };
private:
    void assign(const affine_transform& rhs)
    {
        std::copy(rhs.data,rhs.data+total_size,data);
    }
public:
    affine_transform(void)
    {
        std::fill(data,data+total_size,0);
        std::fill(scaling,scaling+dimension,1);
    }
    affine_transform(const value_type* data_)
    {
        std::copy(data_,data_+total_size,data);
    }
    affine_transform(const affine_transform& rhs)
    {
        assign(rhs);
    }

    const affine_transform& operator=(const affine_transform& rhs)
    {
        assign(rhs);
        return *this;
    }
    value_type operator[](unsigned int i) const{return data[i];}
    value_type& operator[](unsigned int i) {return data[i];}
    const value_type* begin(void) const{return data;}
    const value_type* end(void) const{return data+total_size;}
    value_type* begin(void) {return data;}
    value_type* end(void) {return data+total_size;}
    unsigned int size(void) const{return total_size;}

};




template<class value_type_ = float>
struct transformation_matrix
{
    typedef value_type_ value_type;
    static const unsigned int dimension = 3;
    static const unsigned int sr_size = 9;
    static const unsigned int total_size = 12;
public:
    union
    {
        struct
        {
            value_type sr[9];
            value_type shift[3];
        };
        value_type data[12];
    };

public:
    transformation_matrix(void)
    {
        std::fill((value_type*)data,(value_type*)data+total_size,0);
    }

    // (Affine*Scaling*R1*R2*R3*vs*Translocation*shift_center)*from = (vs*shift_center)*to;
    transformation_matrix(const affine_transform<value_type>& rb,
                          const image::geometry<3>& from,
                          const image::vector<3>& from_vs,
                          const image::geometry<3>& to,
                          const image::vector<3>& to_vs)
    {
        //now sr = Affine*Scaling*R1*R2*R3
        rotation_scaling_affine_matrix(rb.rotation,rb.scaling,rb.affine,sr,vdim<dimension>());
        // calculate (vs*Translocation*shift_center)
        image::vector<3> t(from[0],from[1],from[2]);
        t *= -0.5;
        t += rb.translocation;
        t[0] *= from_vs[0];
        t[1] *= from_vs[1];
        t[2] *= from_vs[2];
        // (Affine*Scaling*R1*R2*R3)*(vs*Translocation*shift_center)
        shift[0] = sr[0]*t[0]+sr[1]*t[1]+sr[2]*t[2];
        shift[1] = sr[3]*t[0]+sr[4]*t[1]+sr[5]*t[2];
        shift[2] = sr[6]*t[0]+sr[7]*t[1]+sr[8]*t[2];
        sr[0] *= from_vs[0];
        sr[1] *= from_vs[1];
        sr[2] *= from_vs[2];
        sr[3] *= from_vs[0];
        sr[4] *= from_vs[1];
        sr[5] *= from_vs[2];
        sr[6] *= from_vs[0];
        sr[7] *= from_vs[1];
        sr[8] *= from_vs[2];
        // inv(vs) ... = inv(vs)(vs*shift_center)...
        if(to_vs[0] != 1.0)
        {
            sr[0] /= to_vs[0];
            sr[1] /= to_vs[0];
            sr[2] /= to_vs[0];
            shift[0] /= to_vs[0];
        }
        if(to_vs[1] != 1.0)
        {
            sr[3] /= to_vs[1];
            sr[4] /= to_vs[1];
            sr[5] /= to_vs[1];
            shift[1] /= to_vs[1];
        }
        if(to_vs[2] != 1.0)
        {
            sr[6] /= to_vs[2];
            sr[7] /= to_vs[2];
            sr[8] /= to_vs[2];
            shift[2] /= to_vs[2];
        }
        // inv(shift_center) ... = inv(shift_center)(shift_center)...
        shift[0] += to[0]*0.5;
        shift[1] += to[1]*0.5;
        shift[2] += to[2]*0.5;
    }

    const transformation_matrix& operator=(const transformation_matrix& rhs)
    {
        std::copy(rhs.data,rhs.data+total_size,data);
        return *this;
    }
    value_type* get(void){return data;}
    const value_type* get(void) const{return data;}
    value_type operator[](unsigned int i) const{return data[i];}
    value_type& operator[](unsigned int i) {return data[i];}

    // load from 4 x 3 M matrix
    template<class InputIterType>
    void load_from_transform(InputIterType M)
    {
        std::copy(M,M+3,data);
        std::copy(M+4,M+7,data+3);
        std::copy(M+8,M+11,data+6);
        data[9] = M[3];
        data[10] = M[7];
        data[11] = M[11];
    }
    template<class InputIterType>
    void save_to_transform(InputIterType M)
    {
        std::copy(data,data+3,M);
        std::copy(data+3,data+6,M+4);
        std::copy(data+6,data+9,M+8);
        M[3] = data[9];
        M[7] = data[10];
        M[11] = data[11];
    }

    bool inverse(void)
    {
        image::matrix<3,3,value_type> iT(sr);
        if(!iT.inv())
            return false;
        value_type new_shift[3];
        vector_rotation(shift,new_shift,iT.begin(),vdim<3>());
        for(unsigned int d = 0;d < 3;++d)
            shift[d] = -new_shift[d];
        std::copy(iT.begin(),iT.end(),sr);
        return true;
    }

    template<class vtype1,class vtype2>
    void operator()(const vtype1& from,vtype2& to) const
    {
        vector_transformation(from.begin(),to.begin(),sr,shift,vdim<3>());
    }
    template<class vtype>
    void operator()(vtype& pos) const
    {
        vtype result;
        vector_transformation(pos.begin(),result.begin(),sr,shift,vdim<3>());
        pos = result;
    }


};





}
#endif // TRANSFORMATION_HPP_INCLUDED
