#ifndef TRANSFORMATION_HPP_INCLUDED
#define TRANSFORMATION_HPP_INCLUDED
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm>
#include "../numerical/matrix.hpp"
#include "../def.hpp"
#include "../utility/pixel_index.hpp"
#include "../utility/basic_image.hpp"
#include "interpolation.hpp"

namespace tipl{

template<unsigned int dim>
struct vdim {};

template<typename input_iter1,typename input_iter2,typename output_iter>
__DEVICE_HOST__ void vector_transformation(input_iter1 vec_in,output_iter vec_out,input_iter2 trans,vdim<2>)
{
    vec_out[0] = vec_in[0]*trans[0] +
                 vec_in[1]*trans[1] +
                 trans[2];
    vec_out[1] = vec_in[0]*trans[3] +
                 vec_in[1]*trans[4] +
                 trans[5];
}

template<typename input_iter1,typename input_iter2,typename output_iter>
__DEVICE_HOST__ void vector_transformation(input_iter1 vec_in,output_iter vec_out,input_iter2 trans,vdim<3>)
{
    vec_out[0] = vec_in[0]*trans[0] +
                 vec_in[1]*trans[1] +
                 vec_in[2]*trans[2] +
                 trans[3];
    vec_out[1] = vec_in[0]*trans[4] +
                 vec_in[1]*trans[5] +
                 vec_in[2]*trans[6] +
                 trans[7];
    vec_out[2] = vec_in[0]*trans[8] +
                 vec_in[1]*trans[9] +
                 vec_in[2]*trans[10] +
                 trans[11];
}

template<typename input_iter1,typename input_iter2,typename input_iter3,typename output_iter>
__DEVICE_HOST__ void vector_transformation(input_iter1 vec_in,output_iter vec_out,input_iter2 rotation,input_iter3 shift,vdim<2>)
{
    vec_out[0] = vec_in[0]*rotation[0] +
                 vec_in[1]*rotation[1] +
                 shift[0];
    vec_out[1] = vec_in[0]*rotation[2] +
                 vec_in[1]*rotation[3] +
                 shift[1];
}

template<typename input_iter1,typename input_iter2,typename input_iter3,typename output_iter>
__DEVICE_HOST__ void vector_transformation(input_iter1 vec_in,output_iter vec_out,input_iter2 rotation,input_iter3 shift,vdim<3>)
{
    vec_out[0] = vec_in[0]*rotation[0] +
                 vec_in[1]*rotation[1] +
                 vec_in[2]*rotation[2] +
                 shift[0];
    vec_out[1] = vec_in[0]*rotation[3] +
                 vec_in[1]*rotation[4] +
                 vec_in[2]*rotation[5] +
                 shift[1];
    vec_out[2] = vec_in[0]*rotation[6] +
                 vec_in[1]*rotation[7] +
                 vec_in[2]*rotation[8] +
                 shift[2];
}

template<typename input_iter1,typename input_iter2,typename output_iter>
void vector_rotation(input_iter1 vec_in,output_iter vec_out,input_iter2 rotation,vdim<2>)
{
    vec_out[0] = vec_in[0]*rotation[0] +
                 vec_in[1]*rotation[1];
    vec_out[1] = vec_in[0]*rotation[2] +
                 vec_in[1]*rotation[3];
}

template<typename input_iter1,typename input_iter2,typename output_iter>
void vector_rotation(input_iter1 vec_in,output_iter vec_out,input_iter2 rotation,vdim<3>)
{
    vec_out[0] = vec_in[0]*rotation[0] +
                 vec_in[1]*rotation[1] +
                 vec_in[2]*rotation[2];
    vec_out[1] = vec_in[0]*rotation[3] +
                 vec_in[1]*rotation[4] +
                 vec_in[2]*rotation[5];
    vec_out[2] = vec_in[0]*rotation[6] +
                 vec_in[1]*rotation[7] +
                 vec_in[2]*rotation[8];
}

/** Perform C= AB
    A,B,C are 2-by-2 matrices
*/
template<typename input_iterator1,typename input_iterator2,typename output_iterator>
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
template<typename input_iterator1,typename input_iterator2,typename output_iterator>
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

template<typename angle_type,typename output_iter>
void rotation_matrix(angle_type theta,output_iter m,vdim<2>)
{
    auto cos_theta = std::cos(theta);
    auto sin_theta = std::sin(theta);
    m[0] = cos_theta;
    m[1] = -sin_theta;
    m[2] = sin_theta;
    m[3] = cos_theta;
}

// a clockwise/left-handed rotation with Euler angles
template<typename angle_type,typename output_type>
void rotation_matrix(angle_type theta,output_type m,vdim<3>)
{
    auto sin_x = std::sin(theta[0]);
    auto cos_x = std::cos(theta[0]);
    auto sin_y = std::sin(theta[1]);
    auto cos_y = std::cos(theta[1]);
    auto sin_z = std::sin(theta[2]);
    auto cos_z = std::cos(theta[2]);
    auto cos_x_sin_z = cos_x*sin_z;
    auto cos_x_cos_z = cos_x*cos_z;
    auto sin_x_sin_z = sin_x*sin_z;
    auto sin_x_cos_z = sin_x*cos_z;
    m[0] = cos_y*cos_z;
    m[1] = cos_y*sin_z;
    m[2] = -sin_y;
    m[3] = -cos_x_sin_z+ sin_x_cos_z*sin_y;
    m[4] = cos_x_cos_z+ sin_x_sin_z*sin_y;
    m[5] = sin_x*cos_y;
    m[6] = sin_x_sin_z+cos_x_cos_z*sin_y;
    m[7] = -sin_x_cos_z+cos_x_sin_z*sin_y;
    m[8] = cos_x*cos_y;
}

template<typename iterator_type1,typename iterator_type2>
void rotation_matrix_to_angles(iterator_type1 m,iterator_type2 theta,vdim<3>)
{
    double sy = std::sqrt(m[0]*m[0]+m[1]*m[1]);
    if (sy > 1.0e-6)
    {
        theta[0] = atan2(m[5],m[8]);
        theta[2] = atan2(m[1],m[0]);
    }
    else
    {
        theta[0] = atan2(-m[7], m[4]);
        theta[2] = 0.0;
    }
    theta[1] = atan2(-m[2], sy);
}

/* Scaling*Rotate */
template<typename angle_type,typename scale_type,typename output_type>
void rotation_scaling_matrix(angle_type theta,scale_type s,output_type m,vdim<2>)
{
    rotation_matrix(theta,m,vdim<2>());
    m[0] *= s[0];
    m[1] *= s[0];
    m[2] *= s[1];
    m[3] *= s[1];
}

/* Scaling*Rx*Ry*Rz */
template<typename angle_type,typename scale_type,typename output_type>
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

/* Affine*Scaling*R */
template<typename angle_type,typename scale_type,typename affine_type,typename output_type>
void rotation_scaling_affine_matrix(angle_type theta,scale_type s,affine_type a,output_type m,vdim<2>)
{
    rotation_scaling_matrix(theta,s,m,vdim<2>());
    m[0] += m[2]*a;
    m[1] += m[3]*a;
}

/* Affine*Scaling*R1*R2*R3 */
template<typename angle_type,typename scale_type,typename affine_type,typename output_type>
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

template<typename intput_type,typename angle_type,typename scale_type,typename affine_type>
void matrix_to_rotation_scaling_affine(intput_type m,angle_type theta,scale_type s,affine_type a,vdim<3>)
{
    double Q[9];
    tipl::mat::inverse(m,tipl::dim<3,3>());
    tipl::mat::qr_decomposition(m,Q,tipl::dim<3,3>());
    tipl::mat::inverse_upper(m,tipl::dim<3,3>());
    tipl::mat::transpose(Q,tipl::dim<3,3>());
    if(m[0] < 0)
    {
        m[0] = -m[0];
        Q[0] = -Q[0];
        Q[1] = -Q[1];
        Q[2] = -Q[2];
    }
    if(m[4] < 0)
    {
        m[1] = -m[1];
        m[4] = -m[4];
        Q[3] = -Q[3];
        Q[4] = -Q[4];
        Q[5] = -Q[5];
    }
    if(m[8] < 0)
    {
        m[2] = -m[2];
        m[5] = -m[5];
        m[8] = -m[8];
        Q[6] = -Q[6];
        Q[7] = -Q[7];
        Q[8] = -Q[8];
    }
    s[0] = m[0];
    s[1] = m[4];
    s[2] = m[8];
    a[0] = m[1]/m[4];
    a[1] = m[2]/m[8];
    a[2] = m[5]/m[8];
    rotation_matrix_to_angles(Q,theta,tipl::vdim<3>());
}

template<typename output_iter>
void rotation_x_matrix(double theta,output_iter m/*a 3x3 matrix*/)
{
    auto cos_theta = std::cos(theta);
    auto sin_theta = std::sin(theta);
    m[0] = 1.0;    m[1] = 0.0;          m[2] = 0.0;
    m[3] = 0.0;    m[4] = cos_theta;    m[5] = sin_theta;
    m[6] = 0.0;    m[7] = -sin_theta;   m[8] = cos_theta;
}

template<typename output_iter>
void rotation_y_matrix(double theta,output_iter m/*a 3x3 matrix*/)
{
    auto cos_theta = std::cos(theta);
    auto sin_theta = std::sin(theta);
    m[0] = cos_theta;    m[1] = 0.0;    m[2] = -sin_theta;
    m[3] = 0.0;          m[4] = 1.0;    m[5] = 0.0;
    m[6] = sin_theta;    m[7] = 0.0;    m[8] = cos_theta;
}

template<typename output_iter>
void rotation_z_matrix(double theta,output_iter m/*a 3x3 matrix*/)
{
    auto cos_theta = std::cos(theta);
    auto sin_theta = std::sin(theta);
    m[0] = cos_theta;     m[1] = sin_theta;    m[2] = 0.0;
    m[3] = -sin_theta;    m[4] = cos_theta;    m[5] = 0.0;
    m[6] = 0.0;           m[7] = 0.0;          m[8] = 1.0;
}

/** rotate from u to v. R : left roration matrix */
template<typename input_iter1,typename input_iter2,typename output_iter>
void rotation_vector_matrix(output_iter r,input_iter1 u,input_iter2 v)
{
    double value = u[0]*v[0]+u[1]*v[1]+u[2]*v[2]+ 1.0;
    if (value == 0.0)
    {
        r[0] = r[4] = r[8] = -1.0;
        r[1] = r[2] = r[3] = r[5] = r[6] = r[7] = 0.0;
        return;
    }
    double uv[3];
    uv[0] = u[0] + v[0];
    uv[1] = u[1] + v[1];
    uv[2] = u[2] + v[2];
    // R(u->v) = (u+v)*(u+v)T/(uT*v+1) - I
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

template<typename input_iter,typename output_iter>
void rotation_matrix(input_iter uv/*a 3d unit vector as the axis*/,double theta,output_iter m/*a 3x3 matrix*/,vdim<3>)
{
    auto cos_theta = std::cos(theta);
    auto sin_theta = std::sin(theta);
    auto cos_theta_1 = (1.0 - cos_theta);
    auto zs = uv[2]*sin_theta;
    auto ys = uv[1]*sin_theta;
    auto xs = uv[0]*sin_theta;
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

template<typename input_iter,typename output_iter>
void scaling_matrix(input_iter scaling,output_iter m,vdim<2>)
{
    m[0] = scaling[0];
    m[1] = 0.0;
    m[2] = 0.0;
    m[3] = scaling[1];
}

template<typename input_iter,typename output_iter>
void scaling_matrix(input_iter scaling,output_iter m,vdim<3>)
{
    m[0] = scaling[0];    m[1] = 0.0;           m[2] = 0.0;
    m[3] = 0.0;           m[4] = scaling[1];    m[5] = 0.0;
    m[6] = 0.0;           m[7] = 0.0;           m[8] = scaling[2];
}

template<typename input_scaling_iter,typename angle_type,typename output_iter>
void rotation_angle_to_rotation_matrix(input_scaling_iter scaling,angle_type rotation,output_iter m,vdim<2>)
{
    double S[4],R[4];
    scaling_matrix(scaling,S,vdim<2>());
    rotation_matrix(rotation[0],R,vdim<2>());
    matrix_product(R,S,m,vdim<2>());
}

// the rotation is the Euler angles, which has Z-X-Z configuration
template<typename input_scaling_iter,typename input_rotation_iter,typename output_iter>
void rotation_angle_to_rotation_matrix(input_scaling_iter scaling,input_rotation_iter rotation,output_iter m,vdim<3>)
{
    double S[9],R[9],M[9];
    scaling_matrix(scaling,S,vdim<3>());
    rotation_z_matrix(rotation[0],R);
    matrix_product(R,S,M,vdim<3>());
    rotation_x_matrix(rotation[1],R);
    matrix_product(R,M,S,vdim<3>());
    rotation_z_matrix(rotation[2],R);
    matrix_product(R,S,m,vdim<3>());
}

// Output Euler angle from rotation matrix
template<typename input_rotation_iter,typename output_iter>
void rotation_matrix_to_rotation_angle(input_rotation_iter rotation_matrix,output_iter rotation_angle,vdim<3>)
{
    rotation_angle[0] = std::atan2(rotation_matrix[6],rotation_matrix[7]); //Z
    rotation_angle[1] = std::acos(rotation_matrix[8]);//X
    rotation_angle[2] = -std::atan2(rotation_matrix[2],rotation_matrix[5]);//Z
}

template<typename input_rotation_iter,typename input_shift_iter,typename output_iter>
void create_affine_param(input_rotation_iter rotation_scaling,input_shift_iter shift,output_iter m,vdim<3>)
{
    std::copy_n(rotation_scaling,3,m);
    std::copy_n(rotation_scaling+3,3,m+4);
    std::copy_n(rotation_scaling+6,3,m+8);
    m[3] = shift[0];
    m[7] = shift[1];
    m[11] = shift[2];
    m[12] = m[13] = m[14] = 0;
    m[15] = 1;
}

template<typename input_scaling_iter,typename input_rotation_iter,typename input_shift_iter,typename output_iter>
void create_affine_param(input_scaling_iter scaling,input_rotation_iter rotation,input_shift_iter shift,output_iter m,vdim<3>)
{
    double M[9];
    rotation_angle_to_rotation_matrix(scaling,rotation,M,vdim<3>());
    create_affine_param(M,shift,m,vdim<3>());
}

template<typename value_type_ = float,int dim = 3>
class affine_param
{
public:
    using value_type = value_type_;
    static constexpr int dimension = dim;
    static constexpr int total_size = (dim-1)*6;
    union
    {
        struct
        {
            value_type translocation[dim];
            value_type rotation[dim == 3 ? 3 : 1];
            value_type scaling[dim];
            value_type affine[dim == 3 ? 3 : 1];
        };
        value_type data_[total_size];
    };
public:
    affine_param(void)
    {
        clear();
    }
    affine_param(std::initializer_list<value_type> rhs)
    {
        size_t i = 0;
        for(const auto& v : rhs) {
            if (i >= total_size) break;
            data_[i++] = v;
        }
    }
    template<typename pointer_type>
    explicit affine_param(const pointer_type* rhs)
    {
        std::copy_n(rhs, total_size, data_);
    }
    template<typename pointer_type>
    const affine_param& operator=(const pointer_type* rhs)
    {
        return std::copy_n(rhs, total_size, data_),*this;
    }

    affine_param(const std::vector<value_type>& rhs)
    {
        std::copy_n(rhs.data(), total_size, data_);
    }
    const affine_param& operator=(const std::vector<value_type> rhs)
    {
        return std::copy_n(rhs.data(), total_size, data_),*this;
    }

    affine_param& operator=(std::initializer_list<value_type> rhs)
    {
        size_t i = 0;
        for(const auto& v : rhs) {
            if (i >= total_size) break;
            data_[i++] = v;
        }
        return *this;
    }
public:
    void clear(void)
    {
        std::fill_n(data_, total_size, value_type(0));
        scaling[0] = 1;
        scaling[1] = 1;
        if constexpr(dimension == 3)
            scaling[2] = 1;
    }
    value_type operator[](unsigned int i) const { return data_[i]; }
    value_type& operator[](unsigned int i) { return data_[i]; }
    const value_type* begin(void) const { return data_; }
    const value_type* end(void) const { return data_ + total_size; }
    value_type* begin(void) { return data_; }
    value_type* end(void) { return data_ + total_size; }
    unsigned int size(void) const { return total_size; }
    value_type* data(void) { return data_; }
    const value_type* data(void) const { return data_; }

    bool operator==(const affine_param& rhs) const
    {
        return std::equal(data_, data_ + total_size, rhs.data_);
    }
    bool operator!=(const affine_param& rhs) const
    {
        return !(*this == rhs);
    }

    friend std::ostream& operator<<(std::ostream& out, const affine_param& T)
    {
        if(typeid(out) == typeid(std::ofstream))
        {
            if constexpr(dimension == 3)
            {
                out << "translocation: " << T.translocation[0] << " " << T.translocation[1] << " " << T.translocation[2] << std::endl;
                out << "rotation: " << T.rotation[0] << " " << T.rotation[1] << " " << T.rotation[2] << std::endl;
                out << "scaling: " << T.scaling[0] << " " << T.scaling[1] << " " << T.scaling[2] << std::endl;
                out << "affine: " << T.affine[0] << " " << T.affine[1] << " " << T.affine[2] << std::endl;
            }
            else
            {
                out << "translocation: " << T.translocation[0] << " " << T.translocation[1] << std::endl;
                out << "rotation: " << T.rotation[0] << std::endl;
                out << "scaling: " << T.scaling[0] << " " << T.scaling[1] << std::endl;
                out << "affine: " << T.affine[0] << std::endl;
            }
            return out;
        }

        if constexpr(dimension == 3)
        {
            out << std::fixed << std::setprecision(2)
                <<  "t:" << T.translocation[0]
                << " " << T.translocation[1]
                << " " << T.translocation[2]
                << std::fixed << std::setprecision(3)
                << " r:" << T.rotation[0]
                << " " << T.rotation[1]
                << " " << T.rotation[2];
            if(T.scaling[0] != 1.0f || T.scaling[1] != 1.0f || T.scaling[2] != 1.0f ||
               T.affine[0] != 0.0f || T.affine[1] != 0.0f || T.affine[2] != 0.0f)
            out << std::fixed << std::setprecision(2)
                << " s:" << T.scaling[0]
                << " " << T.scaling[1]
                << " " << T.scaling[2]
                << std::fixed << std::setprecision(3)
                << " a:" << T.affine[0]
                << " " << T.affine[1]
                << " " << T.affine[2] << std::endl;
        }
        else
        {
            out << std::fixed << std::setprecision(2)
                <<  "t:" << T.translocation[0]
                << " " << T.translocation[1]
                << std::fixed << std::setprecision(3)
                << " r:" << T.rotation[0]
                << std::fixed << std::setprecision(2);
            if(T.scaling[0] != 1.0f || T.scaling[1] != 1.0f || T.affine[0] != 0.0f)
            out << " s:" << T.scaling[0]
                << " " << T.scaling[1]
                << std::fixed << std::setprecision(3)
                << " a:" << T.affine[0] << std::endl;
        }
        return out;
    }
    friend std::istream& operator>>(std::istream& in, affine_param& T)
    {
        std::string text;
        if constexpr(dimension == 3)
        {
            if(in) in >> text >> T.translocation[0] >> T.translocation[1] >> T.translocation[2];
            if(in) in >> text >> T.rotation[0] >> T.rotation[1] >> T.rotation[2];
            if(in) in >> text >> T.scaling[0] >> T.scaling[1] >>  T.scaling[2];
            if(in) in >> text >> T.affine[0] >> T.affine[1] >> T.affine[2];
        }
        else
        {
            if(in) in >> text >> T.translocation[0] >> T.translocation[1];
            if(in) in >> text >> T.rotation[0] >> T.rotation[1];
            if(in) in >> text >> T.scaling[0] >> T.scaling[1];
            if(in) in >> text >> T.affine[0];
        }
        return in;
    }

};

template<typename value_type_ = float,int dim = 3>
struct transformation_matrix
{
    typedef value_type_ value_type;
    static constexpr int dimension = dim;
    static constexpr int sr_size = dim*dim;
    static constexpr int total_size = (dim+1)*dim;
public:
    union
    {
        struct
        {
            value_type sr[dim*dim];
            value_type shift[dim];
        };
        value_type data_[total_size];
    };
public:
    __INLINE__ const value_type& operator[](unsigned int index) const { return data_[index]; }
    __INLINE__ value_type& operator[](unsigned int index) { return data_[index]; }
    __INLINE__ value_type* data(void) { return data_; }
    __INLINE__ value_type* begin(void) { return data_; }
    __INLINE__ value_type* end(void) { return data_ + total_size; }
    __INLINE__ const value_type* data(void) const { return data_; }
    __INLINE__ const value_type* begin(void) const { return data_; }
    __INLINE__ const value_type* end(void) const { return data_ + total_size; }
    __INLINE__ size_t size(void) const { return total_size; }

    __INLINE__ void identity(void)
    {
        std::fill_n(data_, total_size, value_type(0));
        if constexpr(dimension == 3)
            sr[0] = sr[4] = sr[8] = 1;
        else
            sr[0] = sr[3] = 1;
    }
public:
    transformation_matrix(void)
    {
        identity();
    }
    ~transformation_matrix(void){}
    template<typename rhs_value_type>
    transformation_matrix(const transformation_matrix<rhs_value_type,dimension>& M){*this = M;}
    template<typename rhs_value_type>
    transformation_matrix(const tipl::matrix<dimension+1,dimension+1,rhs_value_type>& M){*this = M;}

    template<typename geo_type,typename vs_type>
    transformation_matrix(const affine_param<value_type,dimension>& rb,
                          const geo_type& from,
                          const vs_type& from_vs,
                          const geo_type& to,
                          const vs_type& to_vs)
    {
        if constexpr(dimension==3)
        {
            rotation_scaling_affine_matrix(rb.rotation,rb.scaling,rb.affine,sr,vdim<dimension>());
            vs_type t(from.width(),from.height(),from.depth());
            t *= value_type(-0.5);
            t[0] *= from_vs[0];
            t[1] *= from_vs[1];
            t[2] *= from_vs[2];
            t += rb.translocation;

            shift[0] = sr[0]*t[0]+sr[1]*t[1]+sr[2]*t[2];
            shift[1] = sr[3]*t[0]+sr[4]*t[1]+sr[5]*t[2];
            shift[2] = sr[6]*t[0]+sr[7]*t[1]+sr[8]*t[2];
            sr[0] *= from_vs[0];  sr[1] *= from_vs[1];  sr[2] *= from_vs[2];
            sr[3] *= from_vs[0];  sr[4] *= from_vs[1];  sr[5] *= from_vs[2];
            sr[6] *= from_vs[0];  sr[7] *= from_vs[1];  sr[8] *= from_vs[2];

            if(to_vs[0] != value_type(1))
            {
                sr[0] /= to_vs[0]; sr[1] /= to_vs[0]; sr[2] /= to_vs[0]; shift[0] /= to_vs[0];
            }
            if(to_vs[1] != value_type(1))
            {
                sr[3] /= to_vs[1]; sr[4] /= to_vs[1]; sr[5] /= to_vs[1]; shift[1] /= to_vs[1];
            }
            if(to_vs[2] != value_type(1))
            {
                sr[6] /= to_vs[2]; sr[7] /= to_vs[2]; sr[8] /= to_vs[2]; shift[2] /= to_vs[2];
            }
            shift[0] += value_type(to.width())*value_type(0.5);
            shift[1] += value_type(to.height())*value_type(0.5);
            shift[2] += value_type(to.depth())*value_type(0.5);
        }
        else
        {
            rotation_scaling_affine_matrix(rb.rotation[0],rb.scaling,rb.affine[0],sr,vdim<2>());
            vs_type t(from.width(),from.height());
            t *= value_type(-0.5);
            t[0] *= from_vs[0];
            t[1] *= from_vs[1];
            t += rb.translocation;

            shift[0] = sr[0]*t[0]+sr[1]*t[1];
            shift[1] = sr[2]*t[0]+sr[3]*t[1];
            sr[0] *= from_vs[0]; sr[1] *= from_vs[1];
            sr[2] *= from_vs[0]; sr[3] *= from_vs[1];

            if(to_vs[0] != value_type(1))
            {
                sr[0] /= to_vs[0]; sr[1] /= to_vs[0]; shift[0] /= to_vs[0];
            }
            if(to_vs[1] != value_type(1))
            {
                sr[2] /= to_vs[1]; sr[3] /= to_vs[1]; shift[1] /= to_vs[1];
            }
            shift[0] += value_type(to.width())*value_type(0.5);
            shift[1] += value_type(to.height())*value_type(0.5);
        }
    }

    template<typename geo_type,typename vs_type>
    auto to_affine_param(
                          const geo_type& from,
                          const vs_type& from_vs,
                          const geo_type& to,
                          const vs_type& to_vs) const
    {
        affine_param<value_type,3> rb;
        tipl::matrix<3,3,value_type> R,iR;
        std::copy_n(sr,9,R.begin());

        value_type t[3];
        t[0] = shift[0]-value_type(to.width())*value_type(0.5);
        t[1] = shift[1]-value_type(to.height())*value_type(0.5);
        t[2] = shift[2]-value_type(to.depth())*value_type(0.5);

        if(to_vs[2] != value_type(1))
        {
            R[6] *= to_vs[2]; R[7] *= to_vs[2]; R[8] *= to_vs[2]; t[2] *= to_vs[2];
        }
        if(to_vs[1] != value_type(1))
        {
            R[3] *= to_vs[1]; R[4] *= to_vs[1]; R[5] *= to_vs[1]; t[1] *= to_vs[1];
        }
        if(to_vs[0] != value_type(1))
        {
            R[0] *= to_vs[0]; R[1] *= to_vs[0]; R[2] *= to_vs[0]; t[0] *= to_vs[0];
        }
        R[0] /= from_vs[0]; R[1] /= from_vs[1]; R[2] /= from_vs[2];
        R[3] /= from_vs[0]; R[4] /= from_vs[1]; R[5] /= from_vs[2];
        R[6] /= from_vs[0]; R[7] /= from_vs[1]; R[8] /= from_vs[2];

        iR = tipl::inverse(R);
        rb.translocation[0] = (iR[0]*t[0]+iR[1]*t[1]+iR[2]*t[2])+value_type(from.width())*value_type(0.5)*from_vs[0];
        rb.translocation[1] = (iR[3]*t[0]+iR[4]*t[1]+iR[5]*t[2])+value_type(from.height())*value_type(0.5)*from_vs[1];
        rb.translocation[2] = (iR[6]*t[0]+iR[7]*t[1]+iR[8]*t[2])+value_type(from.depth())*value_type(0.5)*from_vs[2];
        matrix_to_rotation_scaling_affine(R.begin(),rb.rotation,rb.scaling,rb.affine,vdim<dimension>());
        return rb;
    }
public:
    template<typename rhs_value_type>
    auto& operator=(const transformation_matrix<rhs_value_type,dimension>& M)
    {
        std::copy_n(M.data_, total_size, data_);
        return *this;
    }
    template<typename rhs_value_type>
    auto& operator=(const tipl::matrix<dimension+1,dimension+1,rhs_value_type>& M)
    {
        if constexpr(dimension == 3)
        {
            data_[0] = M[0];  data_[1] = M[1];  data_[2] = M[2];
            data_[3] = M[4];  data_[4] = M[5];  data_[5] = M[6];
            data_[6] = M[8];  data_[7] = M[9];  data_[8] = M[10];
            data_[9] = M[3];  data_[10] = M[7]; data_[11] = M[11];
        }
        else
        {
            data_[0] = M[0];  data_[1] = M[1];
            data_[2] = M[3];  data_[3] = M[4];
            data_[4] = M[2];  data_[5] = M[5];
        }
        return *this;
    }
public:

    transformation_matrix& accumulate(const transformation_matrix& rhs)
    {
        tipl::matrix<dimension,dimension,value_type> sr_tmp(sr);
        tipl::mat::product(rhs.sr,sr_tmp.begin(),sr,tipl::dim<dimension,dimension>(),tipl::dim<dimension,dimension>());
        value_type shift_t[dimension];
        for(char d = 0;d < dimension;++d)
            shift_t[d] = shift[d];
        vector_transformation(shift_t,shift,rhs.sr,rhs.shift,vdim<dimension>());
        return *this;
    }

    auto to_matrix(void) const
    {
        tipl::matrix<dimension+1,dimension+1,value_type> M;
        if constexpr(dimension == 3)
        {
            M[0] = data_[0];  M[1] = data_[1];   M[2] = data_[2];
            M[4] = data_[3];  M[5] = data_[4];   M[6] = data_[5];
            M[8] = data_[6];  M[9] = data_[7];   M[10] = data_[8];
            M[3] = data_[9];  M[7] = data_[10];  M[11] = data_[11];
            M[12] = 0.0f;     M[13] = 0.0f;      M[14] = 0.0f;     M[15] = 1.0f;
        }
        else
        {
            M[0] = data_[0];  M[1] = data_[1];
            M[3] = data_[2];  M[4] = data_[3];
            M[2] = data_[4];  M[5] = data_[5];
            M[6] = 0.0f;      M[7] = 0.0f;       M[8] = 1.0f;
        }
        return M;
    }

    void to(tipl::matrix<dimension+1,dimension+1,value_type>& M) const
    {
        if constexpr(dimension == 3)
        {
            M[0] = data_[0];  M[1] = data_[1];   M[2] = data_[2];
            M[4] = data_[3];  M[5] = data_[4];   M[6] = data_[5];
            M[8] = data_[6];  M[9] = data_[7];   M[10] = data_[8];
            M[3] = data_[9];  M[7] = data_[10];  M[11] = data_[11];
            M[12] = 0.0f;     M[13] = 0.0f;      M[14] = 0.0f;     M[15] = 1.0f;
        }
        else
        {
            M[0] = data_[0];  M[1] = data_[1];
            M[3] = data_[2];  M[4] = data_[3];
            M[2] = data_[4];  M[5] = data_[5];
            M[6] = 0.0f;      M[7] = 0.0f;       M[8] = 1.0f;
        }
    }

    bool inverse(void)
    {
        tipl::matrix<dimension,dimension,value_type> iT(sr);
        if(!iT.inv())
            return false;
        value_type new_shift[dimension];
        vector_rotation(shift,new_shift,iT.begin(),vdim<dimension>());
        for(unsigned int d = 0;d < dimension;++d)
            shift[d] = -new_shift[d];
        for(unsigned int d = 0;d < dimension*dimension;++d)
            sr[d] = iT[d];
        return true;
    }

    [[nodiscard]] __INLINE__ auto operator()(const pixel_index<dimension>& from) const
    {
        vector<dimension> to;
        vector_transformation(vector<dimension>(from).begin(),to.begin(),sr,shift,vdim<dimension>());
        return to;
    }
    [[nodiscard]] __INLINE__ auto operator()(const vector<dimension>& from) const
    {
        vector<dimension> to;
        vector_transformation(from.begin(),to.begin(),sr,shift,vdim<dimension>());
        return to;
    }
    __INLINE__ void operator()(vector<dimension>& pos) const
    {
        vector<dimension> result(pos);
        vector_transformation(result.begin(),pos.begin(),sr,shift,vdim<dimension>());
    }
    __INLINE__ void operator()(value_type* pos) const
    {
        vector<dimension> result(pos);
        vector_transformation(result.begin(),pos,sr,shift,vdim<dimension>());
    }

    template <tipl::interpolation itype = linear,typename image_type,
              std::enable_if_t<tipl::is_image_v<image_type>, bool> = true>
    [[nodiscard]] __INLINE__ typename image_type::buffer_type operator()(const image_type& I,const shape<dim>& sp) const;


    template <tipl::interpolation itype = linear,typename image_type1, typename image_type2,
                  std::enable_if_t<tipl::is_image_v<image_type1> && tipl::is_image_v<image_type2>, bool> = true>
    __INLINE__ void operator()(const image_type1& I,image_type2&& I2) const;

    friend std::ostream & operator<<(std::ostream& out, const transformation_matrix& T)
    {
        if constexpr(dimension==3)
        {
            out << T.data_[0] << " " << T.data_[1] << " " << T.data_[2] << " " << T.data_[9] << std::endl;
            out << T.data_[3] << " " << T.data_[4] << " " << T.data_[5] << " " << T.data_[10] << std::endl;
            out << T.data_[6] << " " << T.data_[7] << " " << T.data_[8] << " " << T.data_[11] << std::endl;
        }
        else
        {
            out << T.data_[0] << " " << T.data_[1] << " " << T.data_[4] << std::endl;
            out << T.data_[2] << " " << T.data_[3] << " " << T.data_[5] << std::endl;
        }
        return out;
    }
    friend std::istream& operator>>(std::istream& in, transformation_matrix& T)
    {
        if constexpr(dimension==3)
        {
            in >> T.data_[0] >> T.data_[1] >> T.data_[2] >> T.data_[9];
            in >> T.data_[3] >> T.data_[4] >> T.data_[5] >> T.data_[10];
            in >> T.data_[6] >> T.data_[7] >> T.data_[8] >> T.data_[11];
        }
        else
        {
            in >> T.data_[0] >> T.data_[1] >> T.data_[4];
            in >> T.data_[2] >> T.data_[3] >> T.data_[5];
        }
        return in;
    }
};

template<typename geo_type,typename vs_type,typename value_type>
void inverse(affine_param<value_type,3>& arg,
                      const geo_type& from,
                      const vs_type& from_vs,
                      const geo_type& to,
                      const vs_type& to_vs)
{
    auto T = tipl::transformation_matrix<value_type,3>(arg,from,from_vs,to,to_vs);
    T.inverse();
    arg = T.to_affine_param(to,to_vs,from,from_vs);
}

class from_space : public tipl::matrix<4,4,float>{
private:
    const tipl::matrix<4,4,float>& origin;
public:
    from_space(const tipl::matrix<4,4,float>& space_):tipl::matrix<4,4,float>(),origin(space_){}
    from_space& to(const tipl::matrix<4,4,float>& target)
    {
        for(unsigned int i = 0;i < 16;++i)
            (*this)[i] = target[i];
        tipl::matrix<4,4,float>::inv();
        (*this) *= origin;
        return *this;
    }
};

template<typename value_type>
inline std::vector<value_type> to_vs(const tipl::matrix<4,4,value_type>& trans)
{
    return {std::sqrt(trans[0]*trans[0]+trans[4]*trans[4]+trans[8]*trans[8]),
            std::sqrt(trans[1]*trans[1]+trans[5]*trans[5]+trans[9]*trans[9]),
            std::sqrt(trans[2]*trans[2]+trans[6]*trans[6]+trans[10]*trans[10])};
}

template<typename image_type, typename v_type>
void estimate_affine_param(const image_type& source, const v_type& source_vs,
                               const image_type& target, const v_type& target_vs,
                               affine_param<float, 3>& arg)
{
    arg.clear();
    tipl::vector<3, double> c_s, c_t;
    tipl::matrix<3, 3, double> cov_s, cov_t;

    auto compute_moments = [&](const image_type& img, const v_type& vs,
                               tipl::vector<3, double>& center, tipl::matrix<3, 3, double>& cov)
    {
        size_t sum(0);
        center = {0.0, 0.0, 0.0};
        const size_t sz = img.size(); // HOISTED size evaluation out of loop

        for (tipl::pixel_index<3> i(img.shape()); i < sz; ++i)
            if (img[i.index()] > 0)
            {
                center[0] += i[0] * vs[0];
                center[1] += i[1] * vs[1];
                center[2] += i[2] * vs[2];
                ++sum;
            }

        if (sum == 0)
            return sum;

        center /= double(sum);
        std::fill(cov.begin(), cov.end(), 0.0);

        for (tipl::pixel_index<3> i(img.shape()); i < sz; ++i)
            if (img[i.index()] > 0)
            {
                double d[3] = {i[0] * vs[0] - center[0], i[1] * vs[1] - center[1], i[2] * vs[2] - center[2]};
                cov[0] += d[0] * d[0];
                cov[1] += d[0] * d[1];
                cov[2] += d[0] * d[2];
                cov[4] += d[1] * d[1];
                cov[5] += d[1] * d[2];
                cov[8] += d[2] * d[2];
            }

        cov[3] = cov[1];
        cov[6] = cov[2];
        cov[7] = cov[5];

        tipl::divide_constant(cov.begin(), cov.end(), double(sum));
        return sum;
    };

    if (compute_moments(source, source_vs, c_s, cov_s) <= 0 || compute_moments(target, target_vs, c_t, cov_t) <= 0)
        return;

    double Vs[9] = {0}, Vt[9] = {0}, Ls[3] = {0}, Lt[3] = {0}, R[9] = {0};
    tipl::mat::eigen_decomposition_sym(cov_s.begin(), Vs, Ls, tipl::dim<3, 3>());
    tipl::mat::eigen_decomposition_sym(cov_t.begin(), Vt, Lt, tipl::dim<3, 3>());

    for (int i = 0; i < 3; ++i)
        arg.scaling[i] = (float)std::sqrt(std::max(Lt[i], 0.0) / std::max(Ls[i], 1e-6));

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                R[i * 3 + j] += Vt[i + k * 3] * Vs[j + k * 3];

    if (tipl::mat::determinant(R,tipl::dim<3, 3>()) < 0.0)
        for (int i = 0; i < 3; ++i)
            R[i * 3 + 2] = -R[i * 3 + 2];

    arg.rotation[0] = (float)std::atan2(R[7], R[8]);
    arg.rotation[1] = (float)std::atan2(-R[6], std::sqrt(R[7] * R[7] + R[8] * R[8]));
    arg.rotation[2] = (float)std::atan2(R[3], R[0]);

    // Constrain rotation to [-pi/4, pi/4] to correct 90-degree flipped principal axes
    const float pi_half = 1.570796327f;
    for(int i = 0; i < 3; ++i)
        arg.rotation[i] = std::remainder(arg.rotation[i], pi_half);

    double dt[3];
    for (int i = 0; i < 3; ++i)
        dt[i] = (c_t[i] - target.shape()[i] * target_vs[i] * 0.5) / std::max<double>(arg.scaling[i], 1e-6);

    for (int i = 0; i < 3; ++i)
    {
        double dt_rot = R[0 * 3 + i] * dt[0] + R[1 * 3 + i] * dt[1] + R[2 * 3 + i] * dt[2];
        arg.translocation[i] = (float)(dt_rot - (c_s[i] - source.shape()[i] * source_vs[i] * 0.5));
    }
}

}

#endif // TRANSFORMATION_HPP_INCLUDED
