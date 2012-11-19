#ifndef TRANSFORMATION_HPP_INCLUDED
#define TRANSFORMATION_HPP_INCLUDED

namespace image{



template<unsigned int dim>
struct vdim {};

template<typename input_iter1,typename input_iter2,typename output_iter>
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

template<typename input_iter1,typename input_iter2,typename output_iter>
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

template<typename input_iter1,typename input_iter2,typename input_iter3,typename output_iter>
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


template<typename input_iter1,typename input_iter2,typename input_iter3,typename output_iter>
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

template<typename input_iter1,typename input_iter2,typename output_iter>
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

template<typename output_iter>
void rotation_matrix(typename std::iterator_traits<output_iter>::value_type theta,output_iter m,vdim<2>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    value_type cos_theta = std::cos(theta);
    value_type sin_theta = std::sin(theta);
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
template<typename angle_type,typename output_type>
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
}

/*
 Rx*Ry*Rz*Scaling
 */
template<typename angle_type,typename scale_type,typename output_type>
void rotation_scaling_matrix(angle_type theta,scale_type s,output_type m,vdim<3>)
{
    rotation_matrix(theta,m,vdim<3>());
    m[0] *= s[0];
    m[1] *= s[1];
    m[2] *= s[2];
    m[3] *= s[0];
    m[4] *= s[1];
    m[5] *= s[2];
    m[6] *= s[0];
    m[7] *= s[1];
    m[8] *= s[2];
}
/*
 Rx*Ry*Rz*Scaling*Affine

Affine   = [1   	a[0]    a[1]   0;
            0   	1 	a[2]   0;
            0   	0   	1      0;
            0    	0    	0      1];
 */
template<typename angle_type,typename scale_type,typename affine_type,typename output_type>
void rotation_scaling_affine_matrix(angle_type theta,scale_type s,affine_type a,output_type m,vdim<3>)
{
    rotation_scaling_matrix(theta,s,m,vdim<3>());
    m[2] += m[0]*a[1]+m[1]*a[2];
    m[1] += m[0]*a[0];

    m[5] += m[3]*a[1]+m[4]*a[2];
    m[4] += m[3]*a[0];

    m[8] += m[6]*a[1]+m[7]*a[2];
    m[7] += m[6]*a[0];
}

template<typename output_iter>
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

template<typename output_iter>
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

template<typename output_iter>
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

template<typename input_iter1,typename input_iter2,typename output_iter>
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


template<typename input_iter,typename output_iter>
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



template<typename input_scaling_iter,typename angle_type,typename output_iter>
void rotation_angle_to_rotation_matrix(input_scaling_iter scaling,angle_type rotation,output_iter m,vdim<2>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    value_type S[4],R[4];
    scaling_matrix(scaling,S,vdim<2>());
    rotation_matrix(rotation[0],R,vdim<2>());
    matrix_product(R,S,m,vdim<2>());
}

// the rotation is the Euler angles, which has Z-X-Z configuration
template<typename input_scaling_iter,typename input_rotation_iter,typename output_iter>
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
template<typename input_rotation_iter,typename output_iter>
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
template<typename input_rotation_iter,typename output_iter>
void rotation_matrix_to_rotation_angle(input_rotation_iter rotation_matrix,output_iter rotation_angle,vdim<3>)
{
    rotation_angle[0] = std::atan2(rotation_matrix[6],rotation_matrix[7]); //Z
    rotation_angle[1] = std::acos(rotation_matrix[8]);//X
    rotation_angle[2] = -std::atan2(rotation_matrix[2],rotation_matrix[5]);//Z
}


template<typename input_rotation_iter,typename input_shift_iter,typename output_iter>
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

template<typename input_scaling_iter,typename input_rotation_iter,typename input_shift_iter,typename output_iter>
void create_affine_transformation_matrix(input_scaling_iter scaling,input_rotation_iter rotation,input_shift_iter shift,output_iter m,vdim<3>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    value_type M[9];
    scaling_rotation_matrix(scaling,rotation,M);
    create_transformation_matrixation_matrix(M,shift,m,vdim<3>());
}

template<typename input_rotation_iter,typename input_shift_iter,typename output_iter>
void create_rigidbody_transformation_matrix(input_rotation_iter rotation,input_shift_iter shift,output_iter m,vdim<3>)
{
    typedef typename std::iterator_traits<output_iter>::value_type value_type;
    value_type M[9];
    rotation_matrix(rotation,M);
    create_transformation_matrixation_matrix_3d(M,shift,m,vdim<3>());
}


template<unsigned int dim,typename value_type_ = float>
struct rigid_body_transform
{
    typedef value_type_ value_type;
    static const unsigned int dimension = dim;
    static const unsigned int total_size = dimension+dimension;
    union
    {
        struct
        {
            value_type translocation[dimension];
            value_type rotation[dimension];
        };
        value_type data[total_size];
    };
private:
    void assign(const rigid_body_transform& rhs)
    {
        std::copy(rhs.data,rhs.data+total_size,data);
    }
public:
    rigid_body_transform(void)
    {
        std::fill(data,data+total_size,0);
    }
    rigid_body_transform(const rigid_body_transform& rhs)
    {
        assign(rhs);
    }

    const rigid_body_transform& operator=(const rigid_body_transform& rhs)
    {
        assign(rhs);
        return *this;
    }
    value_type operator[](unsigned int i) const{return data[i];}
    value_type& operator[](unsigned int i) {return data[i];}

};


template<unsigned int dim,typename value_type_ = float>
struct rigid_scaling_transform
{
    typedef value_type_ value_type;
    static const unsigned int dimension = dim;
    static const unsigned int total_size = dimension+dimension+dimension;
    union
    {
        struct
        {
            value_type translocation[dimension];
            value_type rotation[dimension];
            value_type scaling[dimension];
        };
        value_type data[total_size];
    };
private:
    void assign(const rigid_scaling_transform& rhs)
    {
        std::copy(rhs.data,rhs.data+total_size,data);
    }
public:
    rigid_scaling_transform(void)
    {
        std::fill(data,data+dimension+dimension,0);
        std::fill(scaling,scaling+dimension,1);
    }
    rigid_scaling_transform(const rigid_scaling_transform& rhs)
    {
        assign(rhs);
    }

    const rigid_scaling_transform& operator=(const rigid_scaling_transform& rhs)
    {
        assign(rhs);
        return *this;
    }
    value_type operator[](unsigned int i) const{return data[i];}
    value_type& operator[](unsigned int i) {return data[i];}


};

template<unsigned int dim,typename value_type_ = float>
struct affine_transform
{
    typedef value_type_ value_type;
    static const unsigned int dimension = dim;
    static const unsigned int affine_dim = (dimension-1)*(dimension)/2;
    static const unsigned int total_size = dimension+dimension+dimension+affine_dim;
    union
    {
        struct
        {
            value_type translocation[dimension];
            value_type rotation[dimension];
            value_type scaling[dimension];
            value_type affine[affine_dim];
        };
        value_type data[total_size];
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

};




template<unsigned int dim,typename value_type_ = float>
struct transformation_matrix
{
    typedef value_type_ value_type;
    static const unsigned int dimension = dim;
    static const unsigned int scaling_rotation_size = dimension*dimension;
    static const unsigned int total_size = dimension*dimension+dimension;
public:
    union
    {
        struct
        {
            value_type scaling_rotation[scaling_rotation_size];
            value_type shift[dimension];
        };
        value_type data[total_size];
    };

public:
    transformation_matrix(void)
    {
        std::fill((value_type*)data,(value_type*)data+total_size,0);
    }

    // m = T*R1*R2*R3*Scaling*Affine;
    transformation_matrix(const affine_transform<dim,value_type>& rb)
    {
        rotation_scaling_affine_matrix(rb.rotation,rb.scaling,rb.affine,scaling_rotation,vdim<dimension>());
        std::copy(rb.translocation,rb.translocation+dimension,shift);
    }
    transformation_matrix(const rigid_scaling_transform<dim,value_type>& rb)
    {
        rotation_scaling_matrix(rb.rotation,rb.scaling,scaling_rotation,vdim<dimension>());
        std::copy(rb.translocation,rb.translocation+dimension,shift);
    }
    transformation_matrix(const rigid_body_transform<dim,value_type>& rb)
    {
        rotation_matrix(rb.rotation,scaling_rotation,vdim<dim>());
        std::copy(rb.translocation,rb.translocation+dimension,shift);
    }
    const transformation_matrix& operator=(const transformation_matrix& rhs)
    {
        std::copy(rhs.data,rhs.data+total_size,data);
        return *this;
    }
    value_type* get(void)
    {
        return data;
    }
    value_type operator[](unsigned int i) const{return data[i];}
    value_type& operator[](unsigned int i) {return data[i];}
 
    // load from 4 x 3 M matrix
    template<typename InputIterType>
    void load_from_transform(InputIterType M)
    {
        std::copy(M,M+3,data);
        std::copy(M+4,M+7,data+3);
        std::copy(M+8,M+11,data+6);
        data[9] = M[3];
        data[10] = M[7];
        data[11] = M[11];
    }
    template<typename InputIterType>
    void save_to_transform(InputIterType M)
    {
        std::copy(data,data+3,M);
        std::copy(data+3,data+6,M+4);
        std::copy(data+6,data+9,M+8);
        M[3] = data[9];
        M[7] = data[10];
        M[11] = data[11];
    }
    void inverse(void)
    {
        std::vector<value_type> T(16);
        save_to_transform(T.begin());
        T[15] = 1.0;
        math::matrix_inverse(T.begin(),math::dim<4,4>());
        load_from_transform(T.begin());
    }

    template<typename InputIterType,typename OutputIterType>
    void operator()(InputIterType in_iter,OutputIterType out_iter) const
    {
        vector_transformation(in_iter,out_iter,scaling_rotation,shift,vdim<dimension>());
    }

    template<typename IterType>
    void operator()(IterType in_iter) const
    {
        value_type value[dimension];
        vector_transformation(in_iter,value,scaling_rotation,shift,vdim<dimension>());
        std::copy(value,value+dimension,in_iter);
    }

};





}
#endif // TRANSFORMATION_HPP_INCLUDED
