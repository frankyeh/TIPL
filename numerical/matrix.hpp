#ifndef IMAGE_MATRIX_HPP
#define IMAGE_MATRIX_HPP
// Copyright Fang-Cheng Yeh 2010
// Distributed under the BSD License
//
/*
Copyright (c) 2010, Fang-Cheng Yeh
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <iterator>
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>

namespace image
{


// data type for specifying the matrix dimension in compiler time
template<unsigned int row,unsigned int col>
struct dim
{
    unsigned int row_count(void)const
    {
        return row;
    }
    unsigned int col_count(void)const
    {
        return col;
    }
    unsigned int size(void)const
    {
        return row*col;
    }
};


template<class value_type>
struct one{
    value_type operator[](unsigned int)const{return value_type(1);}
    value_type operator*(void) const{return value_type(1);}
    value_type operator[](unsigned int){return value_type(1);}
    value_type operator*(void){return value_type(1);}
};

template<class value_type>
struct zero{
    value_type operator[](unsigned int)const{return value_type(0);}
    value_type operator*(void) const{return value_type(0);}
    value_type operator[](unsigned int){return value_type(0);}
    value_type operator*(void){return value_type(0);}
};

// data type for specifying the matrix dimension in runtime

struct dyndim
{
    unsigned int row,col;
    dyndim(void) {}
    dyndim(unsigned int row_):row(row_),col(1) {}
    dyndim(unsigned int row_,unsigned int col_):row(row_),col(col_) {}
    unsigned int row_count(void)const
    {
        return row;
    }
    unsigned int col_count(void)const
    {
        return col;
    }
    unsigned int size(void)const
    {
        return row*col;
    }
};

namespace vec
{


template<class lhs_type,class rhs_type>
typename std::iterator_traits<lhs_type>::value_type
dot(lhs_type v1,lhs_type v1_end,rhs_type v2)
{
    typedef typename std::iterator_traits<lhs_type>::value_type value_type;
    if (v1 == v1_end)
        return value_type(0);
    value_type sum((*v1)*(*v2));
    if (++v1 != v1_end)
        do
        {
            ++v2;
            sum += (*v1)*(*v2);
        }
        while (++v1 != v1_end);
    return sum;
}

/*
perform x <- -x
*/
template<class lhs_type>
void negate(lhs_type x_begin,lhs_type x_end)
{
    for (;x_begin != x_end;++x_begin)
        (*x_begin) = -(*x_begin);
}

/*
perform x <- ax
*/
template<class lhs_type,class scalar_type>
void scale(lhs_type x_begin,lhs_type x_end,scalar_type a)
{
    for (;x_begin != x_end;++x_begin)
        (*x_begin) *= a;
}

/*
perform y <- ax
*/
template<class lhs_type,class rhs_type,class scalar_type>
void scale(lhs_type x_begin,lhs_type x_end,rhs_type y,scalar_type a)
{
    if (x_begin != x_end)
        do
        {
            (*y) = (*x_begin)*a;
            if (++x_begin == x_end)
                return;
            ++y;
        }
        while (1);
}

/*
calculate norm1
*/
template<class lhs_type>
typename std::iterator_traits<lhs_type>::value_type
norm1(lhs_type x_begin,lhs_type x_end)
{
    typedef typename std::iterator_traits<lhs_type>::value_type value_type;
    if (x_begin == x_end)
        return value_type(0);

    value_type sum(std::abs(*x_begin));
    while (++x_begin != x_end)
        sum += std::abs(*x_begin);
    return sum;
}

/*
calculate norm2
*/
template<class lhs_type>
typename std::iterator_traits<lhs_type>::value_type
norm2(lhs_type x_begin,lhs_type x_end)
{
    typedef typename std::iterator_traits<lhs_type>::value_type value_type;
    // dimension = 0
    if (x_begin == x_end)
        return value_type(0);

    // dimension = 1
    value_type sum(*x_begin);
    if (++x_begin == x_end)
        return std::abs(sum);

    // dimension >= 2
    sum *= sum;
    value_type x2(*x_begin);
    x2 *= x2;
    sum += x2;
    while (++x_begin != x_end)
    {
        x2 = *x_begin;
        x2 *= x2;
        sum += x2;
    }
    return std::sqrt(sum);
}

/*
swap vector x, y
*/
template<class lhs_type,class rhs_type>
void swap(lhs_type x_begin,lhs_type x_end,rhs_type y)
{
    if (x_begin != x_end)
        do
        {
            std::swap(*x_begin,*y);
            if (++x_begin == x_end)
                return;
            ++y;
        }
        while (1);
}

/*
perform y <- y+x
*/
template<class lhs_type,class rhs_type>
void add(lhs_type y_begin,lhs_type y_end,rhs_type x)
{
    if (y_begin != y_end)
        do
        {
            *y_begin += (*x);
            if (++y_begin == y_end)
                return;
            ++x;
        }
        while (1);
}
/*
perform y <- y-x
*/
template<class lhs_type,class rhs_type>
void minus(lhs_type y_begin,lhs_type y_end,rhs_type x)
{
    if (y_begin != y_end)
        do
        {
            *y_begin -= (*x);
            if (++y_begin == y_end)
                return;
            ++x;
        }
        while (1);
}


/*
perform y <- ax+y
*/
template<class lhs_type,class rhs_type,class scalar_type>
void axpy(lhs_type y_begin,lhs_type y_end,scalar_type a,rhs_type x)
{
    if (y_begin != y_end)
        do
        {
            *y_begin += (*x)*a;
            if (++y_begin == y_end)
                return;
            ++x;
        }
        while (1);
}


/*
perform x <- ay+x
*/
template<class lhs_type,class rhs_type,class scalar_type>
void aypx(lhs_type y_begin,lhs_type y_end,scalar_type a,rhs_type x)
{
    if (y_begin != y_end)
        do
        {
            (*x) += (*y_begin)*a;
            if (++y_begin == y_end)
                return;
            ++x;
        }
        while (1);
}



/*
perform x <- c*x+s*y
perform y <- c*y-s*x
*/
template<class lhs_type,class rhs_type,class scalar_type>
void rot(lhs_type x_begin,lhs_type x_end,rhs_type y,scalar_type c,scalar_type s)
{
    typename std::iterator_traits<lhs_type>::value_type x_temp;
    if (x_begin != x_end)
        do
        {

            x_temp = (*x_begin)*c + (*y)*s;
            *y     = (*y)*c       - (*x_begin)*s;
            *x_begin = x_temp;

            if (++x_begin == x_end)
                return;
            ++y;
        }
        while (1);
}



/*
perform A=x*y'
*/

template<class left_input_iterator,
typename right_input_iterator,
typename output_iterator>
void gen(left_input_iterator x,left_input_iterator x_end,right_input_iterator y,output_iterator out)
{
    unsigned int dim = (x_end-x);
    right_input_iterator y_end = y + dim;
    if (x != x_end)
        do
        {
            scale(y,y_end,out,*x);
            if (++x == x_end)
                return;
            out += dim;
        }
        while (1);
}

/*
perform A=x*y'
*/

template<class left_input_iterator,
typename right_input_iterator,
typename output_iterator>
void gen(left_input_iterator x,left_input_iterator x_end,right_input_iterator y,right_input_iterator y_end,output_iterator out)
{
    unsigned int dim = (y_end-y);
    if (x != x_end)
        do
        {
            scale(y,y_end,out,*x);
            if (++x == x_end)
                return;
            out += dim;
        }
        while (1);
}


}

namespace mat
{



/*
perform y = Ax
*/

template<class left_input_iterator,
typename right_input_iterator,
typename output_iterator,
typename left_dim_type>
void vector_product(left_input_iterator A,right_input_iterator x,output_iterator y,const left_dim_type& ldim)
{
    left_input_iterator A_end = A + ldim.size();
    if (A == A_end)
        return;
    unsigned int common_col_count = ldim.col_count();
    left_input_iterator A_next;
    do
    {
        A_next = A + common_col_count;
        *y = image::vec::dot(A,A_next,x);
        if (A_next == A_end)
            return;
        A = A_next;
        ++y;
    }
    while (1);
}

/*
perform y = xA
*/

template<class left_input_iterator,
typename right_input_iterator,
typename output_iterator,
typename left_dim_type>
void left_vector_product(left_input_iterator A,right_input_iterator x,output_iterator y,const left_dim_type& ldim)
{
    std::fill(y,y+ldim.col_count(),0);
    for(left_input_iterator A_end = A + ldim.size();A != A_end;A+=ldim.col_count(),++x)
    {
        typename std::iterator_traits<left_input_iterator>::value_type x_row = *x;
        for(unsigned int col = 0;col < ldim.col_count();++col)
            y[col] += x_row*A[col];
    }
}


/**
perform A*B

INPUT: must be random access iterator
OUTPUT: random access iterator or bidirectional iterator

*/


template<class left_input_iterator,
typename right_input_iterator,
typename output_iterator,
typename left_dim_type,
typename right_dim_type>
void product(left_input_iterator lhs			    /*A*/,
                    right_input_iterator rhs			/*B*/,
                    output_iterator out					/*output*/,
                    const left_dim_type& ldim			/* the dimension of A*/,
                    const right_dim_type& rdim			/* the dimension of B*/)
{
    unsigned int common_col_count = ldim.col_count();
    unsigned int right_col_count = rdim.col_count();
    left_input_iterator lhs_end = lhs + ldim.size();
    right_input_iterator rhs_end = rhs + rdim.col_count();

    while (lhs != lhs_end)
    {
        left_input_iterator lhs_to = lhs + common_col_count;
        for (right_input_iterator rhs_col = rhs;rhs_col != rhs_end;++rhs_col,++out)
        {
            right_input_iterator rhs_from = rhs_col;
            left_input_iterator lhs_from = lhs;
            typename std::iterator_traits<left_input_iterator>::value_type sum((*lhs_from)*(*rhs_from));
            if (++lhs_from != lhs_to)
                do
                {
                    rhs_from += right_col_count;
                    sum += (*lhs_from)*(*rhs_from);
                }
                while (++lhs_from != lhs_to);
            *out = sum;
        }
        lhs = lhs_to;
    }
}

/**
perform A*Bt

INPUT: must be random access iterator
OUTPUT: random access iterator or bidirectional iterator

*/
template<class left_input_iterator,
typename right_input_iterator,
typename output_iterator,
typename left_dim_type,
typename right_dim_type>
void product_transpose(
    left_input_iterator lhs			    /*A*/,
    right_input_iterator rhs			/*B*/,
    output_iterator out					/*output*/,
    const left_dim_type& ldim			/* the dimension of A*/,
    const right_dim_type& rdim			/* the dimension of B*/)
{
    unsigned int common_col_count = ldim.col_count();
    left_input_iterator lhs_end = lhs + ldim.size();
    right_input_iterator rhs_end = rhs + rdim.size();

    for (;lhs != lhs_end;lhs += common_col_count)
        for (right_input_iterator rhs_iter = rhs;rhs_iter != rhs_end;rhs_iter += common_col_count,++out)
            *out = image::vec::dot(lhs,lhs+common_col_count,rhs_iter);

}


/*
perform A*At
INPUT: must be random access iterator
OUTPUT: must be random access iterator
*/

template<class input_iterator,
typename output_iterator,
typename dim_type>
void square(input_iterator lhs,output_iterator out,const dim_type& dim)
{
    output_iterator iter = out;

    unsigned int common_col_count = dim.col_count();
    input_iterator rhs = lhs;
    input_iterator lhs_end = lhs + dim.size();
    input_iterator rhs_end = rhs + dim.size();
    for (unsigned int row = 0;lhs != lhs_end;lhs += common_col_count,++row)
    {
        input_iterator rhs_iter = rhs;
        for (unsigned int col = 0;rhs_iter != rhs_end;rhs_iter += common_col_count,++out,++col)
            if (row <= col)// skip the symmetric part
                *out = image::vec::dot(lhs,lhs+common_col_count,rhs_iter);
    }

    unsigned int row_count = dim.row_count();
    if (row_count > 1)
    {
        input_iterator col_wise = iter + 1;
        input_iterator row_wise = iter + row_count;
        unsigned int shift = row_count + 1;
        for (unsigned int length = row_count - 1;1;col_wise += shift,row_wise += shift)
        {
            input_iterator col_from = col_wise;
            input_iterator col_to = col_wise + length;
            input_iterator row_from = row_wise;
            while (1)
            {
                *row_from = *col_from;
                if (++col_from == col_to)
                    break;
                row_from += row_count;
            }
            if (--length <= 0)
                break;
        }
    }
}







/**
example:
\code
double sym[]={12, 8, 3, 1,
               8, 4, 2, 5,
               3, 2,11, 4,
               1, 5, 4, 7};
is_symmetric(sym,dim<4,4>());
\endcode
*/
template<class input_iterator,class dim_type>
bool is_symmetric(input_iterator iter,const dim_type& dim)
{
    input_iterator col_wise = iter + 1;
    input_iterator row_wise = iter + dim.col_count();
    unsigned int row_count = dim.row_count();
    unsigned int shift = dim.col_count() + 1;
    for (unsigned int length = dim.col_count() - 1;length > 0;--length)
    {
        input_iterator col_from = col_wise;
        input_iterator col_to = col_wise + length;
        input_iterator row_from = row_wise;
        while (1)
        {
            if (*col_from != *row_from)
                return false;
            ++col_from;
            if (col_from == col_to)
                break;
            row_from += row_count;
        }
        col_wise += shift;
        row_wise += shift;
    }
    return true;
}

/**
example:
\code
double sym[]={12, 8, 3, 1,
       9, 4, 2, 5,
       5, 3,11, 4,
       2, 5, 6, 7};
transpose(sym,dim<4,4>());
\endcode
*/
template<class input_iterator,unsigned int matrix_dim>
void transpose(input_iterator A,dim<matrix_dim,matrix_dim>)
{
    if (matrix_dim > 1)
    {
        input_iterator col_wise = A + 1;
        input_iterator row_wise = A + matrix_dim;
        unsigned int shift = matrix_dim + 1;
        for (unsigned int length = matrix_dim - 1;1;col_wise += shift,row_wise += shift)
        {
            input_iterator col_from = col_wise;
            input_iterator col_to = col_wise + length;
            input_iterator row_from = row_wise;
            while (1)
            {
                std::swap(*col_from,*row_from);
                if (++col_from == col_to)
                    break;
                row_from += matrix_dim;
            }
            if (--length <= 0)
                break;
        }
    }
}



template<class input_iterator,class output_iterator,class dim_type>
void transpose(input_iterator in,output_iterator out,const dim_type& dim)
{
    unsigned int col = 0;
    unsigned int out_leap = dim.row_count();
    output_iterator out_col = out + col;
    output_iterator out_end = out + dim.size()-out_leap;// last leap position
    for (input_iterator end = in + dim.size();in != end;++in)
    {
        *out_col = *in;
        if (out_col >= out_end)
        {
            ++col;
            out_col = out + col;
        }
        else
            out_col += out_leap;
    }
}

template<class io_iterator,class dim_type>
void transpose(io_iterator io,const dim_type& dim)
{
    typedef typename std::iterator_traits<io_iterator>::value_type value_type;
    std::vector<value_type> temp(io,io+dim.size());
    transpose(temp.begin(),io,dim);
}

template<class input_iterator,class dim_type>
typename std::iterator_traits<input_iterator>::value_type
trace(input_iterator A,const dim_type& dim)
{
    typedef typename std::iterator_traits<input_iterator>::value_type value_type;
    value_type result = A[0];
    unsigned int leap_size = dim.col_count()+1;
    for (unsigned int index = leap_size;index < dim.size();index += leap_size)
        result += A[index];
    return result;
}


template<class input_iterator,class value_type>
void col_rotate_dyn(input_iterator col1,input_iterator col2,
                           value_type c,value_type s,unsigned int row_count,unsigned int col_count)
{
    value_type temp;
    unsigned int row = 0;
    do
    {
        temp = *col2;
        *col2 = s * (*col1) + c * temp;
        *col1 = c * (*col1) - s * temp;
        if (++row == row_count)
            break;
        col1 += col_count;
        col2 += col_count;
    }
    while (1);
}



template <class iterator_type,class dim_type>
void identity(iterator_type I,const dim_type& dim)
{
    unsigned int size = dim.size();
    typedef typename std::iterator_traits<iterator_type>::value_type value_type;
    std::fill(I,I+size,value_type(0));
    unsigned int leap_size = dim.col_count()+1;
    for (unsigned int index = 0;index < size;index += leap_size)
        I[index] = value_type(1);
}




/**

	double A[]={8, 1, 3,
				7, 0, 2,
                12, 3, 2};
	unsigned int pivot[3];
    math::lu_decomposition(A,pivot,math::dim<3,3>());

*/
template<class io_iterator,class pivot_iterator,class dim_type>
bool lu_decomposition(io_iterator A,pivot_iterator pivot,const dim_type& dim)
{
    typedef typename std::iterator_traits<io_iterator>::value_type value_type;
    const unsigned int dimension = dim.row_count();
    const unsigned int size = dim.size();
    for (unsigned int k = 0;k < dimension;++k)
        pivot[k] = k;
    for (unsigned int k = 0,row_k = 0;k < dimension;++k,row_k+=dimension)
    {
        {
            value_type max_value(0);
            unsigned int max_index = 0;
            unsigned int max_row = k;
            for (unsigned int i = k,index_ik = row_k + k;i < dimension;++i,index_ik += dimension)
            {
                value_type value = A[index_ik];
                if (value < 0)
                    value = -value;
                if (value > max_value)
                {
                    max_value = value;
                    max_index = index_ik;
                    max_row = i;
                }
            }
            if (max_value == 0)
                return false; // singularity
            if (max_row != k) // row swap is needed
            {
                image::vec::swap(A+row_k,A+row_k+dim.col_count(),A+max_index-k);
                std::swap(pivot[k],pivot[max_row]);
            }
        }
        //  reduce the matrix
        value_type bjj = A[row_k + k];
        for (unsigned int row_i = row_k + dimension;row_i < size;row_i += dimension)
        {
            value_type temp = A[row_i + k] /= bjj;
            unsigned int offset = row_i-row_k;
            unsigned int max_row_i_j = row_i + dimension;
            for (unsigned int row_i_j = row_i + k+ 1;row_i_j < max_row_i_j;++row_i_j)
                A[row_i_j] -= temp*A[row_i_j-offset];
        }
    }
    return true;
}


template<class input_iterator,class dim_type>
typename std::iterator_traits<input_iterator>::value_type
lu_determinant(input_iterator A,const dim_type& dim)
{
    typedef typename std::iterator_traits<input_iterator>::value_type value_type;
    value_type result = A[0];
    unsigned int leap_size = dim.col_count()+1;
    for (unsigned int index = leap_size;index < dim.size();index += leap_size)
        result *= A[index];
    return result;
}


template<class io_iterator,class pivot_iterator,class dim_type>
bool ll_decomposition(io_iterator A,pivot_iterator p,const dim_type& dim)
{
    typedef typename std::iterator_traits<io_iterator>::value_type value_type;
    const unsigned int dimension = dim.row_count();
    for (unsigned int i = 0,row_i = 0;i < dimension;i++,row_i += dimension)
    {
        for (unsigned int j = i,row_j = row_i;j < dimension;j++,row_j += dimension)
        {
            unsigned int offset = row_j-row_i;
            value_type sum = A[row_i + j];
            if(i > 0)
            for (unsigned int row_i_k = row_i+i-1;row_i_k >= row_i;--row_i_k)
                sum -= A[row_i_k]*A[row_i_k+offset];
            if(i == j)
            {
                if (sum <= 0.0)
                    return false;
                p[i]=std::sqrt(sum);
            }
            else
                A[row_j+i]=sum/p[i];
        }
    }
    return true;
}

template<class io_iterator,class pivot_iterator,class input_iterator2,
         typename output_iterator,class dim_type>
void ll_solve(io_iterator A,pivot_iterator p,input_iterator2 b,output_iterator x,const dim_type& dim)
{
    typedef typename std::iterator_traits<io_iterator>::value_type value_type;
    const unsigned int dimension = dim.row_count();
    // i = 0;
    x[0] = b[0]/p[0];
    for (unsigned int i = 1,row_i = dimension;
                i < dimension;++i,row_i += dimension)
    {
        value_type sum = b[i];
        for (int k=i-1;k>=0;k--)
            sum -= A[row_i + k]*x[k];
        x[i] = sum/p[i];
    }
    // i = dimension-1
    x[dimension-1] /= p[dimension-1];
    for (int i = dimension-2,row_i = dim.size()-dimension-dimension;
            i >= 0;--i,row_i -= dimension)
    {
        value_type sum = x[i];
        for (unsigned int k=i+1,row_k = row_i + dimension;k < dimension;k++,row_k += dimension)
            sum -= A[row_k + i]*x[k];
        x[i] = sum/p[i];
    }
}



template<class io_iterator,class output_iterator1,class output_iterator2,class dim_type>
bool qr_decomposition(io_iterator A,output_iterator1 c,output_iterator2 d,const dim_type& dim)
{
    typedef typename std::iterator_traits<io_iterator>::value_type value_type;
    bool singular = false;
    unsigned int m = dim.row_count();
    unsigned int n = dim.col_count();
    unsigned int min = std::min<unsigned int>(m,n);
    io_iterator A_row_k = A;
    for (unsigned int k = 0;k < min;k++,A_row_k += n)
    {
        value_type scale(0);
        {
            io_iterator A_i_k = A_row_k+k;
            for (unsigned int i=k;i<m;i++,A_i_k += n)
                scale=std::max<value_type>(scale,*A_i_k < 0 ? -*A_i_k : *A_i_k);
        }
        if (scale == 0.0)
        {
            c[k]=d[k]=0.0;
            singular = true;
        }
        else
        {
            value_type sum(0);
            io_iterator A_i_k = A_row_k+k;
            for (unsigned int i=k;i<m;i++,A_i_k += n)
            {
                value_type t = (*A_i_k /= scale);
                sum += t*t;
            }
            value_type sigma = (A_row_k[k] >= 0) ? std::sqrt(sum):-std::sqrt(sum);
            A_row_k[k] += sigma;
            c[k]=sigma*A_row_k[k];
            d[k] = -scale*sigma;


            for (unsigned int j=k+1;j < n;j++)
            {
                io_iterator A_row_i = A_row_k;
                sum = 0.0;
                for (unsigned int i=k;i<m;i++,A_row_i += n)
                    sum += A_row_i[k]*A_row_i[j];
                value_type tau=sum/c[k];
                A_row_i = A_row_k;
                for (unsigned int i=k;i<m;i++,A_row_i += n)
                    A_row_i[j] -= tau*A_row_i[k];
            }
        }
    }
    return !singular;
}

template<class io_iterator,class output_iterator1,class output_iterator2,class dim_type>
bool lq_decomposition(io_iterator A,output_iterator1 c,output_iterator2 d,const dim_type& dim)
{
    typedef typename std::iterator_traits<io_iterator>::value_type value_type;
    bool singular = false;
    unsigned int m = dim.row_count();
    unsigned int n = dim.col_count();
    unsigned int min = std::min<unsigned int>(m,n);
    io_iterator A_row_k = A;
    for (unsigned int k = 0;k < min;k++,A_row_k += n)
    {
        value_type scale(0);
        {
            io_iterator A_k_i = A_row_k+k;
            io_iterator A_k_n = A_row_k+n;
            for (;A_k_i<A_k_n;A_k_i++)
                scale=std::max<value_type>(scale,*A_k_i < 0 ? -*A_k_i : *A_k_i);
        }
        if (scale == 0.0)
        {
            c[k]=d[k]=0.0;
            singular = true;
        }
        else
        {
            value_type sum(0);
            io_iterator A_k_i = A_row_k+k;
            io_iterator A_k_n = A_row_k+n;
            for (;A_k_i<A_k_n;A_k_i++)
            {
                value_type t = (*A_k_i /= scale);
                sum += t*t;
            }
            value_type sigma = (A_row_k[k] >= 0) ? std::sqrt(sum):-std::sqrt(sum);
            A_row_k[k] += sigma;
            c[k]=sigma*A_row_k[k];
            d[k] = -scale*sigma;

            io_iterator A_row_j = A_row_k+n;
            for (unsigned int j=k+1;j < m;j++,A_row_j += n)
            {
                sum = 0.0;
                for (unsigned int i=k;i<n;i++)
                    sum += A_row_k[i]*A_row_j[i];
                value_type tau=sum/c[k];
                for (unsigned int i=k;i<n;i++)
                    A_row_j[i] -= tau*A_row_k[i];
            }
        }
    }
    return !singular;
}

template<class io_iterator1,class io_iterator2,class output_iterator,class dim_type>
void lq_get_l(io_iterator1 A,io_iterator2 d,output_iterator L,const dim_type& dim)
{
    unsigned int m = dim.row_count();
    unsigned int n = dim.col_count();
    unsigned int min = std::min<unsigned int>(m,n);
    if(A != L)
        std::copy(A,A+m*n,L);
    for(unsigned int i = 0,pos = 0;i < min;++i,pos += n+1)
    {
        L[pos] = d[i];
        for(unsigned int j = 1;j < n-i;++j)
            L[pos+j] = 0;
    }
}

/*
    May modify matrix A and b to enlarge the diagnoal elements
 */
template<class io_iterator,class input_iterator2,class dim_type>
bool jacobi_regularize(io_iterator A,input_iterator2 piv,const dim_type& dim)
{
    typedef typename std::iterator_traits<io_iterator>::value_type value_type;
    const unsigned int dimension = dim.row_count();
    std::vector<unsigned char> selected(dimension);
    for(int row = dimension-1;row >= 0;--row)
    {
        value_type max_value = 0.0;
        unsigned int max_col = 0;
        io_iterator A_row = A + row*dimension;
        for(unsigned int col = 0;col < dimension;++col,++A_row)
            if(std::fabs(*A_row) > max_value && !selected[col])
            {
                max_col = col;
                max_value = std::fabs(*A_row);
            }
        if(max_value == 0.0)
            return false;
        selected[max_col] = 1;
        piv[row] = max_col;
    }
    return true;
}


template<class io_iterator,class pivot_iterator,class input_iterator2,
         typename output_iterator,class dim_type>
bool jacobi_solve(io_iterator A,pivot_iterator p,input_iterator2 b,output_iterator x,const dim_type& dim)
{
    typedef typename std::iterator_traits<output_iterator>::value_type value_type;
    const unsigned int dimension = dim.row_count();
    io_iterator Arow_j = A;
    for(unsigned int i = 0;i < dimension;++i)
    {
        x[i] = b[i];
        value_type scale = 0.0;
        for(unsigned int j = 0;j < dimension;++j,++Arow_j)
            if(j != p[i])
                x[i] -= (*Arow_j)*x[j];
            else
                scale = (*Arow_j);
        if(scale == 0.0)
            return false;
        x[i] /= scale;
    }
    return true;
}

template<class io_iterator,class input_iterator2,class output_iterator,class dim_type>
bool jacobi_solve(io_iterator A,input_iterator2 b,output_iterator x,const dim_type& dim)
{
    typedef typename std::iterator_traits<output_iterator>::value_type value_type;
    const unsigned int dimension = dim.row_count();
    io_iterator Arow_j = A;
    for(unsigned int i = 0;i < dimension;++i)
    {
        x[i] = b[i];
        value_type scale = 0.0;
        for(unsigned int j = 0;j < dimension;++j,++Arow_j)
            if(i != j)
                x[i] -= (*Arow_j)*x[j];
            else
                scale = (*Arow_j);
        if(scale == 0.0)
            return false;
        // can be improve by weighted jacobi method
        x[i] /= scale;
    }
    return true;
}

template<class input_iterator>
typename std::iterator_traits<input_iterator>::value_type
determinant(input_iterator iter,dim<4,4>)
{
    typedef typename std::iterator_traits<input_iterator>::value_type value_type;
    value_type v_9_14_m_10_13  = (*(iter+ 9))*(*(iter+14))-(*(iter+10))*(*(iter+13));
    value_type v_10_15_m_11_14 = (*(iter+10))*(*(iter+15))-(*(iter+11))*(*(iter+14));
    value_type v_11_12_m_8_15 = (*(iter+11))*(*(iter+12))-(*(iter+ 8))*(*(iter+15));
    value_type v_8_13_m_9_12 = ((*(iter+ 8))*(*(iter+13))-(*(iter+ 9))*(*(iter+12)));
    value_type v_11_13_m_9_15 = ((*(iter+11))*(*(iter+13))-(*(iter+ 9))*(*(iter+15)));
    value_type v_8_14_m_10_12 = ((*(iter+ 8))*(*(iter+14))-(*(iter+10))*(*(iter+12)));
    return
        (*iter)*
        (
            (*(iter+ 5))*v_10_15_m_11_14+
            (*(iter+ 6))*v_11_13_m_9_15+
            (*(iter+ 7))*v_9_14_m_10_13
        )-
        *(iter+1)*
        (
            (*(iter+ 4))*v_10_15_m_11_14+
            (*(iter+ 6))*v_11_12_m_8_15+
            (*(iter+ 7))*v_8_14_m_10_12
        )+
        *(iter+2)*
        (
            (*(iter+ 4))*(-v_11_13_m_9_15)+
            (*(iter+ 5))*v_11_12_m_8_15+
            (*(iter+ 7))*v_8_13_m_9_12
        )-
        *(iter+3)*
        (
            (*(iter+ 4))*v_9_14_m_10_13+
            (*(iter+ 5))*(-v_8_14_m_10_12)+
            (*(iter+ 6))*v_8_13_m_9_12
        );
}


/**
example:
\code
double sym[]={12, 8, 3,
               8, 4, 2,
               3, 2,11};
std::cout << la::determinant(sym,la::dim<3,3>());
\endcode
*/
template<class input_iterator>
typename std::iterator_traits<input_iterator>::value_type
determinant(input_iterator iter,dim<3,3>)
{
    return (*(iter  ))*((*(iter+4))*(*(iter+8))-(*(iter+5))*(*(iter+7)))+
           (*(iter+1))*((*(iter+5))*(*(iter+6))-(*(iter+3))*(*(iter+8)))+
           (*(iter+2))*((*(iter+3))*(*(iter+7))-(*(iter+4))*(*(iter+6)));
}

template<class input_iterator>
typename std::iterator_traits<input_iterator>::value_type
determinant(input_iterator iter,dim<2,2>)
{
    return (*iter)*(*(iter+3)) - (*(iter+1))*(*(iter+2));
}
/**
    solve Ax=b

*/
template<class input_iterator1,class input_iterator2,class piv_iterator,class output_iterator,class dim_type>
bool lu_solve(input_iterator1 A,piv_iterator piv,input_iterator2 b,output_iterator x,const dim_type& dim)
{
    typedef typename std::iterator_traits<input_iterator1>::value_type value_type;
    const unsigned int col_count = dim.col_count();
    const unsigned int matrix_size = dim.size();
    for (unsigned int i = 0; i < col_count;++i)
        *(x+i) = b[piv[i]];
    // Solve L*Y = B(piv)
    {
        for (unsigned int j = 0, k = 0;j < matrix_size;j += col_count,++k)
        {
			value_type x_k = *(x+k); 
            for (unsigned int i = j + col_count + k, m = k+1;i < matrix_size;i += col_count,++m)
                *(x+m) -= x_k*(A[i]);  // A[i][k]
        }
    }
    // Solve U*X = Y;
    {
        unsigned int j = matrix_size - 1;
        unsigned int diagonal_shift = col_count + 1;
        for (int k = dim.row_count()-1;k >= 0;--k,j -= diagonal_shift)
        {
            value_type Arowk_value = A[j];
            if (Arowk_value + value_type(1) == value_type(1))
                return false;
            value_type x_k(*(x+k) /= Arowk_value);
            input_iterator1 Arowi = A + k;
            for (unsigned int i = 0;i < k;++i,Arowi += col_count)
                *(x+i) -= x_k*(*Arowi);
        }
    }
    return true;
}

/**
    solve AX=B

*/
template<class input_iterator1,class input_iterator2,class piv_iterator,class output_iterator,class dim_type>
bool lu_solve(input_iterator1 A,piv_iterator piv,input_iterator2 B,output_iterator X,const dim_type& dim,const dim_type& Bdim)
{
    typedef typename std::iterator_traits<input_iterator1>::value_type value_type;
    std::vector<value_type> b(Bdim.row_count());
    std::vector<value_type> x(Bdim.row_count());
    bool result = true;
    for (unsigned int col = 0;col < Bdim.col_count();++col)
    {
        for (unsigned int row = 0,index = col;row < Bdim.row_count();++row,index += Bdim.col_count())
            b[row] = B[index];
        if (lu_solve(A,piv,&*b.begin(),&*x.begin(),dim))
            for (unsigned int row = 0,index = col;row < Bdim.row_count();++row,index += Bdim.col_count())
                X[index] = x[row];
        else
            result = false;
    }
    return result;
}

template<class input_iterator>
bool inverse(input_iterator iter,dim<1,1>)
{
    typedef typename std::iterator_traits<input_iterator>::value_type value_type;
    if (*iter+value_type(1) == value_type(1))
        return false;
    *iter = value_type(1)/(*iter);
    return true;
}

/**
example:
\code
double sym[]={12, 8,
              9, 4};
inverse(sym,dim<2,2>());
\endcode
*/
template<class input_iterator>
bool inverse(input_iterator iter,dim<2,2>)
{
    typedef typename std::iterator_traits<input_iterator>::value_type value_type;
    value_type det = determinant(iter,dim<2,2>());
    if (det+value_type(1) == value_type(1))
        return false;
    std::swap(*(iter),*(iter+3));
    *(iter  ) /= det;
    *(iter+1) /= -det;
    *(iter+2) /= -det;
    *(iter+3) /= det;
    return true;
}

template<class input_iterator>
bool inverse(input_iterator iter,dim<3,3>)
{
    typedef typename std::iterator_traits<input_iterator>::value_type value_type;
    value_type det = determinant(iter,dim<3,3>());
    if (det+value_type(1) == value_type(1))
        return false;
    value_type temp[9];
    temp[0] = iter[4]*iter[8]-iter[5]*iter[7];
    temp[1] = iter[2]*iter[7]-iter[1]*iter[8];
    temp[2] = iter[1]*iter[5]-iter[2]*iter[4];
    temp[3] = iter[5]*iter[6]-iter[3]*iter[8];
    temp[4] = iter[0]*iter[8]-iter[2]*iter[6];
    temp[5] = iter[2]*iter[3]-iter[0]*iter[5];
    temp[6] = iter[3]*iter[7]-iter[4]*iter[6];
    temp[7] = iter[1]*iter[6]-iter[0]*iter[7];
    temp[8] = iter[0]*iter[4]-iter[1]*iter[3];
    iter[0] = temp[0]/det;
    iter[1] = temp[1]/det;
    iter[2] = temp[2]/det;
    iter[3] = temp[3]/det;
    iter[4] = temp[4]/det;
    iter[5] = temp[5]/det;
    iter[6] = temp[6]/det;
    iter[7] = temp[7]/det;
    iter[8] = temp[8]/det;
    return true;
}

template<class input_iterator,class dim_type>
bool inverse(input_iterator A_,dim_type dim)
{
	typedef typename std::iterator_traits<input_iterator>::value_type value_type;
    const unsigned int dimension = dim.row_count();
    const unsigned int matrix_size = dim.size();
	std::vector<value_type> buf(A_,A_+matrix_size);
	std::vector<size_t> piv(dimension);
    value_type* A = &(buf[0]);
    if(!lu_decomposition(A,piv.begin(),dim))
		return false;

    bool result = true;
    std::vector<value_type> x_buf(dimension);
    value_type* x = &(x_buf[0]);    
	for (unsigned int col = 0;col < dimension && result;++col)
    {
		
		// Solve L*Y = B(piv)
        {
			bool find = false;
            for (unsigned int j = 0, k = 0;j < matrix_size;j += dimension,++k)
            {
				if(find)
				{
					value_type x_k = *(x+k); 
            		for (unsigned int i = j + dimension + k, m = k+1;i < matrix_size;i += dimension,++m)
						*(x+m) -= x_k*(A[i]);  // A[i][k]
				}
				else
				{
					if(piv[k] != col)
						continue;
					*(x+k) = 1.0;
					for (unsigned int i = j + dimension + k, m = k+1;i < matrix_size;i += dimension,++m)
						*(x+m) -= A[i];  // A[i][k]
					find = true;
				}
            }
        }
        // Solve U*X = Y;
        {
            unsigned int j = matrix_size - 1;
            unsigned int diagonal_shift = dimension + 1;
            for (int k = dimension-1;k >= 0;--k,j -= diagonal_shift)
            {
                value_type Arowk_value = A[j];
                if (Arowk_value + value_type(1) == value_type(1))
				{
					result = false;
					break;
				}
                value_type x_k(*(x+k) /= Arowk_value);
                value_type* Arowi = A + k;
                for (unsigned int i = 0;i < k;++i,Arowi += dimension)
                    *(x+i) -= x_k*(*Arowi);
            }
        }

        for (unsigned int row = 0,index = col;row < dimension;++row,index += dimension)
			A_[index] = x[row];   
		std::fill(x,x+dimension,0.0);
    }
    return result;
}

template<class input_iterator,class output_iterator,class dim_type>
bool inverse(input_iterator A_,output_iterator A,dim_type dim)
{
    std::copy(A_,A_+dim.size(),A);
    return inverse(A,dim);
}


/** example:
 *
    float A[] = {5, 1, 3, 12, 16,
                 0, 4, 2, -8, -5,
                 0, 0, 12, -1, -4,
                 0, 0, 0,  11, -3,
                 0, 0, 0,   0,  3};
    image::matrix::inverse_upper(A,image::dyndim(5,5));
 */
template<class input_iterator,class dim_type>
bool inverse_upper(input_iterator U,dim_type dim)
{
    typedef typename std::iterator_traits<input_iterator>::value_type value_type;
    unsigned int n = dim.col_count();
    unsigned int n_1 = n+1;
    // inverse the diagonal
    unsigned int pos = 0;
    for(unsigned int col = 0;col < n;++col,pos += n_1)
    {
        if(U[pos] + value_type(1) == value_type(1))
            return false;
        U[pos] = value_type(1)/U[pos];
    }
    // calculate the off diagonals
    input_iterator U_row = U + pos - n - n - n;
    for(int col = n-1;col >= 0;--col,U_row -= n)
    {
        input_iterator U_row_i = U_row;
        for(int row = col - 1;row >= 0;--row,U_row_i -= n)
        {
            // adding up to the diagonal
            value_type sum(0);
            input_iterator iU_row = U_row_i + col + n;
            for(unsigned int i = row + 1;i <= col;++i,iU_row += n)
                sum += U_row_i[i]*(*iU_row);
            U_row_i[col] = -sum*U_row_i[row];
        }
    }

    return true;
}


/** example:
 *
    float A[] = {4, 0, 0, 0, 0,
                 1, 3, 0, 0, 0,
                 -5, 2, 4, 0, 0,
                 1,  -6, 2, 10, 0,
                 4, -10, 11, -30,22};
    image::matrix::inverse_lower(A,image::dyndim(5,5));
 */
template<class input_iterator,class dim_type>
bool inverse_lower(input_iterator U,dim_type dim)
{
    typedef typename std::iterator_traits<input_iterator>::value_type value_type;
    unsigned int n = dim.col_count();
    unsigned int n_1 = n+1;
    // inverse the diagonal
    unsigned int pos = 0;
    for(unsigned int col = 0;col < n;++col,pos += n_1)
    {
        if(U[pos] + value_type(1) == value_type(1))
            return false;
        U[pos] = value_type(1)/U[pos];
    }
    // calculate the off diagonals
    input_iterator U_row = U + pos - n - n;
    for(int row = n-1;row >= 0;--row,U_row -= n)
    {
        for(int col = row - 1;col >= 0;--col)
        {
            // adding up to the diagonal
            value_type sum(0);
            input_iterator iU_row = U_row +col;
            for(unsigned int i = row;i > col;--i, iU_row -= n)
                sum += U_row[i]*(*iU_row);
            U_row[col] = -sum*(*iU_row);
        }
    }
    return true;
}


/** Apply housholder reduction to make column (c) to be 0 below row (r)
		  |  a'				  |				a'
		A=|	   ,a1,a2,a3..	an|		, a0 = | |
		  |  x  			  |				x

	P=1-u*uT/uu_2

	u = x-|x|e0

	Px = |x|e0,

		    |I	0|    | a'| |  a00  |
	make P'=|	 |, P'|   |=|	    |
		    |0  P|    | x | | |x|e0 |
					|  a'				 |
	therefore P'A = |		,P'a1,P'a2...|
	                | |x|e0				 |

	result:
		vector u was stored in the place of x
		right side columns were changed due to P'
	*/
/*
template<class input_iterator,class dim_type>
typename std::iterator_traits<input_iterator>::value_type
household_col_reduction(input_iterator row,unsigned int row_index,unsigned int col_index,const dim_type& dim)
{
typedef typename std::iterator_traits<input_iterator>::value_type value_type;
// calculate all the parameter of Housholder reduction
input_iterator x = row + col_index;
value_type x2 = col_vector_length2(row_iterator(row.index(),x));
if (x2+value_type(1) == value_type(1)) // |x|=0
    return 0;
value_type x_norm = std::sqrt(x2);
//store u in the original place of x
// if x0 > 0, u = x-( |x|)e0, => Px = ( |x|)e0; u^2=2(|x|^2-( |x|)x0)
//    x0 < 0, u = x-(-|x|)e0, => Px = (-|x|)e0; u^2=2(|x|^2-(-|x|)x0)
//
if (x[0] < 0)
    x_norm = -x_norm;
value_type uu_2 = x2 - x_norm * x[0];
x[0] -= x_norm;

// calculate P'a1,P'a2..., upper part is not change for I in P'
// only lower part of vector a is changed
// Pa' = a'-u*uT*a'/uu_2

for (unsigned int i = col_index + 1; i < dim.col_count();++i)
{
    // perform a' <- a' - (u*a/uu_2)u
    row_iterator a(row.index(),row.iterator()+i);
    la::col_add(a,x,
                       -col_product<double*,dim_type>(a,x,dim_type())
                       /uu_2,dim_type());
}
return x_norm;
}*/

template <class input_iterator,class output_iterator>
void eigen_decomposition_sym(input_iterator A,
                                    output_iterator V,
                                    output_iterator d,dim<2,2>)
{
    double b = A[1];
    if (b + 1.0 == 1.0)
    {
        d[0] = A[0];
        d[1] = A[3];
        V[0] = 1.0;
        V[1] = 0.0;
        V[2] = 0.0;
        V[3] = 1.0;
        return;
    }
    double a = A[0];
    double b2 = b*b;
    double c = A[3];
    double a_c = a-c;
    double t = std::sqrt(a_c*a_c+4.0*b2);
    d[0] = (a+c+t)*0.5;
    d[1] = d[0]-t;

    double a_l1 = a-d[0];
    double a_l2 = a-d[1];
    double l1 = sqrt(b2+a_l1*a_l1);
    double l2 = sqrt(b2+a_l2*a_l2);
    V[0] = -b/l1;
    V[2] = a_l1/l1;
    V[1] = -b/l2;
    V[3] = a_l2/l2;
}


template<class input_iterator,class dim_type>
void col_swap(input_iterator i1,input_iterator i2,const dim_type& dim)
{
    unsigned int col_count = dim.col_count();
    // started from 1 for leap iterator problem
    for (unsigned int i = 1;i < dim.row_count();++i,i1 += col_count, i2 += col_count)
        std::swap(*i1,*i2);
    std::swap(*i1,*i2);
}

template <class input_iterator,class output_iterator,class dym_type>
void eigenvalue(input_iterator A,output_iterator d,const dym_type& dimension)
{
    typedef typename std::iterator_traits<input_iterator>::value_type value_type;
    typedef value_type* iterator_type;
    const unsigned int size = dimension.size();
    const unsigned int dim = dimension.col_count();
    const unsigned int shift = dim + 1;
    std::vector<value_type> V_(size);
    std::vector<value_type> e_(dim);
    value_type* V = &*V_.begin();
    value_type* e = &*e_.begin();

    std::copy(A,A+size,V);
    std::fill(d,d+dim,value_type(0));

    //void tridiagonalize(void)
    {
        //  This is derived from the Algol procedures tred2 by
        //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.
        // Householder reduction to tridiagonal form.
        {
            iterator_type Vrowi = V+size-dim;//n-1 row
            for (unsigned int i = dim-1;i > 1;--i,Vrowi -= dim)
            {
                value_type h(0),g,f;
                // Generate Householder vector.u
                // x is the lower i-1 row vector of row i
                // h = |x|^2
                for (unsigned int k = 0; k < i; k++)
                    h += Vrowi[k] * Vrowi[k];
                if (h+value_type(1.0) == value_type(1.0))
                    //if(h < la::eps<value_type>::value)
                {
                    e[i] = Vrowi[i-1];
                    continue;
                }

                f = Vrowi[i-1];
                g = std::sqrt(h);			// g = |x|
                if (f >= value_type(0))
                    g = -g;	// choose sign of g accoring to V[i][i-1]
                e[i] = g;
                h -= f * g;	// h = 1/2|u|^2=1/2(|x-|x|e|^2)=1/2(|x|^2-2*|x|*x0+|x|^2)
                //   = |x|^2-|x|*x0;
                Vrowi[i-1] -= g; // Vrowi x becomes u, u = x-|x|e
                f = value_type(0);
                {
                    iterator_type Vrowj = V;// from the first row
                    for (unsigned int j = 0; j < i; ++j, Vrowj += dim)
                    {
                        unsigned int j_1 = j+1;
                        g = value_type(0);
                        iterator_type rowj_1 = V;
                        for (unsigned int k = 0;k < j_1;++k,rowj_1+=dim)
                            g += Vrowj[k]*Vrowi[k];
                        if (j_1 < i)
                        {
                            iterator_type Vrowk = rowj_1+j; //row j +1 , col j
                            for (unsigned int k = j_1;k < i;++k,Vrowk += dim)
                                g += (*Vrowk)*Vrowi[k];
                        }
                        e[j] = g/h;
                        f += e[j] * Vrowi[j];
                    }
                }
                d[i] = h;
                {
                    value_type hh = f / (h + h);
                    iterator_type Vrowj = V;// from the first row
                    for (unsigned int j = 0; j < i; ++j, Vrowj += dim)
                    {
                        f = Vrowi[j];
                        g = e[j]-hh * f;
                        e[j] = g;
                        for (unsigned int k = 0;k < j+1;++k)
                            Vrowj[k] -= (f * e[k] + g * Vrowi[k]);
                    }
                }
            }
        }

        e[1] = V[dim];
        iterator_type Vdia = V+size-1;
        output_iterator d_iter = d + dim -1;
        d[0] = V[0];
        while (Vdia != V)
        {
            *d_iter = *Vdia;
            --d_iter;
            Vdia -= shift;
        }
    }

    // Symmetric tridiagonal QL algorithm.

    //	void diagonalize(void)
    {
        using namespace std; // for hypot

        //  This is derived from the Algol procedures tql2, by
        //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.
        std::copy(e+1,e+dim,e);
        e[dim-1] = value_type(0);

        value_type p,r,b,f(0);
        for (int l = 0,iter = 0;l < dim && iter < 30;++iter)
        {
            unsigned int m = l;
            // Find small subdiagonal element
            for (;m < dim-1;++m)
            {
                /*
                tst1 = ((d[m+1] > 0) ?
                std::max(tst1,la::abs(d[m]) + d[m+1]):
                std::max(tst1,la::abs(d[m]) - d[m+1]);
                if(tst1+e[m] == tst1)
                break;*/
                if (d[m]+e[m] == d[m])
                    break;

            }
            // If m == l, d[l] is an eigenvalue, go for next
            if ((int)m == l)
            {
                ++l;
                iter = 0;
                continue;
            }

            // Compute implicit shift
            p = (d[l+1]-d[l])/(e[l]*value_type(2));
            r = std::sqrt(1+p*p);
            p = d[m]-d[l]+e[l]/(p+((p<0)? -r:r));

            value_type s(1),c(1),g(0);
            int i = m-1;
            for (; i >= l; i--)
            {
                f = s*e[i];
                b = c*e[i];
                e[i+1] = r = hypot(f,p);
                if (r+f == f && r+p == p)
                {
                    d[i-1] -= g;
                    e[m] = value_type(0);
                    break;
                }
                s = f/r;
                c = p/r;
                p = d[i+1]-g;
                r = (d[i]-p)*s+c*b*value_type(2);
                g = s*r;
                d[i+1] = p+g;
                p = c*r-b;
            } // i loop
            if (r != value_type(0) || i < l)
            {
                e[l] = p;
                e[m] = value_type(0);
                d[l] -= g;
            }
        } // l loop


    }
    std::sort(d,d+dim,std::greater<class std::iterator_traits<output_iterator>::value_type>());
}

/**
A must be a symmetric matrix
Output V:eigenvectors, *stored in colomn major
Output d:eigenvalues
*/
template <class input_iterator,class output_iterator1,class output_iterator2,class dym_type>
void eigen_decomposition_sym(input_iterator A,
                                    output_iterator1 V,
                                    output_iterator2 d,const dym_type& dimension)
{
    typedef typename std::iterator_traits<input_iterator>::value_type value_type;
    if(dimension.col_count() == 3)
    {
        {
            value_type I1 = A[0] + A[4] + A[8];
            value_type I2 = A[0]*A[4] + A[0]*A[8]+ A[4]*A[8]-A[1]*A[1]-A[2]*A[2]-A[5]*A[5];
            value_type I3 = A[0]*A[4]*A[8]+2.0*A[1]*A[2]*A[5]-(A[8]*A[1]*A[1]+A[4]*A[2]*A[2]+A[0]*A[5]*A[5]);
            value_type I1_3 = (I1/3.0);
            value_type v = I1_3*I1_3-I2/3.0;
            value_type s = I1_3*I1_3*I1_3-I1*I2/6.0+I3/2.0;
            if(v + 1.0 == 1.0)
            {
                std::fill(d,d+3,0.0);
                std::fill(V,V+9,0.0);
                V[0] = V[4] = V[8] = 1.0;
                return;
            }
            value_type sqrt_v = std::sqrt(v);
            value_type angle = std::acos(std::max<value_type>(-1.0,std::min<value_type>(1.0,s/v/sqrt_v)))/3.0;
            d[0] = I1_3 + 2.0*sqrt_v*std::cos(angle);
            d[1] = I1_3 - 2.0*sqrt_v*std::cos(3.14159265358979323846/3.0+angle);
            d[2] = I1-d[0]-d[1];
        }

        for(int i = 0;i < 3;++i,V+=3)
        {
            value_type Ai = A[0]-d[i];
            value_type Bi = A[4]-d[i];
            value_type Ci = A[8]-d[i];
            value_type q1 = (A[2]*A[1]-Ai*A[5]);
            value_type q2 = (A[1]*A[5]-Bi*A[2]);
            value_type q3 = (A[2]*A[5]-Ci*A[1]);
            V[0] = q2*q3;
            V[1] = q3*q1;
            V[2] = q2*q1;
            value_type length = std::sqrt(V[0]*V[0]+V[1]*V[1]+V[2]*V[2]);
            V[0] = V[0]/length;
            V[1] = V[1]/length;
            V[2] = V[2]/length;
        }
        return;
    }
    const unsigned int size = dimension.size();
    const unsigned int dim = dimension.col_count();
    std::vector<value_type> e_(dim+1);
    value_type* e = &*e_.begin();

    std::copy(A,A+size,V);
    std::fill(d,d+dim,value_type(0));

    //void tridiagonalize(void)
    {
        //  This is derived from the Algol procedures tred2 by
        //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.
        // Householder reduction to tridiagonal form.
        {
            output_iterator1 Vrowi = V+size-dim;//n-1 row
            for (unsigned int i = dim-1;i > 1;--i,Vrowi -= dim)
            {
                value_type h(0),g,f;
                // Generate Householder vector.u
                // x is the lower i-1 row vector of row i
                // h = |x|^2
                for (unsigned int k = 0; k < i; k++)
                    h += Vrowi[k] * Vrowi[k];
                if (h+value_type(1.0) == value_type(1.0))
                    //if(h < la::eps<value_type>::value)
                {
                    e[i] = Vrowi[i-1];
                    continue;
                }

                f = Vrowi[i-1];
                g = std::sqrt(h);			// g = |x|
                if (f >= value_type(0))
                    g = -g;	// choose sign of g accoring to V[i][i-1]
                e[i] = g;
                h -= f * g;	// h = 1/2|u|^2=1/2(|x-|x|e|^2)=1/2(|x|^2-2*|x|*x0+|x|^2)
                //   = |x|^2-|x|*x0;
                Vrowi[i-1] -= g; // Vrowi x becomes u, u = x-|x|e
                f = value_type(0);
                {
                    output_iterator1 Vrowj = V;// from the first row
                    for (unsigned int j = 0; j < i; ++j, Vrowj += dim)
                    {
                        unsigned int j_1 = j+1;
                        Vrowj[i] = Vrowi[j]/h;
                        g = value_type(0);
                        output_iterator1 rowj_1 = V;
                        for (unsigned int k = 0;k < j_1;++k,rowj_1+=dim)
                            g += Vrowj[k]*Vrowi[k];
                        if (j_1 < i)
                        {
                            output_iterator1 Vrowk = rowj_1+j; //row j +1 , col j
                            for (unsigned int k = j_1;k < i;++k,Vrowk += dim)
                                g += (*Vrowk)*Vrowi[k];
                        }
                        e[j] = g/h;
                        f += e[j] * Vrowi[j];
                    }
                }
                d[i] = h;
                {
                    value_type hh = f / (h + h);
                    output_iterator1 Vrowj = V;// from the first row
                    for (unsigned int j = 0; j < i; ++j, Vrowj += dim)
                    {
                        f = Vrowi[j];
                        g = e[j]-hh * f;
                        e[j] = g;
                        for (unsigned int k = 0;k < j+1;++k)
                            Vrowj[k] -= (f * e[k] + g * Vrowi[k]);
                    }
                }
            }
        }

        e[0] = value_type(0);
        d[0] = V[0];
        e[1] = V[dim];
        d[1] = value_type(0);
        V[0] = value_type(1);
        // Accumulate transformations.
        // Also change V from column major to row major
        // Elements in V(j<i,k<i) is row major,
        // Elements in V(j>=i,k>=i) is column major,
        {
            output_iterator1 Vrowi = V;// from the second row
            for (unsigned int i = 1; i < dim; ++i)
            {
                Vrowi += dim;
                if (d[i] != value_type(0))
                {
                    for (output_iterator1 Vrowj = V;Vrowj != Vrowi;Vrowj += dim)
                    {
                        value_type g = image::vec::dot(Vrowi,Vrowi+i,Vrowj);
                        {
                            output_iterator1 Vrowk = V;
                            for (int k = 0;k < i;++k,Vrowk += dim)
                                Vrowj[k] -= g * Vrowk[i];
                        }
                    }
                }
                d[i] = Vrowi[i];
                Vrowi[i] = value_type(1);
                {
                    output_iterator1 Vrowk = V;
                    for (int k = 0;k < i;++k,Vrowk += dim)
                        Vrowk[i] = Vrowi[k] = value_type(0);

                }
            }
        }


    }

    // Symmetric tridiagonal QL algorithm.

    //	void diagonalize(void)
    {
        using namespace std; // for hypot

        //  This is derived from the Algol procedures tql2, by
        //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.
        ++e;
        value_type p,r,b,f(0);
        for (int l = 0,iter = 0;l < dim && iter < 30;++iter)
        {
            unsigned int m = l;
            // Find small subdiagonal element
            for (;m < dim-1;++m)
            {
                /*
                tst1 = ((d[m+1] > 0) ?
                std::max(tst1,la::abs(d[m]) + d[m+1]):
                std::max(tst1,la::abs(d[m]) - d[m+1]);
                if(tst1+e[m] == tst1)
                break;*/
                if (d[m]+e[m] == d[m])
                    break;

            }
            // If m == l, d[l] is an eigenvalue, go for next
            if ((int)m == l)
            {
                ++l;
                iter = 0;
                continue;
            }

            // Compute implicit shift
            p = (d[l+1]-d[l])/(e[l]*value_type(2));
            r = std::sqrt(1+p*p);
            p = d[m]-d[l]+e[l]/(p+((p<0)? -r:r));

            value_type s(1),c(1),g(0);
            int i = m-1;
            output_iterator1 Vrowi = V+i*dim;
            output_iterator1 Vrowi_end = Vrowi+dim;
            do
            {
                f = s*e[i];
                b = c*e[i];
                e[i+1] = r = hypot(f,p);
                if (r+f == f && r+p == p)
                {
                    d[i-1] -= g;
                    e[m] = value_type(0);
                    break;
                }
                s = f/r;
                c = p/r;
                p = d[i+1]-g;
                r = (d[i]-p)*s+c*b*value_type(2);
                g = s*r;
                d[i+1] = p+g;
                p = c*r-b;
                // Accumulate transformation.
                image::vec::rot(Vrowi,Vrowi_end,Vrowi_end,c,-s);
                if (--i < l)
                    break;
                Vrowi_end = Vrowi;
                Vrowi -= dim;
            } // i loop
            while (1);

            if (r != value_type(0) || i < l)
            {
                e[l] = p;
                e[m] = value_type(0);
                d[l] -= g;
            }
        } // l loop


        // Sort eigenvalues and corresponding vectors.
        output_iterator1 Vrowi = V;
        for (unsigned int i = 0; i < dim-1; ++i,Vrowi += dim)
        {
            unsigned int k = std::max_element(d+i,d+dim)-d;
            if (k != i)
            {
                std::swap(d[k],d[i]);
                image::vec::swap(Vrowi,Vrowi+dim,V+k*dim);
            }
        }
    }
}

/*
Input:
    A: n-by-m matrix
	n <= m
Operation:
    calculate A=U'*S*V
Output:
    A: n singular vector having m dimensions (right side matrix)
	U: n singular vector having n dimensions (left side matrix)
	s: n singular values
*/

template <class input_iterator,class output_iterator1,class output_iterator2,class dym_type>
void svd(input_iterator A,output_iterator1 U,output_iterator2 s,dym_type dimension)
{
    using namespace std;
    typedef typename std::iterator_traits<input_iterator>::value_type value_type;
    int n = dimension.row_count();
    int m = dimension.col_count();
	if(n > m)
		return;
    int nu = n;
    std::vector<value_type> e_(n);
    value_type* e = &*e_.begin();
    value_type* e_end = e+n;
    std::vector<value_type> w_(m);
    value_type* w = &*w_.begin();
    value_type* w_end = w+m;
    value_type max_value = pow(2.0,-52.0);

    // Reduce A to bidiagonal form, storing the diagonal elements
    // in s and the super-diagonal elements in e.

    int nct = std::min(m-1,n);
    int nrt = std::max(0,nu-2);
    {
        int k = 0;
        input_iterator Arowk = A;
        input_iterator Arowk_end = Arowk+m;
        do
        {
            int k_1 = k+1;
            {
                input_iterator Arowk_k = Arowk+k;//A[k][k];

                value_type s_k = image::vec::norm2(Arowk_k,Arowk_end);
                if (s_k == 0.0)
                {
                    s[k] = 0.0;
                    int j = k_1;
                    if (j < n)
                    {
                        input_iterator Arowj_k = Arowk_end+k; // A[k][j];
                        do
                        {
                            e[j] = *Arowj_k;
                            if (++j == n)
                                break;
                            Arowj_k += m;
                        }
                        while (1);
                    }
                }
                else
                {
                    if (Arowk[k] < 0.0)
                        s_k = -s_k;
                    image::vec::scale(Arowk_k,Arowk_end,1.0/s_k);
                    *Arowk_k += 1.0;
                    s[k] = -s_k;
                    int j = k_1;
                    if (j < n)
                    {
                        input_iterator Arowj_k = Arowk_end+k; // A[k][j];
                        do
                        {
                            image::vec::aypx(Arowk_k,Arowk_end,image::vec::dot(Arowk_k,Arowk_end,Arowj_k)/-*Arowk_k,Arowj_k);
                            e[j] = *Arowj_k;
                            if (++j == n)
                                break;
                            Arowj_k += m;
                        }
                        while (1);
                    }
                }
            }
            //now U[k:m][k] stores in A[k:m][k];

            if (k < nrt)
            {
                value_type* e_k_1 = e + k_1;
                value_type e_k_value = image::vec::norm2(e_k_1,e_end);
                if (e_k_value != 0.0)
                {
                    if (*e_k_1 < 0.0)
                        e_k_value = -e_k_value;

                    image::vec::scale(e_k_1,e_end,1.0/e_k_value);
                    *e_k_1 += 1.0;
                    e_k_value = -e_k_value;
                    // Apply the transformation.
                    if (k+1 < m)
                    {
                        value_type* w_k_1 = w+k_1;
                        std::fill(w_k_1,w_end,0.0);

                        //if (j < n)   no need this check
                        {
                            int j = k_1;
                            input_iterator Arowj_k_1 = Arowk_end+k_1;
                            do
                            {
                                image::vec::axpy(w_k_1,w_end,e[j],Arowj_k_1);
                                if (++j == n)
                                    break;
                                Arowj_k_1 += m;
                            }
                            while (1);
                        }

                        //if (j < n)   no need this check
                        {
                            int j = k_1;
                            input_iterator Arowj_k_1 = Arowk_end+k_1;
                            value_type e_k_1_value = *e_k_1;
                            do
                            {
                                image::vec::aypx(w_k_1,w_end,-e[j]/e_k_1_value,Arowj_k_1);
                                if (++j == n)
                                    break;
                                Arowj_k_1 += m;
                            }
                            while (1);
                        }
                    }
                }

                e[k] = e_k_value;
                // store U[k+1:n][k] to A[k][k+1:n]
                std::copy(e+k_1,e_end,U+k*n+k_1);
            }
            max_value=std::max(max_value,std::abs(e[k])+std::abs(s[k]));
            if (++k == nct)
                break;
            Arowk = Arowk_end;
            Arowk_end += m;
        }
        while (1);
    }

    if (m == n)
        s[nct] = *(A+nct*m+nct);
    e[nrt] = A[(n-1)*m + nrt];

    {
        int k = nu-1;
        output_iterator1 Urowk = U+k*n;
        output_iterator1 Urowk_end = Urowk+n;
        while (1)
        {
            int k_1 = k+1;
            output_iterator1 Urowk_k_1 = Urowk+k_1;
            if (k < nrt && (e[k] != 0.0))
            {
                int j = k_1;
                output_iterator1 Urowj_k_1 = Urowk_end+k_1;
                do
                {
                    image::vec::aypx(Urowk_k_1,Urowk_end,image::vec::dot(Urowk_k_1,Urowk_end,Urowj_k_1)/-(*Urowk_k_1),Urowj_k_1);
                    if (++j == nu)
                        break;
                    Urowj_k_1 += n;
                }
                while (1);
            }
            std::fill(Urowk,Urowk_end,0.0);
            Urowk[k] = 1.0;
            if (--k < 0)
                break;
            Urowk_end = Urowk;
            Urowk -= n;
        }
    }

    // If required, generate U.
    {
        int k = nu-1;
        input_iterator Arowk = A+k*m;
        input_iterator Arowk_end = Arowk+m;
        if (k >= 0)
            while (1)
            {
                input_iterator Arowk_k = Arowk + k;
                if (s[k] != 0.0 && k != nct)
                {
                    int j = k + 1;
                    if (j < nu)
                    {
                        input_iterator Arowj_k = Arowk_end+k;
                        while (1)
                        {
                            image::vec::aypx(Arowk_k,Arowk_end,image::vec::dot(Arowk_k,Arowk_end,Arowj_k)/-(*Arowk_k),Arowj_k);
                            if (++j == nu)
                                break;
                            Arowj_k += m;
                        }
                    }
                    image::vec::negate(Arowk_k,Arowk_end);
                    *Arowk_k += 1.0;
                    std::fill(Arowk,Arowk_k,0.0);
                }
                else
                {
                    std::fill(Arowk,Arowk_end,0.0);
                    *Arowk_k = 1.0;
                }
                if (--k < 0)
                    break;
                Arowk_end = Arowk;
                Arowk -= m;
            }
    }
    for (int pp = nu-1;pp > 0;)
    {
        int pp_1 = pp-1;
        if (max_value + e[pp_1] == max_value)
        {
            e[pp_1] = 0.0;
            --pp;
            continue;
        }
        int k;
        for (k = pp-2; k >= 0; k--)
            if (max_value + e[k] == max_value)
            {
                e[k] = 0.0;
                break;
            }
        int ks;
        for (ks = pp; ks > k; ks--)
            if (max_value + s[ks] == max_value)// singularity
            {
                s[ks] = 0.0;
                if (ks == pp)
                {
                    ++k;
                    // Deflate negligible s(p).
                    value_type f(e[pp_1]);
                    e[pp_1] = 0.0;

                    output_iterator1 Urowpp = U+pp*n;
                    output_iterator1 Urowj = Urowpp-n;
                    value_type t,cs,sn;
                    int j = pp_1;
                    while (1)
                    {
                        t = hypot(s[j],f);
                        cs = s[j]/t;
                        sn = f/t;
                        s[j] = t;
                        image::vec::rot(Urowj,Urowj+n,Urowpp,cs,sn);
                        if (j-- != k)
                        {
                            f = -sn*e[j];
                            e[j] *= cs;
                        }
                        if (j < k)
                            break;
                        Urowj -= n;
                    }
                }
                else
                {
                    // Split at negligible s(k).
                    value_type f(e[ks]);
                    e[ks] = 0.0;
                    input_iterator Arowks = A+ks*m;
                    input_iterator Arowj = Arowks;
                    value_type t,cs,sn;
                    int j = ks;
                    while (1)
                    {
                        t = hypot(s[j],f);
                        cs = s[j]/t;
                        sn = f/t;
                        s[j] = t;
                        f = -sn*e[j];
                        e[j] = cs*e[j];
                        image::vec::rot(Arowj,Arowj+m,Arowks,cs,sn);
                        if (++j > pp)
                            break;
                        Arowj += m;
                    }
                }

                break;
            }
        if (ks == k)
        {
            // Perform one qr step.
            ++k;
            // Calculate the shift.
            value_type sp = s[pp];
            value_type sk = s[k];
            value_type ek = e[k];
            value_type b = e[pp_1];
            b *= b;
            b += (s[pp_1] + sp)*(s[pp_1] - sp);
            b /= 2.0;
            value_type c = sp*e[pp_1];
            value_type shift = 0.0;
            c *= c;
            if ((b != 0.0) || (c != 0.0))
            {
                shift = sqrt(b*b + c);
                if (b < 0.0)
                    shift = -shift;
                shift = c/(b + shift);
            }
            value_type f = (sk + sp)*(sk - sp) + shift;
            value_type g = sk*ek;
            value_type t,cs,sn;
            int j = k;
            input_iterator Arowj_1 = A+k*m+m;
            output_iterator1 Urowj_1 = U+k*n+n;
            while (1)
            {
                int j_1 = j+1;
                t = hypot(f,g);
                cs = f/t;
                sn = g/t;

                if (j != k)
                    e[j-1] = t;
                f = cs*s[j] + sn*e[j];
                e[j] = e[j]*cs-sn*s[j];
                g = sn*s[j_1];
                s[j_1] *= cs;

                image::vec::rot(Urowj_1-n,Urowj_1,Urowj_1,cs,sn);

                t = hypot(f,g);
                cs = f/t;
                sn = g/t;

                s[j] = t;
                f = cs*e[j] + sn*s[j_1];
                s[j_1] = s[j_1] * cs - sn*e[j];
                g = sn*e[j_1];
                e[j_1] *= cs;

                image::vec::rot(Arowj_1-m,Arowj_1,Arowj_1,cs,sn);
                if (++j == pp)
                    break;
                Arowj_1 += m;
                Urowj_1 += n;
            }
            e[pp_1] = f;

        }
    } // while

    // make singular value positive
    {
        output_iterator1 Urowk = U;
        output_iterator2 s_iter = s;
        output_iterator2 s_end = s+nu;
        while (1)
        {
            if (*s_iter < 0.0)
            {
                *s_iter = -*s_iter;
                image::vec::negate(Urowk,Urowk+n);
            }
            if (++s_iter == s_end)
                break;
            Urowk += n;
        }
    }
    // sort the values

    {
        input_iterator Arow = A;
        output_iterator1 Urow = U;
        output_iterator2 s_end = s+nu;
        for (output_iterator2 s_to = s+nu-1;s != s_to;++s,Arow += m,Urow += n)
        {
            output_iterator2 s_max= std::max_element(s,s_end);
            if (s_max == s)
                continue;
            std::swap(*s,*s_max);
            unsigned int dif = s_max-s;
            image::vec::swap(Urow,Urow+n,Urow+dif*n);
            image::vec::swap(Arow,Arow+m,Arow+dif*m);
        }
    }
    if (n != dimension.row_count())
    {
        inplace_transpose(A,dimension);
    }
}

template <class input_iterator,class output_iterator2,class dym_type>
void svd(input_iterator A,output_iterator2 s,dym_type dimension)
{
    typedef typename std::iterator_traits<input_iterator>::value_type value_type;
    int n = dimension.row_count();
    int m = dimension.col_count();
    int nu = n;
    std::vector<value_type> e_(n);
    value_type* e = &*e_.begin();
    value_type* e_end = e+n;
    std::vector<value_type> w_(m);
    value_type* w = &*w_.begin();
    value_type* w_end = w+m;
    value_type max_value = pow(2.0,-52.0);

    // Reduce A to bidiagonal form, storing the diagonal elements
    // in s and the super-diagonal elements in e.

    int nct = std::min(m-1,n);
    int nrt = std::max(0,nu-2);
    {
        int k = 0;
        input_iterator Arowk = A;
        input_iterator Arowk_end = Arowk+m;
        do
        {
            int k_1 = k+1;
            {
                input_iterator Arowk_k = Arowk+k;//A[k][k];

                value_type s_k = image::vec::norm2(Arowk_k,Arowk_end);
                if (s_k == 0.0)
                {
                    s[k] = 0.0;
                    int j = k_1;
                    if (j < n)
                    {
                        input_iterator Arowj_k = Arowk_end+k; // A[k][j];
                        do
                        {
                            e[j] = *Arowj_k;
                            if (++j == n)
                                break;
                            Arowj_k += m;
                        }
                        while (1);
                    }
                }
                else
                {
                    if (Arowk[k] < 0.0)
                        s_k = -s_k;
                    image::vec::scale(Arowk_k,Arowk_end,1.0/s_k);
                    *Arowk_k += 1.0;
                    s[k] = -s_k;
                    int j = k_1;
                    if (j < n)
                    {
                        input_iterator Arowj_k = Arowk_end+k; // A[k][j];
                        do
                        {
                            image::vec::aypx(Arowk_k,Arowk_end,image::vec::dot(Arowk_k,Arowk_end,Arowj_k)/-*Arowk_k,Arowj_k);
                            e[j] = *Arowj_k;
                            if (++j == n)
                                break;
                            Arowj_k += m;
                        }
                        while (1);
                    }
                }
            }
            //now U[k:m][k] stores in A[k:m][k];

            if (k < nrt)
            {
                value_type* e_k_1 = e + k_1;
                value_type e_k_value = image::vec::norm2(e_k_1,e_end);
                if (e_k_value != 0.0)
                {
                    if (*e_k_1 < 0.0)
                        e_k_value = -e_k_value;

                    image::vec::scale(e_k_1,e_end,1.0/e_k_value);
                    *e_k_1 += 1.0;
                    e_k_value = -e_k_value;
                    // Apply the transformation.
                    if (k+1 < m)
                    {
                        value_type* w_k_1 = w+k_1;
                        std::fill(w_k_1,w_end,0.0);

                        //if (j < n)   no need this check
                        {
                            int j = k_1;
                            input_iterator Arowj_k_1 = Arowk_end+k_1;
                            do
                            {
                                image::vec::axpy(w_k_1,w_end,e[j],Arowj_k_1);
                                if (++j == n)
                                    break;
                                Arowj_k_1 += m;
                            }
                            while (1);
                        }

                        //if (j < n)   no need this check
                        {
                            int j = k_1;
                            input_iterator Arowj_k_1 = Arowk_end+k_1;
                            value_type e_k_1_value = *e_k_1;
                            do
                            {
                                image::vec::aypx(w_k_1,w_end,-e[j]/e_k_1_value,Arowj_k_1);
                                if (++j == n)
                                    break;
                                Arowj_k_1 += m;
                            }
                            while (1);
                        }
                    }
                }

                e[k] = e_k_value;
            }
            max_value=std::max(max_value,std::abs(e[k])+std::abs(s[k]));
            if (++k == nct)
                break;
            Arowk = Arowk_end;
            Arowk_end += m;
        }
        while (1);
    }

    if (m == n)
        s[nct] = *(A+nct*m+nct);
    e[nrt] = A[(n-1)*m + nrt];
    for (int pp = nu-1;pp > 0;)
    {
        int pp_1 = pp-1;
        if (max_value + e[pp_1] == max_value)
        {
            e[pp_1] = 0.0;
            --pp;
            continue;
        }
        int k;
        for (k = pp-2; k >= 0; k--)
            if (max_value + e[k] == max_value)
            {
                e[k] = 0.0;
                break;
            }
        int ks;
        for (ks = pp; ks > k; ks--)
            if (max_value + s[ks] == max_value)// singularity
            {
                if (ks == pp)
                {
                    ++k;
                    // Deflate negligible s(p).
                    value_type f(e[pp_1]);
                    e[pp_1] = 0.0;
                    value_type t,cs,sn;
                    for (int j = pp_1;j >= k;)
                    {
                        t = hypot(s[j],f);
                        cs = s[j]/t;
                        sn = f/t;
                        s[j] = t;
                        if (j-- != k)
                        {
                            f = -sn*e[j];
                            e[j] *= cs;
                        }
                    }
                }
                else
                {
                    // Split at negligible s(k).
                    value_type f(e[ks]);
                    e[ks] = 0.0;
                    value_type t,cs,sn;
                    for (int j = ks;j <= pp;++j)
                    {
                        t = hypot(s[j],f);
                        cs = s[j]/t;
                        sn = f/t;
                        s[j] = t;
                        f = -sn*e[j];
                        e[j] = cs*e[j];
                    }
                }

                break;
            }
        if (ks == k)
        {
            // Perform one qr step.
            ++k;
            // Calculate the shift.
            value_type sp = s[pp];
            value_type sk = s[k];
            value_type ek = e[k];
            value_type b = e[pp_1];
            b *= b;
            b += (s[pp_1] + sp)*(s[pp_1] - sp);
            b /= 2.0;
            value_type c = sp*e[pp_1];
            value_type shift = 0.0;
            c *= c;
            if ((b != 0.0) || (c != 0.0))
            {
                shift = sqrt(b*b + c);
                if (b < 0.0)
                    shift = -shift;
                shift = c/(b + shift);
            }
            value_type f = (sk + sp)*(sk - sp) + shift;
            value_type g = sk*ek;
            value_type t,cs,sn;
            for (int j = k;j < pp;++j)
            {
                int j_1 = j+1;
                t = hypot(f,g);
                cs = f/t;
                sn = g/t;

                if (j != k)
                    e[j-1] = t;
                f = cs*s[j] + sn*e[j];
                e[j] = e[j]*cs-sn*s[j];
                g = sn*s[j_1];
                s[j_1] *= cs;
                t = hypot(f,g);
                cs = f/t;
                sn = g/t;

                s[j] = t;
                f = cs*e[j] + sn*s[j_1];
                s[j_1] = s[j_1] * cs - sn*e[j];
                g = sn*e[j_1];
                e[j_1] *= cs;
            }
            e[pp_1] = f;

        }
    } // while
    for (unsigned int i = 0;i < nu;++i)
        if (s[i] < 0.0)
            s[i] = -s[i];
    std::sort(s,s+nu,std::greater<value_type>());
}

template<class input_iterator1,class input_iterator2,class output_iterator,class dim_type>
void pseudo_inverse_solve(input_iterator1 At,input_iterator2 y,output_iterator x,dim_type dim)
{
    typedef typename std::iterator_traits<output_iterator>::value_type value_type;
    unsigned int n = dim.row_count();
    unsigned int m = dim.col_count();
    std::vector<value_type> tmp(n),AtA(n*n);
    std::vector<int> pv(n);
    vector_product(At,y,&*tmp.begin(),dim);
    product_transpose(At,At,&*AtA.begin(),dim,dim);
    lu_decomposition(&*AtA.begin(),&*pv.begin(),dyndim(n,n));
    lu_solve(&*AtA.begin(),&*pv.begin(),&*tmp.begin(),x,dyndim(n,n));
}


template<class input_iterator,class output_iterator,class dim_type>
void pseudo_inverse(input_iterator A_,output_iterator A,dim_type dim)
{
    typedef typename std::iterator_traits<output_iterator>::value_type value_type;
    unsigned int n = dim.row_count();
    unsigned int m = dim.col_count();
	unsigned int size = dim.size();
	if(n > m)
		return;
    //A: n singular vector having m dimensions (right side matrix)
	//U: n singular vector having n dimensions (left side matrix)
	//s: n singular values
	std::vector<value_type> U_buffer(n*n);
	value_type* U = &*U_buffer.begin();
	std::vector<value_type> s(n);
	std::copy(A_,A_+size,A);
    svd(A,U,&*s.begin(),dim);

	value_type threshold = std::numeric_limits<value_type>::epsilon()*(value_type)m*s[0];
    
	std::vector<value_type> At_buffer(size),tmp_buffer(size);
	value_type* At = &*At_buffer.begin();
	value_type* tmp = &*tmp_buffer.begin();
	
	output_iterator out = A;
	for(unsigned int index = 0;index < n;++index)
		if(s[index] > threshold)
		{
            image::vec::gen(A,A+m,U,U+n,tmp);
            image::vec::axpy(At,At+size,value_type(1.0)/s[index],tmp);
			A += m;
			U += n;
		}

	std::copy(At,At+size,out);
}


}//namespace mat




template<class iterator_type>
struct inverse_delegate{
    iterator_type iter;
    inverse_delegate(iterator_type iter_):iter(iter_){;}
    const inverse_delegate& operator=(const inverse_delegate& rhs)
    {
        iter = rhs.iter;
        return *this;
    }
};

template<class right_type>
inverse_delegate<typename right_type::const_iterator> inverse(const right_type& rhs)
{
    return inverse_delegate<typename right_type::const_iterator>(rhs.begin());
}
template<class right_type>
inverse_delegate<const right_type*> inverse(const right_type* rhs)
{
    return inverse_delegate<const right_type*>(rhs);
}


template<int c,class left_iterator_type,class right_iterator_type>
struct product_delegate{
    left_iterator_type lhs;
    right_iterator_type rhs;
    product_delegate(left_iterator_type lhs_,right_iterator_type rhs_):lhs(lhs_),rhs(rhs_){;}
    const product_delegate& operator=(const product_delegate& rhs)
    {
        lhs = rhs.lhs;
        rhs = rhs.rhs;
        return *this;
    }
};


template<int row_count,int col_count,class value_type>
struct matrix{
    static const unsigned int mat_size = row_count*col_count;
    value_type value[mat_size];
    typedef value_type* iterator;
    typedef const value_type* const_iterator;
    typedef matrix<row_count,col_count,value_type> type;
    typedef dim<row_count,col_count> dim_type;
public:
    matrix(void){}
    matrix(const matrix& rhs){std::copy(rhs.begin(),rhs.end(),value);}
    template<class iterator_type>
    matrix(iterator_type iter){std::copy(iter,iter+mat_size,value);}
    template<int c,class lhs_type,class rhs_type>
    matrix(const product_delegate<c,lhs_type,rhs_type>& prod)
    {
        image::mat::product(prod.lhs,prod.rhs,value,dim<row_count,c>(),dim<c,col_count>());
    }
    template<class rhs_type>
    matrix(const inverse_delegate<rhs_type>& inv)
    {
        image::mat::inverse(inv.iter,value,dim_type());
    }
public:
    value_type& operator[](unsigned int index){return value[index];}
    const value_type& operator[](unsigned int index) const{return value[index];}
    value_type* begin(void){return value;}
    const value_type* begin(void) const{return value;}
    value_type* end(void){return value+mat_size;}
    const value_type* end(void) const{return value+mat_size;}
public:
    const matrix& operator=(const matrix& rhs)
    {
        std::copy(rhs.begin(),rhs.end(),value);
        return *this;
    }
    template<class rhs_type>
    const matrix& operator=(const rhs_type& rhs)
    {
        std::copy(rhs.begin(),rhs.end(),value);
        return *this;
    }
    template<class rhs_type>
    const matrix& operator=(const rhs_type* rhs)
    {
        std::copy(rhs,rhs+row_count*col_count,value);
        return *this;
    }
    template<class rhs_type>
    const matrix& operator*=(const rhs_type& rhs)
    {
        value_type old_value[mat_size];
        std::copy(value,value+mat_size,old_value);
        image::mat::product(old_value,rhs.begin(),value,dim_type(),dim_type());
        return *this;
    }

    template<class rhs_type>
    product_delegate<col_count,const_iterator,typename rhs_type::const_iterator> operator*(const rhs_type& rhs)
    {
        return product_delegate<col_count,const_iterator,typename rhs_type::const_iterator>(value,rhs.begin());
    }
    template<class pointer_type>
    product_delegate<col_count,const_iterator,const pointer_type*> operator*(const pointer_type* rhs)
    {
        return product_delegate<col_count,const_iterator,const pointer_type*>(value,rhs);
    }
    template<int c,class lhs_type,class rhs_type>
    const matrix& operator=(const product_delegate<c,lhs_type,rhs_type>& prod)
    {
        image::mat::product(prod.lhs,prod.rhs,value,dim<row_count,c>(),dim<c,col_count>());
        return *this;
    }
    template<class rhs_type>
    const matrix& operator=(const inverse_delegate<rhs_type>& inv)
    {
        image::mat::inverse(inv.iter,value,dim_type());
        return *this;
    }
    bool inv(void)
    {
        return image::mat::inverse(value,dim_type());
    }
    value_type det(void) const
    {
        return image::mat::determinant(value,dim_type());
    }
    void zero(void)
    {
        std::fill(value,value+mat_size,value_type(0));
    }
    void identity(void)
    {
        image::mat::identity(value,dim_type());
    }
    void swap(matrix& rhs)
    {
        for(int i = 0;i < mat_size;++i)
            std::swap(value[i],rhs.value[i]);
    }
};


template<int row_count,int col_count,class iterator_type>
struct matrix_buf{
    static const unsigned int mat_size = row_count*col_count;
    typedef iterator_type iterator;
    typedef const iterator_type const_iterator;
    typedef typename std::iterator_traits<iterator_type>::value_type value_type;
    typedef matrix_buf<row_count,col_count,iterator_type> type;
    typedef dim<row_count,col_count> dim_type;
public:
    iterator_type iter;
public:
    matrix_buf(void){}
    matrix_buf(const matrix_buf& rhs):iter(rhs.iter){}
    matrix_buf(iterator_type iter_):iter(iter_){}
public:
    value_type& operator[](unsigned int index){return iter[index];}
    const value_type& operator[](unsigned int index) const{return iter[index];}
    iterator_type begin(void){return iter;}
    const iterator_type begin(void) const{return iter;}
    iterator_type end(void){return iter+mat_size;}
    const iterator_type end(void) const{return iter+mat_size;}
public:
    const matrix_buf& operator=(const matrix_buf& rhs)
    {
        iter = rhs.iter;
        return *this;
    }

    template<class rhs_type>
    const matrix_buf& operator=(const rhs_type& rhs)
    {
        std::copy(rhs.begin(),rhs.end(),iter);
        return *this;
    }
    template<class rhs_type>
    const matrix_buf& operator=(const rhs_type* rhs)
    {
        std::copy(rhs,rhs+row_count*col_count,iter);
        return *this;
    }

    template<class rhs_type>
    product_delegate<col_count,const_iterator,typename rhs_type::const_iterator> operator*(const rhs_type& rhs)
    {
        return product_delegate<col_count,const_iterator,typename rhs_type::const_iterator>(iter,rhs.begin());
    }
    template<class pointer_type>
    product_delegate<col_count,const_iterator,const pointer_type*> operator*(const pointer_type* rhs)
    {
        return product_delegate<col_count,const_iterator,const pointer_type*>(iter,rhs);
    }
    template<int c,class lhs_type,class rhs_type>
    const matrix_buf& operator=(const product_delegate<c,lhs_type,rhs_type>& prod)
    {
        image::mat::product(prod.lhs,prod.rhs,iter,dim<row_count,c>(),dim<c,col_count>());
        return *this;
    }
    template<class rhs_type>
    const matrix_buf& operator=(const inverse_delegate<rhs_type>& inv)
    {
        image::mat::inverse(inv.iter,iter,rhs_type::dim_type());
        return *this;
    }
    bool inv(void)
    {
        return image::mat::inverse(iter,dim_type());
    }
    value_type det(void)
    {
        return image::mat::determinant(iter,dim_type());
    }
    void identity(void)
    {
        image::mat::identity(iter,dim_type());
    }
};

template<int row_count,int col_count,class iterator_type>
matrix_buf<row_count,col_count,iterator_type> make_matrix(iterator_type iterator,dim<row_count,col_count>)
{
    return matrix_buf<row_count,col_count,iterator_type>(iterator);
}

}

#endif
