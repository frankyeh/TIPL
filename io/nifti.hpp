#ifndef TIPL_NIFTI_HPP
#define TIPL_NIFTI_HPP
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
#include <string>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <filesystem>
#include <stdint.h>
#include "interface.hpp"
#include "../numerical/basic_op.hpp"
#include "../numerical/numerical.hpp"
#include "../numerical/resampling.hpp"

namespace tipl
{

namespace io
{

struct header_key /* header key */
{
    /* off + size */
    int sizeof_hdr; /* 0 + 4 */
    char data_type[10]; /* 4 + 10 */
    char db_name[18]; /* 14 + 18 */
    int extents; /* 32 + 4 */
    short int session_error; /* 36 + 2 */
    char regular; /* 38 + 1 */
    char hkey_un0; /* 39 + 1 */
}; /* total=40 bytes */
struct image_dimension
{
    /* off + size */
    short int dim[8]; /* 0 + 16 */
    short int unused8; /* 16 + 2 */
    short int unused9; /* 18 + 2 */
    short int unused10; /* 20 + 2 */
    short int unused11; /* 22 + 2 */
    short int unused12; /* 24 + 2 */
    short int unused13; /* 26 + 2 */
    short int unused14; /* 28 + 2 */
    short int datatype; /* 30 + 2 */
    short int bitpix; /* 32 + 2 */
    short int dim_un0; /* 34 + 2 */
    float pixdim[8]; /* 36 + 32 */
    /*
    pixdim[] specifies the voxel dimensitons:
    pixdim[1] - voxel width
    pixdim[2] - voxel height
    pixdim[3] - interslice distance
    ...etc
    */
    float vox_offset; /* 68 + 4 */
    float funused1; /* 72 + 4 */
    float funused2; /* 76 + 4 */
    float funused3; /* 80 + 4 */
    float cal_max; /* 84 + 4 */
    float cal_min; /* 88 + 4 */
    float compressed; /* 92 + 4 */
    float verified; /* 96 + 4 */
    int glmax,glmin; /* 100 + 8 */
}; /* total=108 bytes */
struct data_history
{
    /* off + size */
    char descrip[80]; /* 0 + 80 */
    char aux_file[24]; /* 80 + 24 */
    char orient; /* 104 + 1 */
    char originator[10]; /* 105 + 10 */
    char generated[10]; /* 115 + 10 */
    char scannum[10]; /* 125 + 10 */
    char patient_id[10]; /* 135 + 10 */
    char exp_date[10]; /* 145 + 10 */
    char exp_time[10]; /* 155 + 10 */
    char hist_un0[3]; /* 165 + 3 */
    int views; /* 168 + 4 */
    int vols_added; /* 172 + 4 */
    int start_field; /* 176 + 4 */
    int field_skip; /* 180 + 4 */
    int omax, omin; /* 184 + 8 */
    int smax, smin; /* 192 + 8 */
};
struct dsr
{
    struct header_key hk; /* 0 + 40 */
    struct image_dimension dime; /* 40 + 108 */
    struct data_history hist; /* 148 + 200 */
}; /* total= 348 bytes */
/* Acceptable values for datatype */




template<typename fun_type>
struct nifti_type_info;

template<>
struct nifti_type_info<unsigned char>
{
    static const int16_t data_type = 2;
    static const int16_t bit_pix = 8;
};
template<>
struct nifti_type_info<char>
{
    static const int16_t data_type = 2;
    static const int16_t bit_pix = 8;
};

template<>
struct nifti_type_info<int16_t>
{
    static const int16_t data_type = 4;
    static const int16_t bit_pix = 16;
};
template<>
struct nifti_type_info<uint16_t>
{
    static const int16_t data_type = 4;
    static const int16_t bit_pix = 16;
};

template<>
struct nifti_type_info<int32_t>
{
    static const int16_t data_type = 8;
    static const int16_t bit_pix = 32;
};
template<>
struct nifti_type_info<uint32_t>
{
    static const int16_t data_type = 8;
    static const int16_t bit_pix = 32;
};
template<>
struct nifti_type_info<float>
{
    static const int16_t data_type = 16;
    static const int16_t bit_pix = 32;
};

template<>
struct nifti_type_info<double>
{
    static const int16_t data_type = 64;
    static const int16_t bit_pix = 64;
};

template<>
struct nifti_type_info<int64_t>
{
    static const int16_t data_type = 1024;
    static const int16_t bit_pix = 64;
};

template<>
struct nifti_type_info<uint64_t>
{
    static const int16_t data_type = 1280;
    static const int16_t bit_pix = 64;
};

template<>
struct nifti_type_info<rgb>
{
    static const int16_t data_type = 128;
    static const int16_t bit_pix = 24;
};

typedef struct
{
    float real;
    float imag;
} complex;

template<>
struct nifti_type_info<complex>
{
    static const int16_t data_type = 32;
    static const int16_t bit_pix = 64;
};


/*************************/  /************************/
struct nifti_1_header
{
    /* NIFTI-1 usage         */  /* nifti 7.5 field(s) */
    /*************************/  /************************/

    /*--- was header_key substruct ---*/
    int   sizeof_hdr;    /*!< MUST be 348           */  /* int sizeof_hdr;      */
    char  data_type[10]; /*!< ++UNUSED++            */  /* char data_type[10];  */
    char  db_name[18];   /*!< ++UNUSED++            */  /* char db_name[18];    */
    int   extents;       /*!< ++UNUSED++            */  /* int extents;         */
    short session_error; /*!< ++UNUSED++            */  /* short session_error; */
    char  regular;       /*!< ++UNUSED++            */  /* char regular;        */
    char  dim_info;      /*!< MRI slice ordering.   */  /* char hkey_un0;       */

    /*--- was image_dimension substruct ---*/
    short dim[8];        /*!< Data array dimensions.*/  /* short dim[8];        */
    float intent_p1 ;    /*!< 1st intent parameter. */  /* short unused8;       */
    /* short unused9;       */
    float intent_p2 ;    /*!< 2nd intent parameter. */  /* short unused10;      */
    /* short unused11;      */
    float intent_p3 ;    /*!< 3rd intent parameter. */  /* short unused12;      */
    /* short unused13;      */
    short intent_code ;  /*!< NIFTI_INTENT_* code.  */  /* short unused14;      */
    short datatype;      /*!< Defines data type!    */  /* short datatype;      */
    short bitpix;        /*!< Number bits/voxel.    */  /* short bitpix;        */
    short slice_start;   /*!< First slice index.    */  /* short dim_un0;       */
    float pixdim[8];     /*!< Grid spacings.        */  /* float pixdim[8];     */
    float vox_offset;    /*!< Offset into .nii file */  /* float vox_offset;    */
    float scl_slope ;    /*!< Data scaling: slope.  */  /* float funused1;      */
    float scl_inter ;    /*!< Data scaling: offset. */  /* float funused2;      */
    short slice_end;     /*!< Last slice index.     */  /* float funused3;      */
    char  slice_code ;   /*!< Slice timing order.   */
    char  xyzt_units ;   /*!< Units of pixdim[1..4] */
    float cal_max;       /*!< Max display intensity */  /* float cal_max;       */
    float cal_min;       /*!< Min display intensity */  /* float cal_min;       */
    float slice_duration;/*!< Time for 1 slice.     */  /* float compressed;    */
    float toffset;       /*!< Time axis shift.      */  /* float verified;      */
    int   glmax;         /*!< ++UNUSED++            */  /* int glmax;           */
    int   glmin;         /*!< ++UNUSED++            */  /* int glmin;           */

    /*--- was data_history substruct ---*/
    char  descrip[80];   /*!< any text you like.    */  /* char descrip[80];    */
    char  aux_file[24];  /*!< auxiliary filename.   */  /* char aux_file[24];   */

    short qform_code ;   /*!< NIFTI_XFORM_* code.   */  /*-- all nifti 7.5 ---*/
    short sform_code ;   /*!< NIFTI_XFORM_* code.   */  /*   fields below here  */
    /*   are replaced       */
    float quatern_b ;    /*!< Quaternion b param.   */
    float quatern_c ;    /*!< Quaternion c param.   */
    float quatern_d ;    /*!< Quaternion d param.   */
    float qoffset_x ;    /*!< Quaternion x shift.   */
    float qoffset_y ;    /*!< Quaternion y shift.   */
    float qoffset_z ;    /*!< Quaternion z shift.   */

    float srow_x[4] ;    /*!< 1st row affine transform.   */
    float srow_y[4] ;    /*!< 2nd row affine transform.   */
    float srow_z[4] ;    /*!< 3rd row affine transform.   */

    char intent_name[16];/*!< 'name' or meaning of data.  */

    char magic[4] ;      /*!< MUST be "ni1\0" or "n+1\0". */

} ;                   /**** 348 bytes total ****/

struct nifti_2_header {  /* NIFTI-2 usage           */ /* NIFTI-1 usage      */ /*  offset  */
                         /***************************/ /**********************/ /************/
   int   sizeof_hdr;     /*!< MUST be 540           */ /* int sizeof_hdr; (348) */   /*   0 */
   char  magic[8] ;      /*!< MUST be valid signature. */  /* char magic[4];    */   /*   4 */
   int16_t datatype;     /*!< Defines data type!    */ /* short datatype;       */   /*  12 */
   int16_t bitpix;       /*!< Number bits/voxel.    */ /* short bitpix;         */   /*  14 */
   int64_t dim[8];       /*!< Data array dimensions.*/ /* short dim[8];         */   /*  16 */
   double intent_p1 ;    /*!< 1st intent parameter. */ /* float intent_p1;      */   /*  80 */
   double intent_p2 ;    /*!< 2nd intent parameter. */ /* float intent_p2;      */   /*  88 */
   double intent_p3 ;    /*!< 3rd intent parameter. */ /* float intent_p3;      */   /*  96 */
   double pixdim[8];     /*!< Grid spacings.        */ /* float pixdim[8];      */   /* 104 */
   int64_t vox_offset;   /*!< Offset into .nii file */ /* float vox_offset;     */   /* 168 */
   double scl_slope ;    /*!< Data scaling: slope.  */ /* float scl_slope;      */   /* 176 */
   double scl_inter ;    /*!< Data scaling: offset. */ /* float scl_inter;      */   /* 184 */
   double cal_max;       /*!< Max display intensity */ /* float cal_max;        */   /* 192 */
   double cal_min;       /*!< Min display intensity */ /* float cal_min;        */   /* 200 */
   double slice_duration;/*!< Time for 1 slice.     */ /* float slice_duration; */   /* 208 */
   double toffset;       /*!< Time axis shift.      */ /* float toffset;        */   /* 216 */
   int64_t slice_start;  /*!< First slice index.    */ /* short slice_start;    */   /* 224 */
   int64_t slice_end;    /*!< Last slice index.     */ /* short slice_end;      */   /* 232 */
   char  descrip[80];    /*!< any text you like.    */ /* char descrip[80];     */   /* 240 */
   char  aux_file[24];   /*!< auxiliary filename.   */ /* char aux_file[24];    */   /* 320 */
   int qform_code ;      /*!< NIFTI_XFORM_* code.   */ /* short qform_code;     */   /* 344 */
   int sform_code ;      /*!< NIFTI_XFORM_* code.   */ /* short sform_code;     */   /* 348 */
   double quatern_b ;    /*!< Quaternion b param.   */ /* float quatern_b;      */   /* 352 */
   double quatern_c ;    /*!< Quaternion c param.   */ /* float quatern_c;      */   /* 360 */
   double quatern_d ;    /*!< Quaternion d param.   */ /* float quatern_d;      */   /* 368 */
   double qoffset_x ;    /*!< Quaternion x shift.   */ /* float qoffset_x;      */   /* 376 */
   double qoffset_y ;    /*!< Quaternion y shift.   */ /* float qoffset_y;      */   /* 384 */
   double qoffset_z ;    /*!< Quaternion z shift.   */ /* float qoffset_z;      */   /* 392 */
   double srow_x[4] ;    /*!< 1st row affine transform. */  /* float srow_x[4]; */   /* 400 */
   double srow_y[4] ;    /*!< 2nd row affine transform. */  /* float srow_y[4]; */   /* 432 */
   double srow_z[4] ;    /*!< 3rd row affine transform. */  /* float srow_z[4]; */   /* 464 */
   int slice_code ;      /*!< Slice timing order.   */ /* char slice_code;      */   /* 496 */
   int xyzt_units ;      /*!< Units of pixdim[1..4] */ /* char xyzt_units;      */   /* 500 */
   int intent_code ;     /*!< NIFTI_INTENT_* code.  */ /* short intent_code;    */   /* 504 */
   char intent_name[16]; /*!< 'name' or meaning of data. */ /* char intent_name[16]; */  /* 508 */
   char dim_info;        /*!< MRI slice ordering.   */      /* char dim_info;        */  /* 524 */
   char unused_str[15];  /*!< unused, filled with \0 */                                  /* 525 */
} ;                   /**** 540 bytes total ****/


/*

*/
template<typename input_interface = std_istream,
         typename output_interface = std_ostream>
class nifti_base
{

public:
    union
    {
        struct dsr header;
        struct nifti_1_header nif_header;
    };
    bool is_nii; // backward compatibility to ANALYE 7.5
    mutable std::string error_msg;
public:
    std::shared_ptr<input_interface> input_stream;
private:
    bool big_endian;
private:
    std::vector<char> rgb_write_buf;
    const void* write_buf = nullptr;
    size_t write_size = 0;
private:
    bool compatible(long type1,long type2) const
    {
        if(type1 == type2)
            return true;
        if((type1 == 2 && type2 == 256) || (type1 == 256 && type2 == 2))
            return true;
        if((type1 == 4 && type2 == 512) || (type1 == 512 && type2 == 4))
            return true;
        if((type1 == 8 && type2 == 768) || (type1 == 768 && type2 == 8))
            return true;
        if((type1 == 1024 && type2 == 1280) || (type1 == 1280 && type2 == 1024))
            return true;
        return false;
    }
    void convert_to_small_endian(void)
    {
        change_endian(header.hk.sizeof_hdr);
        change_endian(header.hk.extents); /* 32 + 4 */
        change_endian(header.hk.session_error); /* 36 + 2 */
        change_endian(header.dime.dim,8); /* 0 + 16 */
        if (is_nii)
        {
            change_endian(nif_header.intent_p1) ;    /*!< 1st intent parameter. */  /* short unused8;       */
            change_endian(nif_header.intent_p2) ;    /*!< 2nd intent parameter. */  /* short unused10;      */
            change_endian(nif_header.intent_p3) ;    /*!< 3rd intent parameter. */  /* short unused12;      */
        }
        else
        {
            change_endian(header.dime.unused8); /* 16 + 2 */
            change_endian(header.dime.unused9); /* 18 + 2 */
            change_endian(header.dime.unused10); /* 20 + 2 */
            change_endian(header.dime.unused11); /* 22 + 2 */
            change_endian(header.dime.unused12); /* 24 + 2 */
            change_endian(header.dime.unused13); /* 26 + 2 */
        }
        change_endian(header.dime.unused14); /* 28 + 2 */
        change_endian(header.dime.datatype); /* 30 + 2 */
        change_endian(header.dime.bitpix); /* 32 + 2 */
        change_endian(header.dime.dim_un0); /* 34 + 2 */
        change_endian(header.dime.pixdim,8); /* 36 + 32 */
        change_endian(header.dime.vox_offset); /* 68 + 4 */
        change_endian(header.dime.funused1); /* 72 + 4 */
        change_endian(header.dime.funused2); /* 76 + 4 */
        if (is_nii)
            change_endian(nif_header.slice_end);     /*!< Last slice index.     */  /* float funused3;      */
        else
            change_endian(header.dime.funused3); /* 80 + 4 */
        change_endian(header.dime.cal_max); /* 84 + 4 */
        change_endian(header.dime.cal_min); /* 88 + 4 */
        change_endian(header.dime.compressed); /* 92 + 4 */
        change_endian(header.dime.verified); /* 96 + 4 */
        change_endian(header.dime.glmax);
        change_endian(header.dime.glmin); /* 100 + 8 */
        if (is_nii)
        {

            change_endian(nif_header.qform_code) ;   /*!< NIFTI_XFORM_* code.   */  /*-- all nifti 7.5 ---*/
            change_endian(nif_header.sform_code) ;   /*!< NIFTI_XFORM_* code.   */  /*   fields below here  */
            /*   are replaced       */
            change_endian(nif_header.quatern_b) ;    /*!< Quaternion b param.   */
            change_endian(nif_header.quatern_c) ;    /*!< Quaternion c param.   */
            change_endian(nif_header.quatern_d) ;    /*!< Quaternion d param.   */
            change_endian(nif_header.qoffset_x) ;    /*!< Quaternion x shift.   */
            change_endian(nif_header.qoffset_y) ;    /*!< Quaternion y shift.   */
            change_endian(nif_header.qoffset_z) ;    /*!< Quaternion z shift.   */

            change_endian(nif_header.srow_x,4) ;    /*!< 1st row affine transform.   */
            change_endian(nif_header.srow_y,4) ;    /*!< 2nd row affine transform.   */
            change_endian(nif_header.srow_z,4) ;    /*!< 3rd row affine transform.   */
        }
        else
        {
            change_endian(header.hist.views); /* 168 + 4 */
            change_endian(header.hist.vols_added); /* 172 + 4 */
            change_endian(header.hist.start_field); /* 176 + 4 */
            change_endian(header.hist.field_skip); /* 180 + 4 */
            change_endian(header.hist.omax);
            change_endian(header.hist.omin); /* 184 + 8 */
            change_endian(header.hist.smax);
            change_endian(header.hist.smin); /* 192 + 8 */
        }
    }

    void convert_to_small_endian2(nifti_2_header& nif_header2)
    {
        change_endian(nif_header2.datatype);
        change_endian(nif_header2.bitpix);
        change_endian(nif_header2.dim,8);
        change_endian(nif_header2.intent_p1) ;    /*!< 1st intent parameter. */  /* short unused8;       */
        change_endian(nif_header2.intent_p2) ;    /*!< 2nd intent parameter. */  /* short unused10;      */
        change_endian(nif_header2.intent_p3) ;    /*!< 3rd intent parameter. */  /* short unused12;      */
        change_endian(nif_header2.pixdim,8);

        change_endian(nif_header2.vox_offset);
        change_endian(nif_header2.scl_slope);
        change_endian(nif_header2.scl_inter);
        change_endian(nif_header2.cal_max);
        change_endian(nif_header2.cal_min);

        change_endian(nif_header2.slice_duration);
        change_endian(nif_header2.toffset);
        change_endian(nif_header2.slice_start);
        change_endian(nif_header2.slice_end);     /*!< Last slice index.     */  /* float funused3;      */
        change_endian(nif_header2.qform_code) ;   /*!< NIFTI_XFORM_* code.   */  /*-- all nifti 7.5 ---*/
        change_endian(nif_header2.sform_code) ;   /*!< NIFTI_XFORM_* code.   */  /*   fields below here  */
        /*   are replaced       */
        change_endian(nif_header2.quatern_b) ;    /*!< Quaternion b param.   */
        change_endian(nif_header2.quatern_c) ;    /*!< Quaternion c param.   */
        change_endian(nif_header2.quatern_d) ;    /*!< Quaternion d param.   */
        change_endian(nif_header2.qoffset_x) ;    /*!< Quaternion x shift.   */
        change_endian(nif_header2.qoffset_y) ;    /*!< Quaternion y shift.   */
        change_endian(nif_header2.qoffset_z) ;    /*!< Quaternion z shift.   */

        change_endian(nif_header2.srow_x,4) ;    /*!< 1st row affine transform.   */
        change_endian(nif_header2.srow_y,4) ;    /*!< 2nd row affine transform.   */
        change_endian(nif_header2.srow_z,4) ;    /*!< 3rd row affine transform.   */

        change_endian(nif_header2.slice_code);
        change_endian(nif_header2.xyzt_units);
        change_endian(nif_header2.intent_code);
    }
    nifti_base(const nifti_base&) = delete;
    nifti_base& operator=(const nifti_base&) = delete;
public:
    bool load_from_file(const std::string& file_name)
    {
        if (!input_stream->open(file_name))
        {
            error_msg = "Cannot read the file. No reading privilege or the file does not exist.";
            return false;
        }
        int size_of_header = 0;
        input_stream->read(&size_of_header,sizeof(int));
        if(size_of_header != 540 && size_of_header != 348)
        {
            change_endian(size_of_header);
            if(size_of_header != 540 && size_of_header != 348)
            {
                error_msg = "Invalid NIFTI format. Size of header is not 540 or 348";
                return false;
            }
            big_endian = true;
        }
        else
            big_endian = false;

        if(size_of_header == 540) // nifti2
        {
            nifti_2_header nif_header2;
            init_header(); // clear nifti1 headers
            input_stream->read(((char*)&nif_header2)+sizeof(int),sizeof(nifti_2_header)-sizeof(int));
            if (big_endian) // big endian condition
                convert_to_small_endian2(nif_header2);
            if (nif_header2.magic[0] != 'n' ||
                nif_header2.magic[1] != '+' ||
                    nif_header2.magic[2] != '2')
            {
                error_msg = "Invalid NIFTI format. No NIFTI tag found.";
                return false;
            }
            input_stream->seek(size_t(nif_header2.vox_offset));

            nif_header.sizeof_hdr = 348;

            // convert NIFTI2 to NIFTI1
            nif_header.datatype         = nif_header2.datatype;
            nif_header.bitpix           = nif_header2.bitpix;
            std::copy(nif_header2.dim,nif_header2.dim+8,nif_header.dim);
            nif_header.intent_p1        = nif_header2.intent_p1;
            nif_header.intent_p2        = nif_header2.intent_p2;
            nif_header.intent_p3        = nif_header2.intent_p3;
            std::copy(nif_header2.pixdim,nif_header2.pixdim+8,nif_header.pixdim);

            nif_header.vox_offset      = nif_header2.vox_offset;
            nif_header.scl_slope       = nif_header2.scl_slope;
            nif_header.scl_inter       = nif_header2.scl_inter;
            nif_header.cal_max         = nif_header2.cal_max;
            nif_header.cal_min         = nif_header2.cal_min;
            nif_header.slice_duration  = nif_header2.slice_duration;
            nif_header.toffset         = nif_header2.toffset;
            nif_header.slice_start     = nif_header2.slice_start;
            nif_header.slice_end       = nif_header2.slice_end;
            std::copy(nif_header2.descrip,nif_header2.descrip+80,nif_header.descrip);
            std::copy(nif_header2.aux_file,nif_header2.aux_file+24,nif_header.aux_file);
            nif_header.qform_code      = nif_header2.qform_code;
            nif_header.sform_code      = nif_header2.sform_code;
            nif_header.quatern_b       = nif_header2.quatern_b;
            nif_header.quatern_c       = nif_header2.quatern_c;
            nif_header.quatern_d       = nif_header2.quatern_d;

            nif_header.qoffset_x       = nif_header2.qoffset_x;
            nif_header.qoffset_y       = nif_header2.qoffset_y;
            nif_header.qoffset_z       = nif_header2.qoffset_z;

            std::copy(nif_header2.srow_x,nif_header2.srow_x+4,nif_header.srow_x);
            std::copy(nif_header2.srow_y,nif_header2.srow_y+4,nif_header.srow_y);
            std::copy(nif_header2.srow_z,nif_header2.srow_z+4,nif_header.srow_z);

            nif_header.slice_code       = nif_header2.slice_code;
            nif_header.xyzt_units       = nif_header2.xyzt_units;
            nif_header.intent_code      = nif_header2.intent_code;

            std::copy(nif_header2.intent_name,nif_header2.intent_name+16,nif_header.intent_name);
            nif_header.dim_info         = nif_header2.dim_info;

            return (*input_stream);
        }
        else
        // "ni1\0" or "n+1\0"
        {
            input_stream->read(((char*)&nif_header)+sizeof(int),sizeof(nifti_1_header)-sizeof(int));
            if (big_endian) // big endian condition
                convert_to_small_endian();

            if (nif_header.magic[0] == 'n' &&
                    (nif_header.magic[2] == '1' || nif_header.magic[2] == '2') &&
                    (nif_header.magic[1] == 'i' || nif_header.magic[1] == '+'))
                is_nii = true;
            else
                is_nii = false;


            if (is_nii && nif_header.magic[1] == '+')
            {
                //int padding = 0;
                //input_stream->read((char*)&padding,4);
                input_stream->seek(size_t(nif_header.vox_offset));
            }
            else
            {
                // find the img file
                if (file_name.size() < 4)
                {
                    error_msg = "Failed to find the img file.";
                    return false;
                }
                auto file_name_no_ext = std::string(file_name.begin(),file_name.end()-4);
                auto data_file = file_name_no_ext;
                data_file += ".img";
                input_stream.reset(new input_interface);
                if(!input_stream->open(data_file.c_str()))
                {
                    error_msg = "Failed to open the img file.";
                    return false;
                }
            }
        }
        return (*input_stream);
    }
    const char* get_descrip(void) const{return nif_header.descrip;}
    void set_descrip(const char* des)
    {
        std::copy(des,des+80,nif_header.descrip);
    }
    unsigned short width(void) const
    {
        return nif_header.dim[1];
    }
    unsigned short height(void) const
    {
        return nif_header.dim[2];
    }
    unsigned short depth(void) const
    {
        return nif_header.dim[3];
    }
    unsigned short dim(unsigned int index) const
    {
        return nif_header.dim[index];
    }
    bool select_volume(size_t i)
    {
        const size_t byte_per_pixel = nif_header.bitpix/8;
        tipl::shape<3> geo(nif_header.dim+1);
        size_t volume_size = byte_per_pixel*geo.size();
        input_stream->clear();
        input_stream->seek(size_t(nif_header.vox_offset)+i*volume_size);
        return (*input_stream);
    }
    template<typename shape_type>
    void set_dim(const shape_type& geo)
    {
        std::fill(nif_header.dim,nif_header.dim+8,1);
        std::copy(geo.begin(),geo.end(),nif_header.dim+1);
        nif_header.dim[0] = shape_type::dimension;
    }

    template<int dim>
    void set_voxel_size(const tipl::vector<dim,float>& pixel_size_from)
    {
        double pixdim[8];
        std::fill(pixdim,pixdim+8,1);
        std::copy(pixel_size_from.begin(),pixel_size_from.end(),pixdim+1);
        pixdim[0] = dim;
        std::copy(pixdim,pixdim+8,nif_header.pixdim);
        if(nif_header.srow_x[0] == 1.0f)
        {
            nif_header.srow_x[0] = pixel_size_from[0];
            nif_header.srow_y[1] = pixel_size_from[1];
            nif_header.srow_z[2] = pixel_size_from[2];
        }
    }
    template<typename matrix_type>
    void set_image_transformation(matrix_type& R,bool mni = false)
    {
        nif_header.sform_code = mni ? 4:1;
        nif_header.qform_code = 0;
        std::copy(R.begin(),R.begin()+4,nif_header.srow_x);
        std::copy(R.begin()+4,R.begin()+8,nif_header.srow_y);
        std::copy(R.begin()+8,R.begin()+12,nif_header.srow_z);
    }
    template<typename matrix_type,typename geo_type>
    void set_LPS_transformation(matrix_type& R,const geo_type& out)
    {
        set_image_transformation(R);
        nif_header.srow_x[3] += nif_header.srow_x[0]*(out.width()-1);
        nif_header.srow_x[0] = -nif_header.srow_x[0];
        nif_header.srow_y[0] = -nif_header.srow_y[0];
        nif_header.srow_z[0] = -nif_header.srow_z[0];
        nif_header.srow_y[3] += nif_header.srow_y[1]*(out.height()-1);
        nif_header.srow_x[1] = -nif_header.srow_x[1];
        nif_header.srow_y[1] = -nif_header.srow_y[1];
        nif_header.srow_z[1] = -nif_header.srow_z[1];
    }

    template<int dim>
    void get_voxel_size(tipl::vector<dim,float>& pixel_size_from) const
    {
        std::copy(nif_header.pixdim+1,nif_header.pixdim+1+dim,pixel_size_from.begin());
    }

    template<typename float_type>
    void get_image_orientation(float_type R)
    {
        handle_qform();
        std::copy(nif_header.srow_x,nif_header.srow_x+3,R);
        std::copy(nif_header.srow_y,nif_header.srow_y+3,R+3);
        std::copy(nif_header.srow_z,nif_header.srow_z+3,R+6);
    }
    template<typename matrix_type>
    void get_image_transformation(matrix_type& R)
    {
        handle_qform();
        R.identity();
        std::copy(nif_header.srow_x,nif_header.srow_x+12,R.begin());
    }
    bool is_mni(void) const
    {
        return nif_header.sform_code >= 4 &&
               (nif_header.srow_x[3] != 0.0f && nif_header.srow_y[3] != 0.0f && nif_header.srow_z[3] != 0.0f && // translocation is not zero
                nif_header.srow_x[1] == 0.0f && nif_header.srow_x[2] == 0.0f && // off diagnonal is zero
                nif_header.srow_y[0] == 0.0f && nif_header.srow_y[2] == 0.0f && // off diagnonal is zero
                nif_header.srow_z[0] == 0.0f && nif_header.srow_z[1] == 0.0f);   // off diagnonal is zero
    }
    const float* get_transformation(void)
    {
        handle_qform();
        return nif_header.srow_x;
    }

    unsigned short get_bit_count(void)
    {
        return nif_header.bitpix;
    }
public:
    nifti_base(void):input_stream(new input_interface)
    {
        init_header();
    }
    void init_header(void)
    {
        using namespace std;
        std::fill((char*)&nif_header,(char*)&nif_header + sizeof(nifti_1_header),0);
        nif_header.sizeof_hdr = 348;
        nif_header.vox_offset = 352;
        nif_header.scl_slope = 1.0;
        nif_header.sform_code = 1;
        nif_header.quatern_c = 1;
        nif_header.srow_x[0] = 1.0;
        nif_header.srow_y[1] = 1.0;
        nif_header.srow_z[2] = 1.0;
        nif_header.magic[0] = 'n';
        nif_header.magic[1] = '+';
        nif_header.magic[2] = '1';
        nif_header.magic[3] = 0;
        is_nii = true;
        set_voxel_size(tipl::vector<3>(1.0f,1.0f,1.0f));
    }
public:
    template<int dimension>
    void get_image_dimension(shape<dimension>& geo) const
    {
        std::copy(nif_header.dim+1,nif_header.dim+1+dimension,geo.begin());
    }
    bool is_integer(void) const
    {
        return nif_header.datatype != 16 && nif_header.datatype != 64;
    }
    bool is_int8(void) const
    {
        return nif_header.datatype == 2;
    }
    template<typename image_type>
    void load_from_image(const image_type& source)
    {
        nif_header.datatype = nifti_type_info<typename image_type::value_type>::data_type;
        nif_header.bitpix = nifti_type_info<typename image_type::value_type>::bit_pix;

        set_dim(source.shape());
        write_size = source.size()*(size_t)(nif_header.bitpix/8);
        if(nif_header.datatype == 128 && nif_header.bitpix == 24)
        {
            rgb_write_buf.resize(source.size()*3);
            for(size_t i = 0,j = 0; i < source.size();++i,j += 3)
            {
                tipl::rgb c = source[i];
                rgb_write_buf[j] = c.r;
                rgb_write_buf[j+1] = c.g;
                rgb_write_buf[j+2] = c.b;
            }
            write_buf = rgb_write_buf.data();
        }
        else
            write_buf = source.data();
        is_nii = true;
    }
    template<typename image_type,typename srow_type>
    static bool load_to_space(const std::string& file_name,image_type& I,const srow_type& I_T)
    {
        nifti_base nii;
        image_type J;
        if(!nii.load_from_file(file_name) ||
           !nii.toLPS(J))
            return false;
        srow_type J_T;
        nii.get_image_transformation(J_T);
        if(I.shape() == J.shape() && I_T == J_T)
            I.swap(J);
        else
        {
            if(is_label_image(J))
                resample<nearest>(J,I,from_space(I_T).to(J_T));
            else
                resample<linear>(J,I,from_space(I_T).to(J_T));
        }
        return true;
    }
    template<typename image_type>
    static bool load_from_file(const std::string& file_name,image_type& I)
    {
        nifti_base nii;
        if(!nii.load_from_file(file_name) ||
           !nii.toLPS(I))
            return false;
        return true;
    }
    template<typename image_type,typename vs_type>
    static bool load_from_file(const std::string& file_name,image_type& I,vs_type& vs)
    {
        nifti_base nii;
        if(!nii.load_from_file(file_name) ||
           !nii.toLPS(I))
            return false;
        nii.get_voxel_size(vs);
        return true;
    }
    template<typename image_type,typename vs_type,typename srow_type>
    static bool load_from_file(const std::string& file_name,image_type& I,vs_type& vs,srow_type& T,bool& is_mni)
    {
        nifti_base nii;
        if(!nii.load_from_file(file_name) ||
           !nii.toLPS(I))
            return false;
        nii.get_voxel_size(vs);
        nii.get_image_transformation(T);
        is_mni = nii.is_mni(); // NIFTI_XFORM_MNI_152
        return true;
    }
    template<typename image_type,typename vs_type,typename srow_type>
    static inline bool load_from_file(const std::string& file_name,image_type& I,vs_type& vs,srow_type& T)
    {
        bool is_mni = false;
        return load_from_file(file_name,I,vs,T,is_mni);
    }
    template<typename image_type,typename vs_type,typename srow_type>
    static inline bool load_to_space(const std::string& file_name,image_type& I,tipl::matrix<4,4>& space_to)
    {
        nifti_base nii;
        image_type I_;
        if(!nii.load_from_file(file_name) ||
           !nii.toLPS(I_))
            return false;
        tipl::matrix<4,4> space_from;
        nii.get_image_transformation(space_from);
        if(space_from == space_to && I.shape() == I_.shape())
        {
            I.swap(I_);
            return true;
        }
        tipl::resample(I_,I,tipl::from_space(space_to).to(space_from));
        return true;
    }
    template<typename image_type,typename vs_type,typename srow_type>
    static bool save_to_file(const std::string& file_name,const image_type& I,const srow_type& T,
                             bool is_mni_152 = false,const char* descript = nullptr)
    {
        nifti_base nii;
        nii.set_voxel_size(tipl::to_vs(T));
        nii.set_image_transformation(T,is_mni_152);
        nii.load_from_image(I);
        if(descript)
            nii.set_descrip(descript);
        return nii.save_to_file(file_name);
    }
    template<typename image_type,typename vs_type,typename srow_type>
    static bool save_to_file(const std::string& file_name,const image_type& I,const vs_type& vs,const srow_type& T,
                             bool is_mni_152 = false,const char* descript = nullptr)
    {
        nifti_base nii;
        nii.set_voxel_size(vs);
        nii.set_image_transformation(T,is_mni_152);
        nii.load_from_image(I);
        if(descript)
            nii.set_descrip(descript);
        return nii.save_to_file(file_name);
    }
    template<typename prog_type = default_prog_type>
    bool save_to_file(const std::string& file_name,prog_type&& prog = prog_type())
    {
        if(!write_buf)
        {
            error_msg = "no image data for saving";
            return false;
        }
        if (!is_nii)// is the header from the analyze format?
        {
            //yes, then change the header to the NIFTI format
            nif_header.sizeof_hdr = 348;
            nif_header.vox_offset = 352;
            nif_header.sform_code = 1;
            nif_header.magic[0] = 'n';
            nif_header.magic[1] = '+';
            nif_header.magic[2] = '1';
            nif_header.magic[3] = 0;
        }
        output_interface out;
        if(!out.open(file_name))
        {
            error_msg = "Could not open the file ";
            error_msg += file_name;
            error_msg += ". Please check if the file path is correct and you have the necessary permissions to write to this file.";
            return false;
        }
        out.write((const char*)&nif_header,sizeof(nif_header));
        int padding = 0;
        out.write((const char*)&padding,4);

        if(!save_stream_with_prog(prog,out,write_buf,write_size,error_msg))
            return false;
        write_buf = nullptr;
        return out;
    }
    template<typename iterator_type1,typename iterator_type2,typename int_type>
    static void copy_ptr(iterator_type1 iter1,iterator_type2 iter2,int_type size)
    {
        for(iterator_type1 end = iter1+size; iter1 != end; ++iter1,++iter2)
            *iter2 = typename std::iterator_traits<iterator_type2>::value_type(*iter1);
    }
    template<typename lhs_type,typename rhs_type>
    static void copy_data(const void* lhs,rhs_type rhs,size_t size)
    {
        copy_ptr(reinterpret_cast<const lhs_type*>(lhs),rhs,size);
    }
public:
    template<typename pointer_type,typename prog_type>
    bool save_to_buffer(pointer_type ptr,size_t pixel_count,prog_type& prog) const
    {
        if(!input_stream.get() || !(*input_stream))
        {
            error_msg = "end of file reached. The file is smaller than expected.";
            return false;
        }
        const size_t byte_per_pixel = nif_header.bitpix/8;
        if(byte_per_pixel == 0 || pixel_count == 0)
        {
            error_msg = "invalid pixel count.";
            return false;
        }
        typedef typename std::iterator_traits<pointer_type>::value_type value_type;
        if(compatible(nifti_type_info<value_type>::data_type,nif_header.datatype))
        {
            if(!read_stream_with_prog(prog,*input_stream.get(),&*ptr,pixel_count*byte_per_pixel,error_msg))
                return false;
            if (big_endian)
                change_endian(&*ptr,pixel_count);
            return true;
        }
        else
        {
            std::vector<char> buf(pixel_count*byte_per_pixel);
            void* buf_ptr = buf.data();
            if(!read_stream_with_prog(prog,*input_stream.get(),buf_ptr,buf.size(),error_msg))
                return false;
            if (big_endian)
            {
                switch (byte_per_pixel)
                {
                    case 2:
                        change_endian<int16_t>(buf_ptr,buf.size()/2);
                        break;
                    case 4:
                        change_endian<int32_t>(buf_ptr,buf.size()/4);
                        break;
                    case 8:
                        change_endian<double>(buf_ptr,buf.size()/8);
                        break;
                }
            }
            switch (nif_header.datatype)
            {
            case 2://DT_SIGNED_CHAR 2
                copy_data<unsigned char>(buf_ptr,ptr,pixel_count);
                break;
            case 4://DT_SIGNED_SHORT 4
                copy_data<int16_t>(buf_ptr,ptr,pixel_count);
                break;
            case 8://DT_SIGNED_INT 8
                copy_data<int32_t>(buf_ptr,ptr,pixel_count);
                break;
            case 16://DT_FLOAT 16
                copy_data<float>(buf_ptr,ptr,pixel_count);
                break;
            case 64://DT_DOUBLE 64
                copy_data<double>(buf_ptr,ptr,pixel_count);
                break;
            case 128://DT_RGB
                for(size_t index = 0;index < buf.size();index +=3,++ptr)
                    *ptr = uint32_t(tipl::rgb(buf[index],buf[index+1],buf[index+2]));
                break;
            case 256: // DT_INT8
                copy_data<char>(buf_ptr,ptr,pixel_count);
                break;
            case 512: // DT_UINT16
                copy_data<uint16_t>(buf_ptr,ptr,pixel_count);
                break;
            case 768: // DT_UINT32
                copy_data<uint32_t>(buf_ptr,ptr,pixel_count);
                break;
            case 1024: // DT_INT64
                copy_data<int64_t>(buf_ptr,ptr,pixel_count);
                break;
            case 1280: // DT_UINT64
                copy_data<uint64_t>(buf_ptr,ptr,pixel_count);
                break;
            }
            return true;
        }
    }
    template<typename image_type,typename prog_type = tipl::io::default_prog_type>
    bool get_untouched_image(image_type& out,prog_type&& prog = prog_type()) const
    {
        try{
            if(untouched_dim.empty())
                out.resize(tipl::shape<image_type::dimension>(nif_header.dim+1));
            else
                out.resize(tipl::shape<image_type::dimension>(untouched_dim.data()+1));
        }
        catch(...)
        {
            error_msg = "insufficient memory";
            return false;
        }
        if(!save_to_buffer(out.begin(),out.size(),prog))
            return false;
        if(nif_header.scl_slope != 0)
        {
            if(nif_header.scl_slope != 1.0f)
                tipl::multiply_constant(out,nif_header.scl_slope);
            if(nif_header.scl_inter != 0.0f)
                tipl::add_constant(out,nif_header.scl_inter);
        }
        return true;
    }

    template<typename image_type>
    bool save_to_image(image_type& out)
    {
        return toLPS(out);
    }
    template<typename image_type>
    bool operator>>(image_type& source)
    {
        return toLPS(source);
    }
    template<typename image_type>
    const image_type& operator<<(const image_type& source)
    {
        load_from_image(source);
        return source;
    }

    void handle_qform(void)
    {
        if(nif_header.qform_code > 0 && nif_header.sform_code == 0)
        {
            float b = nif_header.quatern_b;
            float c = nif_header.quatern_c;
            float d = nif_header.quatern_d;
            float b2 = b*b;
            float c2 = c*c;
            float d2 = d*d;
            float sum = b2+c2+d2;
            float a2 = (sum > 1.0f) ? 0.0f:1.0f-sum;
            float a = std::sqrt(a2);
            float ab2 = 2.0f*a*b;
            float ac2 = 2.0f*a*c;
            float ad2 = 2.0f*a*d;
            float bc2 = 2.0f*b*c;
            float bd2 = 2.0f*b*d;
            float cd2 = 2.0f*c*d;

            float qfac = nif_header.pixdim[0];
            if(qfac == 0.0f)
                qfac = 1.0f;
            nif_header.srow_x[0] = (a2+b2-c2-d2)*nif_header.pixdim[1];
            nif_header.srow_x[1] = (bc2-ad2)*nif_header.pixdim[2];
            nif_header.srow_x[2] = (bd2+ac2)*nif_header.pixdim[3]*qfac;
            nif_header.srow_x[3] = nif_header.qoffset_x;
            nif_header.srow_y[0] = (bc2+ad2)*nif_header.pixdim[1];
            nif_header.srow_y[1] = (a2+c2-b2-d2)*nif_header.pixdim[2];
            nif_header.srow_y[2] = (cd2-ab2)*nif_header.pixdim[3]*qfac;
            nif_header.srow_y[3] = nif_header.qoffset_y;
            nif_header.srow_z[0] = (bd2-ac2)*nif_header.pixdim[1];
            nif_header.srow_z[1] = (cd2+ab2)*nif_header.pixdim[2];
            nif_header.srow_z[2] = (a2+d2-c2-b2)*nif_header.pixdim[3]*qfac;
            nif_header.srow_z[3] = nif_header.qoffset_z;
            nif_header.sform_code = nif_header.qform_code;
        }
    }

public:
    // 0: flip x  1: flip y  2: flip z
    // 3: swap xy 4: swap yz 5: swap xz
    std::vector<short> untouched_dim;
    std::vector<char> flip_swap_seq;
    template<typename image_type>
    void apply_flip_swap_seq(image_type& out,bool reverse = false)
    {
        for(auto type : ((reverse) ? std::vector<char>(flip_swap_seq.rbegin(),flip_swap_seq.rend()) : flip_swap_seq))
            tipl::flip(out,type);
    }
public:
    template<typename T>
    auto toImage(void)
    {
        T I;
        *this >> I;
        return I;
    }
    //from RAS to LPS
    void toLPS(void)
    {
        handle_qform();
        // swap x y
        untouched_dim = std::vector<short>(nif_header.dim,nif_header.dim+8);
        if(std::fabs(nif_header.srow_y[1]) < std::fabs(nif_header.srow_x[1]) &&
           std::fabs(nif_header.srow_z[1]) < std::fabs(nif_header.srow_x[1]))
        {
            flip_swap_seq.push_back(3);
            {
                std::swap(nif_header.srow_x[0],nif_header.srow_x[1]);
                std::swap(nif_header.srow_y[0],nif_header.srow_y[1]);
                std::swap(nif_header.srow_z[0],nif_header.srow_z[1]);
                std::swap(nif_header.pixdim[1],nif_header.pixdim[2]);
                std::swap(nif_header.dim[1],nif_header.dim[2]);
            }
        }
        // swap x z
        if(std::fabs(nif_header.srow_y[2]) < std::fabs(nif_header.srow_x[2]) &&
           std::fabs(nif_header.srow_z[2]) < std::fabs(nif_header.srow_x[2]))
        {
            flip_swap_seq.push_back(5);
            {
                std::swap(nif_header.srow_x[0],nif_header.srow_x[2]);
                std::swap(nif_header.srow_y[0],nif_header.srow_y[2]);
                std::swap(nif_header.srow_z[0],nif_header.srow_z[2]);
                std::swap(nif_header.pixdim[1],nif_header.pixdim[3]);
                std::swap(nif_header.dim[1],nif_header.dim[3]);
            }
        }
        // swap y z
        if(std::fabs(nif_header.srow_x[2]) < std::fabs(nif_header.srow_y[2]) &&
           std::fabs(nif_header.srow_z[2]) < std::fabs(nif_header.srow_y[2]))
        {
            flip_swap_seq.push_back(4);
            {
                std::swap(nif_header.srow_x[1],nif_header.srow_x[2]);
                std::swap(nif_header.srow_y[1],nif_header.srow_y[2]);
                std::swap(nif_header.srow_z[1],nif_header.srow_z[2]);
                std::swap(nif_header.pixdim[2],nif_header.pixdim[3]);
                std::swap(nif_header.dim[2],nif_header.dim[3]);
            }
        }

        // from +x = Right  +y = Anterior +z = Superior
        // to +x = Left  +y = Posterior +z = Superior
        if(nif_header.srow_x[0] > 0)
        {
            flip_swap_seq.push_back(0);
            {
                nif_header.srow_x[3] += nif_header.srow_x[0]*(nif_header.dim[1]-1);
                nif_header.srow_x[0] = -nif_header.srow_x[0];
                nif_header.srow_y[0] = -nif_header.srow_y[0];
                nif_header.srow_z[0] = -nif_header.srow_z[0];
            }
        }

        if(nif_header.srow_y[1] > 0)
        {
            flip_swap_seq.push_back(1);
            {
                nif_header.srow_y[3] += nif_header.srow_y[1]*(nif_header.dim[2]-1);
                nif_header.srow_x[1] = -nif_header.srow_x[1];
                nif_header.srow_y[1] = -nif_header.srow_y[1];
                nif_header.srow_z[1] = -nif_header.srow_z[1];
            }
        }

        if(nif_header.srow_z[2] < 0)
        {
            flip_swap_seq.push_back(2);
            {
                nif_header.srow_z[3] += nif_header.srow_z[2]*(nif_header.dim[3]-1);
                nif_header.srow_x[2] = -nif_header.srow_x[2];
                nif_header.srow_y[2] = -nif_header.srow_y[2];
                nif_header.srow_z[2] = -nif_header.srow_z[2];
            }
        }
    }
    template<typename image_type,typename prog_type = tipl::io::default_prog_type>
    bool toLPS(image_type& out,prog_type&& prog = prog_type())
    {
        if(!write_buf)
        {
            if(!get_untouched_image(out,prog))
                return false;
        }
        if(flip_swap_seq.empty())
            toLPS();
        apply_flip_swap_seq(out);
        return true;
    }
    friend std::ostream& operator<<(std::ostream& out,const nifti_base& nii)
    {
        out << "sizeof_hdr=" << nii.nif_header.sizeof_hdr << std::endl;
        out << "ndim_info=" << (int)nii.nif_header.dim_info << std::endl;
        for(unsigned int i = 0;i < 8;++i)
            out << "dim[" << i << "]=" << nii.nif_header.dim[i] << std::endl;
        out << "intent_p1=" << nii.nif_header.intent_p1 << std::endl;
        out << "intent_p2=" << nii.nif_header.intent_p2 << std::endl;
        out << "intent_p3=" << nii.nif_header.intent_p3 << std::endl;
        out << "intent_code=" << nii.nif_header.intent_code << std::endl;
        out << "datatype=" << nii.nif_header.datatype << std::endl;
        out << "bitpix=" << nii.nif_header.bitpix << std::endl;
        out << "slice_start=" << nii.nif_header.slice_start << std::endl;

        for(unsigned int i = 0;i < 8;++i)
            out << "pixdim[" << i << "]=" << nii.nif_header.pixdim[i] << std::endl;

        out << "vox_offset=" << nii.nif_header.vox_offset << std::endl;
        out << "scl_slope=" << nii.nif_header.scl_slope << std::endl;
        out << "scl_inter=" << nii.nif_header.scl_inter << std::endl;
        out << "slice_end=" << nii.nif_header.slice_end << std::endl;
        out << "slice_code=" << (int)nii.nif_header.slice_code << std::endl;
        out << "xyzt_units=" << (int)nii.nif_header.xyzt_units << std::endl;
        out << "cal_max=" << nii.nif_header.cal_max << std::endl;
        out << "cal_min=" << nii.nif_header.cal_min << std::endl;
        out << "slice_duration=" << nii.nif_header.slice_duration << std::endl;
        out << "toffset=" << nii.nif_header.toffset << std::endl;
        out << "descrip=" << nii.nif_header.descrip << std::endl;
        out << "aux_file=" << nii.nif_header.aux_file << std::endl;
        out << "qform_code=" << nii.nif_header.qform_code << std::endl;
        out << "sform_code=" << nii.nif_header.sform_code << std::endl;
        out << "quatern_b=" << nii.nif_header.quatern_b << std::endl;
        out << "quatern_c=" << nii.nif_header.quatern_c << std::endl;
        out << "quatern_d=" << nii.nif_header.quatern_d << std::endl;
        out << "qoffset_x=" << nii.nif_header.qoffset_x << std::endl;
        out << "qoffset_y=" << nii.nif_header.qoffset_y << std::endl;
        out << "qoffset_z=" << nii.nif_header.qoffset_z << std::endl;

        for(unsigned int i = 0;i < 4;++i)
            out << "srow_x[" << i << "]=" << nii.nif_header.srow_x[i] << std::endl;
        for(unsigned int i = 0;i < 4;++i)
            out << "srow_y[" << i << "]=" << nii.nif_header.srow_y[i] << std::endl;
        for(unsigned int i = 0;i < 4;++i)
            out << "srow_z[" << i << "]=" << nii.nif_header.srow_z[i] << std::endl;
        out << "intent_name=" << nii.nif_header.intent_name << std::endl;
        return out;
    }

};

typedef nifti_base<> nifti;

} // io
} // tipl

#endif//TIPL_NIFTI_HPP

#ifdef TIPL_GZ_STREAM_HPP
namespace tipl{namespace io{
typedef nifti_base<gz_istream,gz_ostream> gz_nifti;
}}
#endif

