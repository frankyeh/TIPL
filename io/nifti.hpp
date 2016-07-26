#ifndef nifti_HEADER_HPP
#define nifti_HEADER_HPP
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
#include <stdint.h>
#include "interface.hpp"
#include "image/utility/basic_image.hpp"
#include "image/numerical/basic_op.hpp"

namespace image
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




template<class fun_type>
struct nifti_type_info;

template<>
struct nifti_type_info<unsigned char>
{
    static const long data_type = 2;
    static const long bit_pix = 8;
};
template<>
struct nifti_type_info<char>
{
    static const long data_type = 2;
    static const long bit_pix = 8;
};

template<>
struct nifti_type_info<int16_t>
{
    static const long data_type = 4;
    static const long bit_pix = 16;
};
template<>
struct nifti_type_info<uint16_t>
{
    static const long data_type = 4;
    static const long bit_pix = 16;
};

template<>
struct nifti_type_info<int32_t>
{
    static const long data_type = 8;
    static const long bit_pix = 32;
};
template<>
struct nifti_type_info<uint32_t>
{
    static const long data_type = 8;
    static const long bit_pix = 32;
};
template<>
struct nifti_type_info<float>
{
    static const long data_type = 16;
    static const long bit_pix = 32;
};

template<>
struct nifti_type_info<double>
{
    static const long data_type = 64;
    static const long bit_pix = 64;
};

template<>
struct nifti_type_info<int64_t>
{
    static const long data_type = 1024;
    static const long bit_pix = 64;
};

template<>
struct nifti_type_info<uint64_t>
{
    static const long data_type = 1280;
    static const long bit_pix = 64;
};

template<>
struct nifti_type_info<rgb_color>
{
    static const long data_type = 128;
    static const long bit_pix = 24;
};

typedef struct
{
    float real;
    float imag;
} complex;

template<>
struct nifti_type_info<complex>
{
    static const long data_type = 32;
    static const long bit_pix = 64;
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


/*

*/
template<class input_interface = std_istream,class output_interface = std_ostream>
class nifti_base
{

public:
    union
    {
        struct dsr header;
        struct nifti_1_header nif_header;
    };
    bool is_nii; // backward compatibility to ANALYE 7.5
private:
    std::auto_ptr<input_interface> input_stream;
    bool big_endian;
private:
    const void* write_buf;
    size_t write_size;
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
    const char* get_header_name(char)const{return ".hdr";}
    const wchar_t* get_header_name(wchar_t)const{return L".hdr";}
    const char* get_image_name(char)const{return ".img";}
    const wchar_t* get_image_name(wchar_t)const{return L".img";}
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
private:
    nifti_base(const nifti_base& rhs);
    const nifti_base& operator=(const nifti_base& rhs);
public:
    bool load_from_file(const std::string& file_name)
    {
        return load_from_file(file_name.c_str());
    }
    template<class char_type>
    bool load_from_file(const char_type* pfile_name)
    {
        input_stream.reset(new input_interface);
        if (!input_stream->open(pfile_name))
        {
            input_stream.reset(0);
            return false;
        }
        input_stream->read(&header,sizeof(header));
        // "ni1\0" or "n+1\0"
        if (nif_header.magic[0] == 'n' &&
                nif_header.magic[2] == '1' &&
                (nif_header.magic[1] == 'i' || nif_header.magic[1] == '+'))
            is_nii = true;
        else
            is_nii = false;

        big_endian = false;
        if (nif_header.sizeof_hdr == 1543569408) // big endian condition
        {
            convert_to_small_endian();
            big_endian = true;
        }
        if (nif_header.sizeof_hdr != 348)
            return false;


        if (is_nii && nif_header.magic[1] == '+')
        {
            //int padding = 0;
            //input_stream->read((char*)&padding,4);
            input_stream->seek(nif_header.vox_offset);
        }
        else
        {
            // find the img file
            typedef std::basic_string<char_type, std::char_traits<char_type>,std::allocator<char_type> > string_type;
            string_type file_name(pfile_name);
            if (file_name.size() < 4)
                return false;
            string_type file_name_no_ext(file_name.begin(),file_name.end()-4);
            string_type data_file(file_name_no_ext);
            data_file += get_image_name(char_type());
            input_stream.reset(new input_interface);
            if(!input_stream->open(data_file.c_str()))
            {
                input_stream.reset(0);
                return false;
            }
        }
        return (*input_stream);
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

    template<class geometry_type>
    void set_dim(const geometry_type& geo)
    {
        std::fill(nif_header.dim,nif_header.dim+8,1);
        std::copy(geo.begin(),geo.end(),nif_header.dim+1);
        nif_header.dim[0] = std::find(nif_header.dim+1,nif_header.dim+8,1)-(nif_header.dim+1);
    }

    template<class pixel_size_type>
    void set_voxel_size(pixel_size_type pixel_size_from)
    {
        float pixdim[8];
        std::fill(pixdim,pixdim+8,1);
        pixdim[1] = pixel_size_from[0];
        pixdim[2] = pixel_size_from[1];
        pixdim[3] = pixel_size_from[2];
        std::copy(pixdim,pixdim+8,nif_header.pixdim);
        if(nif_header.srow_x[0] == 1.0)
        {
            nif_header.srow_x[0] = pixel_size_from[0];
            nif_header.srow_y[1] = pixel_size_from[1];
            nif_header.srow_z[2] = pixel_size_from[2];
        }
    }

    template<class float_type>
    void set_image_transformation(float_type R)
    {
        nif_header.sform_code = 1.0;
        nif_header.qform_code = 0.0;
        std::copy(R,R+4,nif_header.srow_x);
        std::copy(R+4,R+8,nif_header.srow_y);
        std::copy(R+8,R+12,nif_header.srow_z);
    }

    template<class pixel_size_type>
    void get_voxel_size(pixel_size_type pixel_size_from) const
    {
        std::copy(nif_header.pixdim+1,nif_header.pixdim+1+nif_header.dim[0],pixel_size_from);
    }

    template<class float_type>
    void get_image_orientation(float_type R)
    {
        handle_qform();
        std::copy(nif_header.srow_x,nif_header.srow_x+3,R);
        std::copy(nif_header.srow_y,nif_header.srow_y+3,R+3);
        std::copy(nif_header.srow_z,nif_header.srow_z+3,R+6);
    }
    template<class float_type>
    void get_image_transformation(float_type R)
    {
        handle_qform();
        std::copy(nif_header.srow_x,nif_header.srow_x+12,R);
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
    void set_description(const char* des)
    {
        using namespace std;
        strcpy(nif_header.descrip,des);
    }
public:
    nifti_base(void)
    {
        init_header();
    }
    void init_header(void)
    {
        using namespace std;
        std::fill((char*)&nif_header,(char*)&nif_header + sizeof(nifti_1_header),0);
        nif_header.sizeof_hdr = 348;
        nif_header.vox_offset = 352;
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
    }
public:
    template<int dimension>
    void get_image_dimension(geometry<dimension>& geo) const
    {
        std::copy(nif_header.dim+1,nif_header.dim+1+dimension,geo.begin());
    }

    template<class image_type>
    void load_from_image(const image_type& source)
    {
        nif_header.datatype = nifti_type_info<typename image_type::value_type>::data_type;
        nif_header.bitpix = nifti_type_info<typename image_type::value_type>::bit_pix;
        set_dim(source.geometry());
        write_size = source.size()*(size_t)(nif_header.bitpix/8);
        write_buf = &*source.begin();
        is_nii = true;
    }

    template<class char_type>
    bool save_to_file(const char_type* pfile_name)
    {
        if(!write_buf)
            return false;
        if (!is_nii)// is the header from the analyze format?
        {
            //yes, then change the header to the NIFTI format
            nif_header.sizeof_hdr = 348;
            nif_header.vox_offset = 352;
            nif_header.qform_code = 1;
            strcpy(nif_header.magic,"n+1");
        }
        output_interface out;
        if(!out.open(pfile_name))
            return false;
        out.write((const char*)&nif_header,sizeof(nif_header));
        int padding = 0;
        out.write((const char*)&padding,4);
        out.write((const char*)write_buf,write_size);
        write_buf = 0;
        return out;
    }
    template<class pointer_type>
    bool save_to_buffer(pointer_type ptr,unsigned int pixel_count) const
    {
        const int byte_per_pixel = header.dime.bitpix/8;
        typedef typename std::iterator_traits<pointer_type>::value_type value_type;
        if(compatible(nifti_type_info<value_type>::data_type,nif_header.datatype))
        {
            if(!input_stream->read((char*)&*ptr,pixel_count*byte_per_pixel))
                return false;
            if (big_endian)
                change_endian(&*ptr,pixel_count);
            return true;
        }
        else
        {
            std::vector<char> buf(pixel_count*byte_per_pixel);
            if(buf.empty())
                return false;
            void* buf_ptr = &*buf.begin();
            if(!input_stream->read((char*)buf_ptr,buf.size()))
                return false;
            if (big_endian)
            {
                switch (byte_per_pixel)
                {
                    case 2:
                        change_endian((int16_t*)buf_ptr,buf.size()/2);
                        break;
                    case 4:
                        change_endian((int32_t*)buf_ptr,buf.size()/4);
                        break;
                    case 8:
                        change_endian((double*)buf_ptr,buf.size()/8);
                        break;
                }
            }
            switch (nif_header.datatype)
            {
            case 2://DT_UNSIGNED_CHAR 2
                image::copy_ptr((const unsigned char*)buf_ptr,ptr,pixel_count);
                break;
            case 4://DT_SIGNED_SHORT 4
                image::copy_ptr((const int16_t*)buf_ptr,ptr,pixel_count);
                break;
            case 8://DT_SIGNED_INT 8
                image::copy_ptr((const int32_t*)buf_ptr,ptr,pixel_count);
                break;
            case 16://DT_FLOAT 16
                image::copy_ptr((const float*)buf_ptr,ptr,pixel_count);
                break;
            case 64://DT_DOUBLE 64
                image::copy_ptr((const double*)buf_ptr,ptr,pixel_count);
                break;
            case 128://DT_RGB
                for(unsigned int index = 0;index < buf.size();index +=3,++ptr)
                    *ptr = (short)image::rgb_color(buf[index],buf[index+1],buf[index+2]);
                break;
            case 256: // DT_INT8
                image::copy_ptr((const char*)buf_ptr,ptr,pixel_count);
                break;
            case 512: // DT_UINT16
                image::copy_ptr((const uint16_t*)buf_ptr,ptr,pixel_count);
                break;
            case 768: // DT_UINT32
                image::copy_ptr((const uint32_t*)buf_ptr,ptr,pixel_count);
                break;
            case 1024: // DT_UINT32
                image::copy_ptr((const int64_t*)buf_ptr,ptr,pixel_count);
                break;
            case 1280: // DT_UINT32
                image::copy_ptr((const uint64_t*)buf_ptr,ptr,pixel_count);
                break;
            }
            return true;
        }
    }

    bool has_data(void) const
    {
        if(!input_stream.get() || !(*input_stream))
            return false;
        return true;
    }

    template<class image_type>
    bool save_to_image(image_type& out) const
    {
        if(!has_data())
            return false;
        out.resize(image::geometry<image_type::dimension>(nif_header.dim+1));
        if(!save_to_buffer(out.begin(),out.size()))
            return false;
        if(nif_header.scl_slope != 0)
        {
            image::multiply_constant(out,nif_header.scl_slope);
            image::add_constant(out,nif_header.scl_inter);
        }
        return true;
    }
    template<class image_type>
    const nifti_base& operator>>(image_type& source) const
    {
        save_to_image(source);
        return *this;
    }
    template<class image_type>
    nifti_base& operator<<(const image_type& source)
    {
        load_from_image(source);
        return *this;
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
            float a2 = (sum > 1.0) ? 0.0:1.0-sum;
            float a = std::sqrt(a2);
            float ab2 = 2.0*a*b;
            float ac2 = 2.0*a*c;
            float ad2 = 2.0*a*d;
            float bc2 = 2.0*b*c;
            float bd2 = 2.0*b*d;
            float cd2 = 2.0*c*d;

            float qfac = nif_header.pixdim[0];
            if(qfac == 0.0)
                qfac = 1.0;
            nif_header.srow_x[0] = (a2+b2-c2-d2)*nif_header.pixdim[1];
            nif_header.srow_x[1] = bc2-ad2;
            nif_header.srow_x[2] = bd2+ac2;
            nif_header.srow_x[3] = nif_header.qoffset_x;
            nif_header.srow_y[0] = bc2+ad2;
            nif_header.srow_y[1] = (a2+c2-b2-d2)*nif_header.pixdim[2];
            nif_header.srow_y[2] = cd2-ab2;
            nif_header.srow_y[3] = nif_header.qoffset_y;
            nif_header.srow_z[0] = bd2-ac2;
            nif_header.srow_z[1] = cd2+ab2;
            nif_header.srow_z[2] = (a2+d2-c2-b2)*nif_header.pixdim[3]*qfac;
            nif_header.srow_z[3] = nif_header.qoffset_z;
            nif_header.sform_code = 1;
        }
    }

    //from RAS to LPS
    template<class image_type>
    bool toLPS(image_type& out,bool change_header = true)
    {
        if(!save_to_image(out))
            return false;
        handle_qform();


        if(std::fabs(nif_header.srow_x[0]) < std::fabs(nif_header.srow_x[1]))
        {
            if(change_header)
            {
                std::swap(nif_header.srow_x[0],nif_header.srow_x[1]);
                std::swap(nif_header.srow_y[0],nif_header.srow_y[1]);
                std::swap(nif_header.srow_z[0],nif_header.srow_z[1]);
            }
            image::swap_xy(out);
        }
        if(std::fabs(nif_header.srow_x[0]) < std::fabs(nif_header.srow_x[2]))
        {
            if(change_header)
            {
                std::swap(nif_header.srow_x[0],nif_header.srow_x[2]);
                std::swap(nif_header.srow_y[0],nif_header.srow_y[2]);
                std::swap(nif_header.srow_z[0],nif_header.srow_z[2]);
            }
            image::swap_xz(out);
        }
        if(std::fabs(nif_header.srow_y[1]) < std::fabs(nif_header.srow_y[2]))
        {
            if(change_header)
            {
                std::swap(nif_header.srow_x[1],nif_header.srow_x[2]);
                std::swap(nif_header.srow_y[1],nif_header.srow_y[2]);
                std::swap(nif_header.srow_z[1],nif_header.srow_z[2]);
            }
            image::swap_yz(out);
        }

        // from +x = Right  +y = Anterior +z = Superior
        // to +x = Left  +y = Posterior +z = Superior
        if(nif_header.srow_x[0] > 0)
        {
            image::flip_x(out);
            if(change_header)
            {
                nif_header.srow_x[3] += nif_header.srow_x[0]*(out.width()-1);
                nif_header.srow_x[0] = -nif_header.srow_x[0];
                nif_header.srow_y[0] = -nif_header.srow_y[0];
                nif_header.srow_z[0] = -nif_header.srow_z[0];

            }
        }

        if(nif_header.srow_y[1] > 0)
        {
            image::flip_y(out);
            if(change_header)
            {
                nif_header.srow_y[3] += nif_header.srow_y[1]*(out.height()-1);
                nif_header.srow_x[1] = -nif_header.srow_x[1];
                nif_header.srow_y[1] = -nif_header.srow_y[1];
                nif_header.srow_z[1] = -nif_header.srow_z[1];
            }
        }

        if(nif_header.srow_z[2] < 0)
        {
            image::flip_z(out);
            if(change_header)
            {
                nif_header.srow_z[3] += nif_header.srow_z[2]*(out.depth()-1);
                nif_header.srow_x[2] = -nif_header.srow_x[2];
                nif_header.srow_y[2] = -nif_header.srow_y[2];
                nif_header.srow_z[2] = -nif_header.srow_z[2];
            }
        }
        return true;
    }
};

typedef nifti_base<> nifti;

}
}

#endif
