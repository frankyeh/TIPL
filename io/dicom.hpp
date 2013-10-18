//---------------------------------------------------------------------------
#ifndef dicom_headerH
#define dicom_headerH
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
#include <iomanip>
#include <map>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <algorithm>
#include <memory>
#include "image/numerical/basic_op.hpp"
//---------------------------------------------------------------------------
namespace image
{
namespace io
{

//---------------------------------------------------------------------------
const char dicom_long_flag[] = "OBUNOWSQ";
const char dicom_short_flag[] = "AEASATCSDADSDTFLFDISLOLTPNSHSLSSSTTMUIULUS";
//---------------------------------------------------------------------------
class dicom_group_element
{
public:
    union
    {
        char gel[8];
        struct
        {
            unsigned short group;
            unsigned short element;
            union
            {
                unsigned int length;
                struct
                {
                    union
                    {
                        unsigned short vr;
                        struct
                        {
                            char lt0;
                            char lt1;
                        };
                    };
                    union
                    {
                        unsigned short new_length;
                        struct
                        {
                            char lt2;
                            char lt3;
                        };
                    };
                };
            };
        };
    };
    std::vector<unsigned char> data;
private:
    void assign(const dicom_group_element& rhs)
    {
        std::copy(rhs.gel,rhs.gel+8,gel);
        data = rhs.data;
    }
    bool flag_contains(const char* flag,unsigned int flag_size)
    {
        for (unsigned int index = 0; index < flag_size; ++index)
        {
            char lb = flag[index << 1];
            char hb = flag[(index << 1)+1];
            if (lt0 == lb && lt1 == hb)
                return true;
        }
        return false;
    }

public:
    dicom_group_element(void) {}
    dicom_group_element(const dicom_group_element& rhs)
    {
        assign(rhs);
    }
    const dicom_group_element& operator=(const dicom_group_element& rhs)
    {
        assign(rhs);
        return *this;
    }

    bool read(std::ifstream& in)
    {
        if (!in.read(gel,8))
            return false;
        unsigned int read_length = length;
        if (flag_contains(dicom_long_flag,4))
        {
            if (!in.read((char*)&read_length,4))
                return false;
        }
        else if (flag_contains(dicom_short_flag,21))
            read_length = new_length;

        if (read_length == 0xFFFFFFFF)
            read_length = 0;
        if (read_length)
        {
            if(group == 0x7FE0 && element == 0x0010)
            {
                length = read_length;
                return false;
            }
            data.resize(read_length);
            in.read((char*)&*(data.begin()),read_length);
        }
        return !(!in);
    }

    unsigned int get_order(void) const
    {
        unsigned int order = group;
        order <<= 16;
        order |= element;
        return order;
    }
    const std::vector<unsigned char>& get(void) const
    {
        return data;
    }
    unsigned short get_vr(void) const
    {
        return vr;
    }

    bool is_string(void) const
    {
        return (lt0 == 'D' ||  // DA DS DT
                lt0 == 'P' ||  // PN
                lt0 == 'T' ||  // TM
                lt0 == 'L' ||  // LO LT
                lt1 == 'I' ||  // UI
                lt1 == 'H' ||  // SH
                (lt0 != 'A' && lt1 == 'T') || // ST UT LT
                (lt0 == 'A' && lt1 == 'E') || // AE
                (lt0 == 'A' || lt0 == 'C' || lt0 == 'I') && lt1 == 'S');//AS CS IS
    }
    bool is_int16(void) const
    {
        return (lt0 == 'A' && lt1 == 'T') ||
                (lt0 == 'O' && lt1 == 'W') ||
                (lt0 == 'S' && lt1 == 'S') ||
                (lt0 == 'U' && lt1 == 'S');
    }
    bool is_int32(void) const
    {
        return (lt0 == 'S' && lt1 == 'L') ||
                (lt0 == 'U' && lt1 == 'L');
    }
    bool is_float(void) const
    {
        //FL
        return (lt0 == 'F' && lt1 == 'L') || (lt0 == 'O' && lt1 == 'F');
    }
    bool is_double(void) const
    {
        //FD
        return (lt0 == 'F' && lt1 == 'D');
    }

    template<typename value_type>
    void get_value(value_type& value) const
    {
        if(data.empty())
            return;
        if (is_string())
        {
            std::istringstream in((const char*)&*data.begin());
            in >> value;
            return;
        }
        if (is_float() && data.size() >= 4) // float
        {
            value = *(const float*)&*data.begin();
            return;
        }
        if (is_double() && data.size() >= 8) // double
        {
            value = *(const double*)&*data.begin();
            return;
        }
        if (is_int16() && data.size() >= 2) // uint16type
        {
            value = *(const short*)&*data.begin();
            return;
        }
        if (is_int32() && data.size() >= 4)
        {
            value = *(const int*)&*data.begin();
            return;
        }

        if (data.size() == 2) // uint16type
        {
            value = *(const short*)&*data.begin();
            return;
        }
        if (data.size() == 4)
        {
            value = *(const int*)&*data.begin();
            return;
        }
        if (data.size() == 8)
        {
            value = *(const double*)&*data.begin();
            return;
        }
    }

    template<typename stream_type>
    void operator>> (stream_type& out) const
    {
        if (data.empty())
        {
            out << "(null)";
            return;
        }
        if (is_float() && data.size() >= 4) // float
        {
            const float* iter = (const float*)&*data.begin();
            for (unsigned int index = 3;index < data.size();index += 4,++iter)
                out << *iter << " ";
            return;
        }
        if (is_double() && data.size() >= 8) // double
        {
            const double* iter = (const double*)&*data.begin();
            for (unsigned int index = 7;index < data.size();index += 8,++iter)
                out << *iter << " ";
            return;
        }

        bool is_ascii = true;
        if (is_string()) // String
            {/* ascii*/}
        else
            for (unsigned int index = 0;index < data.size() && (data[index] || index <= 2);++index)
                if (!::isprint(data[index]))
                {
                    is_ascii = false;
                    break;
                }
        if (is_ascii)
        {
            for (unsigned int index = 0;index < data.size();++index)
            {
                char ch = data[index];
                if (!ch)
                    break;
                out << ch;
            }
            return;
        }
        if (is_int16() && data.size() >= 2)
        {
            for (unsigned int index = 1;index < data.size();index+=2)
                out << *(const short*)&*(data.begin()+index-1) << " ";
            return;
        }
        if (is_int32() && data.size() == 4)
        {
            for (unsigned int index = 3;index < data.size();index+=4)
                out << *(const int*)&*(data.begin()+index-3) << " ";
            return;
        }
        out << "..." << data.size() << " bytes";
        return;
    }

};

struct dicom_csa_header
{
    char name[64];
    int vm;
    char vr[4];
    int syngodt;
    int nitems;
    int xx;
};

class dicom_csa_data
{
private:
    dicom_csa_header header;
    std::vector<std::string> vals;
    void assign(const dicom_csa_data& rhs)
    {
        std::copy(rhs.header.name,rhs.header.name+64,header.name);
        std::copy(rhs.header.vr,rhs.header.vr+4,header.vr);
        header.vm = rhs.header.vm;
        header.syngodt = rhs.header.syngodt;
        header.nitems = rhs.header.nitems;
        header.xx = rhs.header.xx;
        vals = rhs.vals;
    }
public:
    dicom_csa_data(void) {}
    dicom_csa_data(const dicom_csa_data& rhs)
    {
        assign(rhs);
    }
    const dicom_csa_data& operator=(const dicom_csa_data& rhs)
    {
        assign(rhs);
        return *this;
    }
    bool read(const std::vector<unsigned char>& data,unsigned int& from)
    {
        if (from + sizeof(dicom_csa_header) >= data.size())
            return false;
        std::copy(data.begin() + from,data.begin() + from + sizeof(dicom_csa_header),(char*)&header);
        from += sizeof(dicom_csa_header);
        int xx[4];
        for (int index = 0; index < header.nitems; ++index)
        {
            if (from + sizeof(xx) >= data.size())
                return false;
            std::copy(data.begin() + from,data.begin() + from + sizeof(xx),(char*)xx);
            from += sizeof(xx);
            if (from + xx[1] >= data.size())
                return false;
            if (xx[1])
                vals.push_back(std::string(data.begin() + from,data.begin() + from + xx[1]-1));
            from += xx[1] + (4-(xx[1]%4))%4;
        }
        return true;
    }
    void write_report(std::string& lines) const
    {
        std::ostringstream out;
        out << "CSA::" << header.name << "::VM-" << header.vm << "::VR-" << header.vr << "::Syn-" << header.syngodt << "::ni-" << header.nitems << " = ";
        for (unsigned int index = 0; index < vals.size(); ++index)
            out << vals[index] << " ";
        lines += out.str();
        lines += "\n";
    }
    const char* get_value(unsigned int index) const
    {
        if (index < vals.size())
            return &*vals[index].begin();
        return 0;
    }
    const char* get_name(void) const
    {
        return header.name;
    }
};

class dicom
{
private:
    std::auto_ptr<std::ifstream> input_io;
    unsigned int image_size;
    bool is_mosaic;
private:
    std::map<unsigned int,unsigned int> ge_map;
    std::vector<dicom_group_element*> data;
private:
    std::map<std::string,unsigned int> csa_map;
    std::vector<dicom_csa_data*> csa_data;
private:
    void assign(const dicom& rhs)
    {
        ge_map = rhs.ge_map;
        csa_map = rhs.csa_map;
        for (unsigned int index = 0;index < rhs.data.size();index++)
            data.push_back(new dicom_group_element(*rhs.data[index]));
        for (unsigned int index = 0;index < rhs.csa_data.size();index++)
            csa_data.push_back(new dicom_csa_data(*rhs.csa_data[index]));
    }
    template<typename iterator_type>
    void handle_mosaic(iterator_type image_buffer) const
    {
        typedef typename std::iterator_traits<iterator_type>::value_type pixel_type;
        // number of image in mosaic
        image::geometry<3> geo;
        get_image_dimension(geo);
        unsigned int mosaic_width = geo[0];
        unsigned int mosaic_height = geo[1];
        unsigned int w = width();
        unsigned int h = height();
        if (!mosaic_width || !mosaic_height ||
                w%mosaic_width || h%mosaic_height ||
                (mosaic_width == w && mosaic_height == h))
            return; //not mosaic

        unsigned int mosaic_size = mosaic_width*mosaic_height;
        std::vector<pixel_type> data(w*h);
        std::copy(image_buffer,image_buffer+data.size(),data.begin());
        // rearrange mosaic

        unsigned int mosaic_col_count = w/mosaic_width;
        unsigned int mosaic_line_size = mosaic_size*mosaic_col_count;


        const pixel_type* slice_end = &*data.begin() + data.size();
        for (const pixel_type* slice_band_pos = &*data.begin(); slice_band_pos < slice_end; slice_band_pos += mosaic_line_size)
        {
            const pixel_type* slice_pos_end = slice_band_pos + w;
            for (const pixel_type* slice_pos = slice_band_pos; slice_pos < slice_pos_end; slice_pos += mosaic_width)
            {
                const pixel_type* slice_line_end = slice_pos + mosaic_line_size;
                for (const pixel_type* slice_line = slice_pos; slice_line < slice_line_end; slice_line += w,image_buffer += mosaic_width)
                    std::copy(slice_line,slice_line+mosaic_width,image_buffer);
            }
        }

    }
public:
    dicom(void) {}
    dicom(const dicom& rhs)
    {
        assign(rhs);
    }
    const dicom& operator=(const dicom& rhs)
    {
        assign(rhs);
        return *this;
    }
    ~dicom(void)
    {
        for (unsigned int index = 0; index < data.size(); ++index)
            delete data[index];
        for (unsigned int index = 0; index < csa_data.size(); ++index)
            delete csa_data[index];
    }
public:
    bool load_from_file(const std::string& file_name)
    {
        return load_from_file(file_name.c_str());
    }
    template<typename char_type>
    bool load_from_file(const char_type* file_name)
    {
        ge_map.clear();
        data.clear();
        input_io.reset(new std::ifstream(file_name,std::ios::binary));
        if (!(*input_io))
            return false;
        input_io->seekg(128);
        unsigned int dicom_mark = 0;
        input_io->read((char*)&dicom_mark,4);
        if (dicom_mark != 0x4d434944) //DICM
        {
            // switch to another DICOM format
            input_io->seekg(0,std::ios::beg);
            input_io->read((char*)&dicom_mark,4);
            if(dicom_mark != 8)
                return false;
            input_io->seekg(0,std::ios::beg);
        }
        while (*input_io)
        {
            std::auto_ptr<dicom_group_element> ge(new dicom_group_element);
            if (!ge->read(*input_io))
            {
                if (!(*input_io))
                    return false;
                image_size = ge->length;
                std::string image_type;
                is_mosaic = get_int(0x0019,0x100A) > 1 ||   // multiple frame (new version)
                            (get_text(0x0008,0x0008,image_type) && image_type.find("MOSAIC") != std::string::npos);
                return true;
            }
            // Deal with CSA
            if (ge->group == 0x0029 && (ge->element == 0x1010 || ge->element == 0x1020))
            {
                std::string SV10(ge->get().begin(),ge->get().begin()+4);
                if (SV10 == "SV10")
                {
                    int count = *(int*)&ge->get()[8];
                    if (count <= 128 && count >= 0)
                    {
                        unsigned int pos = 16;
                        for (unsigned int index = 0; index < (unsigned int)count && pos < ge->get().size(); ++index)
                        {
                            std::auto_ptr<dicom_csa_data> csa(new dicom_csa_data);
                            if (!csa->read(ge->get(),pos))
                                break;
                            csa_data.push_back(csa.release());
                            csa_map[csa_data.back()->get_name()] = csa_data.size()-1;
                        }
                    }
                }

            }

            ge_map[ge->get_order()] = data.size();
            data.push_back(ge.release());
        }
        return false;
    }

    const char* get_csa_data(const std::string& name,unsigned int index) const
    {
        std::map<std::string,unsigned int>::const_iterator iter = csa_map.find(name);
        if (iter == csa_map.end())
            return 0;
        return csa_data[iter->second]->get_value(index);
    }

    const unsigned char* get_data(unsigned short group,unsigned short element,unsigned int& length) const
    {
        std::map<unsigned int,unsigned int>::const_iterator iter =
                ge_map.find(((unsigned int)group << 16) | (unsigned int)element);
        if (iter == ge_map.end())
        {
            length = 0;
            return 0;
        }
        length = data[iter->second]->get().size();
        if (!length)
            return 0;
        return (const unsigned char*)&*data[iter->second]->get().begin();
    }

    bool get_text(unsigned short group,unsigned short element,std::string& result) const
    {
        unsigned int length = 0;
        const char* text = (const char*)get_data(group,element,length);
        if (!text)
            return false;
        result = std::string(text,text+length);
        return true;
    }

    template<typename value_type>
    bool get_value(unsigned short group,unsigned short element,value_type& value) const
    {
        std::map<unsigned int,unsigned int>::const_iterator iter =
                ge_map.find(((unsigned int)group << 16) | (unsigned int)element);
        if (iter == ge_map.end())
            return false;
        data[iter->second]->get_value(value);
        return true;
    }
    unsigned int get_int(unsigned short group,unsigned short element) const
    {
        unsigned int value = 0;
        get_value(group,element,value);
        return value;
    }
    float get_float(unsigned short group,unsigned short element) const
    {
        float value = 0.0;
        get_value(group,element,value);
        return value;
    }
    double get_double(unsigned short group,unsigned short element) const
    {
        double value = 0.0;
        get_value(group,element,value);
        return value;
    }
    template<typename voxel_size_type>
    void get_voxel_size(voxel_size_type voxel_size) const
    {
        std::string slice_dis,pixel_spacing;
        if (!get_text(0x0018,0x0088,slice_dis) && !get_text(0x0018,0x0050,slice_dis))
            return;
        if (!get_text(0x0028,0x0030,pixel_spacing))
            return;
        std::replace(pixel_spacing.begin(),pixel_spacing.end(),'\\',' ');
        std::istringstream(pixel_spacing) >> voxel_size[0] >> voxel_size[1];
        std::istringstream(slice_dis) >> voxel_size[2];
    }

    /**
    The DICOM attribute (0020,0037) "Image Orientation (Patient)" gives the
    orientation of the x- and y-axes of the image data in terms of 2 3-vectors.
    The first vector is a unit vector along the x-axis, and the second is
    along the y-axis.
    */
    template<typename vector_type>
    void get_image_row_orientation(vector_type image_row_orientation) const
    {
        //float image_row_orientation[3];
        std::string image_orientation;
        if (!get_text(0x0020,0x0037,image_orientation) &&
                !get_text(0x0020,0x0035,image_orientation))
            return;
        std::replace(image_orientation.begin(),image_orientation.end(),'\\',' ');
        std::istringstream(image_orientation)
        >> image_row_orientation[0]
        >> image_row_orientation[1]
        >> image_row_orientation[2];
    }
    template<typename vector_type>
    void get_image_col_orientation(vector_type image_col_orientation) const
    {
        //float image_col_orientation[3];
        float temp;
        std::string image_orientation;
        if (!get_text(0x0020,0x0037,image_orientation) &&
                !get_text(0x0020,0x0035,image_orientation))
            return;
        std::replace(image_orientation.begin(),image_orientation.end(),'\\',' ');
        std::istringstream(image_orientation)
        >> temp >> temp >> temp
        >> image_col_orientation[0]
        >> image_col_orientation[1]
        >> image_col_orientation[2];
    }

    template<typename vector_type>
    void get_image_orientation(vector_type orientation_matrix) const
    {
        get_image_row_orientation(orientation_matrix);
        get_image_col_orientation(orientation_matrix+3);
        // get the slice direction
        orientation_matrix[6] =
            (orientation_matrix[1] * orientation_matrix[5])-
            (orientation_matrix[2] * orientation_matrix[4]);
        orientation_matrix[7] =
            (orientation_matrix[2] * orientation_matrix[3])-
            (orientation_matrix[0] * orientation_matrix[5]);
        orientation_matrix[8] =
            (orientation_matrix[0] * orientation_matrix[4])-
            (orientation_matrix[1] * orientation_matrix[3]);

        // the slice ordering is always increamental
        if (orientation_matrix[6] + orientation_matrix[7] + orientation_matrix[8] < 0) // no flip needed
        {
            orientation_matrix[6] = -orientation_matrix[6];
            orientation_matrix[7] = -orientation_matrix[7];
            orientation_matrix[8] = -orientation_matrix[8];
        }
    }
    float get_slice_location(void) const
    {
        std::string slice_location;
        if (!get_text(0x0020,0x1041,slice_location))
            return 0.0;
        float data;
        std::istringstream(slice_location) >> data;
        return data;
    }

    void get_patient(std::string& info)
    {
        std::string date,gender,age,id;
        date = gender = age = id = "_";
        get_text(0x0008,0x0022,date);
        get_text(0x0010,0x0040,gender);
        get_text(0x0010,0x1010,age);
        get_text(0x0010,0x0010,id);
        using namespace std;
        gender.erase(remove(gender.begin(),gender.end(),' '),gender.end());
        id.erase(remove(id.begin(),id.end(),' '),id.end());
        std::replace(id.begin(),id.end(),'-','_');
        std::replace(id.begin(),id.end(),'/','_');
        info = date;
        info += "_";
        info += gender;
        info += age;
        info += "_";
        info += id;
    }
    void get_sequence(std::string& info)
    {
        std::string series_num,series_des;
        series_num = series_des = "_";
        get_text(0x0020,0x0011,series_num);
        get_text(0x0008,0x103E,series_des);
        using namespace std;
        series_num.erase(remove(series_num.begin(),series_num.end(),' '),series_num.end());
        series_des.erase(remove(series_des.begin(),series_des.end(),' '),series_des.end());
        std::replace(series_des.begin(),series_des.end(),'-','_');
        if (series_num.size() == 1)
        {
            info = std::string("0");
            info += series_num;
        }
        else
            info = series_num;

        info += "_";
        info += series_des;
    }
    void get_image_name(std::string& info)
    {
        std::string series_des,image_num;
        series_des = image_num = "_";
        get_text(0x0008,0x103E,series_des);
        get_text(0x0020,0x0013,image_num);
        using namespace std;
        series_des.erase(remove(series_des.begin(),series_des.end(),' '),series_des.end());
        std::replace(series_des.begin(),series_des.end(),'-','_');
        image_num.erase(remove(image_num.begin(),image_num.end(),' '),image_num.end());
        info = series_des;
        info += "_i";
        info += image_num;
        info += ".dcm";
    }

    unsigned int width(void) const
    {
        return get_int(0x0028,0x0011);
    }

    unsigned int height(void) const
    {
        return get_int(0x0028,0x0010);
    }

    unsigned int frame_num(void) const
    {
        return get_int(0x0028,0x0008);
    }

    unsigned int get_bit_count(void) const
    {
        return get_int(0x0028,0x0100);
    }

    void get_image_dimension(image::geometry<2>& geo) const
    {
        geo[0] = width();
        geo[1] = height();
    }

    void get_image_dimension(image::geometry<3>& geo) const
    {
        unsigned int length = 0;
        const short* acq_matrix = (const short*)get_data(0x0018,0x1310,length);
        geo[2] = 0;
        // could be mosaic
        if (is_mosaic)
        {
            // the acq matrix may be at [0][1] or [2][3]
            if (acq_matrix[0])
                geo[1] = acq_matrix[3];
            else
                geo[1] = acq_matrix[1];
            if (acq_matrix[3])
                geo[0] = acq_matrix[0];
            else
                geo[0] = acq_matrix[2];
            // get the number of slices
            geo[2] = get_int(0x0019,0x100A);
        }
        if (!geo[0] || !geo[1])
        {
            geo[0] = width();
            geo[1] = height();
        }

        if (geo[2] <= 1 && geo[0] && geo[1])
        {
            geo[2] = image_size/geo[0]/geo[1]/(get_bit_count()/8);
        }
    }


    template<typename pointer_type>
    void save_to_buffer(pointer_type ptr,unsigned int pixel_count) const
    {
        typedef typename std::iterator_traits<pointer_type>::value_type value_type;
        if(sizeof(value_type) == get_bit_count()/8)
            input_io->read((char*)&*ptr,pixel_count*sizeof(value_type));
        else
        {
            std::vector<char> data(pixel_count*get_bit_count()/8);
            input_io->read((char*)&(data[0]),data.size());
            switch (get_bit_count()) //bit count
            {
            case 8://DT_UNSIGNED_CHAR 2
                std::copy((const unsigned char*)&(data[0]),(const unsigned char*)&(data[0])+pixel_count,ptr);
                return;
            case 16://DT_SIGNED_SHORT 4
                std::copy((const short*)&(data[0]),(const short*)&(data[0])+pixel_count,ptr);
                return;
            case 32://DT_SIGNED_INT 8
                std::copy((const int*)&(data[0]),(const int*)&(data[0])+pixel_count,ptr);
                return;
            case 64://DT_DOUBLE 64
                std::copy((const double*)&(data[0]),(const double*)&(data[0])+pixel_count,ptr);
                return;
            }
        }
    }

    template<typename image_type>
    void save_to_image(image_type& out) const
    {
        if(!input_io.get() || !(*input_io))
            return;
        image::geometry<image_type::dimension> geo;
        get_image_dimension(geo);
        if(is_mosaic)
        {
            unsigned short slice_num = geo[2];
            geo[2] = width()*height()/geo[0]/geo[1];
            out.resize(geo);
            save_to_buffer(out.begin(),out.size());
            handle_mosaic(out.begin());
            geo[2] = slice_num;
            out.resize(geo);
        }
        else
        {
            out.resize(geo);
            save_to_buffer(out.begin(),out.size());
        }
    }

    template<typename image_type>
    const dicom& operator>>(image_type& source) const
    {
        save_to_image(source);
        return *this;
    }
    template<typename image_type>
    dicom& operator<<(const image_type& source)
    {
        load_from_image(source);
        return *this;
    }

    const dicom& operator>>(std::string& report) const
    {
        std::ostringstream out;
        std::map<unsigned int,unsigned int>::const_iterator iter = ge_map.begin();
        std::map<unsigned int,unsigned int>::const_iterator end = ge_map.end();
        for (;iter != end;++iter)
        {
            out << std::setw( 8 ) << std::setfill( '0' ) << std::hex << std::uppercase <<
            iter->first << " ";
            out << std::dec;
            unsigned short vr = data[iter->second]->vr;
            out << (char)(vr & 0xFF) << (char)(vr >> 8) << " ";
            *(data[iter->second]) >> out;
            out << std::endl;
        }
        report = out.str();
        for(unsigned int index = 0;index < csa_data.size();++index)
            csa_data[index]->write_report(report);
        return *this;
    }
};

}

}
#endif
