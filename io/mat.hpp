#ifndef MAT_HPP
#define MAT_HPP
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
#include <vector>
#include <fstream>
#include <iterator>
#include <map>
#include "interface.hpp"
namespace image
{

namespace io
{

template<class fun_type>
struct mat_type_info;

template<>
struct mat_type_info<double>
{
    static const unsigned int type = 0;
};
template<>
struct mat_type_info<float>
{
    static const unsigned int type = 10;
};
template<>
struct mat_type_info<unsigned int>
{
    static const unsigned int type = 20;
};
template<>
struct mat_type_info<int>
{
    static const unsigned int type = 20;
};
template<>
struct mat_type_info<short>
{
    static const unsigned int type = 30;
};
template<>
struct mat_type_info<unsigned short>
{
    static const unsigned int type = 40;
};
template<>
struct mat_type_info<unsigned char>
{
    static const unsigned int type = 50;
};
template<>
struct mat_type_info<char>
{
    static const unsigned int type = 50;
};



class mat_matrix
{
private:
    unsigned int type;
    unsigned int rows;
    unsigned int cols;
    std::string name;
    unsigned int namelen;
private:
    std::vector<unsigned char> data_buf;
    void* data_ptr; // for read
private:
    void copy(const mat_matrix& rhs)
    {
        type = rhs.type;
        rows = rhs.rows;
        cols = rhs.cols;
        name = rhs.name;
        namelen = rhs.namelen;
        data_buf = rhs.data_buf;
        data_ptr = data_buf.empty() ? 0 : &*data_buf.begin();
    }

    unsigned int get_total_size(unsigned int ty) const
    {
        unsigned int element_size_array[10] = {8,4,4,2,2,1,0,0,0,0};
        return rows*cols*element_size_array[(ty%100)/10];
    }
public:
    mat_matrix(void):type(0),rows(0),cols(0),namelen(0),data_ptr(0){}
    mat_matrix(const std::string& name_):type(0),rows(0),cols(0),namelen((unsigned int)name_.size()+1),name(name_),data_ptr(0) {}
    mat_matrix(const mat_matrix& rhs){copy(rhs);}
    const mat_matrix& operator=(const mat_matrix& rhs)
    {
        copy(rhs);
        return *this;
    }
    template<class Type>
    void assign(const Type* data_ptr_,unsigned int rows_,unsigned int cols_)
    {
        data_ptr = (void*)data_ptr_;
        rows = rows_;
        cols = cols_;
        type = mat_type_info<Type>::type;
    }
    template<class Type>
    void assign(const char* name_,const Type* data_ptr_,unsigned int rows_,unsigned int cols_)
    {
        name = name_;
        type = mat_type_info<Type>::type;
        namelen = name.length();
        rows = rows_;
        cols = cols_;
        data_buf.resize(rows*cols*sizeof(Type));
        std::copy((const char*)data_ptr_,(const char*)data_ptr_+data_buf.size(),data_buf.begin());
        data_ptr = &*data_buf.begin();
    }

    template<class OutType>
    void copy_data(OutType out)
    {
        switch (type)
        {
        case 0://double
            std::copy((const double*)data_ptr,((const double*)data_ptr) + rows*cols,out);
            break;
        case 10://float
            std::copy((const float*)data_ptr,((const float*)data_ptr) + rows*cols,out);
            break;
        case 20://unsigned int
            std::copy((const unsigned int*)data_ptr,((const unsigned int*)data_ptr) + rows*cols,out);
            break;
        case 30://short
            std::copy((const short*)data_ptr,((const short*)data_ptr) + rows*cols,out);
            break;
        case 40://unsigned short
            std::copy((const unsigned short*)data_ptr,((const unsigned short*)data_ptr) + rows*cols,out);
            break;
        case 50://unsigned char
            std::copy((const unsigned char*)data_ptr,((const unsigned char*)data_ptr) + rows*cols,out);
            break;
        }
    }
    const void* get_data(unsigned int get_type)
    {
        if(get_type != 0 && get_type != 10 && get_type != 20 && get_type != 30 && get_type != 40 && get_type != 50)
            return 0;
        // same type or unsigned short v.s. short
        if (get_type == type || (type == 40 && get_type == 30) || (type == 30 && get_type == 40))
            return data_ptr;
        std::vector<unsigned char> allocator(get_total_size(get_type));
        void* new_data = (void*)&*allocator.begin();
        switch (get_type)
        {
        case 0://double
            copy_data((double*)new_data);
            break;
        case 10://float
            copy_data((float*)new_data);
            break;
        case 20://unsigned int
            copy_data((unsigned int*)new_data);
            break;
        case 30://short
            copy_data((short*)new_data);
            break;
        case 40://unsigned short
            copy_data((unsigned short*)new_data);
            break;
        case 50://unsigned char
            copy_data((unsigned char*)new_data);
            break;
        }
        std::swap(data_ptr,new_data);
        allocator.swap(data_buf);
        type = get_type;
        return data_ptr;
    }
    unsigned int get_rows(void) const
    {
        return rows;
    }
    unsigned int get_cols(void) const
    {
        return cols;
    }
    const std::string& get_name(void) const
    {
        return name;
    }
    template<class stream_type>
    bool read(stream_type& in)
    {
        unsigned int imagf = 0;
        in.read((char*)&type,4);
        if (!in || type > 100 || type % 10 > 1)
            return false;
        if (type % 10) // text
	    type = 0;
        in.read((char*)&rows,4);
        in.read((char*)&cols,4);
        in.read((char*)&imagf,4);
        in.read((char*)&namelen,4);
        if(!in || namelen == 0 || namelen > 255)
            return false;
        std::vector<char> buffer(namelen+1);
        in.read((char*)&*buffer.begin(),namelen);
        if(rows*cols == 0)
            return false;
        name = &*buffer.begin();
        if (!in)
            return false;

		try{
			std::vector<unsigned char> allocator(get_total_size(type));
			allocator.swap(data_buf);
		}
		catch(...)
		{
			return false;
        }
        data_ptr = &*data_buf.begin();
        in.read((char*)data_ptr,get_total_size(type));
        return true;
    }
    template<class stream_type>
    bool write(stream_type& out) const
    {
        unsigned int imagf = 0;
        out.write((const char*)&type,4);
        out.write((const char*)&rows,4);
        out.write((const char*)&cols,4);
        out.write((const char*)&imagf,4);
        out.write((const char*)&namelen,4);
        out.write((const char*)&*name.begin(),namelen);
        out.write((const char*)data_ptr,get_total_size(type));
        return out;
    }
    void get_info(std::string& info) const
    {
        std::ostringstream out;
        unsigned int out_count = std::min<int>(20,rows*cols);
        switch (type)
        {
        case 0://double
            std::copy((const double*)data_ptr,((const double*)data_ptr) + out_count,std::ostream_iterator<double>(out," "));
            break;
        case 10://float
            std::copy((const float*)data_ptr,((const float*)data_ptr) + out_count,std::ostream_iterator<float>(out," "));
            break;
        case 20://unsigned int
            std::copy((const unsigned int*)data_ptr,((const unsigned int*)data_ptr) + out_count,std::ostream_iterator<unsigned int>(out," "));
            break;
        case 30://short
            std::copy((const short*)data_ptr,((const short*)data_ptr) + out_count,std::ostream_iterator<short>(out," "));
            break;
        case 40://unsigned short
            std::copy((const unsigned short*)data_ptr,((const unsigned short*)data_ptr) + out_count,std::ostream_iterator<unsigned short>(out," "));
            break;
        case 50://unsigned char
            std::copy((const unsigned char*)data_ptr,((const unsigned char*)data_ptr) + out_count,std::ostream_iterator<unsigned char>(out," "));
            break;
        }
        info = out.str();
        if(rows*cols > 20)
            info += "...";
    }
};

template<class input_interface = std_istream>
class mat_read_base
{
private:
    std::vector<mat_matrix*> dataset;
    std::map<std::string,int> name_table;
private:
    void copy(const mat_read_base& rhs)
    {
        for(unsigned int index = 0;index < dataset.size();++index)
        {
            std::auto_ptr<mat_matrix> matrix(new mat_matrix);
            *(matrix.get()) = *(rhs.dataset[index]);
            dataset.push_back(matrix.release());
        }
        name_table = rhs.name_table;
    }

public:
    mat_read_base(void){}
    mat_read_base(const mat_read_base& rhs){copy(rhs);}
    const mat_read_base& operator=(const mat_read_base& rhs)
    {
        copy(rhs);
        return *this;
    }
    bool has(const char* name) const
    {
        return name_table.find(name) != name_table.end();
    }
    const void* read_as_type(unsigned int index,unsigned int& rows,unsigned int& cols,unsigned int type) const
    {
        if (index >= dataset.size())
            return 0;
        rows = dataset[index]->get_rows();
        cols = dataset[index]->get_cols();
        return dataset[index]->get_data(type);
    }
    const void* read_as_type(const char* name,unsigned int& rows,unsigned int& cols,unsigned int type) const
    {
        std::map<std::string,int>::const_iterator iter = name_table.find(name);
        if (iter == name_table.end())
            return 0;
        return read_as_type(iter->second,rows,cols,type);
    }
    template<class out_type>
    const out_type*& read(int index,unsigned int& rows,unsigned int& cols,const out_type*& out) const
    {
        return out = (const out_type*)read_as_type(index,rows,cols,mat_type_info<out_type>::type);
    }
    template<class out_type>
    const out_type*& read(const char* name,unsigned int& rows,unsigned int& cols,const out_type*& out) const
    {
        return out = (const out_type*)read_as_type(name,rows,cols,mat_type_info<out_type>::type);
    }
	template<class data_type>
    void read_as_image(int index,image::basic_image<data_type,2>& image_buf) const
	{
		if (index >= dataset.size())
            return;
		image_buf.resize(image::geometry<2>(dataset[index]->get_cols(),dataset[index]->get_rows()));
        dataset[index]->copy_data(image_buf.begin());
	}

	template<class data_type>
    void read_as_image(const char* name,image::basic_image<data_type,2>& image_buf) const
    {
		std::map<std::string,int>::const_iterator iter = name_table.find(name);
        if (iter == name_table.end())
            return;
        read_as_image(iter->second,image_buf);
    }

public:
    ~mat_read_base(void)
    {
        clear();
    }
    void clear(void)
    {
        for(unsigned int index = 0; index < dataset.size(); ++index)
            delete dataset[index];
        dataset.clear();
    }
    template<class char_type>
    bool load_from_file(const char_type* file_name,unsigned int max_count,std::string stop_name)
    {
        input_interface in;
        if(!in.open(file_name))
            return false;
        clear();
        for(unsigned int i = 0;i < max_count && in;++i)
        {
            std::auto_ptr<mat_matrix> matrix(new mat_matrix);
            if (!matrix->read(in))
                break;
            dataset.push_back(matrix.release());
            if(dataset.back()->get_name() == stop_name)
                break;
        }
        for (unsigned int index = 0; index < dataset.size(); ++index)
            name_table[dataset[index]->get_name()] = index;
        return true;
    }
    template<class char_type>
    bool load_from_file(const char_type* file_name)
    {
        input_interface in;
        if(!in.open(file_name))
            return false;
        clear();
        while(in)
        {
            std::auto_ptr<mat_matrix> matrix(new mat_matrix);
            if (!matrix->read(in))
                break;
            dataset.push_back(matrix.release());
        }
        for (unsigned int index = 0; index < dataset.size(); ++index)
            name_table[dataset[index]->get_name()] = index;
        return true;
    }
    template<class Type>
    void add(const char* name_,const Type* data_ptr,unsigned int rows,unsigned int cols)
    {
        std::auto_ptr<mat_matrix> matrix(new mat_matrix);
        matrix->assign(name_,data_ptr,rows,cols);
        dataset.push_back(matrix.release());
        name_table[name_] = dataset.size()-1;
    }

    template<class image_type>
    void save_to_image(image_type& image_data) const
    {
        unsigned int r,c;
        const unsigned short* m = 0;
        read("dimension",r,c,m);
        if(!m || r*c != image_type::dimension)
            return;
        image_data.resize(image::geometry<image_type::dimension>(m));
        const typename image_type::value_type* buf = 0;
        read("image",r,c,buf);
        if(!buf || r*c != image_data.size())
            return;
        std::copy(buf,buf+image_data.size(),image_data.begin());
    }
    template<class voxel_size_type>
    void get_voxel_size(voxel_size_type voxel_size) const
    {
        unsigned int r,c,r2,c2;
        const unsigned short* m = 0;
        read("dimension",r,c,m);
        if(!m)
            return;
        const float* vs = 0;
        read("dimension",r2,c2,vs);
        if(!vs || r*c != r2*c2)
            return;
        for(unsigned int i = 0;i < r*c;++i)
            voxel_size[i] = vs[i];
    }
    const char* name(unsigned int index) const
    {
        return dataset[index]->get_name().c_str();
    }

    unsigned int size(void) const
    {
        return dataset.size();
    }
    mat_matrix& operator[](unsigned int index){return *dataset[index];}
    const mat_matrix& operator[](unsigned int index) const {return *dataset[index];}

    template<class image_type>
    const mat_read_base& operator>>(image_type& source) const
    {
        save_to_image(source);
        return *this;
    }
};



template<class output_interface = std_ostream>
class mat_write_base
{
    output_interface out;
public:
    mat_write_base(const char* file_name)
    {
        out.open(file_name);
    }
public:
    template<class Type>
    bool write(const char* name_,const Type* data_ptr,unsigned int rows,unsigned int cols)
    {
        unsigned int imagf = 0;
        unsigned int type = mat_type_info<Type>::type;
        std::string name(name_);
        unsigned int namelen = name.length()+1;
        out.write((const char*)&type,4);
        out.write((const char*)&rows,4);
        out.write((const char*)&cols,4);
        out.write((const char*)&imagf,4);
        out.write((const char*)&namelen,4);
        out.write((const char*)&*name.begin(),namelen);
        out.write((const char*)data_ptr,rows*cols*sizeof(Type));
        return out;
    }
    bool write(const mat_matrix& data)
    {
        return data.write(out);
    }

public:

    template<class image_type>
    void load_from_image(const image_type& image_data)
    {
        write("dimension",&*image_data.geometry().begin(),1,image_type::dimension);
        write("image",&*image_data.begin(),1,image_data.size());
    }

    template<class image_type>
    mat_write_base& operator<<(image_type& source)
    {
        load_from_image(source);
        return *this;
    }
    operator bool() const	{return !(!out);}
    bool operator!() const	{return !out;}

};

typedef mat_write_base<> mat_write;
typedef mat_read_base<> mat_read;



}
}

#endif//MAT_FILE_HPP
