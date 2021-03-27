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
#include <sstream>
#include <map>
#include "interface.hpp"
namespace tipl
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
    union{
        struct{
            unsigned int type;
            unsigned int rows;
            unsigned int cols;
            unsigned int imagf;
            unsigned int namelen;
        };
        char buf[20];
    };

private:
    std::string name;
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
        data_ptr = data_buf.empty() ? nullptr : &*data_buf.begin();
    }

    size_t get_total_size(unsigned int ty) const
    {
        unsigned int element_size_array[10] = {8,4,4,2,2,1,0,0,0,0};
        return size_t(rows)*size_t(cols)*size_t(element_size_array[(ty%100)/10]);
    }
public:
    mat_matrix(void):type(0),rows(0),cols(0),namelen(0),data_ptr(nullptr){}
    mat_matrix(const std::string& name_):type(0),rows(0),cols(0),namelen(uint32_t(name_.size()+1)),name(name_),data_ptr(nullptr) {}
    mat_matrix(const mat_matrix& rhs){copy(rhs);}
    const mat_matrix& operator=(const mat_matrix& rhs)
    {
        copy(rhs);
        return *this;
    }
    template<class Type>
    void assign(const Type* data_ptr_,unsigned int rows_,unsigned int cols_)
    {
        data_ptr = data_ptr_;
        rows = rows_;
        cols = cols_;
        type = mat_type_info<Type>::type;
    }
    template<class Type>
    void assign(const char* name_,const Type* data_ptr_,unsigned int rows_,unsigned int cols_)
    {
        name = name_;
        type = mat_type_info<Type>::type;
        namelen = uint32_t(name.length());
        rows = rows_;
        cols = cols_;
        data_buf.resize(size_t(rows)*size_t(cols)*size_t(sizeof(Type)));
        std::copy(reinterpret_cast<const char*>(data_ptr_),reinterpret_cast<const char*>(data_ptr_)+data_buf.size(),data_buf.begin());
        data_ptr = &*data_buf.begin();
    }

    template<class OutType>
    void copy_data(OutType out)
    {
        size_t size = size_t(rows)*size_t(cols);
        switch (type)
        {
        case 0://double
            std::copy(reinterpret_cast<const double*>(data_ptr),
                      reinterpret_cast<const double*>(data_ptr)+size,out);
            break;
        case 10://float
            std::copy(reinterpret_cast<const float*>(data_ptr),
                      reinterpret_cast<const float*>(data_ptr)+size,out);
            break;
        case 20://unsigned int
            std::copy(reinterpret_cast<const unsigned int*>(data_ptr),
                      reinterpret_cast<const unsigned int*>(data_ptr)+size,out);
            break;
        case 30://short
            std::copy(reinterpret_cast<const short*>(data_ptr),
                      reinterpret_cast<const short*>(data_ptr)+size,out);
            break;
        case 40://unsigned short
            std::copy(reinterpret_cast<const unsigned short*>(data_ptr),
                      reinterpret_cast<const unsigned short*>(data_ptr)+size,out);
            break;
        case 50://unsigned char
            std::copy(reinterpret_cast<const unsigned char*>(data_ptr),
                      reinterpret_cast<const unsigned char*>(data_ptr)+size,out);
            break;
        }
    }
    const void* get_data(unsigned int get_type)
    {
        if(get_type != 0 && get_type != 10 && get_type != 20 && get_type != 30 && get_type != 40 && get_type != 50)
            return nullptr;
        // same type or unsigned short v.s. short
        if (get_type == type || (type == 40 && get_type == 30) || (type == 30 && get_type == 40))
            return data_ptr;
        std::vector<unsigned char> allocator(get_total_size(get_type));
        void* new_data = &*allocator.begin();
        switch (get_type)
        {
        case 0://double
            copy_data(reinterpret_cast<double*>(new_data));
            break;
        case 10://float
            copy_data(reinterpret_cast<float*>(new_data));
            break;
        case 20://unsigned int
            copy_data(reinterpret_cast<unsigned int*>(new_data));
            break;
        case 30://short
            copy_data(reinterpret_cast<short*>(new_data));
            break;
        case 40://unsigned short
            copy_data(reinterpret_cast<unsigned short*>(new_data));
            break;
        case 50://unsigned char
            copy_data(reinterpret_cast<unsigned char*>(new_data));
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
        in.read(buf,sizeof(buf));
        if (!in || type > 100 || type % 10 > 1)
            return false;
        if (type % 10) // text
	    type = 0;
        if(!in || namelen == 0 || namelen > 255)
            return false;
        std::vector<char> buffer(namelen+1);
        in.read(reinterpret_cast<char*>(&*buffer.begin()),namelen);
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
        in.read(reinterpret_cast<char*>(data_ptr),get_total_size(type));
        return true;
    }
    template<class stream_type>
    bool write(stream_type& out) const
    {
        unsigned int imagf = 0;
        out.write(reinterpret_cast<const char*>(&type),4);
        out.write(reinterpret_cast<const char*>(&rows),4);
        out.write(reinterpret_cast<const char*>(&cols),4);
        out.write(reinterpret_cast<const char*>(&imagf),4);
        out.write(reinterpret_cast<const char*>(&namelen),4);
        out.write(reinterpret_cast<const char*>(&*name.begin()),namelen);
        out.write(reinterpret_cast<const char*>(data_ptr),get_total_size(type));
        return out;
    }
    void get_info(std::string& info) const
    {
        std::ostringstream out;
        unsigned int out_count = rows*cols;
        if(out_count > 20)
            out_count = 20;
        switch (type)
        {
        case 0://double
            std::copy(reinterpret_cast<const double*>(data_ptr),
                      reinterpret_cast<const double*>(data_ptr)+out_count,
                      std::ostream_iterator<double>(out," "));
            break;
        case 10://float
            std::copy(reinterpret_cast<const float*>(data_ptr),
                      reinterpret_cast<const float*>(data_ptr)+out_count,
                      std::ostream_iterator<float>(out," "));
            break;
        case 20://unsigned int
            std::copy(reinterpret_cast<const unsigned int*>(data_ptr),
                      reinterpret_cast<const unsigned int*>(data_ptr)+out_count,
                      std::ostream_iterator<unsigned int>(out," "));
            break;
        case 30://short
            std::copy(reinterpret_cast<const short*>(data_ptr),
                      reinterpret_cast<const short*>(data_ptr)+out_count,
                      std::ostream_iterator<short>(out," "));
            break;
        case 40://unsigned short
            std::copy(reinterpret_cast<const unsigned short*>(data_ptr),
                      reinterpret_cast<const unsigned short*>(data_ptr)+out_count,
                      std::ostream_iterator<unsigned short>(out," "));
            break;
        case 50://unsigned char
            std::copy(reinterpret_cast<const unsigned char*>(data_ptr),
                      reinterpret_cast<const unsigned char*>(data_ptr)+out_count,
                      std::ostream_iterator<unsigned char>(out," "));
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
    std::vector<std::shared_ptr<mat_matrix> > dataset;
    std::map<std::string,int> name_table;
private:
    void copy(const mat_read_base& rhs)
    {
        for(unsigned int index = 0;index < dataset.size();++index)
        {
            std::shared_ptr<mat_matrix> matrix(new mat_matrix);
            *(matrix.get()) = *(rhs.dataset[index]);
            dataset.push_back(matrix);
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
            return nullptr;
        rows = dataset[index]->get_rows();
        cols = dataset[index]->get_cols();
        return dataset[index]->get_data(type);
    }
    const void* read_as_type(const char* name,unsigned int& rows,unsigned int& cols,unsigned int type) const
    {
        std::map<std::string,int>::const_iterator iter = name_table.find(name);
        if (iter == name_table.end())
            return nullptr;
        return read_as_type(iter->second,rows,cols,type);
    }
    template<class out_type>
    const out_type*& read(unsigned int index,unsigned int& rows,unsigned int& cols,const out_type*& out) const
    {
        return out = reinterpret_cast<const out_type*>(read_as_type(index,rows,cols,mat_type_info<out_type>::type));
    }
    template<class out_type>
    const out_type*& read(const char* name,unsigned int& rows,unsigned int& cols,const out_type*& out) const
    {
        return out = reinterpret_cast<const out_type*>(read_as_type(name,rows,cols,mat_type_info<out_type>::type));
    }
    template<class iterator>
    bool read(const char* name,iterator first,iterator last) const
    {
        unsigned int rows,cols,size(std::distance(first,last));
        const typename std::iterator_traits<iterator>::value_type* ptr = nullptr;
        if(read(name,rows,cols,ptr) == nullptr || rows*cols < size)
            return false;
        std::copy(ptr,ptr+size,first);
        return true;
    }
    template<class class_type>
    bool read(const char* name,class_type& data) const
    {
        return read(name,data.begin(),data.end());
    }
    bool read(const char* name,std::string& str) const
    {
        const char* buf = nullptr;
        unsigned int row,col;
        if(!read(name,row,col,buf))
            return false;
        if(buf[row*col-1] == 0)
            str = buf;
        else
            str = std::string(buf,buf+row*col);
        return true;
    }
    std::string read_string(const char* name) const
    {
        const char* buf = nullptr;
        unsigned int row,col;
        if(!read(name,row,col,buf))
            return std::string();
        if(buf[row*col-1] == 0)
            return std::string(buf);
        return std::string(buf,buf+row*col);
    }


    template<class data_type,class image_type>
    void read_as_image(int index,image_type& image_buf) const
    {
        if (index >= dataset.size())
            return;
        image_buf.resize(typename image_type::geometry_type(dataset[index]->get_cols(),dataset[index]->get_rows()));
        dataset[index]->copy_data(image_buf.begin());
    }

    template<class data_type,class image_type>
    void read_as_image(const char* name,image_type& image_buf) const
    {
        std::map<std::string,int>::const_iterator iter = name_table.find(name);
        if (iter == name_table.end())
            return;
        read_as_image(iter->second,image_buf);
    }

public:

    template<class char_type>
    bool load_from_file(const char_type* file_name,unsigned int max_count,std::string stop_name)
    {
        input_interface in;
        if(!in.open(file_name))
            return false;
        dataset.clear();
        for(unsigned int i = 0;i < max_count && in;++i)
        {
            std::shared_ptr<mat_matrix> matrix(new mat_matrix);
            if (!matrix->read(in))
                break;
            dataset.push_back(matrix);
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
        dataset.clear();
        while(in)
        {
            std::shared_ptr<mat_matrix> matrix(new mat_matrix);
            if (!matrix->read(in))
                break;
            dataset.push_back(matrix);
        }
        for (unsigned int index = 0; index < dataset.size(); ++index)
            name_table[dataset[index]->get_name()] = index;
        return !dataset.empty();
    }
    template<class Type>
    void add(const char* name_,const Type* data_ptr,unsigned int rows,unsigned int cols)
    {
        std::shared_ptr<mat_matrix> matrix(new mat_matrix);
        matrix->assign(name_,data_ptr,rows,cols);
        dataset.push_back(matrix);
        name_table[name_] = dataset.size()-1;
    }

    template<class image_type>
    bool save_to_image(image_type& image_data,const char* image_name) const
    {
        unsigned int r,c;
        const unsigned short* m = nullptr;
        read("dimension",r,c,m);
        if(!m || r*c != image_type::dimension)
            return false;
        image_data.resize(typename image_type::geometry_type(m));
        const typename image_type::value_type* buf = 0;
        read(image_name,r,c,buf);
        if(!buf || size_t(r)*size_t(c) != image_data.size())
            return false;
        std::copy(buf,buf+image_data.size(),image_data.begin());
        return true;
    }
    template<class vec_type>
    bool get_voxel_size(vec_type& vs) const
    {
        const float* vs_ptr = nullptr;
        unsigned int r,c;
        read("voxel_size",r,c,vs_ptr);
        if(!vs_ptr || r*c != 3)
            return false;
        for(unsigned int i = 0;i < 3;++i)
            vs[i] = vs_ptr[i];
        return true;
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
        save_to_image(source,"image");
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
    template<class Type,class size_type,class size_type2>
    bool write(const char* name_,const Type* data_ptr,size_type rows_,size_type2 cols_)
    {
        if(!rows_ || !cols_)
            return false;
        unsigned int imagf = 0;
        unsigned int type = mat_type_info<Type>::type;
        std::string name(name_);
        unsigned int namelen = uint32_t(name.length()+1);
        unsigned int rows = uint32_t(rows_);
        unsigned int cols = uint32_t(cols_);
        out.write(reinterpret_cast<const char*>(&type),4);
        out.write(reinterpret_cast<const char*>(&rows),4);
        out.write(reinterpret_cast<const char*>(&cols),4);
        out.write(reinterpret_cast<const char*>(&imagf),4);
        out.write(reinterpret_cast<const char*>(&namelen),4);
        out.write(reinterpret_cast<const char*>(&*name.begin()),namelen);
        out.write(reinterpret_cast<const char*>(data_ptr),size_t(rows)*size_t(cols)*sizeof(Type));
        return out;
    }
    template<class container_type>
    bool write(const char* name_,const container_type& data)
    {
        return write(name_,&data[0],1,uint32_t(data.size()));
    }
    template<class container_type>
    bool write(const char* name_,const container_type& data,uint32_t d)
    {
        return write(name_,&data[0],d,uint32_t(data.size()/d));
    }
    bool write(const char* name_,const std::string text)
    {
        if(text.empty())
            return false;
        return write(name_,text.c_str(),1,text.size()+1);
    }
    bool write(const mat_matrix& data)
    {
        return data.write(out);
    }

public:

    template<class image_type>
    void load_from_image(const image_type& image_data)
    {
        unsigned short dim[image_type::dimension];
        std::copy(image_data.geometry().begin(),image_data.geometry().end(),dim);
        write("dimension",dim,1,image_type::dimension);
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
