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
#include <map>

namespace image
{

namespace io
{

template<typename fun_type>
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



class mat_matrix
{
private:
    unsigned int type;
    unsigned int rows;
    unsigned int cols;
    unsigned int count;
    unsigned int namelen;
private:
    std::string name;
    std::vector<unsigned char> data_buf;
    unsigned char* data; // for read
private:
    unsigned int get_total_size(unsigned int ty)
    {
        unsigned int element_size_array[10] = {8,4,4,2,2,1,0,0,0,0};
        return count*element_size_array[(ty%100)/10];
    }
private:
    void* data_ptr; // for write
    unsigned int element_size;
public:
    mat_matrix(void):type(0),rows(0),cols(0),data(0)
    {}
    mat_matrix(const std::string& name_):namelen(name_.size()+1),name(name_),data(0) {}

    template<typename Type>
    void assign(const Type* data_ptr_,unsigned int rows_,unsigned int cols_)
    {
        data_ptr = (void*)data_ptr_;
        rows = rows_;
        cols = cols_;
        type = mat_type_info<Type>::type;
        element_size = sizeof(Type);
    }
	template<typename OutType>
    void copy_data(OutType out)
    {
        switch (type)
        {
        case 0://double
            std::copy((const double*)data,((const double*)data) + count,out);
            break;
        case 10://float
            std::copy((const float*)data,((const float*)data) + count,out);
            break;
        case 20://unsigned int
            std::copy((const unsigned int*)data,((const unsigned int*)data) + count,out);
            break;
        case 30://short
            std::copy((const short*)data,((const short*)data) + count,out);
            break;
        case 40://unsigned short
            std::copy((const unsigned short*)data,((const unsigned short*)data) + count,out);
            break;
        case 50://unsigned char
            std::copy((const unsigned char*)data,((const unsigned char*)data) + count,out);
            break;
        }
    }
    const void* get_data(unsigned int get_type)
    {
        if(get_type != 0 && get_type != 10 && get_type != 20 && get_type != 30 && get_type != 40 && get_type != 50)
            return 0;
        // same type or unsigned short v.s. short
        if (get_type == type || (type == 40 && get_type == 30) || (type == 30 && get_type == 40))
            return data;
        std::vector<unsigned char> allocator(get_total_size(get_type));
        unsigned char* new_data = &*allocator.begin();
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
        std::swap(data,new_data);
        allocator.swap(data_buf);
        type = get_type;
        return data;
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
    template<typename stream_type>
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
        std::vector<char> buffer(namelen+1);
        in.read((char*)&*buffer.begin(),namelen);
        count = rows*cols;
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
		data = &*data_buf.begin();
        in.read((char*)data,get_total_size(type));
        return in;
    }
    template<typename stream_type>
    bool write(stream_type& out)
    {
        unsigned int imagf = 0;
        out.write((const char*)&type,4);
        out.write((const char*)&rows,4);
        out.write((const char*)&cols,4);
        out.write((const char*)&imagf,4);
        out.write((const char*)&namelen,4);
        out.write((const char*)&*name.begin(),namelen);
        out.write((const char*)data_ptr,element_size*rows*cols);
        return out;
    }
};

class mat
{
private:
    std::vector<mat_matrix*> dataset;
    std::map<std::string,int> name_table;
public:
    const void* get_matrix_as_type(int index,unsigned int& rows,unsigned int& cols,unsigned int type) const
    {
        if (index >= dataset.size())
            return 0;
        rows = dataset[index]->get_rows();
        cols = dataset[index]->get_cols();
        return dataset[index]->get_data(type);
    }
    const void* get_matrix_as_type(const char* name,unsigned int& rows,unsigned int& cols,unsigned int type) const
    {
        std::map<std::string,int>::const_iterator iter = name_table.find(name);
        if (iter == name_table.end())
            return 0;
        return get_matrix_as_type(iter->second,rows,cols,type);
    }
    template<typename out_type>
    void get_matrix(int index,unsigned int& rows,unsigned int& cols,const out_type*& out) const
    {
        out = (const out_type*)get_matrix_as_type(index,rows,cols,mat_type_info<out_type>::type);
    }
    template<typename out_type>
    void get_matrix(const char* name,unsigned int& rows,unsigned int& cols,const out_type*& out) const
    {
        out = (const out_type*)get_matrix_as_type(name,rows,cols,mat_type_info<out_type>::type);
    }
	template<typename data_type>
	void get_matrix_as_image(int index,image::basic_image<data_type,2>& image_buf) const
	{
		if (index >= dataset.size())
            return;
		image_buf.resize(image::geometry<2>(dataset[index]->get_cols(),dataset[index]->get_rows()));
        dataset[index]->copy_data(image_buf.begin());
	}

	template<typename data_type>
	void get_matrix_as_image(const char* name,image::basic_image<data_type,2>& image_buf) const
    {
		std::map<std::string,int>::const_iterator iter = name_table.find(name);
        if (iter == name_table.end())
            return 0;
		get_matrix_as_image(iter->second,image_buf);
    }

    template<typename Type>
    void add_matrix(const char* name,const Type* data_ptr,unsigned int rows,unsigned int cols)
    {
        std::auto_ptr<mat_matrix> matrix(new mat_matrix(name));
        matrix->assign(data_ptr,rows,cols);
        name_table[matrix->get_name()] = dataset.size();
        dataset.push_back(matrix.release());
    }
public:
    ~mat(void)
    {
        clear();
    }
    void clear(void)
    {
        for(unsigned int index = 0; index < dataset.size(); ++index)
            delete dataset[index];
        dataset.clear();
    }
    template<typename char_type>
    bool load_from_file(const char_type* file_name)
    {
        std::ifstream in(file_name, std::ios::binary);
        if(!in)
            return false;
        in.seekg(0,std::ios::end);
        long unsigned int file_size = in.tellg();
        in.seekg(0,std::ios::beg);
        while(in.tellg() < file_size)
        {
            std::auto_ptr<mat_matrix> matrix(new mat_matrix);
            if (!matrix->read(in))
                return false;
            dataset.push_back(matrix.release());
        }
        for (unsigned int index = 0; index < dataset.size(); ++index)
            name_table[dataset[index]->get_name()] = index;
        return true;
    }
    template<typename char_type>
    bool save_to_file(const char_type* file_name) const
    {
        std::ofstream out(file_name, std::ios::binary);
        if(!out)
            return false;
        for (unsigned int index = 0; index < dataset.size(); ++index)
            if(!dataset[index]->write(out))
                return false;
        return true;
    }
    template<typename image_type>
    void load_from_image(const image_type& image_data)
    {
        clear();
        add_matrix("dimension",&*image_data.geometry().begin(),1,image_type::dimension);
        add_matrix("image",&*image_data.begin(),1,image_data.size());
    }

    template<typename image_type>
    void save_to_image(image_type& image_data) const
    {
        unsigned int r,c;
        const unsigned short* m = 0;
        get_matrix("dimension",r,c,m);
        if(!m || r*c != image_type::dimension)
            return;
        image_data.resize(image::geometry<image_type::dimension>(m));
        const typename image_type::value_type* buf = 0;
        get_matrix("image",r,c,buf);
        if(!buf || r*c != image_data.size())
            return;
        std::copy(buf,buf+image_data.size(),image_data.begin());
    }

    unsigned int get_matrix_count(void) const
    {
        return dataset.size();
    }
    const char* get_matrix_name(unsigned int index) const
    {
        return dataset[index]->get_name().c_str();
    }

    template<typename image_type>
    const mat& operator>>(image_type& source) const
    {
        save_to_image(source);
        return *this;
    }
    template<typename image_type>
    mat& operator<<(image_type& source)
    {
        load_from_image(source);
        return *this;
    }

};




}
}

#endif//MAT_FILE_HPP
