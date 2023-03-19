#ifndef MAT_HPP
#define MAT_HPP
#include <vector>
#include <fstream>
#include <iterator>
#include <sstream>
#include <filesystem>
#include <map>
#include "interface.hpp"
namespace tipl
{

namespace io
{

template<typename fun_type>
struct mat_type_info
{
    static const unsigned int type = 60;
};

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
    void* data_ptr = nullptr; // for read
    size_t delay_read_pos = 0;
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
    mat_matrix(void):type(0),rows(0),cols(0),namelen(0){}
    mat_matrix(const std::string& name_):type(0),rows(0),cols(0),namelen(uint32_t(name_.size()+1)),name(name_){}
    mat_matrix(const mat_matrix& rhs){copy(rhs);}
    const mat_matrix& operator=(const mat_matrix& rhs)
    {
        copy(rhs);
        return *this;
    }
    template<typename Type>
    void assign(const Type* data_ptr_,unsigned int rows_,unsigned int cols_)
    {
        data_ptr = data_ptr_;
        rows = rows_;
        cols = cols_;
        type = mat_type_info<Type>::type;
    }
    template<typename Type>
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

    template<typename OutType>
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
    template<typename T>
    bool type_compatible(void) const
    {
        // same type or unsigned short v.s. short
        return mat_type_info<T>::type == type || (type == 40 && mat_type_info<T>::type == 30) || (type == 30 && mat_type_info<T>::type == 40);
    }
    template<typename T>
    T* get_data(void)
    {
        constexpr auto get_type = mat_type_info<T>::type;
        if(get_type == 60)
            return nullptr;
        if (!type_compatible<T>())
        {
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
        }
        return const_cast<T*>(reinterpret_cast<const T*>(data_ptr));
    }
    unsigned int get_rows(void) const
    {
        return rows;
    }
    unsigned int get_cols(void) const
    {
        return cols;
    }
    template<typename T>
    void resize(const T& size)
    {
        rows = size[0];
        cols = size[1];
    }
    void set_name(const std::string& new_name)
    {
        name = new_name;
        namelen = uint32_t(name.length());
    }
    void set_text(const std::string& text)
    {
        type = 50;
        rows = 1;
        cols = text.length()+1;
        data_buf.clear();
        data_buf.resize(cols);
        std::copy(text.c_str(),text.c_str()+cols-1,data_buf.begin());
        data_ptr = &*data_buf.begin();
    }
    const std::string& get_name(void) const
    {
        return name;
    }
    template<typename T>
    bool is_type(void) const
    {
        return type == mat_type_info<T>::type;
    }
    bool has_delay_read(void) const
    {
        return delay_read_pos;
    }
    template<typename stream_type>
    bool read(stream_type& in,bool delayed_read = false)
    {
        // second time read the buffer
        if(delay_read_pos)
        {
            in.clear();
            in.seek(delay_read_pos);
            delay_read_pos = 0;
            goto read;
        }
        in.read(buf,sizeof(buf));
        if (!in || type > 100 || type % 10 > 1)
            return false;
        if (type % 10) // text
            type = 0;
        if(!in || namelen == 0 || namelen > 255)
            return false;
        {
            std::vector<char> buffer(namelen+1);
            in.read(reinterpret_cast<char*>(&*buffer.begin()),namelen);
            if(rows*cols == 0)
                return false;
            name = &*buffer.begin();
        }

        // first time, do not read the data
        if(delayed_read && get_total_size(type) > 16777216) // 16MB
        {
            delay_read_pos = in.tell();
            data_ptr = nullptr;
            in.seek(delay_read_pos + get_total_size(type));
            return true;
        }

        // read buffer
        read:
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
        return in.read(reinterpret_cast<char*>(data_ptr),get_total_size(type));
    }
    template<typename stream_type>
    bool write(stream_type& out) const
    {
        unsigned int imagf = 0;
        out.write(reinterpret_cast<const char*>(&type),4);
        out.write(reinterpret_cast<const char*>(&rows),4);
        out.write(reinterpret_cast<const char*>(&cols),4);
        out.write(reinterpret_cast<const char*>(&imagf),4);
        out.write(reinterpret_cast<const char*>(&namelen),4);
        out.write(reinterpret_cast<const char*>(&*name.begin()),namelen);
        return out.write(reinterpret_cast<const char*>(data_ptr),get_total_size(type));
    }
    void get_info(std::string& info) const
    {
        std::ostringstream out;
        unsigned int out_count = rows*cols;
        out << name <<"= type:" << type << " data[" << rows << "][" << cols << "]:";
        if(out_count > 10)
            out_count = 10;
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

template<typename input_interface = std_istream>
class mat_read_base
{
private:
    std::vector<std::shared_ptr<mat_matrix> > dataset;
    std::map<std::string,size_t> name_table;
private:
    void copy(const mat_read_base& rhs)
    {
        for(unsigned int index = 0;index < dataset.size();++index)
        {
            std::shared_ptr<mat_matrix> matrix(new mat_matrix);
            *(matrix.get()) = *(rhs.dataset[index]);
            dataset.push_back(matrix);
        }
    }

public:
    std::string error_msg;
    mat_read_base(void):in(new input_interface){}
    mat_read_base(const mat_read_base& rhs){copy(rhs);}
    void remove(size_t index)
    {
        dataset.erase(dataset.begin()+index);
        name_table.clear();
        for(size_t index = 0;index < dataset.size();++index)
            name_table[dataset[index]->get_name()] = index;
    }
    const mat_read_base& operator=(const mat_read_base& rhs)
    {
        copy(rhs);
        return *this;
    }
    void swap(mat_read_base& rhs)
    {
        dataset.swap(rhs.dataset);
        name_table.swap(rhs.name_table);
        in.swap(rhs.in);
        std::swap(delay_read,rhs.delay_read);
    }
    bool has(const char* name) const
    {
        return name_table.find(name) != name_table.end();
    }
    template<typename T>
    bool type_compatible(const char* name)
    {
        auto iter = name_table.find(name);
        if(iter == name_table.end())
            return false;
        return dataset[iter->second]->template type_compatible<T>();
    }
    bool get_col_row(const char* name,unsigned int& rows,unsigned int& cols)
    {
        auto iter = name_table.find(name);
        if(iter == name_table.end())
            return false;
        else
        {
            rows = dataset[iter->second]->get_rows();
            cols = dataset[iter->second]->get_cols();
            return true;
        }
    }
    template<typename T>
    const T* read_as_type(unsigned int index,unsigned int& rows,unsigned int& cols) const
    {
        if (index >= dataset.size())
            return nullptr;
        if(dataset[index]->has_delay_read())
        {
            if(!dataset[index]->read(*in.get()))
                return nullptr;
            // if type is not compatible, make sure all data are flushed before calling get_data
            if(!dataset[index]->template type_compatible<T>())
                in->flush();
        }
        rows = dataset[index]->get_rows();
        cols = dataset[index]->get_cols();
        return dataset[index]->template get_data<T>();
    }
    template<typename T>
    const T* read_as_type(const char* name,unsigned int& rows,unsigned int& cols) const
    {
        auto iter = name_table.find(name);
        if (iter == name_table.end())
            return nullptr;
        return read_as_type<T>(iter->second,rows,cols);
    }
    template<typename T>
    const T*& read(unsigned int index,unsigned int& rows,unsigned int& cols,const T*& out) const
    {
        return out = read_as_type<T>(index,rows,cols);
    }
    template<typename T>
    const T*& read(const char* name,unsigned int& rows,unsigned int& cols,const T*& out) const
    {
        return out = read_as_type<T>(name,rows,cols);
    }
    template<typename iterator>
    bool read(const char* name,iterator first,iterator last) const
    {
        unsigned int rows,cols,size(std::distance(first,last));
        const typename std::iterator_traits<iterator>::value_type* ptr = nullptr;
        if(read(name,rows,cols,ptr) == nullptr || rows*cols < size)
            return false;
        std::copy(ptr,ptr+size,first);
        return true;
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
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    bool read(const char* name,T& value) const
    {
        const T* ptr = nullptr;
        unsigned int rows,cols;
        if(!read(name,rows,cols,ptr))
            return false;
        value = *ptr;
        return true;
    }
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    bool read(const char* name,T& data) const
    {
        return read(name,data.begin(),data.end());
    }
    template<typename T>
    T read(const char* name) const
    {
        T data;
        read(name,data);
        return data;
    }

public:
    std::shared_ptr<input_interface> in;
    bool delay_read = false;
    template<typename char_type,typename prog_type = tipl::io::default_prog_type>
    bool load_from_file(const char_type* file_name,prog_type&& prog = prog_type())
    {
        if(!in->open(file_name))
        {
            error_msg = "cannot open file at ";
            error_msg += file_name;
            return false;
        }
        dataset.clear();
        while(in->good() && !in->eof())
        {
            if(!prog(int(in->cur_size()*99/in->size()),100))
                return false;
            std::shared_ptr<mat_matrix> matrix(new mat_matrix);
            if (!matrix->read(*in.get(),delay_read))
                break;
            name_table[matrix->get_name()] = dataset.size();
            dataset.push_back(matrix);
        }    
        return !dataset.empty();
    }
    void add(const char* name_,const mat_matrix& matrix)
    {
        std::shared_ptr<mat_matrix> new_matrix(new mat_matrix);
        *(new_matrix.get()) = matrix;
        dataset.push_back(new_matrix);
        name_table[name_] = dataset.size()-1;
    }

    template<typename container_type>
    void add(const char* name_,const container_type& container)
    {
        std::shared_ptr<mat_matrix> matrix(new mat_matrix);
        matrix->assign(name_,&*container.begin(),1,uint32_t(container.end()-container.begin()));
        dataset.push_back(matrix);
        name_table[name_] = dataset.size()-1;
    }

    template<typename Type>
    void add(const char* name_,const Type* data_ptr,unsigned int rows,unsigned int cols)
    {
        std::shared_ptr<mat_matrix> matrix(new mat_matrix);
        matrix->assign(name_,data_ptr,rows,cols);
        dataset.push_back(matrix);
        name_table[name_] = dataset.size()-1;
    }

    template<typename image_type>
    bool save_to_image(image_type& image_data,const char* image_name) const
    {
        typename image_type::shape_type s;
        if(!get_dimension(s))
            return false;
        image_data.resize(s);
        unsigned int r,c;
        const typename image_type::value_type* buf = 0;
        read(image_name,r,c,buf);
        if(!buf || size_t(r)*size_t(c) != image_data.size())
            return false;
        std::copy(buf,buf+image_data.size(),image_data.begin());
        return true;
    }
    template<typename dim_type>
    bool get_dimension(dim_type& dim) const
    {
        const float* dim_ptr = nullptr;
        unsigned int r,c;
        read("dimension",r,c,dim_ptr);
        if(!dim_ptr || r*c != 3)
            return false;
        for(unsigned int i = 0;i < 3;++i)
            dim[i] = dim_ptr[i];
        return true;
    }
    template<typename vec_type>
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

    auto size(void) const
    {
        return dataset.size();
    }

    auto& operator[](size_t index){return *dataset[index];}
    const auto& operator[](size_t index) const {return *dataset[index];}

    template<typename image_type>
    const mat_read_base& operator>>(image_type& source) const
    {
        save_to_image(source,"image");
        return *this;
    }
};



template<typename output_interface = std_ostream>
class mat_write_base
{
    output_interface out;
public:
    mat_write_base(const char* file_name)
    {
        out.open(file_name);
    }
public:
    template<typename Type,typename size_type,typename size_type2>
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
    template<typename container_type>
    bool write(const char* name_,const container_type& data)
    {
        return write(name_,&*data.begin(),1,uint32_t(data.end()-data.begin()));
    }
    template<typename T>
    bool write(const char* name_,const std::initializer_list<T>& data)
    {
        return write(name_,&*data.begin(),1,uint32_t(data.end()-data.begin()));
    }
    template<typename container_type>
    bool write(const char* name_,const container_type& data,uint32_t d)
    {
        if(data.empty())
            return false;
        return write(name_,&data[0],d,uint32_t((data.end()-data.begin())/d));
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

    template<typename image_type>
    void load_from_image(const image_type& image_data)
    {
        unsigned short dim[image_type::dimension];
        std::copy(image_data.shape().begin(),image_data.shape().end(),dim);
        write("dimension",dim,1,image_type::dimension);
        write("image",&*image_data.begin(),1,image_data.size());
    }

    template<typename image_type>
    mat_write_base& operator<<(const image_type& source)
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

#ifdef TIPL_GZ_STREAM_HPP
namespace tipl{namespace io{
typedef mat_write_base<gz_ostream> gz_mat_write;
typedef mat_read_base<gz_istream> gz_mat_read;
}}
#endif
