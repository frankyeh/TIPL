#ifndef MAT_HPP
#define MAT_HPP
#include <vector>
#include <fstream>
#include <iterator>
#include <sstream>
#include <filesystem>
#include <map>
#include "interface.hpp"
#include "../mt.hpp"
namespace tipl
{

namespace io
{

template<typename fun_type>
struct mat_type_info
{
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
public:
    alignas(4) unsigned int type = 50;
    alignas(4) unsigned int rows = 0;
    alignas(4) unsigned int cols = 0;
    alignas(4) unsigned int imagf = 0;
    alignas(4) unsigned int namelen = 0;
public:
    std::string name;
    std::vector<std::shared_ptr<mat_matrix> > sub_data;
public:
    mutable std::vector<unsigned char> data_buf,converted_data_buf;
    mutable size_t delay_read_pos = 0;
public:
    void copy(const mat_matrix& rhs)
    {
        type = rhs.type;
        rows = rhs.rows;
        cols = rhs.cols;
        name = rhs.name;
        namelen = rhs.namelen;
        data_buf = rhs.data_buf;
        converted_data_buf = rhs.converted_data_buf;
        delay_read_pos = rhs.delay_read_pos;
    }
    size_t get_total_size(unsigned int ty) const
    {
        unsigned int element_size_array[10] = {8,4,4,2,2,1,0,0,0,0};
        return size()*size_t(element_size_array[(ty%100)/10]);
    }
public:
    mat_matrix(void){}
    mat_matrix(const std::string& name_):namelen(name_.size()+1),name(name_){}
    mat_matrix(const mat_matrix& rhs){copy(rhs);}
    template <typename T,typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = true>
    mat_matrix(const std::string& name_,T value):type(mat_type_info<T>::type),rows(1),cols(1),namelen(name_.size()+1),name(name_)
    {
        data_buf.resize(sizeof(T));
        *reinterpret_cast<T*>(data_buf.data()) = value;
    }
    template<typename T>
    mat_matrix(const std::string& name_,const T* ptr,unsigned int rows_,unsigned int cols_):
        type(mat_type_info<T>::type),rows(rows_),cols(cols_),namelen(name_.size()+1),name(name_)
    {
        data_buf.resize(size()*size_t(sizeof(T)));
        std::copy(ptr,ptr+size(),reinterpret_cast<T*>(data_buf.data()));
    }
    const mat_matrix& operator=(const mat_matrix& rhs)
    {
        copy(rhs);
        return *this;
    }
    template<typename T,typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = true>
    void copy_data(T& value) const
    {
        switch (type)
        {
        case 0://double
            value = *reinterpret_cast<const double*>(data_buf.data());
            return;
        case 10://float
            value = *reinterpret_cast<const float*>(data_buf.data());
            return;
        case 20://unsigned int
            value = *reinterpret_cast<const unsigned int*>(data_buf.data());
            return;
        case 30://short
            value = *reinterpret_cast<const short*>(data_buf.data());
            return;
        case 40://unsigned short
            value = *reinterpret_cast<const unsigned short*>(data_buf.data());
            return;
        case 50://unsigned char
            value = *reinterpret_cast<const unsigned char*>(data_buf.data());
            return;
        }
    }
    template<typename T>
    void copy_data(T* out)  const
    {
        switch (type)
        {
        case 0://double
            std::copy(reinterpret_cast<const double*>(data_buf.data()),
                      reinterpret_cast<const double*>(data_buf.data())+size(),out);
            return;
        case 10://float
            std::copy(reinterpret_cast<const float*>(data_buf.data()),
                      reinterpret_cast<const float*>(data_buf.data())+size(),out);
            return;
        case 20://unsigned int
            std::copy(reinterpret_cast<const unsigned int*>(data_buf.data()),
                      reinterpret_cast<const unsigned int*>(data_buf.data())+size(),out);
            return;
        case 30://short
            std::copy(reinterpret_cast<const short*>(data_buf.data()),
                      reinterpret_cast<const short*>(data_buf.data())+size(),out);
            return;
        case 40://unsigned short
            std::copy(reinterpret_cast<const unsigned short*>(data_buf.data()),
                      reinterpret_cast<const unsigned short*>(data_buf.data())+size(),out);
            return;
        case 50://unsigned char
            std::copy(reinterpret_cast<const unsigned char*>(data_buf.data()),
                      reinterpret_cast<const unsigned char*>(data_buf.data())+size(),out);
            return;
        }
    }
    template<typename T>
    bool type_compatible(void) const
    {
        // same type or unsigned short v.s. short
        return mat_type_info<T>::type == type || (type == 40 && mat_type_info<T>::type == 30) || (type == 30 && mat_type_info<T>::type == 40);
    }
    template<typename stream_type>
    void flush(stream_type& in,bool flush)
    {
        if(has_delay_read())
        {
            if(!read(*in.get()))
                return;
            if(flush)
                in->flush();
        }
    }
    template<typename T>
    bool get_sub_data(const std::string& name, T& value) const
    {
        for(auto each : sub_data)
            if(each->name == name)
            {
                each->copy_data(value);
                return true;
            }
        return false;
    }
    template<typename T>
    T* get_data(void) const
    {
        if (type_compatible<T>())
            return const_cast<T*>(reinterpret_cast<const T*>(data_buf.data()));
        converted_data_buf.resize(get_total_size(mat_type_info<T>::type));
        auto new_data = const_cast<T*>(reinterpret_cast<const T*>(converted_data_buf.data()));
        copy_data(new_data);
        if constexpr (sizeof(T) >= 2)
        {
            if(!sub_data.empty() && type == mat_type_info<char>::type)
            {
                float slope = 0.0f;
                T inter = 0;
                if(get_sub_data(name+".slope",slope) && get_sub_data(name+".inter",inter) && slope != 0.0f)
                    tipl::par_for(size(),[&](size_t i){
                            new_data[i] = new_data[i]*slope+inter;
                    });
            }
        }
        return new_data;
    }
    template<typename T>
    T* get_data(const std::vector<size_t>& si2vi,size_t total_size) const
    {
        auto ptr = get_data<T>();
        if(!ptr)
            return nullptr;
        std::vector<unsigned char> sparse_data(total_size*sizeof(T)*rows);
        auto sparse_ptr = reinterpret_cast<T*>(sparse_data.data());
        if constexpr(std::is_floating_point_v<T>)
            std::fill(sparse_ptr,sparse_ptr+total_size*rows,T());
        size_t total = std::min<size_t>(si2vi.size(),cols);
        if(rows == 1)
            for(size_t index = 0;index < total;++index)
                sparse_ptr[si2vi[index]] = ptr[index];
        else
        {
            for(size_t index = 0,from = 0;index < total;++index,from += rows)
                std::copy(ptr+from,ptr+from+rows,
                          sparse_ptr + si2vi[index]*rows);
        }
        sparse_data.swap(converted_data_buf);
        return sparse_ptr;
    }
    template<typename T>
    void resize(const T& s)
    {
        rows = s[0];
        cols = s[1];
        data_buf.resize(get_total_size(type));
    }
    size_t size(void) const{return size_t(rows)*size_t(cols);}
    void set_name(const std::string& new_name)
    {
        name = new_name;
        namelen = name.size()+1;
    }
    void set_text(const std::string& text)
    {
        type = mat_type_info<char>::type;
        rows = 1;
        cols = text.size()+1;
        data_buf.clear();
        data_buf.resize(cols);
        std::copy(text.begin(),text.end(),data_buf.data());
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
        in.read(reinterpret_cast<char*>(&type),20);
        if (!in || type > 60 || type % 10 || size() == 0 || namelen == 0 || namelen > 255)
            return false;
        {
            std::vector<char> buffer(namelen+1);
            in.read(reinterpret_cast<char*>(buffer.data()),namelen);
            name = buffer.data();
        }

        // first time, do not read the data
        if(delayed_read && get_total_size(type) > 16777216) // 16MB
        {
            delay_read_pos = in.tell();
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
        return in.read(reinterpret_cast<char*>(data_buf.data()),get_total_size(type));
    }
    std::string get_info(void) const
    {
        std::ostringstream out;
        auto out_count = size();
        out << name << "= ";
        if(!out_count)
            return out.str();
        if(out_count > 20)
            out_count = 20;
        out << std::vector<std::string>({"double","float","unsigned int","short","unsigned short","unsigned char"})[type/10];
        if(out_count > 1)
        {
            if(rows == 1)
                out << "[" << cols << "]=";
            else
                out << "[" << rows << "][" << cols << "]=";
        }
        else
            out << "=";
        switch (type)
        {
        case 0://double
            std::copy(reinterpret_cast<const double*>(data_buf.data()),
                      reinterpret_cast<const double*>(data_buf.data())+out_count,
                      std::ostream_iterator<double>(out," "));
            break;
        case 10://float
            std::copy(reinterpret_cast<const float*>(data_buf.data()),
                      reinterpret_cast<const float*>(data_buf.data())+out_count,
                      std::ostream_iterator<float>(out," "));
            break;
        case 20://unsigned int
            std::copy(reinterpret_cast<const unsigned int*>(data_buf.data()),
                      reinterpret_cast<const unsigned int*>(data_buf.data())+out_count,
                      std::ostream_iterator<unsigned int>(out," "));
            break;
        case 30://short
            std::copy(reinterpret_cast<const short*>(data_buf.data()),
                      reinterpret_cast<const short*>(data_buf.data())+out_count,
                      std::ostream_iterator<short>(out," "));
            break;
        case 40://unsigned short
            std::copy(reinterpret_cast<const unsigned short*>(data_buf.data()),
                      reinterpret_cast<const unsigned short*>(data_buf.data())+out_count,
                      std::ostream_iterator<unsigned short>(out," "));
            break;
        case 50://unsigned char
            if(sub_data.empty() && out_count < 4096 && data_buf[0] >= '0' && data_buf[0] <= '~')
            {
                std::string text(reinterpret_cast<const char*>(data_buf.data()));
                std::replace(text.begin(),text.end(),'\n',' ');
                return out.str() + text;
            }
            else
            {
                for(size_t i = 0;i < out_count;++i)
                    out << int(data_buf[i]) << " ";
            }
            break;
        }
        std::string info = out.str();
        if(size() > 10)
            info += "...";
        for(auto each : sub_data)
        {
            info += "\n";
            info += each->get_info();
        }
        return info;
    }
};

template<typename input_interface = std_istream>
class mat_read_base
{
private:
    std::vector<std::shared_ptr<mat_matrix> > dataset;
    std::map<std::string,size_t> name_table;
    mutable std::mutex mat_load;
private:
    void copy(const mat_read_base& rhs)
    {
        for(auto each : dataset)
        {
            each->flush(in,true);
            std::shared_ptr<mat_matrix> matrix(new mat_matrix);
            *(matrix.get()) = *(each.get());
            dataset.push_back(matrix);
        }
    }
public:
    std::string error_msg;
    mat_read_base(void):in(new input_interface){}
    mat_read_base(const mat_read_base& rhs){copy(rhs);}
    void remove(const std::string& name)
    {
        remove(index_of(name));
    }
    void remove(size_t remove_index)
    {
        if(remove_index >= dataset.size())
            return;
        dataset.erase(dataset.begin()+remove_index);
        name_table.clear();
        for(size_t index = 0;index < dataset.size();++index)
            name_table[dataset[index]->name] = index;
    }
    void flush(unsigned int index)
    {
        if(index >= dataset.size())
            return;
        dataset[index]->flush(in,true);
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
    bool has(const std::string& name) const
    {
        return name_table.find(name) != name_table.end();
    }
    size_t index_of(const std::string& name) const
    {
        auto iter = name_table.find(name);
        if(iter == name_table.end())
            return dataset.size();
        return iter->second;
    }
    template<typename T>
    bool type_compatible(const std::string& name) const
    {
        auto iter = name_table.find(name);
        if(iter == name_table.end())
            return false;
        return dataset[iter->second]->template type_compatible<T>();
    }
    bool get_col_row(const std::string& name,unsigned int& rows,unsigned int& cols) const
    {
        auto iter = name_table.find(name);
        if(iter == name_table.end())
            return false;
        else
        {
            rows = dataset[iter->second]->rows;
            cols = dataset[iter->second]->cols;
            return true;
        }
    }
    template<typename T>
    T* read_as_type(unsigned int index) const
    {
        std::lock_guard<std::mutex> lock(mat_load);
        // if type is not compatible, make sure all data are flushed before calling get_data);
        dataset[index]->flush(in,!dataset[index]->type_compatible<T>());
        return dataset[index]->template get_data<T>();
    }
    template<typename T>
    T* read_as_type(unsigned int index,const std::vector<size_t>& si2vi,size_t total_size) const
    {
        if(index >= dataset.size())
            return nullptr;
        if(dataset[index]->cols != si2vi.size())
            return read_as_type<T>(index);
        std::lock_guard<std::mutex> lock(mat_load);
        dataset[index]->flush(in,true);
        return dataset[index]->template get_data<T>(si2vi,total_size);
    }
    template<typename T>
    T* read_as_type(const std::string& name,const std::vector<size_t>& si2vi,size_t total_size) const
    {
        return read_as_type<T>(index_of(name),si2vi,total_size);
    }
    template<typename T>
    const T* read_as_type(unsigned int index,unsigned int& rows,unsigned int& cols) const
    {
        if (index >= dataset.size())
            return nullptr;
        rows = dataset[index]->rows;
        cols = dataset[index]->cols;
        return read_as_type<T>(index);
    }
    template<typename T>
    const T* read_as_type(const std::string& name) const
    {
        auto iter = name_table.find(name);
        if(iter == name_table.end())
            return nullptr;
        return read_as_type<T>(iter->second);
    }
    template<typename T>
    const T* read_as_type(const std::string& name,unsigned int& rows,unsigned int& cols) const
    {
        auto iter = name_table.find(name);
        if (iter == name_table.end())
            return nullptr;
        rows = dataset[iter->second]->rows;
        cols = dataset[iter->second]->cols;
        return read_as_type<T>(iter->second);
    }
    template<typename T>
    T read_as_value(const std::string& name) const
    {
        unsigned int rows,cols;
        auto ptr = read_as_type<T>(name,rows,cols);
        return ptr ? *ptr : T();
    }
    template<typename T>
    auto read_as_vector(const std::string& name) const
    {
        unsigned int rows,cols;
        auto ptr = read_as_type<T>(name,rows,cols);
        return ptr ? std::vector<T>(ptr,ptr+size_t(rows)*size_t(cols)) : std::vector<T>();
    }

    template<typename T,typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = true>
    auto read(unsigned int index,unsigned int& rows,unsigned int& cols,const T*& out) const
    {
        return out = read_as_type<T>(index,rows,cols);
    }
    template<typename T,typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = true>
    auto read(const std::string& name,unsigned int& rows,unsigned int& cols,const T*& out) const
    {
        return out = read_as_type<T>(name,rows,cols);
    }
    template<typename T,typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = true>
    auto read(unsigned int index,const T*& out) const
    {
        return out = read_as_type<T>(index);
    }
    template<typename T,typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = true>
    auto read(const std::string& name,const T*& out) const
    {
        return out = read_as_type<T>(name);
    }
    template<typename T,typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = true>
    auto read(unsigned int index,const T*& out,const std::vector<size_t>& si2vi,size_t total_size) const
    {
        return out = read_as_type<T>(index,si2vi,total_size);
    }
    template<typename T,typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = true>
    const T* read(const std::string& name,const T*& out,const std::vector<size_t>& si2vi,size_t total_size) const
    {
        auto iter = name_table.find(name);
        if (iter == name_table.end())
            return nullptr;
        return read_as_type<T>(iter->second,si2vi,total_size);
    }
    template<typename iterator>
    bool read(unsigned int index,iterator first,iterator last) const
    {
        unsigned int rows,cols,size(std::distance(first,last));
        const typename std::iterator_traits<iterator>::value_type* ptr = nullptr;
        if(read(index,rows,cols,ptr) == nullptr)
            return false;
        std::copy(ptr,ptr+std::min<size_t>(size_t(rows)*size_t(cols),size),first);
        return true;
    }
    template<typename iterator>
    bool read(const std::string& name,iterator first,iterator last) const
    {
        auto iter = name_table.find(name);
        if (iter == name_table.end())
            return false;
        return read(iter->second,first,last);
    }
    bool read(const std::string& name,std::string& str) const
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
    bool read(const std::string& name,T& value) const
    {
        const T* ptr = nullptr;
        unsigned int rows,cols;
        if(!read(name,rows,cols,ptr))
            return false;
        value = *ptr;
        return true;
    }
    template<typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    bool read(const std::string& name,T& data) const
    {
        return read(name,data.begin(),data.end());
    }
    template<typename T>
    T read(const std::string& name) const
    {
        T data;
        read(name,data);
        return data;
    }

public:
    std::shared_ptr<input_interface> in;
    bool delay_read = false;
    template<typename prog_type = tipl::io::default_prog_type>
    bool load_from_file(const std::string& file_name,prog_type&& prog = prog_type())
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
            if(!dataset.empty() && matrix->name.find(dataset.back()->name + ".") == 0)
                dataset.back()->sub_data.push_back(matrix);
            else
            {
                name_table[matrix->name] = dataset.size();
                dataset.push_back(matrix);
            }
        }    
        return !dataset.empty();
    }
    void push_back(std::shared_ptr<mat_matrix> mat)
    {
        dataset.push_back(mat);
        name_table[mat->name] = dataset.size()-1;
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

enum storage_type {regular = 0,sloped = 1,masked = 2,masked_sloped = 3} ;
template<typename output_interface = std_ostream>
class mat_write_base
{
    output_interface out;
public:
    bool apply_slope = false;
public:
    unsigned int mask_rows = 0;
    unsigned int mask_cols = 0;
    std::vector<size_t> si2vi;
public:
    mat_write_base(const std::string& file_name)
    {
        out.open(file_name);
    }
public:

    template<storage_type stype = regular,typename T>
    bool write(mat_matrix& mat,const T* ptr)
    {
        if constexpr(stype & sloped)
        {
            if(apply_slope && mat.size() > 4096 && mat.sub_data.empty())
            {
                T inter(ptr[0]),max_v(ptr[0]);
                auto size = mat.size();
                for(size_t i = 0;i < size;++i)
                {
                    auto v = ptr[i];
                    if(v < inter)
                        inter = v;
                    if(v > max_v)
                        max_v = v;
                }
                float slope = float(max_v-inter)/255.99f;
                std::vector<unsigned char> new_data(size);
                if(slope != 0.0f)
                {
                    float scale = 1.0f/slope;
                    tipl::par_for(new_data.size(),[&](size_t i)
                    {
                        new_data[i] = (ptr[i]-inter)*scale;
                    });
                }
                mat.data_buf.swap(new_data);
                ptr = reinterpret_cast<const T*>(mat.data_buf.data());
                mat.type = mat_type_info<char>::type;
                mat.sub_data.push_back(std::make_shared<mat_matrix>(mat.name+".slope",slope));
                mat.sub_data.push_back(std::make_shared<mat_matrix>(mat.name+".inter",inter));
            }
        }
        out.write(reinterpret_cast<const char*>(&mat.type),20);
        out.write(mat.name.data(),mat.namelen);
        out.write(reinterpret_cast<const char*>(ptr),mat.get_total_size(mat.type));
        for(auto each : mat.sub_data)
            write<regular>(*each.get());
        return out;
    }
    template<storage_type stype = regular>
    bool write(mat_matrix& mat)
    {
        return write<stype>(mat,mat.data_buf.data());
    }
    template<storage_type stype = regular,typename T>
    bool write(const std::string& name,const T* ptr,unsigned int rows_,unsigned int cols_)
    {
        if(!rows_ || !cols_)
            return true;
        std::vector<T> buf;
        if constexpr((stype & masked) > 0)
        {
            if(!si2vi.empty() && rows_ == mask_rows && cols_ == mask_cols)
            {
                buf.resize(si2vi.size());
                for(size_t index = 0;index < si2vi.size();++index)
                    buf[index] = ptr[si2vi[index]];
                ptr = buf.data();
                rows_ = 1;
                cols_ = si2vi.size();
            }
        }
        mat_matrix out_data(name);
        out_data.cols = cols_;
        out_data.rows = rows_;
        if constexpr(std::is_class_v<T>)
        {
            out_data.type = mat_type_info<std::remove_const_t<std::remove_reference_t<decltype(*(ptr->data()))> > >::type;
            out_data.rows *= T::dimension;
            write<stype>(out_data,ptr->data());
        }
        else
        {
            out_data.type = mat_type_info<T>::type;
            write<stype>(out_data,ptr);
        }
        return out;
    }

    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    bool write(const std::string& name,T value)
    {
        return write<regular>(name,&value,1,1);
    }
    template<storage_type stype = regular,typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    bool write(const std::string& name,const T& data)
    {
        auto size = uint32_t(data.end()-data.begin());
        if(!size)
            return out;
        return write<stype>(name,data.data(),1,size);
    }
    template<storage_type stype = regular,typename T,typename std::enable_if<std::is_class<T>::value,bool>::type = true>
    bool write(const std::string& name,const T& data,uint32_t d)
    {
        if(data.empty())
            return out;
        return write<stype>(name,data.data(),d,uint32_t((data.end()-data.begin())/d));
    }
    bool write(const std::string& name,const std::string& text)
    {
        if(text.empty())
            return out;
        return write<regular>(name,text.c_str(),1,text.size()+1);
    }
public:
    template<typename image_type>
    void load_from_image(const image_type& image_data)
    {
        unsigned short dim[image_type::dimension];
        std::copy(image_data.shape().begin(),image_data.shape().end(),dim);
        write("dimension",dim,1,image_type::dimension);
        write("image",image_data.data(),1,image_data.size());
    }

    template<typename image_type>
    mat_write_base& operator<<(const image_type& source)
    {
        load_from_image(source);
        return *this;
    }
    operator bool() const	{return !(!out);}
    bool operator!() const	{return !out;}
    void close(void)        {out.close();}
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
