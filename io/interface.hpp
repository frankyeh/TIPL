#ifndef IMAGE_IO_INTERFACE_HPP
#define IMAGE_IO_INTERFACE_HPP
#include <fstream>

namespace tipl
{

namespace io
{

struct default_prog_type{
    default_prog_type(const char*){}
    default_prog_type(void){}
    template<typename T>
    bool operator()(T a,T b){return a < b;}
    bool aborted(void){return false;}
};

template<typename prog_type,typename stream_type,typename ptr_type>
bool read_stream_with_prog(prog_type& prog,
                           stream_type& in,
                           ptr_type* ptr,
                           size_t size_in_byte,
                           std::string& error_msg,
                           size_t buf_size = 1000000)
{
    if(size_in_byte < buf_size || std::is_same<prog_type,default_prog_type>::value)
    {
        if(!in.read(reinterpret_cast<char*>(ptr),size_in_byte))
        {
            if(in.eof())
                return true;
            error_msg = "I/O error";
            return false;
        }
        return true;
    }
    if constexpr(!std::is_same<prog_type,default_prog_type>::value)
    {
        auto buf = reinterpret_cast<char*>(ptr);
        size_t pos = 0;
        while(prog(pos*100/size_in_byte,100))
        {
            if(buf_size < 64000000)
                buf_size *= 2;
            if(!in.read(buf+pos,std::min<size_t>(buf_size,size_in_byte-pos)))
            {
                error_msg = "error reading data";
                return false;
            }
            pos += buf_size;
        }
        if(pos < size_in_byte)
        {
            error_msg = "aborted";
            return false;
        }

    }
    return true;
}

template<typename prog_type,typename stream_type,typename ptr_type>
bool save_stream_with_prog(prog_type& prog,
                           stream_type& out,
                           const ptr_type* ptr,
                           size_t size_in_byte,
                           std::string& error_msg,
                           size_t buf_size = 1000000)
{
    if(size_in_byte < buf_size || std::is_same<prog_type,default_prog_type>::value)
    {
        if(!out.write(reinterpret_cast<const char*>(ptr),size_in_byte))
        {
            error_msg = "insufficient disk space";
            return false;
        }
        return true;
    }

    if constexpr(!std::is_same<prog_type,default_prog_type>::value)
    {
        auto buf = reinterpret_cast<const char*>(ptr);
        size_t pos = 0;
        while(prog(pos*100/size_in_byte,100))
        {
            if(buf_size < 64000000)
                buf_size *= 2;
            if(!out.write(buf+pos,std::min<size_t>(buf_size,size_in_byte-pos)))
            {
                error_msg = "insufficient disk space";
                return false;
            }
            pos += buf_size;
        }
        if(pos < size_in_byte)
        {
            error_msg = "aborted";
            return false;
        }
    }
    return true;
}

class std_istream{
    size_t size_;
    std::ifstream in;
public:
    std_istream(void):size_(0){}
    template<typename char_type>
    bool open(const char_type* file_name)
    {
        in.open(file_name,std::ios::binary);
        if(in)
        {
            in.seekg(0,std::ios::end);
            size_ = in.tellg();
            in.seekg(0,std::ios::beg);
        }
        return in.good();
    }
    bool read(void* buf,size_t size)
    {
        return in.read((char*)buf,size).good();
    }
    void seek(size_t pos)
    {
        in.seekg(pos,std::ios::beg);
    }
    void seek_end(int pos)
    {
        in.seekg(pos,std::ios::end);
    }
    size_t tell(void)
    {
        return in.tellg();
    }
    void clear(void)
    {
        in.clear();
    }
    size_t cur_size(void)
    {
        return in.tellg();
    }
    size_t size(void)
    {
        return size_;
    }
    void flush(void) const
    {
        ;
    }
    bool eof(void) const
    {
        return in.eof();
    }
    bool good(void) const
    {
        return in.good();
    }
    operator bool() const	{return in.good();}
    bool operator!() const	{return !in.good();}
};

class std_ostream{
    std::ofstream out;
public:
    template<typename char_type>
    bool open(const char_type* file_name)
    {
        out.open(file_name,std::ios::binary);
        return out.good();
    }
    bool write(const void* buf,size_t size)
    {
        return out.write((const char*)buf,size).good();
    }
    void close(void)
    {
        out.close();
    }
    operator bool() const	{return out.good();}
    bool operator!() const	{return !out.good();}
};
}
}





#endif//INTERFACE_HPP
