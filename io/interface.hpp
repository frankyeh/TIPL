#ifndef IMAGE_IO_INTERFACE_HPP
#define IMAGE_IO_INTERFACE_HPP
#include <fstream>


namespace tipl
{

namespace io
{

template<typename T> struct is_tuple : std::false_type {};
template<typename... Ts> struct is_tuple<std::tuple<Ts...>> : std::true_type {};



class std_istream{
    size_t size_;
    std::ifstream in;
public:
    std_istream(void):size_(0){}
    bool open(const std::filesystem::path& file_name)
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
    bool open(const std::filesystem::path& file_name)
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
