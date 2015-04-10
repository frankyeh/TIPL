#ifndef IMAGE_IO_INTERFACE_HPP
#define IMAGE_IO_INTERFACE_HPP
#include <fstream>

namespace image
{

namespace io
{

class std_istream{
    std::ifstream in;
public:
    template<typename char_type>
    bool open(const char_type* file_name)
    {
        in.open(file_name,std::ios::binary);
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
    void write(const void* buf,size_t size)
    {
        out.write((const char*)buf,size);
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
