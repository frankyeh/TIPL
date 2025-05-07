#ifndef BITMAP_IO_HPP
#define BITMAP_IO_HPP
#include <vector>
#include <stdexcept>
#include <fstream>
namespace tipl
{

namespace io
{

struct bitmap_file_header
{
    unsigned int bfSize = 0;
    unsigned int bfOffBits = 0;
    bool read(std::istream& in)
    {
        unsigned short bfType;
        in.read((char*)&bfType,2);
        if (bfType !=  0x4D42) // BM
            return false;
        in.read((char*)&bfSize,4);
        unsigned int dummy;
        in.read((char*)&dummy,4);
        in.read((char*)&bfOffBits,4);
        return !(!in);
    }
    bool write(std::ostream& out) const
    {
        unsigned short bfType = 0x4D42;
        out.write((const char*)&bfType,2);
        out.write((const char*)&bfSize,4);
        unsigned int dummy = 0;
        out.write((const char*)&dummy,4);
        dummy = 54;
        out.write((const char*)&dummy,4);//bfOffBits
        return !(!out);
    }
};


struct bitmap_info_header
{
    unsigned int biSize;
    unsigned int biWidth;
    unsigned int biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int biCompression;
    unsigned int biSizeImage;
    unsigned int biXPelsPerMeter;
    unsigned int biYPelsPerMeter;
    unsigned int biClrUsed;
    unsigned int biClrImportant;
};
class bitmap
{
private:
    bitmap_file_header bmfh;
    bitmap_info_header bmih;
    std::vector<unsigned char> data;
public:
    bitmap(void)
    {
        std::fill(reinterpret_cast<char*>(&bmih),reinterpret_cast<char*>(&bmih)+sizeof(bmih),0);
        bmih.biSize = sizeof(bitmap_info_header);
    }
    bitmap(const std::string& file_name)
    {
        if (!load_from_file(file_name))
            throw std::runtime_error("failed to open bitmap file");
    }
    bool save_to_file(const std::string& file_name) const
    {
        std::ofstream out(file_name,std::ios::binary);
        return save_to_stream(out);
    }
    template<typename stream_type>
    bool save_to_stream(stream_type& out) const
    {
        if (!bmfh.write(out))
            return false;
        out.write(reinterpret_cast<const char*>(&bmih),sizeof(bitmap_info_header));
        out.write(reinterpret_cast<const char*>(&*data.begin()),int64_t(data.size()));
        return true;
    }
    bool load_from_file(const std::string& file_name)
    {
        std::ifstream in(file_name,std::ios::binary);
        if (!in || !bmfh.read(in))
            return false;
        in.read(reinterpret_cast<char*>(&bmih),sizeof(bitmap_info_header));
        if (!in || bmih.biWidth <= 0 || bmih.biHeight <= 0 || bmih.biCompression != 0)
            return false;
        try
        {
            data.resize(bmih.biSizeImage);
        }
        catch (...)
        {
            return false;
        }
        in.seekg(bmfh.bfOffBits,std::ios::beg);
        in.read(reinterpret_cast<char*>(&*data.begin()),int64_t(data.size()));
        if (!in)
            return false;
        return true;
    }
    template<typename image_type>
    void load_from_image(const image_type& image)
    {
        bmih.biBitCount = 32;
        bmih.biPlanes = 1;
        bmih.biCompression = 0;//BI_RGB;
        bmih.biWidth = image.width();
        bmih.biHeight = image.size()/image.width(); // this even fits for 3D images
        bmih.biSizeImage = image.size() << 2;
        bmfh.bfSize = 54 + bmih.biSizeImage;
        data.resize(bmih.biSizeImage);

        typename image_type::const_iterator iter = image.begin();
        typename image_type::const_iterator end = image.end();
        int line_width = image.width();
        tipl::rgb* out_line = (rgb*)&*data.begin() + image.size() - line_width;
        for (;iter != end;iter += line_width,out_line -= line_width)
            std::copy_n(iter,line_width,out_line);
    }

    template<typename image_type>
    void save_to_image(image_type& image) const
    {
        typedef typename image_type::value_type pixel_type;
        tipl::shape<image_type::dimension> geo;
        std::fill(geo.begin(),geo.end(),1);
        geo[0] = bmih.biWidth;
        geo[1] = bmih.biHeight;
        image.resize(geo);
        switch (bmih.biBitCount)
        {
        case 8:
            std::copy_n(reinterpret_cast<const unsigned char*>(data.data()),image.size(),image.begin());
            break;
        case 32:
            std::copy_n(reinterpret_cast<const tipl::rgb*>(data.data()),image.size(),image.begin());
            break;
        case 24:
        {
            unsigned int width = bmih.biWidth;
            const unsigned char* iter = &*data.begin();
            unsigned char r,g,b;
            pixel_type* beg = &*image.begin();
            pixel_type* line = beg + image.size()-width;
            pixel_type* iter2 = line;
            unsigned int padding = (4-((bmih.biWidth*3) & 3)) & 3;

            for (unsigned int x = 0;line >= beg;++iter2)
            {
                b = *iter;
                ++iter;
                g = *iter;
                ++iter;
                r = *iter;
                ++iter;
                ++x;
                if (x >= width)
                {
                    x = 0;
                    iter += padding;
                    line -= width;
                    iter2 = line;
                    continue;
                }
                *iter2 = rgb(r,g,b);
            }
        }
        break;
        default:
            throw std::runtime_error("unsupported bit count");
            break;
        }
    }

    template<typename image_type>
    const bitmap& operator>>(image_type& source) const
    {
        save_to_image(source);
        return *this;
    }
    template<typename image_type>
    bitmap& operator<<(const image_type& source)
    {
        load_from_image(source);
        return *this;
    }
};


}
}
#endif//BITMAP_HPP
