#ifndef TIFF_HPP
#define TIFF_HPP
#include <fstream>
#include <vector>
#include <map>
/*
1 = BYTE 8-bit unsigned integer.
2 = ASCII 8-bit byte that contains a 7-bit ASCII code; the last byte must be NUL (binary zero).
3 = SHORT 16-bit (2-byte) unsigned integer.
4 = LONG 32-bit (4-byte) unsigned integer.
5 = RATIONAL Two LONGs: the first represents the numerator of a
fraction; the second, the denominator.
6 = SBYTE An 8-bit signed (twos-complement) integer.
7 = UNDEFINED An 8-bit byte that may contain anything, depending on
the definition of the field.
8 = SHORT A 16-bit (2-byte) signed (twos-complement) integer.
9 = SLONG A 32-bit (4-byte) signed (twos-complement) integer.
10 = SRATIONAL Two SLONG¡¦s: the first represents the numerator of a
fraction, the second the denominator.
11 = FLOAT Single precision (4-byte) IEEE format.
12 = DOUBLE Double precision (8-byte) IEEE format.
*/
namespace image{

namespace io{


const unsigned int sizeof_type[13]={0,1,1,2,4,8,1,1,2,4,8,4,8};
struct ifd
{
    /*
    	Bytes 0-1 The Tag that identifies the field.
    	Bytes 2-3 The field Type.
    	Bytes 4-7 The number of values, Count of the indicated Type.
    	Bytes 8-11 The Value Offset, the file offset (in bytes) of the Value for the field.
    	The Value is expected to begin on a word boundary; the corresponding
    	Value Offset will thus be an even number. This file offset may
    	point anywhere in the file, even after the image data.
    	*/
    unsigned short tag;
    unsigned short type;
    unsigned int count;
    unsigned int offset;
    bool storage_in_offset;
    void read(std::istream& in)
    {
        in.read((char*)&tag,sizeof(tag));
        in.read((char*)&type,sizeof(type));
        in.read((char*)&count,sizeof(count));
        in.read((char*)&offset,sizeof(offset));
    }
    template<typename output_type>
    void read(std::istream& in,std::vector<output_type>& out) const
    {
        unsigned int buffer_size = count*sizeof_type[type];
        std::vector<unsigned char> buffer(buffer_size);
        if (buffer_size <= 4)
            std::copy((const char*)&offset,((const char*)&offset)+buffer_size,buffer.begin());
        else
        {
            in.seekg(offset,std::ios::beg);
            in.read((char*)&*buffer.begin(),buffer.size());
        }
        out.resize(count);
        switch (type)
        {
        case 1://BYTE 8-bit unsigned integer.
        case 2://ASCII 8-bit byte that contains a 7-bit ASCII code; the last byte must be NUL (binary zero).
            std::copy(buffer.begin(),buffer.end(),out.begin());
            break;
        case 3://SHORT 16-bit (2-byte) unsigned integer.
            std::copy((const unsigned short*)&*buffer.begin(),(const unsigned short*)&*buffer.begin()+count,out.begin());
            break;
        case 4://LONG 32-bit (4-byte) unsigned integer.
        case 5://RATIONAL Two LONGs: the first represents the numerator of a fraction; the second, the denominator.
            std::copy((const unsigned int*)&*buffer.begin(),(const unsigned int*)&*buffer.begin()+count,out.begin());
            break;
        case 6://SBYTE An 8-bit signed (twos-complement) integer.
        case 7://UNDEFINED An 8-bit byte that may contain anything, depending on the definition of the field.
            std::copy((const char*)&*buffer.begin(),(const char*)&*buffer.begin()+count,out.begin());
            break;
        case 8://SHORT A 16-bit (2-byte) signed (twos-complement) integer.
            std::copy((const short*)&*buffer.begin(),(const short*)&*buffer.begin()+count,out.begin());
            break;
        case 9://SLONG A 32-bit (4-byte) signed (twos-complement) integer.
        case 10://SRATIONAL Two SLONG¡¦s: the first represents the numerator of afraction, the second the denominator.
            std::copy((const int*)&*buffer.begin(),(const int*)&*buffer.begin()+count,out.begin());
            break;
        case 11://FLOAT Single precision (4-byte) IEEE format.
            std::copy((const float*)&*buffer.begin(),(const float*)&*buffer.begin()+count,out.begin());
            break;
        case 12://DOUBLE Double precision (8-byte) IEEE format.
            std::copy((const double*)&*buffer.begin(),(const double*)&*buffer.begin()+count,out.begin());
            break;
        }
    }
};




class tiff
{
private:
    unsigned int width,height;
    unsigned int row_per_strip;
    std::vector<size_t> strip_offset;
    std::vector<size_t> strip_size;
    short photo,compression,unit;
private:
    std::map<unsigned short,ifd> entries;
    std::map<unsigned short,unsigned int> info;
public:
	template<typename char_type>
    bool load_from_file(const char_type* file_name)
    {
        std::ifstream in(file_name,std::ios::binary);
        //Bytes 0-1: The byte order used within the file. Legal values are:"II" (4949.H)
        //Bytes 2-3 An arbitrary but carefully chosen number (42) that further identifies the file as a TIFF file.
        {
            char ii_tag[4];
            in.read(ii_tag,4);
            if (ii_tag[0] != 'I' || ii_tag[1] != 'I' || ii_tag[2] != 42 || ii_tag[3] != 0)
                return false;
        }
        //Bytes 4-7 The offset (in bytes) of the first IFD. The directory may be at any location in the
        //file after the header but must begin on a word boundary. In particular, an Image
        //File Directory may follow the image data it describes. Readers must follow the
        //pointers wherever they may lead.
        {
            unsigned int offset;
            in.read((char*)&offset,4);
            in.seekg(offset,std::ios::beg);
            if (!in)
                return false;
        }

        // IFD
        {
            unsigned short number_entry;
            in.read((char*)&number_entry,sizeof(number_entry));
            for (unsigned int index = 0;index < number_entry;++index)
            {
                ifd ifd_buffer;
                ifd_buffer.read(in);
                switch (ifd_buffer.tag)
                {
                case TiffTag::ImageWidth:
                    width = ifd_buffer.offset;
                    break;
                case TiffTag::ImageLength:
                    height = ifd_buffer.offset;
                    break;
                case TiffTag::RowsPerStrip:
                    row_per_strip = ifd_buffer.offset;
                    break;
                case TiffTag::PhotometricInterpretation:
                    photo = ifd_buffer.offset;
                    break;
                case TiffTag::Compression:
                    compression = ifd_buffer.offset;
                    break;
                case TiffTag::ResolutionUnit:
                    unit = ifd_buffer.offset;
                    break;
                case TiffTag::StripOffsets:
                    ifd_buffer.read(in,strip_offset);
                    break;
                case TiffTag::StripByteCounts:
                    ifd_buffer.read(in,strip_size);
                    break;
                default:
                    if (ifd_buffer.count*sizeof_type[ifd_buffer.type] <= 4)
                        info[ifd_buffer.tag] = ifd_buffer.offset;
                    else
                        entries[ifd_buffer.tag] = ifd_buffer;
                }
            }
        }

    }

};



}
}


#endif