#ifndef AVI_HPP
#define AVI_HPP
#include <iostream>
#include <fstream>
#include <vector>
#include "bitmap.hpp"

namespace image {


namespace io {

struct fourcc {
    union {
        unsigned int value;
        char cc[4];
    };
    fourcc(void):value(0) {}
    fourcc(const char* cc_)
    {
        cc[0] = cc_[0];cc[1] = cc_[1];cc[2] = cc_[2];cc[3] = cc_[3];
    }
    fourcc(const fourcc& rhs):value(rhs.value) {}
    const fourcc& operator=(const fourcc& rhs)
    {
        value = rhs.value;
        return *this;
    }
    const fourcc& operator=(const char* rhs)
    {
        cc[0] = rhs[0];cc[1] = rhs[1];cc[2] = rhs[2];cc[3] = rhs[3];
        return *this;
    }
};

struct avi_header_t
{
    fourcc fcc;
    unsigned int   cb;
    unsigned int   dwMicroSecPerFrame;
    unsigned int   dwMaxBytesPerSec;
    unsigned int   dwPaddingGranularity;
    unsigned int   dwFlags;
    unsigned int   dwTotalFrames;
    unsigned int   dwInitialFrames;
    unsigned int   dwStreams;
    unsigned int   dwSuggestedBufferSize;
    unsigned int   dwWidth;
    unsigned int   dwHeight;
    unsigned int   dwReserved[4];
public:
    avi_header_t(void)
    {
        memset(this,0,sizeof(*this));
        fcc = "avih";
        cb = sizeof(avi_header_t)-8;
        dwFlags = 0x10;
        dwStreams = 1; // no audio supported
    }
};

struct avi_stream_header_t
{
    fourcc fcc;
    unsigned int  cb;
    fourcc fccType;
    fourcc fccHandler;
    unsigned int  dwFlags;
    unsigned short   wPriority;
    unsigned short   wLanguage;
    unsigned int  dwInitialFrames;
    unsigned int  dwScale;
    unsigned int  dwRate;
    unsigned int  dwStart;
    unsigned int  dwLength;
    unsigned int  dwSuggestedBufferSize;
    unsigned int  dwQuality;
    unsigned int  dwSampleSize;
    struct {
        short int left;
        short int top;
        short int right;
        short int bottom;
    } rcFrame;
public:
    avi_stream_header_t(void)
    {
        memset(this,0,sizeof(*this));
        fcc = "strh";
        fccType = "vids";
        cb = sizeof(avi_stream_header_t)-8;
        dwScale = 1;
    }
};

struct avi_stream_format_t
{
    fourcc label;
    unsigned int headersize;
    image::io::bitmap_info_header bh;
public:
    avi_stream_format_t(void)
    {
        memset(this,0,sizeof(*this));
        label = "strf";
        headersize = sizeof(avi_stream_format_t)-8;
        bh.biSize = sizeof(bitmap_info_header);
        bh.biPlanes = 1;
        bh.biBitCount = 24;
    }
};



struct riff_header {
    unsigned int pos;
    std::ofstream *out;
    riff_header(const char* fourcc,std::ofstream *out_):out(out_)
    {
        out->write((const char*)fourcc,4);
        pos = (unsigned int)out->tellp();
        unsigned int dummy = 0;
        out->write((const char*)&dummy,4);
    }

    ~riff_header(void)
    {
        unsigned int cur_pos = (unsigned int)out->tellp();
        out->seekp(pos);
        unsigned int size = cur_pos-pos-4;
        out->write((const char*)&size,4);
        out->seekp(cur_pos);
    }
};

class avi {
    std::auto_ptr<std::ofstream> out;
    std::vector<unsigned int> offsets;
    std::vector<riff_header> riff;
    void write(unsigned int value) {
        out->write((const char*)&value,4);
    }
    void write(const char* cc) {
        out->write(cc,4);
    }
private:
    unsigned int frame_count;
    unsigned int number_of_frames_pos;
    unsigned int data_length_pos;
public:
    avi(void):out(0),frame_count(0)
    {
    }
    bool open(const char *filename, unsigned int width, unsigned int height,
              fourcc codec, unsigned int fps)
    {
        out.reset(new std::ofstream(filename,std::ios::binary));
        if(!out->good())
            return false;
        avi_header_t ah;
        avi_stream_header_t sh;
        avi_stream_format_t sf;

        /* set avi header */
        unsigned int frame_size = width*height*3;
        ah.dwMicroSecPerFrame= 1000000 / fps;
        ah.dwMaxBytesPerSec = frame_size;
        ah.dwWidth = width;
        ah.dwHeight = height;
        ah.dwSuggestedBufferSize = frame_size;
        sh.fccHandler = codec;
        sh.dwRate = fps;
        sh.dwSuggestedBufferSize = frame_size;
        sf.bh.biWidth = width;
        sf.bh.biHeight = height;
        sf.bh.biCompression = codec.value;
        sf.bh.biSizeImage = frame_size;


        riff.push_back(riff_header("RIFF",out.get()));
        write("AVI ");
        riff.push_back(riff_header("LIST",out.get()));
        write("hdrl");
        number_of_frames_pos = (unsigned int)out->tellp()+24; // the number of frame will be updated at close
        out->write((const char*)&ah,sizeof(ah));
        riff.push_back(riff_header("LIST",out.get()));
        write("strl");
        data_length_pos = (unsigned int)out->tellp()+40; // the data length will be updated at close
        out->write((const char*)&sh,sizeof(sh));
        out->write((const char*)&sf,sizeof(sf));
        riff.pop_back(); // "LIST"
        riff.pop_back(); // "LIST"
        riff.push_back(riff_header("LIST",out.get()));
        write("movi");
        return true;
    }
    void add_frame(unsigned char *buffer, unsigned int len, bool compressed)
    {
        if(!buffer)
            return;
        ++frame_count;
        unsigned int pad;
        pad = len % 4;
        if (pad > 0)
            pad = 4 - pad;
        write(compressed ? "00dc" : "00db");
        offsets.push_back(len + pad);
        write(offsets.back());
        out->write((const char*)buffer,len);
        unsigned int dummy = 0;
        out->write((const char*)&dummy,pad);
        return;
    }
    void close(void)
    {
        riff.pop_back(); // "LIST"

        // index
        riff.push_back(riff_header("idx1",out.get()));
        for (unsigned int index = 0,offset = 4; index < offsets.size(); index++)
        {
            write("00dc");
            write(0x10);
            write(offset);
            write(offsets[index]);
            offset += offsets[index] + 8;
        }
        riff.pop_back();//"idx1"

        // update frame count
        {
            unsigned int cur_pos = (unsigned int)out->tellp();
            out->seekp(number_of_frames_pos);
            write(frame_count);
            out->seekp(data_length_pos);
            write(frame_count);
            out->seekp(cur_pos);
        }

        riff.pop_back();
    }
};

}// io
}// image
#endif /* AVI_HPP*/

