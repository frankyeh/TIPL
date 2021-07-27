#ifndef AVI_HPP
#define AVI_HPP
#include <iostream>
#include <fstream>
#include <vector>
#include "bitmap.hpp"

namespace tipl {


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
};

struct avi_header_t
{
    fourcc fcc;
    unsigned int   cb;
    unsigned int   dwMicroSecPerFrame = 0;
    unsigned int   dwMaxBytesPerSec = 0;
    unsigned int   dwPaddingGranularity = 0;
    unsigned int   dwFlags = 0x10;
    unsigned int   dwTotalFrames = 0;
    unsigned int   dwInitialFrames = 0;
    unsigned int   dwStreams = 1; // no audio supported
    unsigned int   dwSuggestedBufferSize = 0;
    unsigned int   dwWidth = 0;
    unsigned int   dwHeight = 0;
    unsigned int   dwReserved[4] = {0,0,0,0};
public:
    avi_header_t(void):fcc("avih"),cb(sizeof(avi_header_t)-8)
    { 
    }
};

struct avi_stream_header_t
{
    fourcc fcc;
    unsigned int  cb = 0;
    fourcc fccType;
    fourcc fccHandler;
    unsigned int  dwFlags = 0;
    unsigned short   wPriority = 0;
    unsigned short   wLanguage = 0;
    unsigned int  dwInitialFrames = 0;
    unsigned int  dwScale = 1;
    unsigned int  dwRate = 0;
    unsigned int  dwStart = 0;
    unsigned int  dwLength = 0;
    unsigned int  dwSuggestedBufferSize = 0;
    unsigned int  dwQuality = 0;
    unsigned int  dwSampleSize = 0;
    struct {
        short int left = 0;
        short int top = 0;
        short int right = 0;
        short int bottom = 0;
    } rcFrame;
public:
    avi_stream_header_t(void):fcc("strh"),cb(sizeof(avi_stream_header_t)-8),fccType("vids")
    {
    }
};

struct avi_stream_format_t
{
    fourcc label;
    unsigned int headersize = 0;
    tipl::io::bitmap_info_header bh;
public:
    avi_stream_format_t(void)
    {
        label = "strf";
        headersize = sizeof(avi_stream_format_t)-8;
        bh.biSize = sizeof(bitmap_info_header);
        bh.biPlanes = 1;
        bh.biBitCount = 24;
    }
};



struct riff_header {
    unsigned int pos = 0;
    std::ofstream *out = nullptr;
    riff_header(const char* fourcc,std::ofstream *out_):out(out_)
    {
        out->write(fourcc,4);
        pos = uint32_t(out->tellp());
        unsigned int dummy = 0;
        out->write(reinterpret_cast<const char*>(&dummy),4);
    }

    ~riff_header(void)
    {
        unsigned int cur_pos = uint32_t(out->tellp());
        out->seekp(pos);
        unsigned int size = cur_pos-pos-4;
        out->write(reinterpret_cast<const char*>(&size),4);
        out->seekp(cur_pos);
    }
};

class avi {
    std::shared_ptr<std::ofstream> out;
    std::vector<unsigned int> offsets;
    std::vector<riff_header> riff;
    void write(unsigned int value) {
        out->write(reinterpret_cast<const char*>(&value),4);
    }
    void write(const char* cc) {
        out->write(cc,4);
    }
private:
    unsigned int frame_count = 0;
    unsigned int number_of_frames_pos = 0;
    unsigned int data_length_pos = 0;
public:
    avi(void){}
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
        number_of_frames_pos = uint32_t(out->tellp())+24; // the number of frame will be updated at close
        out->write(reinterpret_cast<const char*>(&ah),sizeof(ah));
        riff.push_back(riff_header("LIST",out.get()));
        write("strl");
        data_length_pos = uint32_t(out->tellp())+40; // the data length will be updated at close
        out->write(reinterpret_cast<const char*>(&sh),sizeof(sh));
        out->write(reinterpret_cast<const char*>(&sf),sizeof(sf));
        riff.pop_back(); // "LIST"
        riff.pop_back(); // "LIST"
        riff.push_back(riff_header("LIST",out.get()));
        write("movi");
        return true;
    }
    void add_frame(const char *buffer, unsigned int len, bool compressed)
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
        out->write(buffer,len);
        char dummy[4] = {0,0,0,0};
        out->write(dummy,pad);
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
            auto cur_pos = out->tellp();
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

