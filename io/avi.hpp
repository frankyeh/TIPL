#ifndef AVI_HPP
#define AVI_HPP
#include <iostream>
#include <fstream>
#include <vector>

namespace image{


namespace io{

struct fourcc{
    union{
    unsigned int value;
    char cc[4];
    };
    fourcc(void):value(0){}
    fourcc(const char* cc_)
    {
        cc[0] = cc_[0];cc[1] = cc_[1];cc[2] = cc_[2];cc[3] = cc_[3];
    }
    fourcc(const fourcc& rhs):value(rhs.value){}
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
    fourcc label;
    unsigned int headersize;
    unsigned int time_delay;
    unsigned int data_rate;
    unsigned int reserved;
    unsigned int flags;
    unsigned int number_of_frames;
    unsigned int initial_frames;
    unsigned int data_streams;
    unsigned int buffer_size;
    unsigned int width;
    unsigned int height;
    unsigned int time_scale;
    unsigned int playback_data_rate;
    unsigned int starting_time;
    unsigned int data_length;
public:
    avi_header_t(void)
    {
        memset(this,0,sizeof(*this));
        label = "avih";
        headersize = sizeof(avi_header_t)-8;
        flags = 0x10;
        data_streams = 1; // no audio supported
    }

    void write(std::ofstream *out) const
    {
        out->write((char*)this,sizeof(avi_header_t));
    }
};

struct avi_stream_header_t
{
    fourcc label;
    unsigned int headersize;
    fourcc data_type;
    fourcc codec;
    unsigned int flags;
    unsigned int priority;
    unsigned int initial_frames;
    unsigned int time_scale;
    unsigned int data_rate;
    unsigned int start_time;
    unsigned int data_length;
    unsigned int buffer_size;
    unsigned int video_quality;
    unsigned int sample_size;
    unsigned int reserved[2];
public:
    avi_stream_header_t(void)
    {
        memset(this,0,sizeof(*this));
        label = "strh";
        data_type = "vids";
        headersize = sizeof(avi_stream_header_t)-8;
        time_scale = 1;
    }
public:
    void write(std::ofstream *out) const
    {
        out->write((char*)this,sizeof(avi_stream_header_t));
    }
};

struct avi_stream_format_t
{
    fourcc label;
    unsigned int headersize;
    unsigned int header_size;
    unsigned int width;
    unsigned int height;
    unsigned short int num_planes;
    unsigned short int bits_per_pixel;
    fourcc compression_type;
    unsigned int image_size;
    unsigned int x_pels_per_meter;
    unsigned int y_pels_per_meter;
    unsigned int colors_used;
    unsigned int colors_important;
public:
    avi_stream_format_t(void)
    {
        memset(this,0,sizeof(*this));
        label = "strf";
        headersize = sizeof(avi_stream_format_t)-8;
        header_size = 40;
        num_planes = 1;
        bits_per_pixel = 24;
    }
public:
    void write(std::ofstream *out) const
    {
        out->write((char*)this,sizeof(avi_stream_format_t));
    }
};



struct riff_header{
    unsigned int pos;
    std::ofstream *out;
    riff_header(const char* fourcc,std::ofstream *out_):out(out_)
    {
        out->write((const char*)fourcc,4);
        pos = out->tellp();
        unsigned int dummy = 0;
        out->write((const char*)&dummy,4);
    }

    ~riff_header(void)
    {
        unsigned int cur_pos = out->tellp();
        out->seekp(pos);
        unsigned int size = cur_pos-pos-4;
        out->write((const char*)&size,4);
        out->seekp(cur_pos);
    }
};

class avi{
    std::auto_ptr<std::ofstream> out;
    long marker;
    std::vector<unsigned int> offsets;
    std::vector<riff_header> riff;
    void write(unsigned int value){out->write((const char*)&value,4);}
    void write(const char* cc){out->write(cc,4);}
private:
    unsigned int frame_count;
    unsigned int number_of_frames_pos;
    unsigned int data_length_pos;
public:
    avi(void):out(0),marker(0),frame_count(0)
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
        ah.time_delay= 1000000 / fps;
        ah.data_rate = frame_size;
        ah.width = width;
        ah.height = height;
        ah.buffer_size = frame_size;
        sh.codec = codec;
        sh.data_rate = fps;
        sh.buffer_size = frame_size;
        sf.width = width;
        sf.height = height;
        sf.compression_type = codec;
        sf.image_size = frame_size;


        riff.push_back(riff_header("RIFF",out.get()));
        write("AVI ");
        riff.push_back(riff_header("LIST",out.get()));
        write("hdrl");
        number_of_frames_pos = (unsigned int)out->tellp()+24; // the number of frame will be updated at close
        ah.write(out.get());
        riff.push_back(riff_header("LIST",out.get()));
        write("strl");
        data_length_pos = (unsigned int)out->tellp()+40; // the data length will be updated at close
        sh.write(out.get());
        sf.write(out.get());
        riff.pop_back(); // "LIST"
        riff.pop_back(); // "LIST"
        riff.push_back(riff_header("LIST",out.get()));
        write("movi");
        return true;
    }
    void add_frame(unsigned char *buffer, unsigned int len)
    {
        if(!buffer)
            return;
        ++frame_count;
        unsigned int pad;
        pad = len % 4;
        if (pad > 0)
            pad = 4 - pad;
        write("00dc");
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
            unsigned int cur_pos = out->tellp();
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

