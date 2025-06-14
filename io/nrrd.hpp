#ifndef NRRD_HPP
#define NRRD_HPP
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include "interface.hpp"
#include "../numerical/basic_op.hpp"
#include "../numerical/matrix.hpp"
#include "../utility/shape.hpp"
#include "../po.hpp"
#include "gz_stream.hpp"

namespace tipl
{



namespace io
{

template<typename gz_stream_type = void>
class nrrd
{
public:
    std::unordered_map<std::string,std::string> values;
    tipl::matrix<4,4,float> T;
    tipl::vector<3> vs;
    tipl::shape<3> size;
    int dim4 = 1;
    std::string data_file;
    std::string error_msg;
public:
    bool file_series = false;
    size_t from = 0,to = 0,step = 1;
    size_t header_size = 0;
private:
    bool read_v3(std::istream& in,float& vx,float& vy,float& vz)   // read (x,y,z)
    {
        std::string value;
        in >> value;
        auto sep1 = value.find_first_of(',');
        auto sep2 = value.find_last_of(',');
        if(sep1 == std::string::npos ||
           sep2 == std::string::npos)
            return false;
        try{
            vx = std::stof(std::string(value.begin()+1,value.begin()+sep1));
            vy = std::stof(std::string(value.begin()+sep1+1,value.begin()+sep2));
            vz = std::stof(std::string(value.begin()+sep2+1,value.end()));
        }
        catch(...)
        {
            error_msg = "error parsing header value: ";
            error_msg += value;
            return false;
        }
        return true;
    }
    template<typename T>
    bool read_buffer(T& I)
    {       
        if(file_series)
        {
            for(size_t index = from,z = 0;index <= to;index += step)
            {
                std::string file_name;
                file_name.resize(data_file.length()+2);
                sprintf(&file_name[0],data_file.c_str(),index);
                if(!std::filesystem::exists(file_name))
                {
                    error_msg = "file not found ";
                    error_msg += file_name;
                    return false;
                }
                std::ifstream in(file_name,std::ios::binary);
                if(!in.read(reinterpret_cast<char*>(I.data() + I.plane_size()*z),I.plane_size()*sizeof(typename T::value_type)))
                {
                    error_msg = "error reading image data ";
                    error_msg += file_name;
                    return false;
                }
            }
        }
        else
        {
            if(!std::filesystem::exists(data_file))
            {
                error_msg = "data file not found";
                return false;
            }
            if(values["encoding"] == "raw")
            {
                std::ifstream in(data_file,std::ios::binary);
                in.seekg(header_size,std::ios_base::beg);
                if(!in.read(reinterpret_cast<char*>(I.data()),I.size()*sizeof(typename T::value_type)))
                {
                    error_msg = "error reading image data " + data_file;
                    return false;
                }
                goto check_endian;
            }
            if(values["encoding"] == "gzip")
            {
                if constexpr(!std::is_void_v<gz_stream_type>)
                {
                    std::ifstream in(data_file,std::ios::binary);
                    in.seekg(0,std::ios_base::end);
                    std::vector<unsigned char> buf(size_t(in.tellg())-header_size);
                    in.seekg(header_size,std::ios_base::beg);
                    if(!in.read(reinterpret_cast<char*>(buf.data()),buf.size()))
                    {
                        error_msg = "error reading image data: " + data_file;
                        return false;
                    }
                    gz_stream_type istrm;
                    istrm.input(std::move(buf));
                    istrm.output(I.data(),I.size()*sizeof(typename T::value_type));
                    if(istrm.process() > 1) // != Z_OK(0) or Z_STREAM_END(1)
                    {
                        error_msg = "corrupted gzip encoding: " + data_file;
                        return false;
                    }
                    goto check_endian;
                }
            }
            error_msg = "unsupported encoding type: " + values["encoding"];
            return false;
        }
        check_endian:
        if(values["endian"] == "big")
        {
            for(size_t i = 0;i < I.size();++i)
                change_endian(I[i]);
        }
        return true;
    }
    template<typename as_type,typename T>
    bool read_as_type(T& I)
    {
        try{
            tipl::image<T::dimension,as_type> buf;
            if constexpr (T::dimension == 3)
                buf.resize(size);
            else
                buf.resize(tipl::shape<4>(size[0],size[1],size[2],dim4));
            if(!read_buffer(buf))
                return false;
            if constexpr (std::is_same<std::remove_reference_t<decltype(I)>,std::remove_reference_t<decltype(buf)>>::value)
                I.swap(buf);
            else
                I = buf;
        }
        catch(const std::bad_alloc&)
        {
            error_msg = "insufficient memory";
            return false;
        }

        return true;
    }
    template<typename T>
    bool read_image(T& I)
    {
        if(values["type"] == "float")
            return read_as_type<float>(I);
        if(values["type"] == "double")
            return read_as_type<double>(I);
        if(values["type"] == "int" || values["type"] == "unsigned int")
            return read_as_type<uint32_t>(I);
        if(values["type"] == "short" || values["type"] == "unsigned short" || values["type"] == "int16")
            return read_as_type<uint16_t>(I);
        if(values["type"] == "uchar")
            return read_as_type<uint8_t>(I);
        error_msg = "unsupported type:";
        error_msg += values["type"];
        return false;
    }
public:
    bool load_from_file(const std::string& file_name)
    {
        std::ifstream in(file_name,std::ios::binary);
        if(!in)
        {
            error_msg = "cannot open file";
            return false;
        }
        std::string line;
        if(!std::getline(in,line) || !tipl::begins_with(line,"NRRD000"))
        {
            error_msg = "invalid nrrd file format";
            return false;
        }
        T.identity();
        data_file = file_name;
        while(std::getline(in,line) && !line.empty())
        {
            auto sep = line.find(':');
            if(sep == std::string::npos || line.front() == '#')
                continue;
            std::string name = line.substr(0,sep);
            while((line[sep+1] == ' ' || line[sep+1] == '=' ) && sep+1 < line.length())
                ++sep;
            auto v = line.substr(sep+1);
            std::istringstream in2(values[name] = v);
            if(name == "space directions")
            {
                read_v3(in2,T[0],T[1],T[2]);
                read_v3(in2,T[4],T[5],T[6]);
                read_v3(in2,T[8],T[9],T[10]);
                vs[0] = std::abs(T[0]);
                vs[1] = std::abs(T[5]);
                vs[2] = std::abs(T[10]);
            }
            if(name == "space origin")
                read_v3(in2,T[3],T[7],T[11]);
            if(name == "sizes")
                in2 >> size[0] >> size[1] >> size[2] >> dim4;
            if(name == "data file")
            {
                in2 >> data_file;
                if(data_file.find("%") != std::string::npos)
                {
                    file_series = true;
                    in2 >> from >> to >> step;
                }
                data_file = std::filesystem::path(file_name).parent_path().u8string() + "/" + data_file;
            }
        }
        if(data_file == file_name)
            header_size = in.tellg();
        if(!size.size())
        {
            error_msg = "invalid nrrd header size zero";
            return false;
        }
        return true;
    }
    const tipl::shape<3>& shape(void) const{return size;}
    template<typename image_type>
    bool save_to_image(image_type& out)
    {
        if(!read_image(out))
            return false;
        auto space_text = values["space"];
        if(tipl::contains(space_text,"right"))
            tipl::flip_x(out);
        if(tipl::contains(space_text,"anterior"))
            tipl::flip_y(out);
        if(tipl::contains(space_text,"inferior"))
            tipl::flip_z(out);
        return true;
    }

    template<typename image_type>
    bool operator>>(image_type& source)
    {
        return save_to_image(source);
    }

    void get_voxel_size(tipl::vector<3>& vs_) const
    {
        vs_ = vs;
    }

    void get_image_transformation(tipl::matrix<4,4,float>& T_) const
    {
        T_ = T;
    }
};

}//io
}//tipl

#endif//NRRD_HPP

#ifdef TIPL_GZ_STREAM_HPP
namespace tipl{namespace io{
typedef nrrd<tipl::io::inflate_stream> gz_nrrd;
}}
#endif

