#ifndef NRRD_HPP
#define NRRD_HPP
#include <map>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "../numerical/basic_op.hpp"
#include "../numerical/matrix.hpp"
#include "../utility/shape.hpp"

namespace tipl
{



namespace io
{

class nrrd
{
    std::map<std::string,std::string> values;
    tipl::matrix<4,4,float> T;
    tipl::vector<3> vs;
    tipl::shape<3> size;
    std::string data_file;
public:
    std::string error_msg;
private:
    bool read_v3(std::istream& in,float& vx,float& vy,float& vz)   // read (x,y,z)
    {
        std::string value;
        in >> value;
        auto sep1 = value.find_first_of(',');
        auto sep2 = value.find_last_of(',');
        if(sep1 == std::string::npos ||
           sep1 == std::string::npos)
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
    template<typename as_type,typename T>
    bool read_as_type(T& I)
    {
        if(!std::filesystem::exists(data_file))
        {
            error_msg = "data file not found";
            return false;
        }
        std::ifstream in(data_file.c_str(),std::ios::binary);
        tipl::image<3,as_type> buf(size);
        if(!in.read(reinterpret_cast<char*>(&buf[0]),buf.size()*sizeof(as_type)))
        {
            error_msg = "error reading data file";
            return false;
        }
        if constexpr (std::is_same<as_type,typename T::value_type>::value)
            buf.swap(I);
        else
            I = buf;
        return true;
    }
    template<typename T>
    bool read_image(T& I)
    {
        if(values["type"] == "float")
            return read_as_type<float>(I);
        if(values["type"] == "double")
            return read_as_type<double>(I);
        if(values["type"] == "int")
            return read_as_type<int32_t>(I);
        if(values["type"] == "unsigned int")
            return read_as_type<uint32_t>(I);
        if(values["type"] == "short")
            return read_as_type<int16_t>(I);
        if(values["type"] == "unsigned short")
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
        return load_from_file(file_name.c_str());
    }
    template<typename char_type>
    bool load_from_file(const char_type* file_name)
    {
        std::ifstream in(file_name);
        if(!in)
        {
            error_msg = "cannot open file";
            return false;
        }
        std::string line;
        if(!std::getline(in,line) || line.substr(0,7) != "NRRD000")
        {
            error_msg = "invalid nrrd file format";
            return false;
        }
        T.identity();
        while(std::getline(in,line))
        {
            auto sep = line.find(':');
            if(sep == std::string::npos || line.front() == '#')
                continue;
            std::string name = line.substr(0,sep);
            while(line[sep+1] == ' ' && sep+1 < line.length())
                ++sep;
            std::istringstream in2(values[name] = line.substr(sep+1));
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
                in2 >> size[0] >> size[1] >> size[2];
            if(name == "data file")
            {
                in2 >> data_file;
                data_file = std::filesystem::path(file_name).parent_path().string() + "/" + data_file;
            }
        }
        if(!size.size())
        {
            error_msg = "invalid nrrd header size zero";
            return false;
        }
        if(!std::filesystem::exists(data_file))
        {
            error_msg = "data file not found";
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
        if(values["space"].find("right") != std::string::npos)
            tipl::flip_x(out);
        if(values["space"].find("anterior") != std::string::npos)
            tipl::flip_y(out);
        if(values["space"].find("inferior") != std::string::npos)
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
