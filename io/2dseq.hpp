#ifndef BRUKER2DSEQ_HPP
#define BRUKER2DSEQ_HPP
#include <string>
#include <map>
#include <sstream>
#include <iterator>

namespace image
{



namespace io
{

class bruker_info
{
    std::map<std::string,std::string> info;
    /*
    ##$RECO_size=( 2 )
    128 128                 <--info
    */
private:
    void load_info(std::ifstream& in)
    {
        std::string line;
        info.clear();
        while(std::getline(in,line))
        {
            if(line.size() < 4 ||
                    line[0] != '#' ||
                    line[1] != '#' ||
                    line[2] != '$')
                continue;

            std::string::iterator sep = std::find(line.begin(),line.end(),'=');
            if(sep == line.end())
                continue;
            std::string name(line.begin()+3,sep);
            info[name] =
                std::string(sep+1,line.end());
            if(*(sep+1) == '(')
            {
                std::string accumulated_info;
                while(in && in.peek() != '#')
                {
		    std::getline(in,line);
		    if(line[0] == '$')
    			continue;
                    accumulated_info += line;
                    accumulated_info += " ";
                }
                accumulated_info.erase(std::remove(accumulated_info.begin(),accumulated_info.end(),'<'),accumulated_info.end());
                accumulated_info.erase(std::remove(accumulated_info.begin(),accumulated_info.end(),'>'),accumulated_info.end());
                info[name] = accumulated_info;
            }
        }
    }
public:
    template<typename char_type>
    bool load_from_file(const char_type* file_name)

    {
        std::ifstream info(file_name);
        if(!info)
            return false;
        load_info(info);
        return true;
    }
    const std::string& operator[](const std::string& tag)
    {
        return info[tag];
    }
};


class bruker_2dseq
{
    // the 2dseq data
    std::vector<float> data;

    // image dimension
    unsigned short dim[4];

    // spatial resolution
    float resolution[3];

    // pixel data type
    // 0: short
    // 1: int
    unsigned char data_type;
private:
    std::string tmp;
    std::wstring wtmp;

    bool check_name(const char* filename)
    {
        std::string str = filename;
        if(str.length() < 5)
            return false;
        std::string name(str.end()-5,str.end());
        if(name[0] != '2' || name[1] != 'd' || name[2] != 's' || name[3] != 'e' || name[4] != 'q')
            return false;
        return true;
    }
    bool check_name(const wchar_t* filename)
    {
        std::wstring str = filename;
        if(str.length() < 5)
            return false;
        std::wstring name(str.end()-5,str.end());
        if(name[0] != L'2' || name[1] != L'd' || name[2] != L's' || name[3] != L'e' || name[4] != L'q')
            return false;
        return true;
    }

    const char* load_d3proc(const char* filename)
    {
        std::string str = filename;
        tmp = std::string(str.begin(),str.end()-5);
        tmp += "d3proc";
        return tmp.c_str();
    }
    const wchar_t* load_d3proc(const wchar_t* filename)
    {
        std::wstring str = filename;
        wtmp = std::wstring(str.begin(),str.end()-5);
        wtmp += L"d3proc";
        return wtmp.c_str();
    }

    const char* load_reco(const char* filename)
    {
        std::string str = filename;
        tmp = std::string(str.begin(),str.end()-5);
        tmp += "reco";
        return tmp.c_str();
    }
    const wchar_t* load_reco(const wchar_t* filename)
    {
        std::wstring str = filename;
        wtmp = std::wstring(str.begin(),str.end()-5);
        wtmp += L"reco";
        return wtmp.c_str();
    }

    
public:
    

    template<typename char_type>
    bool load_from_file(const char_type* file_name)
    {
        if(!check_name(file_name))
            return false;
        bruker_info info1,info2;
        if(!info1.load_from_file(load_d3proc(file_name)))
            return false;

        // read image dimension
        unsigned int total_size = 0;
        {
            std::fill(dim,dim+4,1);
            std::istringstream(info1["IM_SIX"]) >> dim[0];
            std::istringstream(info1["IM_SIY"]) >> dim[1];
            std::istringstream(info1["IM_SIZ"]) >> dim[2];
            total_size = dim[0]*dim[1]*dim[2];
	    if(!total_size)
                return false;
        }

        // set default type to short
        data_type = 0;
        if(info1["DATTYPE"] == std::string("ip_int"))
            data_type = 1;

        // read 2dseq
        {
            std::vector<char> buffer;
            {
                switch(data_type)
                {
                case 0:
                    buffer.resize(total_size*2);
                    break;
                case 1:
                    buffer.resize(total_size*4);
                    break;
                }

                std::ifstream in(file_name,std::ios::binary);
                in.read((char*)&*buffer.begin(),buffer.size());
            }
            data.resize(total_size);
            switch (data_type)
            {
                case 0:
                    std::copy((short*)&buffer[0],(short*)&buffer[0]+total_size,data.begin());
                    break;
                case 1:
                    std::copy((int*)&buffer[0],(int*)&buffer[0]+total_size,data.begin());
                    break;
            }
        }

        if(info2.load_from_file(load_reco(file_name)))
        {
            {
                unsigned short reco_dim[3] = {0,0,0};
                std::istringstream(info2["RECO_size"]) >> reco_dim[0] >> reco_dim[1] >> reco_dim[2];
                if(reco_dim[2] && reco_dim[2] != dim[2] && (reco_dim[2] % dim[2] == 0))
                {
                    dim[3] = dim[2]/reco_dim[2];
                    dim[2] = reco_dim[2];
                }

                // get sptial resolution
                {
                    std::vector<float> fov_data; // in cm
                    std::istringstream fov_text(info2["RECO_fov"]);
                    std::copy(std::istream_iterator<float>(fov_text),
                              std::istream_iterator<float>(),
                              std::back_inserter(fov_data));
                    std::fill(resolution,resolution+3,0.0);
                    for(unsigned int index = 0;index < 3 && index < fov_data.size();++index)
                        resolution[index] = fov_data[index]*10.0/(float)dim[index]; // in mm
                }

            }
            if(info2["RECO_byte_order"] == std::string("bigEndian"))
            {
                switch (data_type)
                {
                case 0:
                    change_endian((short*)&data[0],total_size);
                    break;
                case 1:
                    change_endian((int*)&data[0],total_size);
                    break;
                }
            }

            // get slope
            std::vector<float> slopes;
            {
                std::istringstream slope_text_parser(info2["RECO_map_slope"]);
                std::copy(std::istream_iterator<double>(slope_text_parser),
                          std::istream_iterator<double>(),
                          std::back_inserter(slopes));
            }

            // correct slope
            if(!slopes.empty())
            {
                unsigned int plane_size = dim[0]*dim[1];
                unsigned int plane_num = total_size/plane_size;
                std::vector<float>::iterator iter = data.begin();
                for(unsigned int z = 0;z < plane_num;++z)
                {
                    int slope_index = std::floor(float(z)*slopes.size()/plane_num);
                    if(slope_index >= slopes.size())
                       slope_index = slopes.size()-1;
                    float s = slopes[slope_index];
                    for(unsigned int index = 0;index < plane_size;++index,++iter)
                        *iter /= s;
                }
            }
        }
        return true;

    }

    template<typename pixel_size_type>
    void get_voxel_size(pixel_size_type pixel_size_from) const
    {
        if(dim[2] >= 1)
            std::copy(resolution,resolution+3,pixel_size_from);
        else
            std::copy(resolution,resolution+2,pixel_size_from);
    }

    template<typename image_type>
    void save_to_image(image_type& out) const
    {
        out.resize(geometry<image_type::dimension>(dim));
        std::copy(data.begin(),data.begin()+out.size(),out.begin());
    }

    template<typename image_type>
    const bruker_2dseq& operator>>(image_type& source) const
    {
        save_to_image(source);
        return *this;
    }
};




}






}





#endif//2DSEQ_HPP
