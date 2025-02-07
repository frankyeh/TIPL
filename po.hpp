#ifndef TIPL_PO_HPP
#define TIPL_PO_HPP
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <filesystem>
#include <map>
#include <set>

namespace tipl{


struct default_output{
        template<typename type>
        auto& operator<<(const type& v)
        {
            std::cout << v;
            return *this;
        }
};


template<typename T>
auto split(const T& s,typename T::value_type delimiter)
{
    std::vector<T> tokens;
    std::stringstream ss(s);
    for (T token; std::getline(ss, token, delimiter); tokens.push_back(token));
    return tokens;
}

inline bool contains(const std::string& str,const std::string& suffix)
{
    return str.find(suffix) != std::string::npos;
}

inline bool ends_with(const std::string& str,const std::string& suffix)
{
    return (str.size() >= suffix.size()) ? (0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix)) : false;
}
inline bool begins_with(const std::string& str,const std::string& suffix)
{
    return (str.size() >= suffix.size()) ? (0 == str.compare(0, suffix.size(), suffix)) : false;
}
inline bool remove_suffix(std::string& str,const std::string& suffix)
{
    if(!ends_with(str,suffix))
        return false;
    str.erase(str.size() - suffix.size());
    return true;
}

inline std::string to_lower(const std::string& str)
{
    std::string result(str);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return result;
}

inline bool contains_case_insensitive(const std::string& str,const std::string& suffix)
{
    return to_lower(str).find(to_lower(suffix)) != std::string::npos;
}
inline bool equal_case_insensitive(const std::string& str,const std::string& suffix)
{
    return to_lower(str) == to_lower(suffix);
}

template<typename T>
T common_prefix(const T& str1,const T& str2)
{
    T result;
    for(size_t cur = 0;cur < str1.length() && cur < str2.length();++cur)
    {
        if(str1[cur] != str2[cur])
            break;
        result.push_back(str1[cur]);
    }
    return result;
}
template<typename T>
T common_prefix(const T& str1,const T& str2,const T& str3)
{
    T result;
    for(size_t cur = 0;cur < str1.length() && cur < str2.length() && cur < str3.length();++cur)
    {
        if(str1[cur] != str2[cur] || str1[cur] != str3[cur])
            break;
        result.push_back(str1[cur]);
    }
    return result;
}


template<typename T>
bool match_strings(const T& str1,const T& str1_match,
                   const T& str2,T& str2_match,bool try_reverse = true,bool try_swap = true)
{

    // A->A
    // B->B
    if(str1 == str1_match)
    {
        str2_match = str2;
        return true;
    }
    // A->B
    // A->B
    if(str1 == str2)
    {
        str2_match = str1_match;
        return true;
    }

    if(str1.length() > 1)
    {
        size_t pos = str1_match.find(str1);
        if(pos != std::string::npos)
        {
            str2_match = str1_match;
            str2_match.replace(pos, str1.length(), str2);
            return true;
        }
        pos = str2.find(str1);
        if(pos != std::string::npos)
        {
            str2_match = str2;
            str2_match.replace(pos, str1.length(), str1_match);
            return true;
        }
    }

    // remove common prefix
    {
        auto cprefix = common_prefix(str1,str1_match,str2);
        if(!cprefix.empty())
        {
            if(!match_strings(str1.substr(cprefix.length()),str1_match.substr(cprefix.length()),
                              str2.substr(cprefix.length()),str2_match))
                return false;
            str2_match = cprefix+str2_match;
            return true;
        }
    }

    // remove common postfix
    {
        auto cpostfix = common_prefix(T(str1.rbegin(),str1.rend()),
                                      T(str1_match.rbegin(),str1_match.rend()),
                                      T(str2.rbegin(),str2.rend()));
        if(!cpostfix.empty())
        {
            if(!match_strings(str1.substr(0,str1.length()-cpostfix.length()),
                              str1_match.substr(0,str1_match.length()-cpostfix.length()),
                              str2.substr(0,str2.length()-cpostfix.length()),str2_match))
                return false;
            str2_match += T(cpostfix.rbegin(),cpostfix.rend());
            return true;
        }
    }




    auto cp1_1 = common_prefix(str1,str1_match);
    auto cp1_2 = common_prefix(str1,str2);


    // A_B->A_C
    // D_B->D_C
    if(cp1_1.length() > cp1_2.length())
    {
        // A->A_C
        // D->D_C
        if(cp1_1 == str1)
        {
            str2_match = str2 + str1_match.substr(cp1_1.size());
            return true;
        }

        if(str1.length() == str2.length())
        {
            if(match_strings(str1.substr(cp1_1.size()),str1_match.substr(cp1_1.size()),
                             str2.substr(cp1_1.size()),str2_match))
            {
                str2_match = str2.substr(0,cp1_1.size()) + str2_match;
                return true;
            }
        }

        if(match_strings(str1.substr(1),str1_match.substr(1),str2,str2_match))
            return true;
    }
    // try reversed
    T rev;
    if(try_reverse && match_strings(T(str1.rbegin(),str1.rend()),
                     T(str1_match.rbegin(),str1_match.rend()),
                     T(str2.rbegin(),str2.rend()),rev,false,try_swap))
    {
        str2_match = T(rev.rbegin(),rev.rend());
        return true;
    }
    // try swap
    if(try_swap)
        return match_strings(str1,str2,str1_match,str2_match,try_reverse,false);
    return false;
}

template<typename T>
bool match_files(const T& file_path1,const T& file_path2,
                 const T& file_path1_others,T& file_path2_gen)
{
    auto name1 = std::filesystem::path(file_path1).filename().u8string();
    auto name2 = std::filesystem::path(file_path2).filename().u8string();
    auto name1_others = std::filesystem::path(file_path1_others).filename().u8string();
    auto path1 = std::filesystem::path(file_path1).parent_path().u8string();
    auto path2 = std::filesystem::path(file_path2).parent_path().u8string();
    auto path1_others = std::filesystem::path(file_path1_others).parent_path().u8string();
    T name2_others,path2_others;
    if(!match_strings(name1,name2,name1_others,name2_others) ||
       !match_strings(path1,path2,path1_others,path2_others))
        return match_strings(file_path1,file_path2,file_path1_others,file_path2_gen);
    if(!path2_others.empty())
        path2_others += "/";
    file_path2_gen = path2_others + name2_others;
    return true;
}
inline bool match_wildcard(const std::string& file_path,const std::string& wild_card)
{
    std::string result;
    return tipl::match_strings(wild_card,file_path,std::string("*"),result,true,false) && !result.empty();
}

inline void search_files(const std::string& search_path,const std::string& wildcard,std::vector<std::string>& results)
{
    if(!std::filesystem::exists(search_path) || !std::filesystem::is_directory(search_path))
        return;
    for (const auto& entry : std::filesystem::directory_iterator(search_path))
    {
        if (!std::filesystem::is_regular_file(entry))
            continue;
        if (wildcard.empty() || tipl::match_wildcard(entry.path().filename().u8string(),wildcard))
            results.push_back(entry.path().u8string());
    }
}
inline std::vector<std::string> search_files(const std::string& search_path,const std::string& wildcard)
{
    std::vector<std::string> results;
    search_files(search_path,wildcard,results);
    return results;
}
inline void search_dirs(const std::string& search_path,const std::string& wildcard,std::vector<std::string>& results)
{
    if(!std::filesystem::exists(search_path) || !std::filesystem::is_directory(search_path))
        return;
    for (const auto& entry : std::filesystem::directory_iterator(search_path))
    {
        if (!std::filesystem::is_directory(entry))
            continue;
        if (wildcard.empty() || tipl::match_wildcard(entry.path().filename().u8string(),wildcard))
            results.push_back(entry.path().u8string());
    }
}
inline std::vector<std::string> search_dirs(const std::string& search_path,const std::string& wildcard)
{
    std::vector<std::string> results;
    search_dirs(search_path,wildcard,results);
    return results;
}

template<typename out_type = void,typename error_type = void>
bool search_filesystem(std::string path,std::vector<std::string>& filenames,bool file = true)
{
    if constexpr(!std::is_void<out_type>::value)
    {
        if(file)
            out_type() << "searching file(s) at: " << path;
        else
            out_type() << "searching directories at: " << path;
    }
    if (path.find('*') == std::string::npos)
    {
        if (std::filesystem::exists(path))
        {
            filenames.push_back(path);
            return true;
        }
        else
        {
            if constexpr(!std::is_void<error_type>::value)
                error_type() << "file not exist: " << path;
            return false;
        }
    }

    auto search_path = std::filesystem::current_path().u8string();
    bool no_path = true;
    if (path.find('/') != std::string::npos)
    {
        no_path = false;
        search_path = path.substr(0, path.find_last_of('/'));
        path = path.substr(path.find_last_of('/') + 1);
        if(search_path.find('*') != std::string::npos)
        {
            std::vector<std::string> dirs;
            search_filesystem<out_type,error_type>(search_path,dirs,false);
            bool result = false;
            for(auto dir : dirs)
                result |= search_filesystem<out_type,error_type>(dir + "/" + path,filenames);
            return result;
        }
    }

    try{
        if(!std::filesystem::exists(search_path) || !std::filesystem::is_directory(search_path))
        {
            if constexpr(!std::is_void<error_type>::value)
                error_type() << "directory not exist: " << search_path;
            return true;
        }
        std::vector<std::string> new_filenames;

        for (const auto& entry : std::filesystem::directory_iterator(search_path))
        {
            if (file && !std::filesystem::is_regular_file(entry))
                continue;
            if (!file && !std::filesystem::is_directory(entry))
                continue;
            if (tipl::match_wildcard(entry.path().filename().u8string(),path))
                new_filenames.push_back(no_path ? entry.path().filename().u8string() : entry.path().u8string());
        }
        if constexpr(!std::is_void<out_type>::value)
            out_type() << new_filenames.size() << " files matching " << path;
        std::sort(new_filenames.begin(),new_filenames.end());
        filenames.insert(filenames.end(),new_filenames.begin(),new_filenames.end());
    }
    catch (const std::filesystem::filesystem_error& e)
    {
        if constexpr(!std::is_void<error_type>::value)
            error_type() << e.what();
        return false;
    }
    catch(const std::runtime_error& e)
    {
        if constexpr(!std::is_void<error_type>::value)
            error_type() << e.what();
        return false;
    }
    catch(...)
    {
        if constexpr(!std::is_void<error_type>::value)
            error_type() << "unknown error when searching files";
        return false;
    }
    return true;
}

template<typename out = default_output>
class program_option{
    std::vector<std::string> names;
    std::vector<std::string> values;
    std::vector<char> used,printed;
    std::set<std::string> not_found_names;
    bool add_option(const std::string& str)
    {
        if(str.length() < 3 || str[0] != '-' || str[1] != '-')
        {
            error_msg = std::string("invalid argument ") + str + " did you forget to put double dash '--' in front of the argument?";
            return false;
        }
        auto pos = std::find(str.begin(),str.end(),'=');
        names.push_back(std::string(str.begin()+2,pos));
        values.push_back(pos == str.end() ? std::string() : std::string(pos+1,str.end()));
        used.push_back(0);
        printed.push_back(0);
        if(values.back().front() == '\"' && values.back().back() == '\"')
        {
            values.back().pop_back();
            values.back() = values.back().substr(1);
        }
        return true;
    }
public:
    struct program_option_assign{
        const char* name = nullptr;
        program_option* po = nullptr;
        program_option_assign(const char* name_,program_option* po_):name(name_),po(po_){}
        template<typename T> void operator=(const T& rhs)
        {
            std::ostringstream str_out;
            str_out << rhs;
            po->set(name,str_out.str());
        }
    };
    inline program_option_assign operator[](const char* name)     {return program_option_assign(name,this);}
public:
    std::string error_msg,exec_path;
    template<typename out_warning = out>
    void check_end_param(void)
    {
        for(size_t i = 0;i < used.size();++i)
            if(!used[i])
            {
                const std::string& str1 = names[i];
                std::map<int,std::string,std::greater<int> > candidate_list;
                for(const auto& str2 : not_found_names)
                {
                    int c = -std::abs(int(str1.length())-int(str2.length()));
                    size_t common_length = std::min(str1.length(),str2.length());
                    for(size_t j = 0;j < common_length;++j)
                    {
                        if(str1[j] == str2[j])
                            ++c;
                        if(str1[str1.length()-1-j] == str2[str2.length()-1-j])
                            ++c;
                    }
                    candidate_list[c] = str2;
                }
                std::string prompt_msg;
                if(!candidate_list.empty() && candidate_list.begin()->first > 0)
                {
                    prompt_msg = "Did you mean --";
                    prompt_msg += candidate_list.begin()->second;
                    prompt_msg += " ?";
                }
                out_warning() << "--" << str1 << " is not used/recognized. " << prompt_msg << std::endl;
            }
    }
    void clear(void)
    {
        names.clear();
        values.clear();
        used.clear();
        printed.clear();
    }

    bool parse(int ac, char *av[])
    {
        exec_path = std::filesystem::absolute(std::filesystem::path(av[0])).parent_path().u8string();
        clear();
        if(ac == 2) // command from log file
        {
            std::ifstream in(av[1]);
            std::string line;
            while(std::getline(in,line))
            {
                line = std::string("--")+line;
                add_option(line);
            }
        }
        else
        for(int i = 1;i < ac;++i)
        {
            std::string str(av[i]);
            if(!add_option(str))
                return false;
        }
        return true;
    }
    bool parse(const std::string& av)
    {
        clear();
        std::istringstream in(av);
        while(in)
        {
            std::string str;
            in >> str;
            if(str.find('"') != std::string::npos)
            {
                str.erase(str.find('"'),1);
                while(in)
                {
                    std::string other_str;
                    in >> other_str;
                    str += " ";
                    str += other_str;
                    if(other_str.find('"') != std::string::npos)
                    {
                        str.erase(str.find('"'),1);
                        break;
                    }
                }
            }
            if(!str.empty() && !add_option(str))
                return false;
        }
        return true;
    }

    bool check(const char* name)
    {
        if(!has(name))
        {
            out() << "please specify --" << name << std::endl;
            return false;
        }
        return true;
    }

    bool has(const char* name)
    {
        std::string str_name(name);
        for(size_t i = 0;i < names.size();++i)
        {
            if(names[i] == str_name)
            {
                used[i] = true;
                return true;
            }
        }
        not_found_names.insert(name);
        return false;
    }

    void get_wildcard_list(std::vector<std::pair<std::string,std::string> >& wlist) const
    {
        for(size_t i = 0;i < names.size();++i)
            if(values[i].find('*') != std::string::npos)
                wlist.push_back(std::make_pair(names[i],values[i]));
    }

    void set_used(char value)
    {
        std::fill(used.begin(),used.end(),value);
    }
    void set(const char* name,const std::string& value)
    {
        std::string str_name(name);
        for(size_t i = 0;i < names.size();++i)
            if(names[i] == str_name)
            {
                values[i] = value;
                used[i] = 0;
                printed[i] = 0;
                return;
            }
        names.push_back(name);
        values.push_back(value);
        used.push_back(0);
        printed.push_back(0);
    }
    template<typename T,typename std::enable_if<std::is_fundamental<T>::value,bool>::type = true>
    void set(const char* name,T value)
    {
        set(name,std::to_string(value));
    }
    template<typename T>
    void set(const char* name,const std::vector<T>& container)
    {
        std::string str;
        for(auto& each : container)
        {
            if(!str.empty())
                str += ",";
            str += std::to_string(each);
        }
        set(name,str);
    }

private:
    template <typename T>
    struct convert_to {
        static T from(const std::string& value) {
            T df;
            std::istringstream(value) >> df;
            return df;
        }
    };
    template <typename T>
    struct convert_to<std::basic_string<T> > {
        static const std::basic_string<T>& from(const std::string& value) {
            return value;
        }
    };
    template <typename T>
    struct convert_to<std::vector<T> > {
        static std::vector<T> from(const std::string& value) {
            std::vector<T> result;
            std::istringstream stream(value);
            std::string element;
            while (std::getline(stream, element, ',')) {
                T parsed_value;
                std::istringstream(element) >> parsed_value;
                result.push_back(parsed_value);
            }
            return result;
        }
    };

public:
    template<typename value_type>
    value_type get(const char* name,value_type df)
    {
        std::string str_name(name);
        for(size_t i = 0;i < names.size();++i)
            if(names[i] == str_name)
            {
                if(!used[i])
                    used[i] = 1;
                if(!printed[i])
                {
                    printed[i] = 1;
                    out() << name << "=" << values[i] << std::endl;
                }
                return convert_to<value_type>::from(values[i]);
            }
        not_found_names.insert(name);
        out() << name << "=" << df << std::endl;
        return df;
    }

    std::string get(const char* name,const char* df_ptr)
    {
        return get(name,std::string(df_ptr));
    }

    std::string get(const char* name)
    {
        return get(name,std::string());
    }
    bool get_files(const char* name,std::vector<std::string>& filenames)
    {
        std::vector<std::string> file_list = tipl::split(get(name),',');
        for(size_t index = 0;index < file_list.size();++index)
        {
            if(file_list[index].find('*') == std::string::npos)
                filenames.push_back(file_list[index]);
            else
            {
                size_t old_size = filenames.size();
                if(search_filesystem(file_list[index],filenames))
                    out() << file_list[index] << ": " << filenames.size()-old_size << " file(s) specified." << std::endl;
            }
        }
        return true;
    }

};


}// namespace tipl

#endif // TIPL_PO_HPP

