#ifndef TIPL_PO_HPP
#define TIPL_PO_HPP
#include <algorithm>
#include <cctype>
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

inline auto get_directories(const std::filesystem::path& dir)
{
    std::vector<std::string> dir_list;
    if(std::filesystem::exists(dir) && std::filesystem::is_directory(dir))
        for(const auto& entry : std::filesystem::directory_iterator(dir))
            if(entry.is_directory())
                dir_list.push_back(entry.path().filename().u8string());

    std::sort(dir_list.begin(),dir_list.end());
    return dir_list;
}

template<typename T>
auto split(const T& s,typename T::value_type delimiter)
{
    std::vector<T> tokens;
    std::stringstream ss(s);
    for (T token; std::getline(ss, token, delimiter); tokens.push_back(token));
    return tokens;
}
template<typename T>
auto to_path(const std::vector<T>& list)
{
    std::vector<std::filesystem::path> file_list(list.size());
    for(size_t i = 0;i < list.size();++i)
        file_list[i] = list[i];
    return file_list;
}

template<typename T>
auto split(const T& s,typename T::value_type delimiter,
           typename T::value_type open,
           typename T::value_type close)
{
    std::vector<T> tokens;
    int depth = 0;
    size_t beg = 0;

    for(size_t i = 0;i < s.size();++i)
    {
        if(s[i] == open)
        {
            if(s.find(close,i+1) != T::npos)
                ++depth;
        }
        else
            if(s[i] == close)
            {
                if(depth)
                    --depth;
            }
            else
                if(s[i] == delimiter && !depth)
                {
                    tokens.push_back(s.substr(beg,i-beg));
                    beg = i+1;
                }
    }

    tokens.push_back(s.substr(beg));
    return tokens;
}

template<typename T>
auto split_by_line_breaks(const T& s,bool ignore_empty_lines = true)
{
    std::vector<T> tokens;
    for(auto t : split(s, '\n'))
    {
        if(!t.empty() && t.back() == '\r')
            t.pop_back();
        if(ignore_empty_lines && t.empty())
            continue;
        tokens.push_back(t);
    }
    return tokens;
}
template<typename T>
auto merge(const std::vector<T>& tokens, typename T::value_type delimiter)
{
    if (tokens.empty()) return T{};
    T result = tokens[0];
    for (size_t i = 1; i < tokens.size(); ++i)
        result += delimiter + tokens[i];
    return result;
}

template<typename T>
bool contains(const T& vs,const typename T::value_type& v)
{
    return std::find(vs.begin(),vs.end(),v) != vs.end();
}

inline bool contains(const std::string& str,const std::string& suffix)
{
    return str.find(suffix) != std::string::npos;
}

inline bool contains(const std::string& str, const std::initializer_list<std::string>& suffixes)
{
    for (const auto& suffix : suffixes)
        if(contains(str,suffix))
            return true;
    return false;
}

inline std::string replace(std::string str,
                           const std::string& old_str,
                           const std::string& new_str)
{
    if(old_str.empty())
        return str;
    for(size_t pos = 0;(pos = str.find(old_str,pos)) != std::string::npos;)
    {
        str.replace(pos,old_str.length(),new_str);
        pos += new_str.length();
    }
    return str;
}

inline bool ends_with(const std::string& str,const std::string& suffix)
{
    return (str.size() >= suffix.size()) ? (0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix)) : false;
}

inline bool ends_with(const std::string& str, const std::initializer_list<std::string>& suffixes)
{
    for (const auto& suffix : suffixes)
        if(ends_with(str,suffix))
            return true;
    return false;
}
inline bool begins_with(const std::string& str,const std::string& suffix)
{
    return (str.size() >= suffix.size()) ? (0 == str.compare(0, suffix.size(), suffix)) : false;
}
inline bool begins_with(const std::string& str, const std::initializer_list<std::string>& prefixes)
{
    for (const auto& prefix : prefixes)
        if(begins_with(str,prefix))
            return true;
    return false;
}

inline std::string trim_space(std::string s)
{
    auto not_space = [](unsigned char c){return !std::isspace(c);};
    s.erase(s.begin(),std::find_if(s.begin(),s.end(),not_space));
    s.erase(std::find_if(s.rbegin(),s.rend(),not_space).base(),s.end());
    return s;
};

inline std::string remove_all_suffix(const std::string& str)
{
    std::string_view s(str);
    while (true)
    {
        size_t pos = s.find_last_of('.');
        if (pos == std::string::npos || pos == 0 || (s.size() - pos) > 6)
            break;

        size_t sep = s.find_last_of("/\\");
        if (sep != std::string::npos && pos < sep)
            break;

        s = s.substr(0, pos);
    }
    return std::string(s);
}
inline std::filesystem::path remove_all_suffix(std::filesystem::path p)
{
    auto name = p.filename();
    for(auto ext = name.extension(); !ext.empty() && ext.native().size() <= 6; ext = name.extension())
        name = name.stem();
    return p.parent_path()/name;
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
inline bool contains_case_insensitive(const std::string& str,const std::initializer_list<std::string>& suffix)
{
    for (const auto& each : suffix)
        if(contains_case_insensitive(str,each))
            return true;
    return false;
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

    if(!str1.empty())
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
                              str2.substr(cprefix.length()),str2_match,try_reverse,try_swap))
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
                              str2.substr(0,str2.length()-cpostfix.length()),str2_match,try_reverse,try_swap))
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
                             str2.substr(cp1_1.size()),str2_match,try_reverse,try_swap))
            {
                str2_match = str2.substr(0,cp1_1.size()) + str2_match;
                return true;
            }
        }

        if(match_strings(str1.substr(1),str1_match.substr(1),str2,str2_match,try_reverse,try_swap))
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
    #ifdef _WIN32
    if(file_path1.find('\\') != std::string::npos ||
       file_path2.find('\\') != std::string::npos ||
       file_path1_others.find('\\') != std::string::npos)
    {
        auto replace_slash = [](auto path)
        {
            std::replace(path.begin(), path.end(), '\\', '/');
            return path;
        };
        return match_files(replace_slash(file_path1),replace_slash(file_path2),replace_slash(file_path1_others),file_path2_gen);
    }
    #endif

    auto name1 = std::filesystem::path(file_path1).filename().u8string();
    auto name2 = std::filesystem::path(file_path2).filename().u8string();
    auto name1_others = std::filesystem::path(file_path1_others).filename().u8string();
    auto path1 = std::filesystem::path(file_path1).parent_path().u8string();
    auto path2 = std::filesystem::path(file_path2).parent_path().u8string();
    auto path1_others = std::filesystem::path(file_path1_others).parent_path().u8string();
    T name2_others,path2_others;
    if(!match_strings(name1,name2,name1_others,name2_others) ||
       !match_strings(path1,path2,path1_others,path2_others))
    {
        if(!match_strings(file_path1,file_path2,file_path1_others,file_path2_gen))
        {
            if(path2 == path1 && path2 != path1_others)
            {
                file_path2_gen = (std::filesystem::path(file_path1_others).parent_path()/
                                 std::filesystem::path(file_path2).filename()).u8string();
                return true;
            }
            return false;
        }
    }
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

inline void search_files(const std::filesystem::path& search_path,
                         const std::string& wildcard,
                         std::vector<std::filesystem::path>& results,
                         bool recursive = false)
{
    if(!std::filesystem::is_directory(search_path))
        return;

    auto add = [&](const auto& entry)
    {
        if(entry.is_regular_file() &&
            (wildcard.empty() || tipl::match_wildcard(entry.path().filename().u8string(),wildcard)))
            results.push_back(entry.path());
    };

    if(recursive)
        for(const auto& entry : std::filesystem::recursive_directory_iterator(search_path))
            add(entry);
    else
        for(const auto& entry : std::filesystem::directory_iterator(search_path))
            add(entry);
}

inline auto search_files(const std::filesystem::path& search_path,
                         const std::string& wildcard,
                         bool recursive = false)
{
    std::vector<std::filesystem::path> results;
    search_files(search_path,wildcard,results,recursive);
    return results;
}
inline void search_dirs(const std::filesystem::path& search_path,const std::string& wildcard,std::vector<std::filesystem::path>& results)
{
    if(!std::filesystem::is_directory(search_path))
        return;

    for(const auto& entry : std::filesystem::directory_iterator(search_path))
        if(entry.is_directory() &&
            (wildcard.empty() || tipl::match_wildcard(entry.path().filename().u8string(),wildcard)))
                results.push_back(entry.path());
}
inline auto search_dirs(const std::filesystem::path& search_path,const std::string& wildcard)
{
    std::vector<std::filesystem::path> results;
    search_dirs(search_path,wildcard,results);
    return results;
}


inline bool search_filesystem(const std::filesystem::path& pat,
                              std::vector<std::filesystem::path>& out,
                              bool want_file = true)
{
    if(pat.empty())
        return false;
    auto old_size = out.size();
    auto match = [](const std::string& p,const std::string& s)
    {
        const char *a = p.c_str(),*b = s.c_str(),*sa = nullptr,*sb = nullptr;
        while(*b)
            if(*a == '?' || *a == *b)
                ++a,++b;
            else if(*a == '*')
                sa = a++,sb = b;
            else if(sa)
                a = sa+1,b = ++sb;
            else
                return false;
        while(*a == '*')
            ++a;
        return !*a;
    };

    std::filesystem::path base;
    std::vector<std::string> parts;

    for(const auto& c : pat)
    {
        auto s = c.u8string();
        if(s.find_first_of("*?") != std::string::npos || !parts.empty())
            parts.push_back(s);
        else
            base /= c;
    }

    std::error_code ec;
    if(parts.empty())
    {
        if((want_file ? std::filesystem::is_regular_file(pat,ec) :
                 std::filesystem::is_directory(pat,ec)) && !ec)
            out.push_back(std::filesystem::weakly_canonical(pat,ec));
        return !out.empty();
    }

    if(base.empty())
        base = std::filesystem::current_path();

    if(!std::filesystem::is_directory(base,ec))
        return false;

    std::vector<std::filesystem::path> dirs{base};

    for(size_t i = 0;i < parts.size();++i)
    {
        std::vector<std::filesystem::path> next;

        for(const auto& d : dirs)
            for(std::filesystem::directory_iterator it(d,ec),end;!ec && it != end;++it)
            {
                const auto& e = *it;
                if(!match(parts[i],e.path().filename().u8string()))
                    continue;

                if(i+1 == parts.size())
                {
                    if(want_file ? e.is_regular_file(ec) : e.is_directory(ec))
                        out.push_back(std::filesystem::weakly_canonical(e.path(),ec));
                }
                else if(e.is_directory(ec))
                    next.push_back(e.path());
            }

        dirs.swap(next);
    }

    return out.size() != old_size;
}



inline std::string complete_suffix(const std::string& file_name)
{
    std::filesystem::path p(file_name);
    std::string ext = p.extension().u8string();
    return (ext == ".gz") ? p.stem().extension().u8string() + ext : ext;
}
inline auto read_text_file(const std::filesystem::path& file_name)
{
    std::ifstream file(file_name);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line))
        lines.push_back(line);
    return lines;
}

template<typename out = default_output>
class program_option{
    std::vector<std::string> names;
    std::vector<std::string> values;
    std::vector<char> used,printed;
    std::set<std::string> not_found_names;
    bool add_option(const std::string& str)
    {
        if(!tipl::begins_with(str,"--"))
        {
            error_msg = std::string("invalid argument ") + str + " did you forget to put double dash '--' in front of the argument?";
            return false;
        }
        auto pos = std::find(str.begin(),str.end(),'=');
        names.push_back(std::string(str.begin()+2,pos));
        if(pos == str.end())
            values.push_back(std::string());
        else
        {
            auto v = std::string(pos+1,str.end());
            v.erase(std::remove(v.begin(),v.end(),'\"'),v.end());
            values.push_back(v);
        }
        used.push_back(0);
        printed.push_back(0);        
        return true;
    }
private:
    void check_interact(void)
    {
        for(size_t i = 0;i < names.size();++i)
            if(names[i] == "interact")
            {
                interact = (values[i].empty() || values[i] == "1" || values[i] == "true");
                used[i] = 1;
                printed[i] = 1;
                break;
            }
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
    bool interact = false;
    template<typename out_warning = out>
    void check_end_param(void)
    {
        for(size_t i = 0;i < used.size();++i)
            if(!used[i])
            {
                const std::string& str1 = names[i];
                std::map<int,std::string,std::greater<int>> candidate_list;
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

        if(interact)
        {
            std::string assigned_param;
            for(size_t i = 0;i < names.size();++i)
                if(names[i] != "interact")
                    assigned_param += "--" + names[i] + "=" + values[i] + " ";
            out() << "parameters: dsi_studio " << assigned_param;
        }
    }

    void clear(void)
    {
        names.clear();
        values.clear();
        used.clear();
        printed.clear();
        not_found_names.clear();
        error_msg.clear();
        interact = false;
    }

    bool parse(int ac, char *av[])
    {
        exec_path = std::filesystem::absolute(std::filesystem::path(av[0])).parent_path().u8string();
        clear();
        for(int i = 1;i < ac;++i)
        {
            std::string str(av[i]);
            if(!add_option(str))
                return false;
        }
        check_interact();
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
            if(str.empty() || !in)
                break;
            if(!values.empty() && str.find("--") != 0)
            {
                values.back() += " " + str;
                continue;
            }
            if(str.find('"') != std::string::npos)
            {
                if(std::count(str.begin(),str.end(),'"') == 1)
                {
                    str.erase(str.find('"'),1);
                    while(in)
                    {
                        std::string other_str;
                        in >> other_str;
                        str += " ";
                        str += other_str;
                        if(other_str.back() == '"')
                        {
                            str.pop_back();
                            break;
                        }
                    }
                }
                else
                    str.erase(std::remove(str.begin(), str.end(), '"'), str.end());
            }
            if(!str.empty() && !add_option(str))
                return false;
        }
        check_interact();
        return true;
    }

    bool check(const char* name)
    {
        if(!has(name) && (!interact || get(name).empty()))
            return out() << (error_msg = "please specify --" + std::string(name)), false;
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
        if(interact)
            out() << "interact mode skips optional parameter --" << name;
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
    void mute(const char* name)
    {
        std::string str_name(name);
        for(size_t i = 0;i < names.size();++i)
            if(names[i] == str_name)
            {
                printed[i] = 1;
                return;
            }
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
    template<typename T> struct is_vector : std::false_type {};
    template<typename T> struct is_vector<std::vector<T>> : std::true_type {};

public:
    template<typename value_type>
    value_type get(const char* name,value_type df)
    {
        for(size_t i = 0;i < names.size();++i)
        {
            if(names[i] == name)
            {
                used[i] = 1;
                if(!printed[i])
                {
                    printed[i] = 1;
                    out() << name << "=" << values[i] << std::endl;
                }

                if(values[i].empty())
                    return df;

                if constexpr(std::is_same_v<value_type,std::string>)
                    return values[i];
                else if constexpr(is_vector<value_type>::value)
                {
                    value_type result;
                    std::istringstream stream(values[i]);
                    std::string element;
                    while(std::getline(stream,element,','))
                    {
                        typename value_type::value_type parsed{};
                        std::istringstream(element) >> parsed;
                        result.push_back(parsed);
                    }
                    return result;
                }
                else
                {
                    value_type result{};
                    std::istringstream(values[i]) >> result;
                    return result;
                }
            }
        }

        // Convert default value to string exactly once
        std::ostringstream stream;
        if constexpr(is_vector<value_type>::value)
            for(size_t i = 0;i < df.size();++i)
                stream << (i ? "," : "") << df[i];
        else
            stream << df;

        std::string df_str = stream.str();

        // Handle interaction if enabled
        if(interact)
        {
            std::cout << "Please specify --" << name << " [" << df_str << "]: ";
            std::string input;
            std::getline(std::cin,input);

            set(name,input.empty() ? df_str : input);
            return get(name,df);
        }

        // Fallback for non-interactive mode
        not_found_names.insert(name);
        out() << name << "=" << df_str << std::endl;
        return df;
    }

    std::string get(const char* name,const std::vector<std::string>& selections,const std::string& default_sel)
    {
        if(!has(name))
        {
            if(!interact)
                return default_sel;

            std::cout << "Please specify --" << name << " (";
            for(size_t i = 0;i < selections.size();++i)
                std::cout << (i ? "," : "") << selections[i];
            std::cout << ") [" << default_sel << "]: ";

            std::string input;
            std::getline(std::cin,input);
            if(input.empty())
                input = default_sel;
            set(name,input);
        }

        auto sel = get(name);
        if(sel.empty())
            return default_sel;

        std::string resolved_sel = sel;

        if(sel[0] >= '0' && sel[0] <= '9')
        {
            size_t idx = size_t(sel[0]-'0');
            if(idx < selections.size())
                resolved_sel = selections[idx];
        }
        else
        {
            auto it = std::find_if(selections.begin(),selections.end(),[&](const auto& s){ return tipl::contains(s,sel); });
            if(it != selections.end())
                resolved_sel = *it;
        }

        if(resolved_sel != sel)
            set(name,resolved_sel);

        return resolved_sel;
    }

    size_t get(const char* name,const std::vector<std::string>& selections,size_t default_sel = 0)
    {
        // 1. Get the string equivalent of the default index
        std::string default_str = default_sel < selections.size() ? selections[default_sel] : "";

        // 2. Route EVERYTHING through the std::string version
        std::string resolved = get(name,selections,default_str);

        // 3. Map the resolved string back to its index
        auto it = std::find(selections.begin(),selections.end(),resolved);
        return it != selections.end() ? size_t(std::distance(selections.begin(),it)) : default_sel;
    }

    std::string get(const char* name,const char* df_ptr)
    {
        return get(name,std::string(df_ptr));
    }

    std::string get(const char* name)
    {
        return get(name,std::string());
    }

    template<typename extension_type>
    std::string get_file(const char* name,const extension_type& extension,std::string default_sel = "")
    {
        if(!has(name))
        {
            if(!interact)
                return default_sel;

            std::vector<std::string> local_files;
            for(const auto& entry : std::filesystem::directory_iterator(std::filesystem::current_path()))
                if(tipl::ends_with(entry.path().filename().u8string(),extension))
                    local_files.push_back(entry.path().filename().u8string());

            std::sort(local_files.begin(),local_files.end());

            if(!local_files.empty())
            {
                std::cout << "\navailable files for --" << name << ":\n";
                for(size_t i = 0;i < local_files.size();++i)
                    std::cout << "[" << (i + 1) << "] " << local_files[i] << (i % 5 == 4 ? '\n' : '\t');
                if(default_sel.empty())
                    default_sel = local_files.front();
            }

            std::cout << "Please specify --" << name << (local_files.empty() ? "" : " (Enter 1-" + std::to_string(local_files.size()) + ", or type path)") << " [" << default_sel << "]: ";

            std::string input;
            std::getline(std::cin,input);

            if(input.empty())
                input = default_sel;
            else if(std::all_of(input.begin(),input.end(),[](unsigned char c){ return std::isdigit(c); }))
            {
                try
                {
                    size_t idx = std::stoul(input);
                    if(idx > 0 && idx <= local_files.size())
                        input = local_files[idx - 1];
                }
                catch(...)
                {
                    // Ignore overflow, treat input as a raw file path
                }
            }

            set(name,input);
        }

        auto sel = get(name);
        return sel.empty() ? default_sel : sel;
    }

    template<typename out_warning = out>
    auto get_files(const char* name,const std::string& default_str = std::string())
    {
        auto search_str = get(name,default_str);
        std::vector<std::filesystem::path> filenames;
        for(const auto& each : tipl::split(search_str,','))
        {
            if(each.find('*') == std::string::npos)
                filenames.push_back(each);
            else
            {
                std::vector<std::filesystem::path> new_files;
                if(search_filesystem(each,new_files))
                {
                    out() << each << ": " << new_files.size() << " file(s) specified by " << each;
                    std::sort(new_files.begin(),new_files.end());
                    filenames.insert(filenames.end(),new_files.begin(),new_files.end());
                }
                else
                    out_warning() << "could not find files matching " << each;
            }
        }
        if(filenames.empty())
            error_msg = "no file found matching " + search_str;
        return filenames;
    }

};


}// namespace tipl

#endif // TIPL_PO_HPP

