#ifndef TIPL_PROG_HPP
#define TIPL_PROG_HPP
#include <memory>
#include <ctime>
#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include <sstream>


#if defined(TIPL_USE_QT) && !defined(__CUDACC__)
#include <QProgressDialog>
#include <QApplication>
#endif

#include "mt.hpp"
#include "po.hpp"

namespace tipl{

inline bool prog_aborted = false;
inline bool show_prog = false;
inline std::vector<std::chrono::high_resolution_clock::time_point> process_time,t_last;
inline std::vector<std::string> status_list,at_list;
inline bool processing_time_less_than(int time)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::high_resolution_clock::now() - process_time.back()).count() < time;
}

inline void update_prog(std::string status,bool show_now = false,uint32_t now = 0,uint32_t total = 0)
{
    if(!show_prog || !tipl::is_main_thread())
        return;
    #if defined(TIPL_USE_QT) && !defined(__CUDACC__)
    static std::shared_ptr<QProgressDialog> progressDialog;
    if(status_list.empty())
    {
        if(progressDialog.get())
        {
            progressDialog->close();
            progressDialog.reset();
            QApplication::processEvents();
        }
        return;
    }

    if(!show_now && processing_time_less_than(250))
        return;

    if(!progressDialog.get())
    {
        progressDialog.reset(new QProgressDialog(status.c_str(),"Cancel",0,100));
        progressDialog->setAttribute(Qt::WA_ShowWithoutActivating);
        progressDialog->activateWindow();
    }
    else
    {
        progressDialog->setLabelText(status.c_str());
        if(progressDialog->wasCanceled())
        {
            prog_aborted = true;
            return;
        }

    }

    if(total != 0)
    {
        progressDialog->setRange(0, int(total));
        progressDialog->setValue(int(now));

    }
    progressDialog->show();
    progressDialog->raise();
    QApplication::processEvents();
    #endif
    return;
}



class progress{
private:
    void begin_prog(const char* status,bool show_now = false)
    {
        if(!tipl::is_main_thread())
            return;
        prog_aborted = false;
        status_list.push_back(status);
        process_time.resize(status_list.size());
        process_time.back() = std::chrono::high_resolution_clock::now();
        t_last.resize(status_list.size());
        t_last.back() = std::chrono::high_resolution_clock::now()+std::chrono::milliseconds(200);
        update_prog(get_status(),show_now);
    }

    static std::string get_status(void)
    {
        std::string result;
        for(size_t i = 0;i < status_list.size();++i)
        {
            if(status_list[i].empty())
                continue;
            if(i && (result.empty() || result.back() != '\n'))
                result += "\n";
            {
                std::string s;
                std::getline(std::istringstream(status_list[i]),s);
                result += s;
            }
            if(i < at_list.size())
            {
                result += " ";
                result += at_list[i];
            }
        }
        if(!result.empty() && result.back() == '\n')
            result.pop_back();
        return result;
    }

    static bool check_prog(unsigned int now,unsigned int total)
    {
        if(!show_prog || !tipl::is_main_thread() || status_list.empty())
        {
            if(prog_aborted)
                return false;
            return now < total;
        }
        if(now >= total || aborted())
        {
            if(at_list.size() == status_list.size())
                at_list.back().clear();
            return false;
        }
        if(std::chrono::high_resolution_clock::now() > t_last.back())
        {
            t_last.back() = std::chrono::high_resolution_clock::now()+std::chrono::milliseconds(200);
            int expected_sec = (
                        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                process_time.back()).count()*int(total-now)/int(now+1)/1000/60);
            at_list.resize(status_list.size());
            std::ostringstream outstr;
            outstr << "(" << now << "/" << total << ")";
            if(expected_sec)
                outstr << " " << expected_sec << " min";
            at_list.back() = outstr.str();
            update_prog(get_status(),false,now,total);
            if(prog_aborted)
                progress::print("operation aborted",false,false,true);
            return !prog_aborted;
        }
        return now < total;
    }

public:
    static void print(const char* status,bool head_node, bool tail_node,unsigned int error_code = 0)
    {
        static const char* color_end = "\033[0m";
        static const char* color31 = "\033[1;31m";
        static const char* color32 = "\033[0;32m";
        static const char* color33 = "\033[0;33m";
        static const char* color34 = "\033[1;34m";
        static const char* color35 = "\033[1;35m";

        std::istringstream in(status);
        std::string line;
        while(std::getline(in,line))
        {
            if(line.empty())
                continue;
            std::string head;
            for(size_t i = 1;i < status_list.size();++i)
                head += "│  ";
            if(!status_list.empty())
            {
                if(head_node)
                    head += "├──┬──";
                else
                    if(tail_node)
                        head += "│  ";
                    else
                        head += "├──";
            }
            else
                if(head_node)
                    head = "┌──";
            if(tail_node)
                head += "└──";


            if(!tipl::is_main_thread())
                head += "[thread]";

            const char* color = nullptr;
            if(error_code)
            {
                line = std::string(reinterpret_cast<const char*>(&error_code))+line;
                color = color31;
            }
            if(!color && tipl::begins_with(line,"sav"))
            {
                line = std::string("💾")+line;
                color = color35;
            }
            if(!color && tipl::begins_with(line,"open"))
            {
                line = std::string("📂")+line;
                color = color35;
            }
            if(!color && head_node)
            {
                line = std::string("📟")+line;
                color = color34;
            }
            if(!show_prog)
            {
                if(color)
                    line = std::string(color) + line + color_end;
                else
                {
                    auto eq_pos = line.find('=');
                    if(eq_pos != std::string::npos)
                        line = std::string(color32) + line.substr(0,eq_pos) + color_end + line.substr(eq_pos);
                    else
                    {
                        auto info_pos = line.find(": ");
                        if(info_pos != std::string::npos)
                            line = std::string(color33) + line.substr(0,info_pos) + color_end + line.substr(info_pos);
                    }
                }
            }
            std::cout << head + line << std::endl;
            head_node = false;
        }
    }
public:
    bool temporary = false;
    progress(void):temporary(true){}
    progress(const char* status,bool show_now = false)
    {
        print(status,true,false);
        begin_prog(status,show_now);
    }
    progress(const char* status1,const char* status2,bool show_now = false)
    {
        std::string s(status1);
        s += status2;
        print(s.c_str(),true,false);
        begin_prog(s.c_str(),show_now);
    }
    static bool is_running(void) {return !status_list.empty();}
    static bool aborted(void) { return prog_aborted;}
    template<typename value_type1,typename value_type2>
    bool operator()(value_type1 now,value_type2 total)
    {
        return check_prog(uint32_t(now),uint32_t(total));
    }
    ~progress(void)
    {
        if(!tipl::is_main_thread() || temporary)
            return;
        std::ostringstream out;

        {
            std::string unit("ms");
            float count = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - process_time.back()).count();
            if(count > 1000.0f)
            {
                count /= 1000.0f;
                unit = "s";
                if(count > 60.0f)
                {
                    count /= 60.0;
                    unit = "m";
                    if(count > 60.0f)
                    {
                        count /= 60.0f;
                        unit = "h";
                    }
                }
            }
            out << "⏱" << count << unit;
        }
        status_list.pop_back();
        print(out.str().c_str(),false,true);

        process_time.pop_back();
        t_last.pop_back();
        if(status_list.empty())
            at_list.clear();
        update_prog(get_status());
    }
};


template<typename fun_type>
bool run(const char* msg,fun_type fun)
{
    if(!show_prog)
    {
        fun();
        return true;
    }
    progress prog(msg);
    bool ended = false;
    tipl::par_for(2,[&](int i)
    {
        if(!i)
        {
            fun();
            ended = true;
        }
        else
        {
            size_t i = 0;
            while(!ended)
            {
                std::this_thread::yield();
                prog(i,i+1);
                if(prog.aborted())
                    return;
                ++i;
            }
        }
    },2);
    return !prog.aborted();
}

template<unsigned int code = 0>
class output{
    std::ostringstream s;
    public:
        ~output()
        {
            auto out = s.str();
            if(out.empty())
                return;
            if(out.back() == '\n')
                out.pop_back();
            progress::print(out.c_str(),false,false,code);
        }
        output& operator<<(std::ostream& (*var)(std::ostream&))
        {
            s << var;
            return *this;
        }
        template<typename type>
        output& operator<<(const type& v)
        {
            s << v;
            return *this;
        }
};

using out = output<>;
using error = output<0x008c9de2>;   //❌ (U+274C)
using warning = output<0x00979de2>; //❗ (U+2757)



}

#endif//TIPL_PROG_HPP
