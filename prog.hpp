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
inline std::vector<std::chrono::high_resolution_clock::time_point> process_time;
inline std::chrono::high_resolution_clock::time_point next_update_time = std::chrono::high_resolution_clock::now();
inline std::vector<std::string> status_list,at_list;
inline std::atomic_int status_count = 0;
inline std::mutex print_mutex,msg_mutex;
inline std::string last_msg;
#if defined(TIPL_USE_QT) && !defined(__CUDACC__)
inline std::shared_ptr<QProgressDialog> progressDialog;
#endif

// can only called in main thread
inline std::string get_prog_status(void)
{
    std::string result;
    for(size_t i = 1;i < status_list.size();++i)
    {
        if(status_list[i].empty())
            continue;
        result += tipl::split(status_list[i],'\n').front();
        if(i < at_list.size())
            result += " " + at_list[i];
        if(result.back() != '\n')
            result += "\n";
    }
    std::scoped_lock<std::mutex> lock2(msg_mutex);
    if(last_msg.empty() && !result.empty())
        result.pop_back();
    return result+last_msg;
}

inline void create_prog(void)
{
    #if defined(TIPL_USE_QT) && !defined(__CUDACC__)
    if(!show_prog || !tipl::is_main_thread())
        return;
    progressDialog.reset(new QProgressDialog(get_prog_status().c_str(),"Cancel",0,100));
    progressDialog->setAttribute(Qt::WA_ShowWithoutActivating);
    progressDialog->activateWindow();
    progressDialog->show();
    progressDialog->raise();
    QApplication::processEvents();
    #endif
}
inline void close_prog(void)
{
    #if defined(TIPL_USE_QT) && !defined(__CUDACC__)
    if(!show_prog || !tipl::is_main_thread() || status_count > 1)
        return;
    if(progressDialog.get())
    {
        progressDialog->close();
        progressDialog.reset();
        QApplication::processEvents();
    }
    #endif
}

class progress{
private:
    void begin_prog(const std::string& status,bool show_now = false)
    {
        if(!tipl::is_main_thread())
            return;
        prog_aborted = false;
        status_list.push_back(status);
        ++status_count;
        process_time.resize(status_list.size());
        process_time.back() = std::chrono::high_resolution_clock::now();
        last_msg.clear();
        if(show_now)
            create_prog();
    }
    static bool check_prog(unsigned int now,unsigned int total)
    {
        if(!show_prog || !tipl::is_main_thread() || !status_count)
        {
            if(prog_aborted)
                return false;
            return now < total;
        }
        if(now >= total || prog_aborted)
        {
            if(at_list.size() == status_list.size())
                at_list.back().clear();
            return false;
        }
        if(std::chrono::high_resolution_clock::now() < next_update_time)
            return true;
        next_update_time = std::chrono::high_resolution_clock::now()+std::chrono::milliseconds(500);
        at_list.resize(status_list.size());
        std::ostringstream outstr;
        outstr << "(" << now << "/" << total << ")";
        {
            int exp_min = (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                       process_time.back()).count()*(total-now)/(now+1)/1000/60);
            if(exp_min)
                outstr << " " << exp_min << " min";
        }
        at_list.back() = outstr.str();
        #if defined(TIPL_USE_QT) && !defined(__CUDACC__)
        if(!progressDialog.get())
            create_prog();
        progressDialog->setLabelText(get_prog_status().c_str());
        progressDialog->adjustSize();
        progressDialog->setRange(0, int(total));
        progressDialog->setValue(int(now));
        if(progressDialog->wasCanceled())
        {
            prog_aborted = true;
            progress::print("operation aborted",false,false,1);
        }
        QApplication::processEvents();
        #endif
        return !prog_aborted;
    }
    static std::string get_head(bool head_node, bool tail_node)
    {
        std::string head;
        for(size_t i = 1;i < status_count;++i)
            head += "│  ";
        if(!tipl::is_main_thread())
        {
            head += "│  ";
            return head;
        }
        if(status_count)
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
        return head;
    }
    static std::string get_color_line(std::string line,bool head_node,int error_code)
    {
        std::string color_end = "\033[0m";
        std::string color31 = "\033[1;31m";
        std::string color32 = "\033[0;32m";
        std::string color33 = "\033[0;33m";
        std::string color34 = "\033[1;34m";
        std::string color35 = "\033[1;35m";
        if(error_code == 1) // ❌
            return color31 + "❌" + line + color_end;
        if(error_code == 2) // ❗
            return color31 + "❗" + line + color_end;
        if(tipl::begins_with(line,"sav"))
            return color35 + "💾" + line + color_end;
        if(tipl::begins_with(line,"open"))
            return color35 + "📂" + line + color_end;
        if(head_node)
            return color34 + std::string("📟") + line + color_end;

        auto eq_pos = line.find('=');
        if(eq_pos != std::string::npos)
            return color32 + line.substr(0,eq_pos) + color_end + line.substr(eq_pos);

        auto info_pos = line.find(": ");
        if(info_pos != std::string::npos)
            return color33 + line.substr(0,info_pos) + color_end + line.substr(info_pos);
        return line;
    }
public:
    static void print(const std::string& status,bool head_node, bool tail_node,int error_code = 0)
    {
        std::scoped_lock<std::mutex> lock(print_mutex);
        std::istringstream in(status);
        std::string line;
        while(std::getline(in,line))
        {
            if(line.empty())
                continue;
            auto new_line = get_color_line(line,head_node,error_code);
            if(!tipl::is_main_thread())
            {
                std::ostringstream out;
                out << "[thread " << std::this_thread::get_id() << "]" << new_line;
                new_line = out.str();
            }   
            std::cout << get_head(head_node,tail_node) + new_line << std::endl;
            head_node = false;
        }
    }
public:
    bool temporary = false;
    progress(void):temporary(true){}
    void operator=(progress&& rhs)
    {
        if(temporary && !rhs.temporary)
        {
            temporary = false;
            rhs.temporary = true;
        }
    }
    progress(const std::string& status,bool show_now = false)
    {
        print(status,true,false);
        begin_prog(status,show_now);
    }
    progress(const std::string& status1,const std::string& status2,bool show_now = false)
    {
        std::string s(status1);
        s += status2;
        print(s.c_str(),true,false);
        begin_prog(s.c_str(),show_now);
    }
    static bool is_running(void) {return status_count > 1;}
    static bool aborted(void) { return prog_aborted;}
    template<typename value_type1,typename value_type2>
    bool operator()(value_type1 now,value_type2 total) const
    {
        if(temporary)
            return now < total;
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
        --status_count;
        print(out.str().c_str(),false,true);
        process_time.pop_back();
        if(status_list.empty())
            at_list.clear();
        {
            std::scoped_lock<std::mutex> lock2(msg_mutex);
            last_msg.clear();
        }
        close_prog();

    }
};


template<typename fun_type>
bool run(const std::string& msg,fun_type fun)
{
    if(!show_prog)
    {
        fun();
        return true;
    }
    progress prog(msg);
    std::atomic_bool ended = false;
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
                prog(i,i+1);
                if(prog.aborted())
                    return;
                ++i;
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
    },2);
    return !prog.aborted();
}

template<int code>
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
            progress::print(out,false,false,code);
            std::scoped_lock<std::mutex> lock2(msg_mutex);
            std::getline(std::istringstream(out),last_msg);
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

using out = output<0>;
using error = output<1>;   //❌
using warning = output<2>; //❗



}

#endif//TIPL_PROG_HPP
