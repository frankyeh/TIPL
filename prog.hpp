#ifndef TIPL_PROG_HPP
#define TIPL_PROG_HPP
#include <memory>
#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include <sstream>
#include <atomic>


#if defined(TIPL_USE_QT) && !defined(__CUDACC__)
#include <QProgressDialog>
#include <QApplication>
#endif

#include "mt.hpp"
#include "po.hpp"

namespace tipl{

struct prog_status{
    std::chrono::high_resolution_clock::time_point next_update_time,start_time;
    std::string status,at;
};
inline std::vector<prog_status> status_list;
inline std::atomic_int status_count = 0;
inline bool prog_aborted = false,show_prog = false;

inline std::mutex print_mutex,msg_mutex;
inline std::string last_msg;
#if defined(TIPL_USE_QT) && !defined(__CUDACC__)
inline std::shared_ptr<QProgressDialog> progressDialog;
#endif

inline void create_prog(void)
{
#if defined(TIPL_USE_QT) && !defined(__CUDACC__)
    if(!show_prog || !tipl::is_main_thread())
        return;
    progressDialog.reset(new QProgressDialog(QString(), "Cancel", 0, 1000));
    progressDialog->setMinimumWidth(500);
    progressDialog->setAutoClose(false);
    progressDialog->setAutoReset(false);
    progressDialog->show();
    QApplication::processEvents(QEventLoop::AllEvents,20);
#endif
}
inline void close_prog(void)
{
#if defined(TIPL_USE_QT) && !defined(__CUDACC__)
    if(!show_prog || !tipl::is_main_thread() || !progressDialog)
        return;
    if(status_count <= 1)
        progressDialog->close(),progressDialog.reset();
    QApplication::processEvents();
#endif
}

class progress{
private:
    void begin_prog(const std::string& status,bool show_now = false)
    {
        if(!tipl::is_main_thread())
            return;
        auto t = std::chrono::high_resolution_clock::now();
        prog_aborted = false;
        status_list.push_back({t,t,status,{}});
        ++status_count;
        last_msg.clear();
        if(show_now)
            create_prog();
    }
    static bool check_prog(unsigned int now,unsigned int total)
    {
        if(!show_prog || !tipl::is_main_thread() || !status_count)
            return !prog_aborted && now < total;
        auto& cur_status = status_list.back();
        if(now >= total || prog_aborted)
        {
            cur_status.at.clear();
            return false;
        }
        auto now_time = std::chrono::high_resolution_clock::now();
        if(now_time < cur_status.next_update_time)
            return true;
        cur_status.next_update_time = now_time+std::chrono::milliseconds(500);
        cur_status.at = "(" + std::to_string(now) + "/" + std::to_string(total) + ")";
        int exp_min = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - cur_status.start_time).count()*(total-now)/(now+1)/1000/60;
        if(exp_min)
            cur_status.at += " " + std::to_string(exp_min) + " min";

#if defined(TIPL_USE_QT) && !defined(__CUDACC__)
        if(!progressDialog.get())
            create_prog();
        if(progressDialog)
        {
            std::string result;
            for(size_t i = 1;i < status_list.size();++i)
                if(!status_list[i].status.empty())
                    result += tipl::split(status_list[i].status,'\n').front() + " " + status_list[i].at + "\n";
            {
                std::scoped_lock<std::mutex> lock(msg_mutex);
                if(last_msg.empty() && !result.empty())
                    result.pop_back();
            }
            progressDialog->setLabelText(QString::fromStdString(result + last_msg));
            progressDialog->setValue(total ? int(std::min<uint64_t>(1000, uint64_t(now)*1000/total)) : 0);
            if(progressDialog->wasCanceled())
            {
                prog_aborted = true;
                progress::print("operation aborted",false,false,1);
            }
            QApplication::processEvents(QEventLoop::AllEvents,20);
        }
#endif
        return !prog_aborted;
    }
    static std::string get_head(bool head_node,bool tail_node)
    {
        int count = status_count.load();
        std::string head;
        for(int i = 1;i < count;++i)
            head += "│  ";
        if(!tipl::is_main_thread())
            return head + "│  ";
        if(!count)
            return head_node ? "┌──" : tail_node ? "└──" : "";
        head += head_node ? "├──┬──" : tail_node ? "│  " : "├──";
        return tail_node ? head + "└──" : head;
    }
    static std::string get_color_line(std::string line,bool head_node,int error_code)
    {
        auto color = [](const char* c,const std::string& s){return std::string(c) + s + "\033[0m";};

        if(error_code == 1) return color("\033[1;31m","❌" + line);
        if(error_code == 2) return color("\033[1;31m","❗" + line);
        if(tipl::begins_with(line,"sav")) return color("\033[1;35m","💾" + line);
        if(tipl::begins_with(line,"open")) return color("\033[1;35m","📂" + line);
        if(head_node) return color("\033[1;34m","📟" + line);

        if(auto p = line.find('=');p != std::string::npos)
            return color("\033[0;32m",line.substr(0,p)) + line.substr(p);

        if(auto p = line.find(": ");p != std::string::npos)
            return color("\033[0;33m",line.substr(0,p)) + line.substr(p);

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
#if defined(TIPL_USE_QT) && !defined(__CUDACC__)
        if(tipl::is_main_thread() && progressDialog.get() && status_count &&
            std::chrono::high_resolution_clock::now() > status_list.back().next_update_time)
        {
            status_list.back().next_update_time = std::chrono::high_resolution_clock::now()+std::chrono::milliseconds(500);
            QApplication::processEvents();
        }
#endif
    }
public:
    bool temporary = false;
    progress(void):temporary(true){}
    void operator=(progress&& rhs)
    {
        if(temporary && !rhs.temporary)
            temporary = false,rhs.temporary = true;
    }
    progress(const std::string& status,bool show_now = false)
    {
        print(status,true,false);
        begin_prog(status,show_now);
    }
    progress(const std::string& a,const std::string& b,bool show_now = false):progress(a + b,show_now) {}
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
        if(!tipl::is_main_thread() || temporary || status_count == 0)
            return;

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::high_resolution_clock::now() - status_list.back().start_time).count();

        auto h = ms/3600000; ms %= 3600000;
        auto m = ms/60000;   ms %= 60000;
        auto s = ms/1000;    ms %= 1000;

        std::string t;
        if(h) t += std::to_string(h) + "h ";
        if(m) t += std::to_string(m) + "m ";
        if(s) t += std::to_string(s) + "s ";
        t += std::to_string(ms) + "ms";

        status_list.pop_back();
        --status_count;
        print("⏱" + t,false,true);


        {
            std::scoped_lock<std::mutex> lock2(msg_mutex);
            last_msg.clear();
        }
        close_prog();

    }
};


template<typename fun_type>
bool run(const std::string& msg, fun_type fun)
{
    if (!show_prog)
    {
        fun();
        return true;
    }

    progress prog(msg);
    std::atomic<bool> ended{false};

    std::thread worker_thread([&]() {
        fun();
        ended = true;
    });

    size_t count = 0;
    while (!ended)
    {
        prog(count, count + 1);
        if (prog.aborted())
            break;
        ++count;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    if (worker_thread.joinable())
        worker_thread.join();
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
    output& operator<<(std::ostream& (*v)(std::ostream&)){s << v; return *this;}
    template<typename T> output& operator<<(const T& v){s << v; return *this;}
};

using out = output<0>;
using error = output<1>;   //❌
using warning = output<2>; //❗



}

#endif//TIPL_PROG_HPP
