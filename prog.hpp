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
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QPainter>
#include <QFontMetrics>
#include <QMouseEvent>
#include <QApplication>
#endif

#include "mt.hpp"
#include "po.hpp"

namespace tipl{

struct prog_status{
    std::chrono::high_resolution_clock::time_point next_update_time,start_time;
    std::string status,at;
    unsigned int now = 0,total = 0;
};
inline std::vector<prog_status> status_list;
inline std::atomic_int status_count = 0;
inline bool prog_aborted = false,show_prog = false;

inline std::mutex print_mutex,msg_mutex;
inline std::string last_msg;

#if defined(TIPL_USE_QT) && !defined(__CUDACC__)
struct progress_dialog : public QDialog{
    struct ring_widget : public QWidget{
        QPushButton cancel{"Cancel",this};
        int active = 0,pulse = 0;

        ring_widget()
        {
            setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
            setMinimumSize(220,220);
            cancel.setFixedSize(140,140);
            cancel.setStyleSheet(
                "QPushButton{border-radius:70px;background:#3b3b3b;color:white;font-weight:bold;font-size:16px;}"
                "QPushButton:hover{background:#555;font-size:18px;}QPushButton:pressed{background:#222;}");
            connect(&cancel,&QPushButton::clicked,[]{prog_aborted = true;});
        }

        void resizeEvent(QResizeEvent*) override
        {
            cancel.move((width()-cancel.width())/2,(height()-cancel.height())/2);
        }

        void paintEvent(QPaintEvent*) override
        {
            QPainter p(this);
            p.setRenderHint(QPainter::Antialiasing);
            QPoint c(width()/2,height()/2);
            int n = std::max(1,active), R = std::min(width(),height())/2 - 12;
            int gap = n > 1 ? std::min(24,(R-58)/(n-1)) : 0;

            for(int i = 1,k = 0;i < int(status_list.size());++i)
            {
                auto& s = status_list[i];
                if(!s.now || !s.total)
                    continue;

                int r = R - (n-1-k)*gap;
                QRect rc(c.x()-r,c.y()-r,r*2,r*2);
                int t = (pulse + k*25)%120;
                int v = 185 + (t < 60 ? t : 120-t);

                p.setPen(QPen(QColor(230,232,235),13,Qt::SolidLine,Qt::RoundCap));
                p.drawEllipse(rc);
                p.setPen(QPen(QColor::fromHsv((205+k*28)%360,155,v),13,Qt::SolidLine,Qt::RoundCap));
                p.drawArc(rc,90*16,-int(360.0*16.0*s.now/s.total));
                ++k;
            }

            if(!active)
            {
                int t = pulse%120;
                p.setPen(QPen(QColor::fromHsv(205,80,175 + (t < 60 ? t : 120-t)),13,Qt::SolidLine,Qt::RoundCap));
                p.drawEllipse(QRect(c.x()-R,c.y()-R,R*2,R*2));
            }
        }
    };

    QVBoxLayout box{this};
    ring_widget rings;
    QLabel text;
    QPoint drag_pos;
    bool dragging = false;

    progress_dialog()
    {
        setAttribute(Qt::WA_TranslucentBackground);
        setWindowFlags(windowFlags() | Qt::FramelessWindowHint);
        box.setContentsMargins(8,8,8,8);
        box.setSpacing(2);
        text.setAlignment(Qt::AlignCenter);
        text.setFixedWidth(310);
        box.addWidget(&rings,1);
        box.addWidget(&text,0,Qt::AlignCenter);
        installEventFilter(this);
        rings.installEventFilter(this);
        text.installEventFilter(this);
        resize(320,360);
    }

    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(255,255,255,210));
        p.drawRoundedRect(rect(),24,24);
    }

    bool eventFilter(QObject*,QEvent* event) override
    {
        if(event->type() == QEvent::MouseButtonPress)
        {
            auto* e = static_cast<QMouseEvent*>(event);
            if(e->button() == Qt::LeftButton)
            {
                dragging = true;
                drag_pos = e->globalPosition().toPoint() - frameGeometry().topLeft();
                setCursor(Qt::SizeAllCursor);
                return true;
            }
        }
        if(event->type() == QEvent::MouseMove && dragging)
        {
            move(static_cast<QMouseEvent*>(event)->globalPosition().toPoint() - drag_pos);
            return true;
        }
        if(event->type() == QEvent::MouseButtonRelease)
        {
            dragging = false;
            unsetCursor();
            return true;
        }
        return false;
    }

    void refresh()
    {
        QStringList lines;
        QFontMetrics fm(text.font());
        rings.active = 0;
        rings.pulse = (rings.pulse + 8)%120;

        for(int i = 1;i < int(status_list.size());++i)
            if(!status_list[i].status.empty())
            {
                auto s = QString::fromStdString(tipl::split(status_list[i].status,'\n').front());
                if(!status_list[i].at.empty())
                    s += " " + QString::fromStdString(status_list[i].at);
                lines << fm.elidedText(s,Qt::ElideRight,text.width()-8);
                rings.active += status_list[i].total != 0;
            }

        text.setText(lines.empty() ? "working..." : lines.join('\n'));
        text.setFixedHeight(fm.lineSpacing()*std::max<int>(1,lines.size())+4);
        rings.update();
    }
};
inline std::shared_ptr<progress_dialog> progressDialog;
#endif

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
#if defined(TIPL_USE_QT) && !defined(__CUDACC__)

        if(show_now && !progressDialog)
        {
            progressDialog.reset(new progress_dialog);
            progressDialog->show();
        }

#endif
    }
    static bool check_prog(unsigned int now,unsigned int total)
    {
        if(prog_aborted)
            return false;
        if(!show_prog || !tipl::is_main_thread() || !status_count)
            return now < total;
        auto& cur_status = status_list.back();
        auto now_time = std::chrono::high_resolution_clock::now();
        if(now == 0 || now_time < cur_status.next_update_time)
            return now < total;
        cur_status.next_update_time = now_time+std::chrono::milliseconds(200);
        cur_status.at = "(" + std::to_string(now) + "/" + std::to_string(total) + ")";
        cur_status.total = total;
        cur_status.now = now;
        if(now < total)
        {
            if(auto e = std::chrono::duration_cast<std::chrono::milliseconds>(now_time-cur_status.start_time).count()*(total-now)/now/60000; e)
                cur_status.at += " "+std::to_string(e)+" min";
        }

#if defined(TIPL_USE_QT) && !defined(__CUDACC__)
        if(!progressDialog)
        {
            progressDialog.reset(new progress_dialog);
            progressDialog->show();
        }
        progressDialog->refresh();
        QApplication::processEvents();
#endif
        if(prog_aborted)
            return progress::print("operation aborted",false,false,1),false;
        return now < total;
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
    template<typename fun_type>
    bool run(size_t total,fun_type&& fun)
    {
        unsigned int dummy = 0;
        if (!show_prog || !tipl::is_main_thread() || status_list.empty())
            return fun(dummy),true;

        status_list.back().total = total;
        status_list.back().now = 1;
        std::atomic<bool> ended{false};

        std::thread worker_thread([&]() {
            fun(status_list.back().now);
            ended = true;
        });

        #if defined(TIPL_USE_QT) && !defined(__CUDACC__)
        while (!ended && !prog_aborted)
        {
            if(status_count && std::chrono::high_resolution_clock::now() > status_list.back().next_update_time)
            {
                status_list.back().next_update_time = std::chrono::high_resolution_clock::now()+std::chrono::milliseconds(200);
                if(progressDialog)
                    progressDialog->refresh();
                QApplication::processEvents();
            }
        }
        #endif
        if (worker_thread.joinable())
            worker_thread.join();
        return !prog_aborted;
    }
    ~progress(void)
    {
        if(!tipl::is_main_thread() || temporary || status_count == 0)
            return;

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::high_resolution_clock::now() - status_list.back().start_time).count();

        auto s=ms/1000; ms%=1000;
        auto m=s/60;    s%=60;
        auto h=m/60;    m%=60;
        std::string t;
        if(h) t += std::to_string(h) + "h";
        if(m) t += std::to_string(m) + "m";
        if(s) t += std::to_string(s) + "s";
        t+=std::to_string(ms)+"ms";

        status_list.pop_back();
        --status_count;
        print("⏱" + t,false,true);

        {
            std::scoped_lock<std::mutex> lock2(msg_mutex);
            last_msg.clear();
        }

#if defined(TIPL_USE_QT) && !defined(__CUDACC__)
        if(!show_prog || !progressDialog)
            return;
        if(status_count <= 1)
            progressDialog->close(),progressDialog.reset();
#endif

    }
};

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
