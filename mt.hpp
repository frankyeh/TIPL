#ifndef MULTI_THREAD_HPP
#define MULTI_THREAD_HPP
#include <future>
#include <iostream>
#include "def.hpp"
namespace tipl{

class time
{
    public:
        time(const char* msg_):msg(msg_),t1(std::chrono::high_resolution_clock::now()){}
        time():  t1(std::chrono::high_resolution_clock::now()){}
        void restart(){t1 = std::chrono::high_resolution_clock::now();}
        void start(){t1 = std::chrono::high_resolution_clock::now();}
        void stop(){t2 = std::chrono::high_resolution_clock::now();}
    public:
        template<typename T>
        auto elapsed(){return std::chrono::duration_cast<T>(std::chrono::high_resolution_clock::now() - t1).count();}
        template<typename T>
        auto total(){stop();return std::chrono::duration_cast<T>(t2 - t1).count();}
        ~time()
        {
            if(!msg.empty())
                std::cout << msg << elapsed<std::chrono::milliseconds>() << std::endl;
        }
    private:
        std::string msg;
        std::chrono::high_resolution_clock::time_point t1, t2;
};

class estimate_time{
    std::string name;
    size_t n = 0;
    double time_total = 0.0;
    std::chrono::high_resolution_clock::time_point s;
public:
    estimate_time(const char* name_):name(name_){}
    ~estimate_time()
    {
        std::cout << name.c_str() << time_total/double(n) << " microseconds" << std::endl;
    }
    void start(void)
    {
        s = std::chrono::high_resolution_clock::now();
    }
    void stop(void)
    {
        auto stop = std::chrono::high_resolution_clock::now();
        time_total += std::chrono::duration_cast<std::chrono::microseconds>(stop-s).count();
        ++n;
    }
};

template<int>
bool is_main_thread(void)
{
    static auto main_thread_id = std::this_thread::get_id();
    return main_thread_id == std::this_thread::get_id();
}


template <bool enabled_mt = true,typename T,typename Func,typename std::enable_if<
              std::is_integral<T>::value ||
              std::is_class<T>::value,bool>::type = true>
void par_for(T from,T to,Func&& f,unsigned int thread_count = std::thread::hardware_concurrency())
{
    if(to == from)
        return;
    if(!is_main_thread<0>() || tipl::use_xeus_cling || !enabled_mt)
    {
        for(;from != to;++from)
            if constexpr(function_traits<Func>::arg_num == 2)
                f(from,0);
            else
                f(from);
        return;
    }

    if constexpr(!tipl::use_xeus_cling && enabled_mt)
    {
        size_t size = to-from;
        if(thread_count > size)
            thread_count = size;
        size_t block_size = size/thread_count;
        size_t remainder = size % thread_count;
        std::vector<std::thread> threads;
        for(unsigned int id = 1; id < thread_count; id++)
        {
            auto block_end = from + block_size + (id <= remainder ? 1 : 0);
            if constexpr(function_traits<Func>::arg_num == 2)
            {
                threads.push_back(std::thread([=,&f]
                {
                    for(auto pos = from;pos != block_end;++pos)
                        f(pos,id);
                }));
            }
            else
            {
                threads.push_back(std::thread([=,&f]
                {
                    for(auto pos = from;pos != block_end;++pos)
                        f(pos);
                }));
            }
            from = block_end;
        }
        for(;from != to;++from)
        {
            if constexpr(function_traits<Func>::arg_num == 2)
                f(from,0);
            else
                f(from);
        }
        for(auto &thread : threads)
            thread.join();
    }
}

template <bool enabled_mt = true,typename T,typename Func,
          typename std::enable_if<std::is_integral<T>::value,bool>::type = true>
inline void par_for(T size, Func&& f,unsigned int thread_count = std::thread::hardware_concurrency())
{
    par_for<enabled_mt>(T(0),size,std::move(f),thread_count);
}

template <bool enabled_mt = true,typename T,typename Func,
          typename std::enable_if<std::is_class<T>::value,bool>::type = true>
inline void par_for(T& c, Func&& f,unsigned int thread_count = std::thread::hardware_concurrency())
{
    par_for<enabled_mt>(c.begin(),c.end(),std::move(f),thread_count);
}

template<typename T>
void aggregate_results(std::vector<std::vector<T> >&& results,std::vector<T>& all_result_)
{
    std::vector<size_t> insert_pos;
    insert_pos.push_back(0);
    for(size_t i = 0;i < results.size();++i)
        insert_pos.push_back(insert_pos.back() + results[i].size());

    std::vector<T> all_result(insert_pos.back());
    tipl::par_for(results.size(),[&](unsigned int index)
    {
        std::move(results[index].begin(),results[index].end(),all_result.begin()+int64_t(insert_pos[index]));
    });
    all_result.swap(all_result_);
}

namespace backend {
    struct seq{
        template<typename Fun>
        inline void operator()(size_t n,Fun&& f)
        {
            for(size_t i = 0;i < n;++i)
                f(i);
        }
    };
    struct mt{
        template<typename Fun>
        inline void operator()(size_t n,Fun&& f)
        {
            par_for(n,std::move(f));
        }
    };
}


class thread{
    std::shared_ptr<std::future<void> > th;
public:
    bool running = false;
    bool terminated = false;
public:
    thread(void){}
    ~thread(void){clear();}
    void clear(void)
    {
        if(th.get())
        {
            terminated = true;
            th->wait();
            th.reset();
        }
        terminated = false;
        running = false;
    }

    template<typename lambda_type>
    void run(lambda_type&& fun)
    {
        if(th.get())
            clear();
        running = true;
        th.reset(new std::future<void>(std::async(std::launch::async,fun)));
    }
    void wait(void)
    {
        th->wait();
    }
};


}
#endif // MULTI_THREAD_HPP

