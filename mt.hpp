#ifndef MULTI_THREAD_HPP
#define MULTI_THREAD_HPP
#include <future>
#include <iostream>
#include "def.hpp"
namespace tipl{

class time
{
    public:
        time():  t1(std::chrono::high_resolution_clock::now()){}
        void restart(){t1 = std::chrono::high_resolution_clock::now();}
        void start(){t1 = std::chrono::high_resolution_clock::now();}
        void stop(){t2 = std::chrono::high_resolution_clock::now();}
    public:
        template<typename T = std::chrono::milliseconds>
        double elapsed(){return std::chrono::duration_cast<T>(std::chrono::high_resolution_clock::now() - t1).count();}
        template<typename T = std::chrono::milliseconds>
        double total(){stop();return std::chrono::duration_cast<T>(t2 - t1).count();}
        ~time(){}
    private:
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

template <typename T,typename Func,typename std::enable_if<
              std::is_integral<T>::value ||
              std::is_class<T>::value,bool>::type = true>
void par_for(T from,T to, Func&& f, unsigned int thread_count = std::thread::hardware_concurrency())
{
    if constexpr(tipl::use_xeus_cling)
    {
        for(;from != to;++from)
            if constexpr(function_traits<Func>::arg_num == 2)
                f(from,0);
            else
                f(from);
        return;
    }
    if(to == from)
        return;
    size_t size = to-from;
    if(thread_count > size)
        thread_count = size;
    size_t block_size = size/thread_count;
    std::vector<std::future<void> > futures;
    for(unsigned int id = 1; id < thread_count; id++)
    {
        auto block_end = from + block_size;
        if constexpr(function_traits<Func>::arg_num == 2)
        {
            futures.push_back(std::move(std::async(std::launch::async, [=,&f]
            {
                auto pos = from;
                for(;pos != block_end;++pos)
                    f(pos,id);
            })));
        }
        else
        {
            futures.push_back(std::move(std::async(std::launch::async, [=,&f]
            {
                auto pos = from;
                for(;pos != block_end;++pos)
                    f(pos);
            })));
        }
        from = block_end;
    }
    for(;from != to;++from)
        if constexpr(function_traits<Func>::arg_num == 2)
            f(from,0);
        else
            f(from);
    for(auto &future : futures)
        future.wait();
}

template <typename T,typename Func,typename std::enable_if<std::is_integral<T>::value,bool>::type = true>
void par_for(T size, Func&& f, unsigned int thread_count = std::thread::hardware_concurrency())
{
    if constexpr(tipl::use_xeus_cling)
    {
        for(T i = 0; i < size;++i)
            if constexpr(function_traits<Func>::arg_num == 2)
                f(i,0);
            else
                f(i);
        return;
    }

    std::vector<std::future<void> > futures;
    if(thread_count > size)
        thread_count = int(size);
    for(unsigned int id = 1; id < thread_count; id++)
    {
        if constexpr(function_traits<Func>::arg_num == 2)
        {
            futures.push_back(std::move(std::async(std::launch::async, [=,&f]
            {
                for(T i = id; i < size; i += thread_count)
                    f(i,id);
            })));
        }
        else
        {
            futures.push_back(std::move(std::async(std::launch::async, [=,&f]
            {
                for(T i = id; i < size; i += thread_count)
                    f(i);
            })));
        }
    }
    for(T i = 0; i < size; i += thread_count)
        if constexpr(function_traits<Func>::arg_num == 2)
            f(i,0);
        else
            f(i);
    for(auto &future : futures)
        future.wait();
}

class thread{
private:
    std::shared_ptr<std::future<void> > th;
    bool started;
public:
    bool terminated;
public:
    thread(void):started(false),terminated(false){}
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
        started = false;
    }

    template<typename lambda_type>
    void run(lambda_type&& fun)
    {
        if(started)
            clear();
        started = true;
        th.reset(new std::future<void>(std::async(std::launch::async,fun)));
    }
    void wait(void)
    {
        th->wait();
    }
    bool has_started(void)const{return started;}
};


}
#endif // MULTI_THREAD_HPP

