#ifndef MULTI_THREAD_HPP
#define MULTI_THREAD_HPP
#include <future>
#include <iostream>
namespace tipl{

class time
{
 public:
    time():  t1(std::chrono::high_resolution_clock::now()){}
    template<typename T> // T: std::chrono::milliseconds
    double elapsed(){return std::chrono::duration_cast<T>(std::chrono::high_resolution_clock::now() - t1).count();}
    void restart(){t1 = std::chrono::high_resolution_clock::now();}
    void start(){t1 = std::chrono::high_resolution_clock::now();}
    void stop(){t2 = std::chrono::high_resolution_clock::now();}
    double total(){stop();return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();}
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

template <typename T,typename Func,typename std::enable_if<std::is_integral<T>::value,bool>::type = true>
void par_for(T size, Func&& f, unsigned int thread_count = std::thread::hardware_concurrency())
{
#ifdef USING_XEUS_CLING
// cling still has an issue using std::future
// https://github.com/root-project/cling/issues/387
    for(T i = 0; i < size;++i)
        f(i);
#else
    std::vector<std::future<void> > futures;
    if(thread_count > size)
        thread_count = int(size);
    for(unsigned int id = 1; id < thread_count; id++)
    {
        futures.push_back(std::move(std::async(std::launch::async, [id,size,thread_count,&f]
        {
            for(T i = id; i < size; i += thread_count)
                f(i);
        })));
    }
    for(T i = 0; i < size; i += thread_count)
        f(i);
    for(auto &future : futures)
        future.wait();
#endif
}


template <typename T,typename Func,typename std::enable_if<std::is_integral<T>::value,bool>::type = true>
void par_for2(T size,Func&& f, unsigned int thread_count = std::thread::hardware_concurrency())
{
#ifdef USING_XEUS_CLING
// cling still has an issue using std::future
// https://github.com/root-project/cling/issues/387
    for(T i = 0; i < size;++i)
        f(i,i%thread_count);
#else
    std::vector<std::future<void> > futures;
    if(thread_count > size)
        thread_count = size;
    for(uint16_t id = 1; id < thread_count; id++)
    {
        futures.push_back(std::move(std::async(std::launch::async, [id,size,thread_count,&f]
        {
            for(T i = id; i < size; i += thread_count)
                f(i,id);
        })));
    }
    for(T i = 0; i < size; i += thread_count)
        f(i,0);
    for(auto &future : futures)
        future.wait();
#endif
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

