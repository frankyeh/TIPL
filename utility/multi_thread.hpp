#ifndef MULTI_THREAD_HPP
#define MULTI_THREAD_HPP
#include <future>
namespace image{
template <class T,class Func>
void par_for(T size, Func f, int thread_count = std::thread::hardware_concurrency())
{
    std::vector<std::future<void> > futures;
    if(thread_count > size)
        thread_count = int(size);
    for(int id = 1; id < thread_count; id++)
    {
        futures.push_back(std::move(std::async(std::launch::async, [id,size,thread_count,&f]
        {
            for(int i = id; i < size; i += thread_count)
                f(i);
        })));
    }
    for(int i = 0; i < size; i += thread_count)
        f(i);
    for(auto &future : futures)
        future.wait();
}

template <class T,class Func>
void par_for2(T size, Func f, int thread_count = std::thread::hardware_concurrency())
{
    std::vector<std::future<void> > futures;
    if(thread_count > size)
        thread_count = size;
    for(int id = 1; id < thread_count; id++)
    {
        futures.push_back(std::move(std::async(std::launch::async, [id,size,thread_count,&f]
        {
            for(int i = id; i < size; i += thread_count)
                f(i,id);
        })));
    }
    for(int i = 0; i < size; i += thread_count)
        f(i,0);
    for(auto &future : futures)
        future.wait();
}

template <class T,class Func>
void par_for_block(T size, Func f, int thread_count = std::thread::hardware_concurrency())
{
    if(!size)
        return;
    std::vector<std::future<void> > futures;
    if(thread_count > size)
        thread_count = size;

    size_t block_size = size/thread_count;
    size_t pos = 0;
    for(int id = 1; id < thread_count; id++)
    {
        size_t end = pos + block_size;
        futures.push_back(std::move(std::async(std::launch::async, [pos,end,&f]
        {
            for(size_t i = pos; i < end;++i)
                f(i);
        })));
        pos = end;
    }
    for(size_t i = pos; i < size;++i)
        f(i);
    for(auto &future : futures)
        future.wait();
}

template <class T,class Func>
void par_for_block2(T size, Func f, int thread_count = std::thread::hardware_concurrency())
{
    if(!size)
        return;
    std::vector<std::future<void> > futures;
    if(thread_count > size)
        thread_count = size;

    size_t block_size = size/thread_count;
    size_t pos = 0;
    for(int id = 1; id < thread_count; id++)
    {
        size_t end = pos + block_size;
        futures.push_back(std::move(std::async(std::launch::async, [id,pos,end,&f]
        {
            for(size_t i = pos; i < end;++i)
                f(i,id);
        })));
        pos = end;
    }
    for(size_t i = pos; i < size;++i)
        f(i,0);
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

    template<class lambda_type>
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

