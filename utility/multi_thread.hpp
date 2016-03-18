#ifndef MULTI_THREAD_HPP
#define MULTI_THREAD_HPP
#include <future>
namespace image{
template <typename T, typename Func>
void par_for(T size, Func f, int thread_count = std::thread::hardware_concurrency())
{
    std::vector<std::future<void> > futures;
    for(int id = 0; id < thread_count; id++)
    {
        futures.push_back(std::move(std::async(std::launch::async, [id,size,thread_count,&f]
        {
            for(int i = id; i < size; i += thread_count)
                f(i);
        })));
    }
    for(auto &future : futures)
        future.wait();
}

template <typename T, typename Func>
void par_for2(T size, Func f, int thread_count = std::thread::hardware_concurrency())
{
    std::vector<std::future<void> > futures;
    for(int id = 0; id < thread_count; id++)
    {
        futures.push_back(std::move(std::async(std::launch::async, [id,size,thread_count,&f]
        {
            for(int i = id; i < size; i += thread_count)
                f(i,id);
        })));
    }
    for(auto &future : futures)
        future.wait();
}


}
#endif // MULTI_THREAD_HPP

