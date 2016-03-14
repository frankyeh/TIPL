#ifndef MULTI_THREAD_HPP
#define MULTI_THREAD_HPP
#include <future>
namespace image{
template <typename T, typename Func>
void par_for(T size, Func f)
{
    std::vector<std::future<void> > futures;
    for(int id = 0; id < std::thread::hardware_concurrency(); id++)
    {
        futures.push_back(std::move(std::async(std::launch::async, [id,size,&f]
        {
            int thread = std::thread::hardware_concurrency();
            for(int i = id; i < size; i += thread)
                f(i);
        })));
    }
    for(auto &future : futures)
        future.wait();
}

}
#endif // MULTI_THREAD_HPP

