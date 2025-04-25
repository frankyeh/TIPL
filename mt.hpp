#ifndef MULTI_THREAD_HPP
#define MULTI_THREAD_HPP
#include <chrono>
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
        std::cout << name << time_total/double(n) << " microseconds" << std::endl;
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


template<typename T>
auto estimate_run_time(T&& fun)
{
    auto start = std::chrono::high_resolution_clock::now();
    fun();
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
}

inline auto main_thread_id = std::this_thread::get_id();
inline bool is_main_thread(void)
{
    return main_thread_id == std::this_thread::get_id();
}

inline int max_thread_count = std::thread::hardware_concurrency();


enum par_for_type{
    sequential = 0,
    sequential_with_id = 1,
    ranged = 2,
    ranged_with_id = 3
};
inline bool par_for_running = false;
template <par_for_type type = sequential,typename T,
          typename Func,typename std::enable_if<
              std::is_integral<T>::value ||
              std::is_class<T>::value ||
              std::is_pointer<T>::value,bool>::type = true>
__HOST__ void par_for(T from,T to,Func&& f,int thread_count)
{
    if(to == from)
        return;
    size_t n = to-from;
    thread_count = std::max<int>(1,std::min<int>(thread_count,n));
    if(par_for_running)
        thread_count = 1;
    if(thread_count > 1)
        par_for_running = true;
    #ifdef __CUDACC__
    int cur_device = 0;
    if constexpr(use_cuda)
    {
        if(thread_count > 1 && cudaGetDevice(&cur_device) != cudaSuccess)
            throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
    }
    #endif

    auto run = [=,&f](T beg,T end,size_t id)
    {
        #ifdef __CUDACC__
        if constexpr(use_cuda)
        {
            if(id && cudaSetDevice(cur_device) != cudaSuccess)
                throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
        }
        #endif
        if constexpr(type >= ranged)
        {
            if constexpr(type == ranged_with_id)
                f(beg,end,id);
            else
                f(beg,end);
        }
        else
        {
            for(;beg != end;++beg)
                if constexpr(type == sequential_with_id)
                    f(beg,id);
                else
                    f(beg);
        }
    };

    std::vector<std::thread> threads;
    if(thread_count > 1)
    {
        size_t block_size = n / thread_count;
        size_t remainder = n % thread_count;
        for(size_t id = 1; id < thread_count; id++)
        {
            auto end = from + block_size + (id <= remainder ? 1 : 0);
            threads.push_back(std::thread(run,from,end,id));
            from = end;
        }
        par_for_running = false;
    }
    run(from,to,0);
    for(auto &thread : threads)
        thread.join();
}


template <par_for_type type = sequential,typename T,
          typename Func,typename std::enable_if<
              std::is_integral<T>::value ||
              std::is_class<T>::value ||
              std::is_pointer<T>::value,bool>::type = true>
void par_for(T from,T to,Func&& f)
{
    if(par_for_running)
    {
        par_for<type>(from,to,std::forward<Func>(f),1);
        return;
    }
    par_for<type>(from,to,std::forward<Func>(f),max_thread_count);
}

template <par_for_type type = sequential,typename T,typename Func,
          typename std::enable_if<std::is_integral<T>::value,bool>::type = true>
inline void par_for(T size, Func&& f,unsigned int thread_count)
{
    par_for<type>(T(),size,std::forward<Func>(f),thread_count);
}
template <par_for_type type = sequential,typename T,typename Func,
          typename std::enable_if<std::is_integral<T>::value,bool>::type = true>
inline void par_for(T size, Func&& f)
{
    par_for<type>(T(),size,std::forward<Func>(f));
}

template <par_for_type type = sequential,typename T,typename Func>
inline typename std::enable_if<
    std::is_same<decltype(std::declval<T>().begin()), decltype(std::declval<T>().end())>::value>::type
par_for(T& c, Func&& f,unsigned int thread_count)
{
    par_for<type>(c.begin(),c.end(),std::forward<Func>(f),thread_count);
}
template <par_for_type type = sequential,typename T,typename Func>
inline typename std::enable_if<
    std::is_same<decltype(std::declval<T>().begin()), decltype(std::declval<T>().end())>::value>::type
par_for(T& c, Func&& f)
{
    par_for<type>(c.begin(),c.end(),std::forward<Func>(f));
}

template <par_for_type type = sequential,typename T, typename Func>
size_t adaptive_par_for(T from, T to, Func&& f)
{
    if(to-from <= 8 || !tipl::is_main_thread() || par_for_running)
    {
        par_for<type>(from,to,std::forward<Func>(f),1);
        return 1;
    }
    auto block_size = std::max<decltype(to-from)>(1,(to-from) >> 6);
    double run_time_per_block,thread_overhead;
    do
    {
        if(from + block_size*6 > to)
        {
            par_for<type>(from,to,std::forward<Func>(f),1);
            return 1;
        }
        // estimate thread overhead burden
        run_time_per_block = estimate_run_time([&](void){par_for<type>(from,from+block_size, std::forward<Func>(f),1);});
        from += block_size;

        thread_overhead = estimate_run_time([&](void){par_for<type>(from,from+block_size+block_size, std::forward<Func>(f),2);});
        from += block_size + block_size;

    }
    while(run_time_per_block >= thread_overhead);

    thread_overhead -= run_time_per_block;

    int64_t num_block = (to-from)/block_size;

    // optimize estimated_time = (num_block / thread_count) * run_time_per_block + (thread_count-1)*thread_overhead;
    // solving a*(x-1)+b/x, where a=thread_overhead and b=num_block*run_time_per_block
    // the x*=sqrt(b/a)
    int optimal_threads = std::min<int>(std::max<int>(1,std::sqrt(num_block*run_time_per_block/thread_overhead)),max_thread_count);

    par_for<type>(from, to,std::forward<Func>(f), optimal_threads);
    return optimal_threads;
}

template <par_for_type type = sequential,typename T,typename Func,
          typename std::enable_if<std::is_integral<T>::value,bool>::type = true>
inline size_t adaptive_par_for(T size, Func&& f)
{
    return adaptive_par_for<type>(T(),size,std::forward<Func>(f));
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
        template<typename Func>
        inline void operator()(size_t n,Func&& f)
        {
            par_for(n,std::forward<Func>(f));
        }
    };
}


class thread{
    std::unique_ptr<std::thread> th;
public:
    bool running = false;
    bool terminated = false;
    #ifdef __CUDACC__
    int cur_device = 0;
    #endif
public:
    thread(void){}
    ~thread(void)
    {
        #ifdef __CUDACC__
        if constexpr(use_cuda)
            cudaDeviceSynchronize();
        #endif
        clear();
    }
    void clear(void)
    {
        if(th)
        {
            terminated = true;
            th->join();
            th.reset();
        }
        terminated = false;
        running = false;
    }
    template<typename lambda_type>
    void run(lambda_type&& fun)
    {
        clear();
        #ifdef __CUDACC__
        if constexpr(use_cuda)
        {
            if(cudaGetDevice(&cur_device) != cudaSuccess)
                throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
        }
        #endif
        th = std::make_unique<std::thread>([this,fun = std::forward<lambda_type>(fun)]()
        {
            running = true;
            #ifdef __CUDACC__
            if constexpr(use_cuda)
            {
                if(cudaSetDevice(cur_device) != cudaSuccess)
                    throw std::runtime_error(cudaGetErrorName(cudaGetLastError()));
            }
            #endif
            try{
            fun();
            }
            catch(...){
                running = false;
                throw;
            }
            running = false;
        });
    }
    void join(void)
    {
        th->join();
    }
};


}
#endif // MULTI_THREAD_HPP

