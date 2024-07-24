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
    static struct thread_opt{
        std::vector<size_t> performance;
        size_t cur_thread_count = 1;
        size_t last_size = 0;
        thread_opt(size_t max_thread = max_thread_count):performance(max_thread+1){}
        std::chrono::high_resolution_clock::time_point beg;
        void start()
        {
            beg = std::chrono::high_resolution_clock::now();
        }
        void end()
        {
            auto time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-beg).count();
            if(!performance[cur_thread_count])
                performance[cur_thread_count] = time;
            else
                performance[cur_thread_count] = (performance[cur_thread_count]+time)/2;
            if(cur_thread_count > 1 && time > performance[cur_thread_count-1])
                --cur_thread_count;
            if(time > performance[cur_thread_count+1] && cur_thread_count < performance.size()-1)
                ++cur_thread_count;
        }
    } thread_optimizer;

    thread_optimizer.start();
    par_for<type>(from,to,std::move(f),thread_optimizer.cur_thread_count);
    thread_optimizer.end();
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
    par_for<type>(c.begin(),c.end(),std::move(f),thread_count);
}
template <par_for_type type = sequential,typename T,typename Func>
inline typename std::enable_if<
    std::is_same<decltype(std::declval<T>().begin()), decltype(std::declval<T>().end())>::value>::type
par_for(T& c, Func&& f)
{
    par_for<type>(c.begin(),c.end(),std::move(f));
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

