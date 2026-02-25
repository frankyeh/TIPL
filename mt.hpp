#ifndef MULTI_THREAD_HPP
#define MULTI_THREAD_HPP
#include <chrono>
#include <future>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

#include "def.hpp"
namespace tipl{



class time
{
public:
    // Constructor starts the timer immediately
    time(const char* msg_) : msg(msg_), t1(std::chrono::high_resolution_clock::now()) {}
    time() : t1(std::chrono::high_resolution_clock::now()) {}

    void restart() { t1 = std::chrono::high_resolution_clock::now(); }
    void start() { t1 = std::chrono::high_resolution_clock::now(); }
    void stop() { t2 = std::chrono::high_resolution_clock::now(); }

    template<typename T = std::chrono::milliseconds>
    auto elapsed() {return std::chrono::duration_cast<T>(std::chrono::high_resolution_clock::now() - t1).count();}

    template<typename T = std::chrono::milliseconds>
    auto total() {stop();return std::chrono::duration_cast<T>(t2 - t1).count();}

    std::string to_string() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now - t1;
        auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
        duration -= hours;
        auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
        duration -= minutes;
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        std::ostringstream oss;
        if (hours.count() > 0) oss << hours.count() << "h";
        if (minutes.count() > 0) oss << minutes.count() << "m";
        oss << seconds.count() << "s";
        return oss.str();
    }

    ~time()
    {
        if (!msg.empty())
            std::cout << msg << elapsed<>() << std::endl;
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
        if(n)
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



inline auto main_thread_id = std::this_thread::get_id();
inline bool is_main_thread(void)
{
    return main_thread_id == std::this_thread::get_id();
}

inline int max_thread_count = std::thread::hardware_concurrency();
enum par_for_type {
    sequential, sequential_with_id,
    ranged, ranged_with_id,
    dynamic, dynamic_with_id
};

inline std::atomic<bool> par_for_running{false};
template <par_for_type type = sequential, typename T, typename Func,
          typename std::enable_if_t<std::is_integral_v<T> || std::is_class_v<T> || std::is_pointer_v<T>, int> = 0>
void par_for(T from, T to, Func&& f, int thread_count) {
    if (from == to) return;
    size_t n = to - from;
    bool is_root = !par_for_running.exchange(true);
    int active = (is_root && n > 1) ? std::min<int>(thread_count, (int)n) : 1;

#ifdef __CUDACC__
    int dev = 0;
    bool has_cuda = (active > 1 && cudaGetDevice(&dev) == 0);
#endif

    // Shared counter for dynamic types
    std::atomic<size_t> next_idx{0};

    auto run = [&](T b, T e, size_t id) -> void
    {
#ifdef __CUDACC__
        if (id && has_cuda) cudaSetDevice(dev);
#endif
        if constexpr (type == dynamic || type == dynamic_with_id) {
            for (size_t i = next_idx++; i < n; i = next_idx++) {
                if constexpr (type == dynamic_with_id) f(from + i, id);
                else f(from + i);
            }
        } else if constexpr (type >= ranged) {
            if constexpr (type == ranged_with_id) f(b, e, id);
            else f(b, e);
        } else for (; b != e; ++b) {
            if constexpr (type == sequential_with_id) f(b, id);
            else f(b);
        }
    };

    if (active > 1) {
        std::vector<std::thread> workers;
        size_t block = n / active, rem = n % active;
        T cursor = from;
        for (int i = 1; i < active; ++i) {
            T next = cursor + block + (i <= rem);
            workers.push_back(std::thread(run, cursor, next, i));
            cursor = next;
        }
        run(cursor, to, 0);
        for (auto& t : workers) t.join();
    } else run(from, to, 0);

    if (is_root) par_for_running = false;
}
// Overload: Automatic thread count
template <par_for_type type = sequential, typename T, typename Func,
          typename std::enable_if_t<std::is_integral_v<T> || std::is_class_v<T> || std::is_pointer_v<T>, int> = 0>
void par_for(T from, T to, Func&& f) {
    par_for<type>(from, to, std::forward<Func>(f), par_for_running ? 1 : max_thread_count);
}

// Overload: Single size (integral)
template <par_for_type type = sequential, typename T, typename Func, typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
void par_for(T size, Func&& f, int tc = max_thread_count) {
    par_for<type>(T(0), size, std::forward<Func>(f), tc);
}

// Overload: Containers
template <par_for_type type = sequential, typename C, typename Func,
          typename = decltype(std::declval<C>().begin())>
void par_for(C& c, Func&& f, int tc = max_thread_count) {
    par_for<type>(c.begin(), c.end(), std::forward<Func>(f), tc);
}


template <typename T>
double estimate_run_time(T&& fun)
{
    auto start = std::chrono::steady_clock::now();
    std::forward<T>(fun)();
    auto end = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::micro>(end - start).count();
}

template <par_for_type type = sequential, typename T, typename Func>
size_t adaptive_par_for(T from, T to, Func&& f)
{
    if (to - from <= 8 || !tipl::is_main_thread() || par_for_running.exchange(true))
    {
        par_for<type>(from, to, f, 1);
        return 1;
    }

    struct scoped_flag
    {
        ~scoped_flag()
        {
            par_for_running = false;
        }
    } guard;

    auto block_size = std::max<decltype(to - from)>(1, (to - from) >> 6);

    if (from + block_size * 3 > to)
    {
        par_for<type>(from, to, f, 1);
        return 1;
    }

    double t1 = estimate_run_time([&]() { par_for<type>(from, from + block_size, f, 1); });
    from += block_size;

    double t2 = estimate_run_time([&]() { par_for<type>(from, from + block_size * 2, f, 2); });
    from += block_size * 2;

    double overhead = t2 - t1;

    if (overhead <= 0)
    {
        par_for<type>(from, to, std::forward<Func>(f), max_thread_count);
        return max_thread_count;
    }

    int64_t num_block = (to - from) / block_size;
    int opt_threads = std::max<int>(1, std::sqrt(num_block * t1 / overhead));
    opt_threads = std::min<int>(opt_threads, max_thread_count);

    par_for<type>(from, to, std::forward<Func>(f), opt_threads);
    return opt_threads;
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

