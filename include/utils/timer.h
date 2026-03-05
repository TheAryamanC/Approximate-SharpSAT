#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <mutex>

namespace sharpsat {

class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::duration<double>;
    
    Timer() { start(); }
    
    void start() {
        start_time_ = Clock::now();
    }
    
    void stop() {
        end_time_ = Clock::now();
    }
    
    double elapsed_seconds() const {
        auto end = end_time_.time_since_epoch().count() == 0 ? Clock::now() : end_time_;
        return std::chrono::duration<double>(end - start_time_).count();
    }
    
    double elapsed_ms() const {
        return elapsed_seconds() * 1000.0;
    }
    
    void reset() {
        start();
        end_time_ = TimePoint();
    }
    
private:
    TimePoint start_time_;
    TimePoint end_time_;
};

class TimerRegistry {
public:
    static TimerRegistry& instance() {
        static TimerRegistry registry;
        return registry;
    }
    
    void start(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        timers_[name].start();
    }
    
    void stop(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = timers_.find(name);
        if (it != timers_.end()) {
            it->second.stop();
        }
    }
    
    double get_elapsed(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = timers_.find(name);
        if (it != timers_.end()) {
            return it->second.elapsed_seconds();
        }
        return 0.0;
    }
    
    void print_all() const;
    
    void reset(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = timers_.find(name);
        if (it != timers_.end()) {
            it->second.reset();
        }
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        timers_.clear();
    }
    
private:
    TimerRegistry() = default;
    TimerRegistry(const TimerRegistry&) = delete;
    TimerRegistry& operator=(const TimerRegistry&) = delete;
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Timer> timers_;
};

// Timer that automatically starts on construction and stops on destruction, useful for timing scopes
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name) : name_(name) {
        TimerRegistry::instance().start(name_);
    }
    
    ~ScopedTimer() {
        TimerRegistry::instance().stop(name_);
    }
    
private:
    std::string name_;
};

} // namespace sharpsat

#endif // TIMER_H
