#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <iostream>
#include <sstream>
#include <mutex>
#include <chrono>
#include <iomanip>

namespace sharpsat {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }
    
    void set_level(LogLevel level) { min_level_ = level; }
    LogLevel get_level() const { return min_level_; }
    
    void set_verbose(bool verbose) { 
        min_level_ = verbose ? LogLevel::DEBUG : LogLevel::INFO; 
    }
    
    template<typename T>
    void log_helper(std::ostringstream& oss, T&& arg) {
        oss << std::forward<T>(arg);
    }
    
    template<typename T, typename... Args>
    void log_helper(std::ostringstream& oss, T&& arg, Args&&... args) {
        oss << std::forward<T>(arg);
        log_helper(oss, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void log(LogLevel level, Args&&... args) {
        if (level < min_level_) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostringstream oss;
        oss << "[" << get_timestamp() << "] ";
        oss << level_to_string(level) << ": ";
        log_helper(oss, std::forward<Args>(args)...);
        oss << std::endl;
        std::cout << oss.str() << std::flush;
    }
    
    template<typename... Args>
    void debug(Args&&... args) {
        log(LogLevel::DEBUG, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void info(Args&&... args) {
        log(LogLevel::INFO, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void warning(Args&&... args) {
        log(LogLevel::WARNING, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void error(Args&&... args) {
        log(LogLevel::ERROR, std::forward<Args>(args)...);
    }
    
private:
    Logger() : min_level_(LogLevel::INFO) {}
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    std::string level_to_string(LogLevel level) const {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARN";
            case LogLevel::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }
    
    std::string get_timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time), "%H:%M:%S");
        oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }
    
    LogLevel min_level_;
    std::mutex mutex_;
};

// macros to help with logging without needing to specify the logger instance and log level each time
#define LOG_DEBUG(...) sharpsat::Logger::instance().debug(__VA_ARGS__)
#define LOG_INFO(...) sharpsat::Logger::instance().info(__VA_ARGS__)
#define LOG_WARNING(...) sharpsat::Logger::instance().warning(__VA_ARGS__)
#define LOG_ERROR(...) sharpsat::Logger::instance().error(__VA_ARGS__)

} // namespace sharpsat

#endif // LOGGER_H
