#include "utils/timer.h"
#include "utils/logger.h"
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>

using namespace sharpsat;

void test_timer_basic() {
    std::cout << "  - test_timer_basic...";
    
    Timer timer;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    double elapsed = timer.elapsed_seconds();
    
    // Should be at least 0.1 seconds (100ms)
    assert(elapsed >= 0.09);  // Allow some tolerance
    
    std::cout << " PASSED\n";
}

void test_timer_start_stop() {
    std::cout << "  - test_timer_start_stop...";
    
    Timer timer;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    timer.stop();
    
    double elapsed1 = timer.elapsed_seconds();
    
    // Wait more but timer is stopped
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    double elapsed2 = timer.elapsed_seconds();
    
    // Elapsed time should not increase after stop
    assert(elapsed2 >= elapsed1 * 0.9 && elapsed2 <= elapsed1 * 1.1);
    
    std::cout << " PASSED\n";
}

void test_timer_reset() {
    std::cout << "  - test_timer_reset...";
    
    Timer timer;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    timer.reset();
    
    double elapsed = timer.elapsed_seconds();
    
    // Should be close to 0 after reset
    assert(elapsed < 0.05);
    
    std::cout << " PASSED\n";
}

void test_timer_elapsed_ms() {
    std::cout << "  - test_timer_elapsed_ms...";
    
    Timer timer;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    double elapsed_ms = timer.elapsed_ms();
    
    // Should be at least 50ms
    assert(elapsed_ms >= 45.0);  // Allow tolerance
    
    std::cout << " PASSED\n";
}

void test_timer_registry() {
    std::cout << "  - test_timer_registry...";
    
    TimerRegistry::instance().clear();
    
    TimerRegistry::instance().start("test_timer");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    TimerRegistry::instance().stop("test_timer");
    
    double elapsed = TimerRegistry::instance().get_elapsed("test_timer");
    
    assert(elapsed >= 0.04);  // At least 40ms
    
    std::cout << " PASSED\n";
}

void test_timer_registry_multiple() {
    std::cout << "  - test_timer_registry_multiple...";
    
    TimerRegistry::instance().clear();
    
    TimerRegistry::instance().start("timer1");
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    TimerRegistry::instance().stop("timer1");
    
    TimerRegistry::instance().start("timer2");
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    TimerRegistry::instance().stop("timer2");
    
    double elapsed1 = TimerRegistry::instance().get_elapsed("timer1");
    double elapsed2 = TimerRegistry::instance().get_elapsed("timer2");
    
    assert(elapsed2 > elapsed1);
    
    std::cout << " PASSED\n";
}

void test_scoped_timer() {
    std::cout << "  - test_scoped_timer...";
    
    TimerRegistry::instance().clear();
    
    {
        ScopedTimer timer("scoped");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }  // Timer stops automatically here
    
    double elapsed = TimerRegistry::instance().get_elapsed("scoped");
    
    assert(elapsed >= 0.04);
    
    std::cout << " PASSED\n";
}

void test_logger_levels() {
    std::cout << "  - test_logger_levels...";
    
    Logger& logger = Logger::instance();
    
    // Set to INFO level
    logger.set_level(LogLevel::INFO);
    assert(logger.get_level() == LogLevel::INFO);
    
    // Set to DEBUG level
    logger.set_level(LogLevel::DEBUG);
    assert(logger.get_level() == LogLevel::DEBUG);
    
    // Set verbose (should set to DEBUG)
    logger.set_verbose(true);
    assert(logger.get_level() == LogLevel::DEBUG);
    
    logger.set_verbose(false);
    assert(logger.get_level() == LogLevel::INFO);
    
    std::cout << " PASSED\n";
}

void test_logger_output() {
    std::cout << "  - test_logger_output...";
    
    Logger& logger = Logger::instance();
    logger.set_level(LogLevel::INFO);
    
    // These should not crash
    logger.info("Test info message");
    logger.warning("Test warning message");
    logger.error("Test error message");
    
    // Debug should be filtered out at INFO level
    logger.debug("Test debug message (should be filtered)");
    
    std::cout << " PASSED\n";
}

void run_utils_tests() {
    std::cout << "\n=== Utils Tests ===\n";
    test_timer_basic();
    test_timer_start_stop();
    test_timer_reset();
    test_timer_elapsed_ms();
    test_timer_registry();
    test_timer_registry_multiple();
    test_scoped_timer();
    test_logger_levels();
    test_logger_output();
}
