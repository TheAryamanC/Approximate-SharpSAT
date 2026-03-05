#include "utils/timer.h"
#include "utils/logger.h"
#include <iomanip>

namespace sharpsat {

void TimerRegistry::print_all() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    LOG_INFO("Timer Statistics:");
    for (const auto& pair : timers_) {
        const std::string& name = pair.first;
        const Timer& timer = pair.second;
        double elapsed = timer.elapsed_seconds();
        LOG_INFO(name, ": ", std::fixed, std::setprecision(3), 
                elapsed, " seconds (", std::setprecision(1), elapsed * 1000, " ms)");
    }
}

} // namespace sharpsat
