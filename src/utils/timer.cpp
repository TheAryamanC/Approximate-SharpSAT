#include "utils/timer.h"
#include <iomanip>
#include <iostream>

using namespace std;
namespace sharpsat {

void TimerRegistry::print_all() const {
    lock_guard<mutex> lock(mutex_);
    
    cout << "Timer Statistics:" << endl;
    for (const auto& pair : timers_) {
        const string& name = pair.first;
        const Timer& timer = pair.second;
        double elapsed = timer.elapsed_seconds();
        cout << name << ": " << fixed << setprecision(3) << elapsed << " seconds (" << setprecision(1) << elapsed * 1000 << " ms)" << endl;
    }
}

} // namespace sharpsat
