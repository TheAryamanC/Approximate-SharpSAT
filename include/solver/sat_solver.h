#ifndef SAT_SOLVER_H
#define SAT_SOLVER_H

#include "cnf/cnf_structure.h"
#include "cnf_simplifier.h"
#include <unordered_map>
#include <optional>

namespace sharpsat {

// Simple DPLL-based SAT solver
class SATSolver {
public:
    SATSolver();
    explicit SATSolver(uint32_t max_decisions);
    explicit SATSolver(uint32_t max_decisions, bool use_gpu);
    
    // Check satisfiability
    bool solve(const CNF& cnf);
    
    // Check satisfiability with partial assignment
    bool solve(const CNF& cnf, const std::unordered_map<Variable, bool>& partial_assignment);
    
    // Get satisfying assignment (if SAT)
    const std::unordered_map<Variable, bool>& get_assignment() const { 
        return assignment_; 
    }
    
    // Set maximum number of decisions - if this is exceeded, the solver will give up and return false
    void set_max_decisions(uint32_t max_decisions) { max_decisions_ = max_decisions; }
    
    // Enable/disable GPU acceleration
    void set_use_gpu(bool use_gpu) { use_gpu_ = use_gpu; }
    bool get_use_gpu() const { return use_gpu_; }
    
    void reset_stats() {
        num_decisions_ = 0;
        num_conflicts_ = 0;
    }
    
private:
    // DPLL search
    bool dpll(CNF& cnf, std::unordered_map<Variable, bool>& assignment);
    
    // Choose next variable to branch on (0 if none available)
    Variable choose_variable(const CNF& cnf, const std::unordered_map<Variable, bool>& assignment);
    
    CNFSimplifier simplifier_;
    std::unordered_map<Variable, bool> assignment_;
    uint32_t max_decisions_;
    double timeout_seconds_;
    uint64_t num_decisions_;
    uint64_t num_conflicts_;
    bool use_gpu_;
};

} // namespace sharpsat

#endif // SAT_SOLVER_H
