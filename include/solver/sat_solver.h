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
    
    // Check satisfiability
    bool solve(const CNF& cnf);
    
    // Check satisfiability with partial assignment
    bool solve(const CNF& cnf, const std::unordered_map<Variable, bool>& partial_assignment);
    
    // Get satisfying assignment (if SAT)
    const std::unordered_map<Variable, bool>& get_assignment() const { 
        return assignment_; 
    }
    
    // Set timeout for finding a solution (this will increase as the CNF gets larger)
    void set_timeout(double timeout) { timeout_seconds_ = timeout; }
    
    // Set maximum number of decisions - if this is exceeded, the solver will give up and return false
    void set_max_decisions(uint32_t max_decisions) { max_decisions_ = max_decisions; }
    
    // Statistics
    uint64_t get_num_decisions() const { return num_decisions_; }
    uint64_t get_num_conflicts() const { return num_conflicts_; }
    
    void reset_stats() {
        num_decisions_ = 0;
        num_conflicts_ = 0;
    }
    
private:
    // DPLL search
    bool dpll(CNF& cnf, std::unordered_map<Variable, bool>& assignment);
    
    // Choose next variable to branch on (0 if none available)
    Variable choose_variable(const CNF& cnf, const std::unordered_map<Variable, bool>& assignment);
    
    // Check if all variables are assigned
    bool all_assigned(const CNF& cnf, const std::unordered_map<Variable, bool>& assignment);
    
    // Check if formula is satisfied
    bool is_satisfied(const CNF& cnf, const std::unordered_map<Variable, bool>& assignment);
    
    CNFSimplifier simplifier_;
    std::unordered_map<Variable, bool> assignment_;
    uint32_t max_decisions_;
    double timeout_seconds_;
    uint64_t num_decisions_;
    uint64_t num_conflicts_;
};

} // namespace sharpsat

#endif // SAT_SOLVER_H
