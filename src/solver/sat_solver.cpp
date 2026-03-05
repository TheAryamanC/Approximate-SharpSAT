#include "solver/sat_solver.h"
#include "utils/logger.h"
#include "cuda_interface.h"
#include "cuda/clause_evaluation.cuh"
#include <algorithm>

using namespace std;
namespace sharpsat {

// Constructors
SATSolver::SATSolver() 
    : max_decisions_(1000000), timeout_seconds_(300.0), 
      num_decisions_(0), num_conflicts_(0), use_gpu_(false) {
}

SATSolver::SATSolver(uint32_t max_decisions) 
    : max_decisions_(max_decisions), timeout_seconds_(300.0),
      num_decisions_(0), num_conflicts_(0), use_gpu_(false) {
}

SATSolver::SATSolver(uint32_t max_decisions, bool use_gpu) 
    : max_decisions_(max_decisions), timeout_seconds_(300.0),
      num_decisions_(0), num_conflicts_(0), use_gpu_(use_gpu) {
}

// Solve without partial assignment
bool SATSolver::solve(const CNF& cnf) {
    unordered_map<Variable, bool> empty_assignment;
    return solve(cnf, empty_assignment);
}

// Solve with partial assignment
bool SATSolver::solve(const CNF& cnf, const unordered_map<Variable, bool>& partial_assignment) {
    // pass by reference to avoid copying assignment map
    reset_stats();
    assignment_ = partial_assignment;
    
    // clone CNF to work on without modifying original
    CNF working_cnf = cnf.clone();
    
    // apply partial assignment to simplify CNF before starting DPLL
    if (!partial_assignment.empty()) {
        working_cnf.apply_assignment(partial_assignment);
    }
    
    // check if trivially SAT/UNSAT
    auto triviality = simplifier_.check_triviality(working_cnf);
    if (triviality == CNFSimplifier::TrivialityCheck::SAT) {
        return true;
    } else if (triviality == CNFSimplifier::TrivialityCheck::UNSAT) {
        return false;
    }
    
    // search
    return dpll(working_cnf, assignment_);
}

// DPLL search
bool SATSolver::dpll(CNF& cnf, unordered_map<Variable, bool>& assignment) {
    // check if reached decision limit
    if (num_decisions_ >= max_decisions_) {
        LOG_DEBUG("SAT solver reached decision limit");
        return false;
    }
    
    // propagate - use GPU if enabled and available
    if (use_gpu_ && sharpsat::cuda::is_cuda_available()) {
        // Convert CNF to GPU format
        vector<vector<int32_t>> clause_vecs;
        for (const auto& clause : cnf.clauses()) {
            clause_vecs.push_back(clause.literals);
        }
        
        vector<int32_t> flat_lits;
        vector<uint32_t> offsets;
        sharpsat::cuda::convert_clauses_to_gpu_format(clause_vecs, flat_lits, offsets);
        
        bool conflict = false;
        if (!sharpsat::cuda::unit_propagation_gpu(flat_lits, offsets, assignment, &conflict)) {
            num_conflicts_++;
            return false;
        }
        
        // Apply assignment to CNF
        cnf.apply_assignment(assignment);
    } else {
        // CPU-based unit propagation
        if (!simplifier_.unit_propagate(cnf, assignment)) {
            num_conflicts_++;
            return false;
        }
    }
    
    // check if SAT/UNSAT after propagation
    if (cnf.is_empty()) {
        return true;
    }
    if (cnf.has_empty_clause()) {
        num_conflicts_++;
        return false;
    }
    
    // choose variable to branch on
    Variable var = choose_variable(cnf, assignment);
    if (var == 0) {
        return true; // if there is no variable left to assign, we must have satisfied all clauses
    }
    num_decisions_++;
    
    // try positive/negative assignments and recurse
    {
        CNF cnf_copy = cnf.clone();
        unordered_map<Variable, bool> assignment_copy = assignment;
        assignment_copy[var] = true;
        cnf_copy.apply_assignment(assignment_copy);
        
        if (dpll(cnf_copy, assignment_copy)) {
            assignment = assignment_copy;
            return true;
        }
    }
    {
        CNF cnf_copy = cnf.clone();
        unordered_map<Variable, bool> assignment_copy = assignment;
        assignment_copy[var] = false;
        cnf_copy.apply_assignment(assignment_copy);
        
        if (dpll(cnf_copy, assignment_copy)) {
            assignment = assignment_copy;
            return true;
        }
    }
    
    num_conflicts_++;
    return false;
}

// Choose variable to branch on using MOMS heuristic (appear most frequently in shortest clauses)
Variable SATSolver::choose_variable(const CNF& cnf, const unordered_map<Variable, bool>& assignment) {
    // find minimum clause size
    size_t min_clause_size = SIZE_MAX;
    for (const auto& clause : cnf.clauses()) {
        if (clause.size() > 0 && clause.size() < min_clause_size) {
            min_clause_size = clause.size();
        }
    }
    
    // no clauses left (should be handled by is_empty check), or all clauses are satisfied (should be handled by is_satisfied check)
    if (min_clause_size == SIZE_MAX) {
        for (Variable var = 1; var <= cnf.num_variables(); var++) {
            if (assignment.find(var) == assignment.end()) {
                return var;
            }
        }
        return 0;  // no unassigned variable
    }
    
    // go through clauses of minimum size and count occurrences of unassigned variables
    unordered_map<Variable, uint32_t> min_clause_occurrences;
    for (const auto& clause : cnf.clauses()) {
        if (clause.size() == min_clause_size) {
            for (Literal lit : clause.literals) {
                Variable v = var(lit);
                if (assignment.find(v) == assignment.end()) {
                    min_clause_occurrences[v]++;
                }
            }
        }
    }
    
    // select the variable with the most occurrences in minimum-size clauses
    Variable best_var = 0;
    uint32_t best_score = 0;
    
    for (const auto& entry : min_clause_occurrences) {
        if (entry.second > best_score) {
            best_score = entry.second;
            best_var = entry.first;
        }
    }
    
    // if no variable found in minimum clauses, fall back to any unassigned variable
    if (best_var == 0) {
        for (Variable var = 1; var <= cnf.num_variables(); var++) {
            if (assignment.find(var) == assignment.end()) {
                return var;
            }
        }
        return 0;  // no unassigned variable
    }
    
    return best_var;
}

} // namespace sharpsat
