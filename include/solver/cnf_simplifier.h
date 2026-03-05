#ifndef CNF_SIMPLIFIER_H
#define CNF_SIMPLIFIER_H

#include "cnf/cnf_structure.h"
#include <unordered_map>

namespace sharpsat {

// Simplifies CNF formulas
class CNFSimplifier {
public:
    CNFSimplifier() = default;
    
    // Apply unit propagation until fixpoint (false if conflict detected)
    bool unit_propagate(CNF& cnf, std::unordered_map<Variable, bool>& assignment);
    
    // Apply pure literal elimination
    bool pure_literal_elimination(CNF& cnf, std::unordered_map<Variable, bool>& assignment);
    
    // Subsumption: remove clauses that are subsumed by others
    void remove_subsumed_clauses(CNF& cnf);
    
    // Remove satisfied clauses based on partial assignment
    void apply_partial_assignment(CNF& cnf, const std::unordered_map<Variable, bool>& assignment);
    
    // Full simplification pipeline
    bool simplify(CNF& cnf, std::unordered_map<Variable, bool>& assignment);
    
    // Check if CNF is trivially SAT or UNSAT
    enum class TrivialityCheck { SAT, UNSAT, UNKNOWN };
    TrivialityCheck check_triviality(const CNF& cnf);
    
private:
    // Find unit clauses
    std::vector<Literal> find_unit_clauses(const CNF& cnf);
    
    // Find pure literals
    std::vector<Literal> find_pure_literals(const CNF& cnf);
    
    // Check if clause is satisfied by assignment
    bool is_clause_satisfied(const Clause& clause, const std::unordered_map<Variable, bool>& assignment);
};

} // namespace sharpsat

#endif // CNF_SIMPLIFIER_H
