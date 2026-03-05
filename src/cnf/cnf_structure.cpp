#include "cnf/cnf_structure.h"
#include <algorithm>

using namespace std;
namespace sharpsat {

// add a clause to the CNF
void CNF::add_clause(const Clause& clause) {
    if (!clause.empty()) { // empty clauses are UNSAT
        clauses_.push_back(clause);
        num_clauses_ = clauses_.size();
        
        // update num_variables
        for (Literal lit : clause.literals) {
            uint32_t v = var(lit);
            if (v > num_variables_) {
                num_variables_ = v;
            }
        }
    }
}

// add a clause - using literals directly
void CNF::add_clause(const vector<Literal>& literals) {
    add_clause(Clause(literals));
}

// compute how many times each variable appears in the CNF
void CNF::compute_variable_occurrences() {
    variable_occurrences_.clear();
    variable_occurrences_.resize(num_variables_ + 1, 0);
    positive_occurrences_.clear();
    negative_occurrences_.clear();
    
    for (const auto& clause : clauses_) {
        for (Literal lit : clause.literals) {
            Variable v = var(lit);
            variable_occurrences_[v]++;
            
            if (sign(lit)) {
                positive_occurrences_[v]++;
            } else {
                negative_occurrences_[v]++;
            }
        }
    }
}

// get positive and negative occurrences of a variable
uint32_t CNF::get_positive_occurrences(Variable var) const {
    auto it = positive_occurrences_.find(var);
    return it != positive_occurrences_.end() ? it->second : 0;
}

uint32_t CNF::get_negative_occurrences(Variable var) const {
    auto it = negative_occurrences_.find(var);
    return it != negative_occurrences_.end() ? it->second : 0;
}

// check if CNF has an empty clause - if so, it's unsatisfiable
bool CNF::has_empty_clause() const {
    return any_of(clauses_.begin(), clauses_.end(), [](const Clause& c) { return c.empty(); });
}

// compute average clause size (total literals / total clauses)
double CNF::avg_clause_size() const {
    if (clauses_.empty()) return 0.0;
    
    size_t total = 0;
    for (const auto& clause : clauses_) {
        total += clause.size();
    }
    return static_cast<double>(total) / clauses_.size();
}

// compute maximum clause size
uint32_t CNF::max_clause_size() const {
    if (clauses_.empty()) return 0;
    
    uint32_t max_size = 0;
    for (const auto& clause : clauses_) {
        if (clause.size() > max_size) {
            max_size = clause.size();
        }
    }
    return max_size;
}

// apply a partial assignment to the CNF
void CNF::apply_assignment(const unordered_map<Variable, bool>& assignment) {
    // clauses after applying assignment
    vector<Clause> new_clauses;
    
    for (auto& clause : clauses_) { // for each clause
        bool satisfied = false;
        vector<Literal> new_literals;
        
        for (Literal lit : clause.literals) { // for each literal in the clause
            // get variable and sign
            Variable v = var(lit);
            bool lit_sign = sign(lit);
            
            // check if variable is assigned
            auto it = assignment.find(v);
            if (it != assignment.end()) {
                // if variable is assigned, check if it satisfies the literal
                if (it->second == lit_sign) {
                    // if literal is satisfied by the assignment, we can skip the rest of the clause
                    // (OR of literals - if one is true, the whole clause is true)
                    satisfied = true;
                    break;
                }
                // if variable is assigned but does not satisfy the literal, we skip this literal (it becomes false)
            } else {
                // if variable is unassigned, we keep it in the new clause
                new_literals.push_back(lit);
            }
        }
        
        // if the clause is not satisfied, we add the new clause (with unassigned literals) to the new CNF
        if (!satisfied) {
            if (new_literals.empty()) {
                // empty clause - formula is UNSAT
                clauses_.clear();
                clauses_.push_back(Clause());
                return;
            }
            new_clauses.push_back(Clause(new_literals));
        }
    }
    
    // replace old clauses with new clauses after applying assignment
    clauses_ = move(new_clauses);
    num_clauses_ = clauses_.size();
}

// clone the CNF (deep copy) - so that we can modify the clone without affecting the original
CNF CNF::clone() const {
    CNF new_cnf;
    new_cnf.num_variables_ = num_variables_;
    new_cnf.num_clauses_ = num_clauses_;
    new_cnf.clauses_ = clauses_;
    new_cnf.variable_occurrences_ = variable_occurrences_;
    new_cnf.positive_occurrences_ = positive_occurrences_;
    new_cnf.negative_occurrences_ = negative_occurrences_;
    return new_cnf;
}

} // namespace sharpsat
