#include "solver/cnf_simplifier.h"
#include "utils/logger.h"
#include <algorithm>

using namespace std;
namespace sharpsat {

// Unit propagation: repeatedly assign variables from unit clauses until no more unit clauses exist
bool CNFSimplifier::unit_propagate(CNF& cnf, unordered_map<Variable, bool>& assignment) {
    bool changed = true;
    
    while (changed) {
        changed = false;
        auto unit_clauses = find_unit_clauses(cnf);
        
        if (unit_clauses.empty()) {
            break; // no more unit clauses
        }
        
        for (Literal lit : unit_clauses) {
            Variable v = var(lit);
            bool value = sign(lit);
            
            // if variable is not assigned, assign it according to the unit clause
            if (assignment.find(v) == assignment.end()) {
                assignment[v] = value;
                changed = true;
            // if variable is already assigned, check for consistency
            } else if (assignment[v] != value) {
                return false;
            }
        }
        
        // if we made any new assignments, we need to apply them to the CNF and check for contradictions
        if (changed) {
            cnf.apply_assignment(assignment);
            if (cnf.has_empty_clause()) {
                return false;
            }
        }
    }
    
    return true;
}

// Pure literal elimination: assign pure literals (those that appear with only one polarity) and simplify the CNF
bool CNFSimplifier::pure_literal_elimination(CNF& cnf, unordered_map<Variable, bool>& assignment) {
    auto pure_lits = find_pure_literals(cnf);
    
    if (pure_lits.empty()) {
        return true; // no pure literals to eliminate
    }
    
    // go through pure literals and assign them, then apply the assignment to the CNF
    for (Literal lit : pure_lits) {
        Variable v = var(lit);
        bool value = sign(lit);
        assignment[v] = value;
    }
    
    cnf.apply_assignment(assignment);
    return !cnf.has_empty_clause(); // if we have an empty clause after applying the assignment, the CNF is unsatisfiable
}

// Subsumed clause elimination: remove clauses that are subsets of others
void CNFSimplifier::remove_subsumed_clauses(CNF& cnf) {
    auto& clauses = cnf.clauses();
    vector<bool> subsumed(clauses.size(), false);
    
    // go through all pairs of clauses
    for (size_t i = 0; i < clauses.size(); i++) {
        if (subsumed[i]) continue; // skip already subsumed clauses
        
        for (size_t j = i + 1; j < clauses.size(); ++j) {
            if (subsumed[j]) continue;
            
            // check if clause i is a superset of clause j
            if (clauses[i].size() <= clauses[j].size()) {
                bool subsumes = true;
                for (Literal lit : clauses[i].literals) {
                    // check if all literals of clause i are in clause j
                    if (find(clauses[j].literals.begin(), clauses[j].literals.end(), lit) == clauses[j].literals.end()) {
                        subsumes = false;
                        break;
                    }
                }
                if (subsumes) {
                    subsumed[j] = true;
                }
            }
            
            // Check if clause j is a subset of clause i
            if (clauses[j].size() < clauses[i].size()) {
                bool subsumes = true;
                for (Literal lit : clauses[j].literals) {
                    // check if all literals of clause j are in clause i
                    if (find(clauses[i].literals.begin(), clauses[i].literals.end(), lit) == clauses[i].literals.end()) {
                        subsumes = false;
                        break;
                    }
                }
                if (subsumes) {
                    subsumed[i] = true;
                    break;
                }
            }
        }
    }
    
    // remove clauses that are subsumed
    vector<Clause> new_clauses;
    for (size_t i = 0; i < clauses.size(); ++i) {
        if (!subsumed[i]) {
            new_clauses.push_back(clauses[i]);
        }
    }
    clauses = move(new_clauses);
}

// Apply a partial assignment to the CNF, simplifying it accordingly - already written in CNF class, just call it here
void CNFSimplifier::apply_partial_assignment(CNF& cnf, const unordered_map<Variable, bool>& assignment) {
    cnf.apply_assignment(assignment);
}

// Main simplification function that applies all simplification techniques iteratively until no more simplifications can be made
bool CNFSimplifier::simplify(CNF& cnf, unordered_map<Variable, bool>& assignment) {
    // unit propagation
    if (!unit_propagate(cnf, assignment)) {
        return false;
    }
    
    // pure literal elimination
    if (!pure_literal_elimination(cnf, assignment)) {
        return false;
    }
    
    // remove subsumed clauses
    remove_subsumed_clauses(cnf);
    
    return !cnf.has_empty_clause();
}

// Check if the CNF is trivially satisfiable (empty) or unsatisfiable (contains an empty clause)
CNFSimplifier::TrivialityCheck CNFSimplifier::check_triviality(const CNF& cnf) {
    if (cnf.is_empty()) {
        return TrivialityCheck::SAT;
    }
    
    if (cnf.has_empty_clause()) {
        return TrivialityCheck::UNSAT;
    }
    
    return TrivialityCheck::UNKNOWN;
}

// Helper functions to find unit clauses and pure literals
vector<Literal> CNFSimplifier::find_unit_clauses(const CNF& cnf) {
    vector<Literal> units;
    
    for (const auto& clause : cnf.clauses()) {
        if (clause.size() == 1) {
            units.push_back(clause.literals[0]);
        }
    }
    
    return units;
}

vector<Literal> CNFSimplifier::find_pure_literals(const CNF& cnf) {
    unordered_map<Variable, bool> literal_polarity;
    unordered_map<Variable, bool> has_both_polarities;
    
    // Collect all literals
    for (const auto& clause : cnf.clauses()) {
        for (Literal lit : clause.literals) {
            Variable v = var(lit);
            bool polarity = sign(lit);
            
            if (literal_polarity.find(v) == literal_polarity.end()) {
                literal_polarity[v] = polarity;
                has_both_polarities[v] = false;
            } else if (literal_polarity[v] != polarity) {
                has_both_polarities[v] = true;
            }
        }
    }
    
    // Find pure literals
    vector<Literal> pure_literals;
    for (const auto& pair : literal_polarity) {
        Variable v = pair.first;
        bool polarity = pair.second;
        if (!has_both_polarities[v]) {
            pure_literals.push_back(make_literal(v, polarity));
        }
    }
    
    return pure_literals;
}

// Check if a clause is satisfied under the current assignment (one of its literals is true)
bool CNFSimplifier::is_clause_satisfied(const Clause& clause, const unordered_map<Variable, bool>& assignment) {
    for (Literal lit : clause.literals) {
        Variable v = var(lit);
        auto it = assignment.find(v);
        if (it != assignment.end() && it->second == sign(lit)) {
            return true;
        }
    }
    return false;
}

} // namespace sharpsat
