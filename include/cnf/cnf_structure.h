#ifndef CNF_STRUCTURE_H
#define CNF_STRUCTURE_H

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

/// <summary>
/// A variable is represented as a positive integer index (>=1)
/// A literal is a variable index with a sign (positive or negative)
/// A clause is a disjunction (OR) of literals
/// A CNF formula is a conjunction (AND) of clauses
/// </summary>

namespace sharpsat {

// Define variables and literals
using Variable = uint32_t;
using Literal = int32_t;

// A clause is an OR of literals
struct Clause {
    std::vector<Literal> literals;
    bool is_satisfied;
    
    Clause() : is_satisfied(false) {}
    explicit Clause(const std::vector<Literal>& lits) 
        : literals(lits), is_satisfied(false) {}
    
    size_t size() const { return literals.size(); }
    bool empty() const { return literals.empty(); }
    void add_literal(Literal lit) { literals.push_back(lit); }
};

// XOR constraint: x1 ^ x2 ^ ... ^ xn = bool
struct XorConstraint {
    std::vector<Variable> variables;
    bool rhs;  // right-hand side (0 or 1)
    
    XorConstraint() : rhs(false) {}
    XorConstraint(const std::vector<Variable>& vars, bool r)
        : variables(vars), rhs(r) {}
    
    size_t size() const { return variables.size(); }
};

// A CNF is an AND of clauses (and XOR constraints)
class CNF {
public:
    CNF() : num_variables_(0), num_clauses_(0) {}
    
    uint32_t num_variables() const { return num_variables_; }
    uint32_t num_clauses() const { return num_clauses_; }
    const std::vector<Clause>& clauses() const { return clauses_; }
    std::vector<Clause>& clauses() { return clauses_; }
    
    void add_clause(const Clause& clause);
    void add_clause(const std::vector<Literal>& literals);
    
    void set_num_variables(uint32_t n) { num_variables_ = n; }
    
    void compute_variable_occurrences();
    const std::vector<uint32_t>& get_variable_occurrences() const { 
        return variable_occurrences_; 
    }
    
    // Get literal occurrences (positive and negative separately)
    uint32_t get_positive_occurrences(Variable var) const;
    uint32_t get_negative_occurrences(Variable var) const;
    
    // Check empty
    bool has_empty_clause() const;
    bool is_empty() const { return clauses_.empty(); }
    
    // Statistics
    double avg_clause_size() const;
    uint32_t max_clause_size() const;
    
    // Apply partial assignment
    void apply_assignment(const std::unordered_map<Variable, bool>& assignment);
    
    // Clone
    CNF clone() const;
    
private:
    uint32_t num_variables_;
    uint32_t num_clauses_;
    std::vector<Clause> clauses_;
    // counts of variable occurrences (for heuristics)
    std::vector<uint32_t> variable_occurrences_;
    std::unordered_map<Variable, uint32_t> positive_occurrences_;
    std::unordered_map<Variable, uint32_t> negative_occurrences_;
};

// Helper functions
inline Variable var(Literal lit) {
    return lit > 0 ? lit : -lit;
}

inline bool sign(Literal lit) {
    return lit > 0;
}

inline Literal make_literal(Variable var, bool positive) {
    return positive ? static_cast<Literal>(var) : -static_cast<Literal>(var);
}

} // namespace sharpsat

#endif // CNF_STRUCTURE_H
