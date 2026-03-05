#include "cnf/cnf_parser.h"
#include "cnf/cnf_structure.h"
#include "solver/sat_solver.h"
#include "solver/cnf_simplifier.h"
#include "utils/logger.h"
#include <iostream>
#include <cassert>

using namespace sharpsat;
using namespace std;

void test_cnf_parser() {
    cout << "Testing CNF parser..." << endl;
    
    // Create test CNF string
    string cnf_str = R"(
c This is a comment
c Another comment
p cnf 3 2
1 -2 3 0
-1 2 0
)";
    
    auto cnf = CNFParser::parse_string(cnf_str);
    assert(cnf != nullptr);
    assert(cnf->num_variables() == 3);
    assert(cnf->num_clauses() == 2);
    assert(cnf->clauses().size() == 2);
    assert(cnf->clauses()[0].size() == 3);
    assert(cnf->clauses()[1].size() == 2);
}

void test_cnf_structure() {
    cout << "Testing CNF structure..." << endl;
    
    CNF cnf;
    cnf.set_num_variables(3);
    cnf.add_clause({1, -2, 3});
    cnf.add_clause({-1, 2});
    
    assert(cnf.num_variables() == 3);
    assert(cnf.num_clauses() == 2);
    assert(!cnf.has_empty_clause());
    
    // Test avg clause size
    double avg = cnf.avg_clause_size();
    assert(avg == 2.5);
}

void test_unit_propagation() {
    cout << "Testing unit propagation..." << endl;
    
    CNF cnf;
    cnf.set_num_variables(3);
    cnf.add_clause({1});  // Unit clause: x1 = true
    cnf.add_clause({1, 2, 3});
    cnf.add_clause({-1, 2});
    
    unordered_map<Variable, bool> assignment;
    CNFSimplifier simplifier;
    bool result = simplifier.unit_propagate(cnf, assignment);
    
    assert(result == true);
    assert(assignment.find(1) != assignment.end());
    assert(assignment[1] == true);
}

void test_sat_solver() {
    cout << "Testing SAT solver..." << endl;
    
    // Satisfiable formula
    CNF cnf1;
    cnf1.set_num_variables(2);
    cnf1.add_clause({1, 2});
    cnf1.add_clause({-1, 2});
    cnf1.add_clause({1, -2});
    
    SATSolver solver1;
    bool is_sat1 = solver1.solve(cnf1);
    assert(is_sat1);
    
    // Unsatisfiable formula
    CNF cnf2;
    cnf2.set_num_variables(1);
    cnf2.add_clause({1});
    cnf2.add_clause({-1});
    
    SATSolver solver2;
    bool is_sat2 = solver2.solve(cnf2);
    assert(!is_sat2);
}

void test_cnf_simplifier() {
    cout << "Testing CNF simplifier..." << endl;
    
    CNFSimplifier simplifier;
    
    // Test with pure literal
    CNF cnf;
    cnf.set_num_variables(3);
    cnf.add_clause({1, 2});
    cnf.add_clause({1, 3});  // Variable 1 appears only positive (pure literal)
    cnf.add_clause({-2, -3});
    
    unordered_map<Variable, bool> assignment;
    cnf.compute_variable_occurrences();
    bool result = simplifier.pure_literal_elimination(cnf, assignment);
    
    assert(result);
    assert(assignment.find(1) != assignment.end());
    assert(assignment[1] == true);  // Pure positive literal
}

int main() {
    Logger::instance().set_verbose(false);
    
    cout << "Running end-to-end tests" << endl;
    
    try {
        test_cnf_parser();
        test_cnf_structure();
        test_unit_propagation();
        test_sat_solver();
        test_cnf_simplifier();
        
        cout << "All tests passed" << endl;
        return 0;
    } catch (const exception& e) {
        cerr << "Test Failed: " << e.what() << endl;
        return 1;
    }
}
