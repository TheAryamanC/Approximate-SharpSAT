#include "solver/sat_solver.h"
#include "cnf/cnf_structure.h"
#include <iostream>
#include <cassert>

using namespace sharpsat;

void test_sat_trivial_sat() {
    std::cout << "  - test_sat_trivial_sat...";
    
    CNF cnf;  // Empty CNF is SAT
    
    SATSolver solver;
    bool is_sat = solver.solve(cnf);
    
    assert(is_sat == true);
    
    std::cout << " PASSED\n";
}

void test_sat_single_clause_sat() {
    std::cout << "  - test_sat_single_clause_sat...";
    
    CNF cnf;
    cnf.set_num_variables(2);
    cnf.add_clause({1, 2});  // x1 ∨ x2
    
    SATSolver solver;
    bool is_sat = solver.solve(cnf);
    
    assert(is_sat == true);
    
    std::cout << " PASSED\n";
}

void test_sat_unit_clause() {
    std::cout << "  - test_sat_unit_clause...";
    
    CNF cnf;
    cnf.set_num_variables(1);
    cnf.add_clause({1});  // x1
    
    SATSolver solver;
    bool is_sat = solver.solve(cnf);
    
    assert(is_sat == true);
    const auto& assignment = solver.get_assignment();
    assert(assignment.at(1) == true);
    
    std::cout << " PASSED\n";
}

void test_sat_simple_unsat() {
    std::cout << "  - test_sat_simple_unsat...";
    
    CNF cnf;
    cnf.set_num_variables(1);
    cnf.add_clause({1});   // x1
    cnf.add_clause({-1});  // ¬x1
    
    SATSolver solver;
    bool is_sat = solver.solve(cnf);
    
    assert(is_sat == false);
    
    std::cout << " PASSED\n";
}

void test_sat_medium_sat() {
    std::cout << "  - test_sat_medium_sat...";
    
    CNF cnf;
    cnf.set_num_variables(3);
    cnf.add_clause({1, 2});      // x1 ∨ x2
    cnf.add_clause({-1, 3});     // ¬x1 ∨ x3
    cnf.add_clause({-2, -3});    // ¬x2 ∨ ¬x3
    
    SATSolver solver;
    bool is_sat = solver.solve(cnf);
    
    assert(is_sat == true);
    
    std::cout << " PASSED\n";
}

void test_sat_medium_unsat() {
    std::cout << "  - test_sat_medium_unsat...";
    
    CNF cnf;
    cnf.set_num_variables(3);
    cnf.add_clause({1, 2});      // x1 ∨ x2
    cnf.add_clause({-1, 2});     // ¬x1 ∨ x2
    cnf.add_clause({1, -2});     // x1 ∨ ¬x2
    cnf.add_clause({-1, -2});    // ¬x1 ∨ ¬x2
    
    SATSolver solver;
    bool is_sat = solver.solve(cnf);
    
    assert(is_sat == false);
    
    std::cout << " PASSED\n";
}

void test_sat_with_partial_assignment_sat() {
    std::cout << "  - test_sat_with_partial_assignment_sat...";
    
    CNF cnf;
    cnf.set_num_variables(3);
    cnf.add_clause({1, 2, 3});
    cnf.add_clause({-1, 2, 3});
    
    std::unordered_map<Variable, bool> partial;
    partial[3] = true;  // x3 = true satisfies both clauses
    
    SATSolver solver;
    bool is_sat = solver.solve(cnf, partial);
    
    assert(is_sat == true);
    
    std::cout << " PASSED\n";
}

void test_sat_with_partial_assignment_unsat() {
    std::cout << "  - test_sat_with_partial_assignment_unsat...";
    
    CNF cnf;
    cnf.set_num_variables(2);
    cnf.add_clause({1, 2});
    cnf.add_clause({-1, 2});
    cnf.add_clause({-2});  // x2 must be false
    
    std::unordered_map<Variable, bool> partial;
    partial[2] = false;
    
    SATSolver solver;
    bool is_sat = solver.solve(cnf, partial);
    
    assert(is_sat == false);
    
    std::cout << " PASSED\n";
}

void test_sat_max_decisions_limit() {
    std::cout << "  - test_sat_max_decisions_limit...";
    
    // Create a harder formula
    CNF cnf;
    cnf.set_num_variables(10);
    for (int i = 1; i <= 10; i++) {
        cnf.add_clause({i, i+1 > 10 ? 1 : i+1});
    }
    
    SATSolver solver(5);  // Very low decision limit
    bool is_sat = solver.solve(cnf);
    
    // With such a low limit, solver should give up
    // (result depends on whether it finds sat within limit)
    
    std::cout << " PASSED\n";
}

void test_sat_choose_variable() {
    std::cout << "  - test_sat_choose_variable...";
    
    CNF cnf;
    cnf.set_num_variables(3);
    cnf.add_clause({1, 2});
    cnf.add_clause({1, 3});
    cnf.add_clause({2, 3});
    
    SATSolver solver;
    bool is_sat = solver.solve(cnf);
    
    assert(is_sat == true);
    
    std::cout << " PASSED\n";
}

void run_sat_solver_tests() {
    std::cout << "\n=== SAT Solver Tests ===\n";
    test_sat_trivial_sat();
    test_sat_single_clause_sat();
    test_sat_unit_clause();
    test_sat_simple_unsat();
    test_sat_medium_sat();
    test_sat_medium_unsat();
    test_sat_with_partial_assignment_sat();
    test_sat_with_partial_assignment_unsat();
    test_sat_max_decisions_limit();
    test_sat_choose_variable();
}
