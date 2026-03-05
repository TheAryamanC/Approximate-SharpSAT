#include "cnf/cnf_parser.h"
#include "cnf/cnf_structure.h"
#include <iostream>
#include <cassert>
#include <sstream>

using namespace sharpsat;

void test_cnf_structure_basic() {
    std::cout << "  - test_cnf_structure_basic...";
    
    CNF cnf;
    cnf.set_num_variables(3);
    cnf.add_clause({1, -2, 3});
    cnf.add_clause({-1, 2});
    
    assert(cnf.num_variables() == 3);
    assert(cnf.num_clauses() == 2);
    assert(!cnf.has_empty_clause());
    assert(!cnf.is_empty());
    assert(cnf.clauses().size() == 2);
    assert(cnf.clauses()[0].size() == 3);
    assert(cnf.clauses()[1].size() == 2);
    
    std::cout << " PASSED\n";
}

void test_cnf_avg_clause_size() {
    std::cout << "  - test_cnf_avg_clause_size...";
    
    CNF cnf;
    cnf.set_num_variables(3);
    cnf.add_clause({1, -2, 3});
    cnf.add_clause({-1, 2});
    
    double avg = cnf.avg_clause_size();
    assert(avg == 2.5);
    
    std::cout << " PASSED\n";
}

void test_cnf_max_clause_size() {
    std::cout << "  - test_cnf_max_clause_size...";
    
    CNF cnf;
    cnf.set_num_variables(5);
    cnf.add_clause({1, -2});
    cnf.add_clause({-1, 2, 3, 4, 5});
    cnf.add_clause({1});
    
    uint32_t max_size = cnf.max_clause_size();
    assert(max_size == 5);
    
    std::cout << " PASSED\n";
}

void test_cnf_empty_clause() {
    std::cout << "  - test_cnf_empty_clause...";
    
    CNF cnf;
    cnf.set_num_variables(2);
    cnf.add_clause({1, 2});
    
    assert(!cnf.has_empty_clause());
    
    // Apply assignment that creates empty clause
    std::unordered_map<Variable, bool> assignment;
    assignment[1] = false;
    assignment[2] = false;
    cnf.apply_assignment(assignment);
    
    assert(cnf.has_empty_clause());
    
    std::cout << " PASSED\n";
}

void test_cnf_clone() {
    std::cout << "  - test_cnf_clone...";
    
    CNF cnf;
    cnf.set_num_variables(3);
    cnf.add_clause({1, -2, 3});
    cnf.add_clause({-1, 2});
    
    CNF clone = cnf.clone();
    assert(clone.num_variables() == cnf.num_variables());
    assert(clone.num_clauses() == cnf.num_clauses());
    assert(clone.clauses().size() == cnf.clauses().size());
    
    std::cout << " PASSED\n";
}

void test_cnf_variable_occurrences() {
    std::cout << "  - test_cnf_variable_occurrences...";
    
    CNF cnf;
    cnf.set_num_variables(3);
    cnf.add_clause({1, -2, 3});
    cnf.add_clause({-1, 2});
    cnf.add_clause({1, 3});
    
    cnf.compute_variable_occurrences();
    
    const auto& occurrences = cnf.get_variable_occurrences();
    assert(occurrences[1] == 3);  // var 1 appears 3 times
    assert(occurrences[2] == 2);  // var 2 appears 2 times
    assert(occurrences[3] == 2);  // var 3 appears 2 times
    
    assert(cnf.get_positive_occurrences(1) == 2);
    assert(cnf.get_negative_occurrences(1) == 1);
    assert(cnf.get_positive_occurrences(2) == 1);
    assert(cnf.get_negative_occurrences(2) == 1);
    
    std::cout << " PASSED\n";
}

void test_cnf_apply_assignment() {
    std::cout << "  - test_cnf_apply_assignment...";
    
    CNF cnf;
    cnf.set_num_variables(3);
    cnf.add_clause({1, -2, 3});
    cnf.add_clause({-1, 2});
    cnf.add_clause({2, 3});
    
    std::unordered_map<Variable, bool> assignment;
    assignment[1] = true;  // satisfies first clause
    cnf.apply_assignment(assignment);
    
    // First clause should be removed as satisfied
    assert(cnf.num_clauses() == 2);
    
    std::cout << " PASSED\n";
}

void test_helper_functions() {
    std::cout << "  - test_helper_functions...";
    
    // Test var()
    assert(var(5) == 5);
    assert(var(-5) == 5);
    
    // Test sign()
    assert(sign(5) == true);
    assert(sign(-5) == false);
    
    // Test make_literal()
    assert(make_literal(5, true) == 5);
    assert(make_literal(5, false) == -5);
    
    std::cout << " PASSED\n";
}

void run_cnf_structure_tests() {
    std::cout << "\n=== CNF Structure Tests ===\n";
    test_cnf_structure_basic();
    test_cnf_avg_clause_size();
    test_cnf_max_clause_size();
    test_cnf_empty_clause();
    test_cnf_clone();
    test_cnf_variable_occurrences();
    test_cnf_apply_assignment();
    test_helper_functions();
}
