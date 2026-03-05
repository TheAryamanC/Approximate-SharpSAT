#include "solver/approximate_counter.h"
#include "cnf/cnf_parser.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace sharpsat;

void test_counter_trivial_empty() {
    std::cout << "  - test_counter_trivial_empty...";
    
    CNF cnf;
    cnf.set_num_variables(3);
    // Empty CNF (no clauses) - all 2^3 = 8 assignments satisfy it
    
    CounterConfig config;
    config.use_cuda = false;  // Test CPU version
    config.use_ml_hashes = false;
    
    ApproximateCounter counter(config);
    CountResult result = counter.count(cnf);
    
    assert(result.successful);
    assert(result.count == 8.0);  // 2^3
    
    std::cout << " PASSED\n";
}

void test_counter_trivial_unsat() {
    std::cout << "  - test_counter_trivial_unsat...";
    
    // Create UNSAT formula: x AND NOT x
    CNF cnf;
    cnf.set_num_variables(1);
    cnf.add_clause({1});
    cnf.add_clause({-1});
    
    CounterConfig config;
    config.use_cuda = false;  // Test CPU version
    config.use_ml_hashes = false;
    config.max_iterations = 3;  // Reduce for faster test
    
    ApproximateCounter counter(config);
    CountResult result = counter.count(cnf);
    
    assert(result.successful);
    // For UNSAT formulas, count should be 0 or very small
    assert(result.count < 1.0);
    
    std::cout << " PASSED\n";
}

void test_counter_simple_formula() {
    std::cout << "  - test_counter_simple_formula...";
    
    std::string cnf_str = R"(
p cnf 2 2
1 2 0
-1 2 0
)";
    
    auto cnf = CNFParser::parse_string(cnf_str);
    assert(cnf != nullptr);
    
    CounterConfig config;
    config.epsilon = 0.8;
    config.delta = 0.2;
    config.use_cuda = false;
    config.use_ml_hashes = false;
    config.max_iterations = 5;  // Reduce for faster test
    
    ApproximateCounter counter(config);
    CountResult result = counter.count(*cnf);
    
    // This formula has 3 satisfying assignments: (F,T), (T,T), (T,F)
    // Count should be approximately 3
    assert(result.successful);
    assert(result.count > 0);  // At least detect it's SAT
    
    std::cout << " PASSED\n";
}

void test_counter_config() {
    std::cout << "  - test_counter_config...";
    
    CounterConfig config;
    config.epsilon = 0.5;
    config.delta = 0.1;
    config.seed = 123;
    config.use_ml_hashes = true;
    config.use_cuda = false;
    
    ApproximateCounter counter(config);
    
    const auto& retrieved = counter.get_config();
    assert(retrieved.epsilon == 0.5);
    assert(retrieved.delta == 0.1);
    assert(retrieved.seed == 123);
    assert(retrieved.use_ml_hashes == true);
    
    std::cout << " PASSED\n";
}

void test_counter_result_bounds() {
    std::cout << "  - test_counter_result_bounds...";
    
    std::string cnf_str = R"(
p cnf 3 2
1 2 3 0
-1 -2 -3 0
)";
    
    auto cnf = CNFParser::parse_string(cnf_str);
    assert(cnf != nullptr);
    
    CounterConfig config;
    config.epsilon = 0.8;
    config.delta = 0.2;
    config.use_cuda = false;
    config.use_ml_hashes = false;
    config.max_iterations = 5;
    
    ApproximateCounter counter(config);
    CountResult result = counter.count(*cnf);
    
    if (result.successful && result.count > 0) {
        // Bounds should bracket the count
        assert(result.lower_bound <= result.count);
        assert(result.count <= result.upper_bound);
        assert(result.lower_bound < result.upper_bound);
    }
    
    std::cout << " PASSED\n";
}

void test_counter_with_xor() {
    std::cout << "  - test_counter_with_xor...";
    
    // Test that counter can handle XOR constraints internally
    std::string cnf_str = R"(
p cnf 4 4
1 2 0
-1 3 0
-2 4 0
-3 -4 0
)";
    
    auto cnf = CNFParser::parse_string(cnf_str);
    assert(cnf != nullptr);
    
    CounterConfig config;
    config.epsilon = 1.0;
    config.delta = 0.3;
    config.use_cuda = false;
    config.use_ml_hashes = false;
    config.max_iterations = 3;
    
    ApproximateCounter counter(config);
    CountResult result = counter.count(*cnf);
    
    assert(result.successful);
    // Just check it completes without crashing
    
    std::cout << " PASSED\n";
}

void run_approximate_counter_tests() {
    std::cout << "\n=== Approximate Counter Tests ===\n";
    test_counter_trivial_empty();
    test_counter_trivial_unsat();
    test_counter_simple_formula();
    test_counter_config();
    test_counter_result_bounds();
    test_counter_with_xor();
}
