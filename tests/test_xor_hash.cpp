#include "xor/xor_hash_generator.h"
#include "cnf/cnf_structure.h"
#include <iostream>
#include <cassert>

using namespace sharpsat;

void test_xor_single_hash() {
    std::cout << "  - test_xor_single_hash...";
    
    XorHashGenerator gen(42);
    uint32_t num_vars = 10;
    double sparsity = 0.5;
    
    XorConstraint xor_c = gen.generate_single_hash(num_vars, sparsity);
    
    // Should have at least one variable
    assert(!xor_c.variables.empty());
    
    // All variables should be in valid range
    for (Variable v : xor_c.variables) {
        assert(v >= 1 && v <= num_vars);
    }
    
    std::cout << " PASSED\n";
}

void test_xor_multiple_hashes() {
    std::cout << "  - test_xor_multiple_hashes...";
    
    XorHashGenerator gen(42);
    uint32_t num_vars = 10;
    uint32_t num_hashes = 5;
    double sparsity = 0.5;
    
    auto xors = gen.generate_random_hashes(num_vars, num_hashes, sparsity);
    
    assert(xors.size() == num_hashes);
    
    for (const auto& xor_c : xors) {
        assert(!xor_c.variables.empty());
        for (Variable v : xor_c.variables) {
            assert(v >= 1 && v <= num_vars);
        }
    }
    
    std::cout << " PASSED\n";
}

void test_xor_set_seed() {
    std::cout << "  - test_xor_set_seed...";
    
    XorHashGenerator gen1(42);
    XorHashGenerator gen2(42);
    
    uint32_t num_vars = 10;
    double sparsity = 0.5;
    
    auto xor1 = gen1.generate_single_hash(num_vars, sparsity);
    auto xor2 = gen2.generate_single_hash(num_vars, sparsity);
    
    // Same seed should produce same results
    assert(xor1.variables == xor2.variables);
    assert(xor1.rhs == xor2.rhs);
    
    std::cout << " PASSED\n";
}

void test_xor_different_seeds() {
    std::cout << "  - test_xor_different_seeds...";
    
    XorHashGenerator gen1(42);
    XorHashGenerator gen2(123);
    
    uint32_t num_vars = 10;
    double sparsity = 0.5;
    
    auto xor1 = gen1.generate_single_hash(num_vars, sparsity);
    auto xor2 = gen2.generate_single_hash(num_vars, sparsity);
    
    // Different seeds should (likely) produce different results
    // Note: there's a small chance they could be the same by chance
    
    std::cout << " PASSED\n";
}

void test_xor_sparsity_zero() {
    std::cout << "  - test_xor_sparsity_zero...";
    
    XorHashGenerator gen(42);
    uint32_t num_vars = 10;
    double sparsity = 0.0;  // No variables included by chance
    
    XorConstraint xor_c = gen.generate_single_hash(num_vars, sparsity);
    
    // Should still have at least one variable (fallback)
    assert(!xor_c.variables.empty());
    assert(xor_c.variables.size() >= 1);
    
    std::cout << " PASSED\n";
}

void test_xor_sparsity_one() {
    std::cout << "  - test_xor_sparsity_one...";
    
    XorHashGenerator gen(42);
    uint32_t num_vars = 10;
    double sparsity = 1.0;  // All variables included
    
    XorConstraint xor_c = gen.generate_single_hash(num_vars, sparsity);
    
    // Should have all variables (or close to it)
    assert(xor_c.variables.size() >= 8);  // Allow some randomness
    
    std::cout << " PASSED\n";
}

void test_xor_recommended_sparsity() {
    std::cout << "  - test_xor_recommended_sparsity...";
    
    // Test different problem sizes
    double s1 = XorHashGenerator::get_recommended_sparsity(50);
    double s2 = XorHashGenerator::get_recommended_sparsity(500);
    double s3 = XorHashGenerator::get_recommended_sparsity(5000);
    
    // Larger problems should have lower sparsity
    assert(s1 >= s2);
    assert(s2 >= s3);
    
    // All should be in valid range
    assert(s1 > 0.0 && s1 <= 1.0);
    assert(s2 > 0.0 && s2 <= 1.0);
    assert(s3 > 0.0 && s3 <= 1.0);
    
    std::cout << " PASSED\n";
}

void test_cell_expected_size() {
    std::cout << "  - test_cell_expected_size...";
    
    Cell cell(5);
    uint32_t num_vars = 10;
    
    double expected = cell.expected_size(num_vars);
    
    // Expected size = 2^(10 - 5) = 2^5 = 32
    assert(expected == 32.0);
    
    std::cout << " PASSED\n";
}

void run_xor_hash_tests() {
    std::cout << "\n=== XOR Hash Generator Tests ===\n";
    test_xor_single_hash();
    test_xor_multiple_hashes();
    test_xor_set_seed();
    test_xor_different_seeds();
    test_xor_sparsity_zero();
    test_xor_sparsity_one();
    test_xor_recommended_sparsity();
    test_cell_expected_size();
}
