#include "utils/logger.h"
#include <iostream>
#include <exception>

using namespace sharpsat;

// Forward declarations of test runners
void run_cnf_structure_tests();
void run_cnf_parser_tests();
void run_cnf_simplifier_tests();
void run_sat_solver_tests();
void run_xor_hash_tests();
void run_utils_tests();
void run_approximate_counter_tests();

int main() {
    // Disable verbose logging during tests
    Logger::instance().set_verbose(false);
    Logger::instance().set_level(LogLevel::ERROR);  // Only show errors
    
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "   SharpSAT Comprehensive Test Suite\n";
    std::cout << "========================================\n";
    
    int failed_suites = 0;
    int total_suites = 0;
    
    try {
        total_suites++;
        run_cnf_structure_tests();
        std::cout << "✓ CNF Structure tests PASSED\n";
    } catch (const std::exception& e) {
        std::cerr << "✗ CNF Structure tests FAILED: " << e.what() << "\n";
        failed_suites++;
    }
    
    try {
        total_suites++;
        run_cnf_parser_tests();
        std::cout << "✓ CNF Parser tests PASSED\n";
    } catch (const std::exception& e) {
        std::cerr << "✗ CNF Parser tests FAILED: " << e.what() << "\n";
        failed_suites++;
    }
    
    try {
        total_suites++;
        run_cnf_simplifier_tests();
        std::cout << "✓ CNF Simplifier tests PASSED\n";
    } catch (const std::exception& e) {
        std::cerr << "✗ CNF Simplifier tests FAILED: " << e.what() << "\n";
        failed_suites++;
    }
    
    try {
        total_suites++;
        run_sat_solver_tests();
        std::cout << "✓ SAT Solver tests PASSED\n";
    } catch (const std::exception& e) {
        std::cerr << "✗ SAT Solver tests FAILED: " << e.what() << "\n";
        failed_suites++;
    }
    
    try {
        total_suites++;
        run_xor_hash_tests();
        std::cout << "✓ XOR Hash Generator tests PASSED\n";
    } catch (const std::exception& e) {
        std::cerr << "✗ XOR Hash Generator tests FAILED: " << e.what() << "\n";
        failed_suites++;
    }
    
    try {
        total_suites++;
        run_utils_tests();
        std::cout << "✓ Utils tests PASSED\n";
    } catch (const std::exception& e) {
        std::cerr << "✗ Utils tests FAILED: " << e.what() << "\n";
        failed_suites++;
    }
    
    try {
        total_suites++;
        run_approximate_counter_tests();
        std::cout << "✓ Approximate Counter tests PASSED\n";
    } catch (const std::exception& e) {
        std::cerr << "✗ Approximate Counter tests FAILED: " << e.what() << "\n";
        failed_suites++;
    }
    
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "   Test Summary\n";
    std::cout << "========================================\n";
    std::cout << "Total test suites: " << total_suites << "\n";
    std::cout << "Passed: " << (total_suites - failed_suites) << "\n";
    std::cout << "Failed: " << failed_suites << "\n";
    
    if (failed_suites == 0) {
        std::cout << "\n✓ ALL TESTS PASSED!\n\n";
        return 0;
    } else {
        std::cout << "\n✗ SOME TESTS FAILED\n\n";
        return 1;
    }
}
