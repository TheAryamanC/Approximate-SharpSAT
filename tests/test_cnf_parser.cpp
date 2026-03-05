#include "cnf/cnf_parser.h"
#include "cnf/cnf_structure.h"
#include <iostream>
#include <cassert>
#include <sstream>

using namespace sharpsat;

void test_parser_basic() {
    std::cout << "  - test_parser_basic...";
    
    std::string cnf_str = R"(
c Test CNF
p cnf 3 2
1 -2 3 0
-1 2 0
)";
    
    auto cnf = CNFParser::parse_string(cnf_str);
    assert(cnf != nullptr);
    assert(cnf->num_variables() == 3);
    assert(cnf->num_clauses() == 2);
    
    std::cout << " PASSED\n";
}

void test_parser_comments() {
    std::cout << "  - test_parser_comments...";
    
    std::string cnf_str = R"(
c This is a comment
c Another comment  
c Yet another comment
p cnf 2 1
1 2 0
)";
    
    auto cnf = CNFParser::parse_string(cnf_str);
    assert(cnf != nullptr);
    assert(cnf->num_variables() == 2);
    assert(cnf->num_clauses() == 1);
    assert(cnf->clauses()[0].size() == 2);
    
    std::cout << " PASSED\n";
}

void test_parser_empty_lines() {
    std::cout << "  - test_parser_empty_lines...";
    
    std::string cnf_str = R"(

p cnf 2 2

1 -2 0

-1 2 0

)";
    
    auto cnf = CNFParser::parse_string(cnf_str);
    assert(cnf != nullptr);
    assert(cnf->num_variables() == 2);
    assert(cnf->num_clauses() == 2);
    
    std::cout << " PASSED\n";
}

void test_parser_single_literal_clauses() {
    std::cout << "  - test_parser_single_literal_clauses...";
    
    std::string cnf_str = R"(
p cnf 3 3
1 0
-2 0
3 0
)";
    
    auto cnf = CNFParser::parse_string(cnf_str);
    assert(cnf != nullptr);
    assert(cnf->num_variables() == 3);
    assert(cnf->num_clauses() == 3);
    assert(cnf->clauses()[0].size() == 1);
    assert(cnf->clauses()[1].size() == 1);
    assert(cnf->clauses()[2].size() == 1);
    
    std::cout << " PASSED\n";
}

void test_parser_large_clauses() {
    std::cout << "  - test_parser_large_clauses...";
    
    std::string cnf_str = R"(
p cnf 10 2
1 2 3 4 5 6 7 8 9 10 0
-1 -2 -3 -4 -5 0
)";
    
    auto cnf = CNFParser::parse_string(cnf_str);
    assert(cnf != nullptr);
    assert(cnf->num_variables() == 10);
    assert(cnf->num_clauses() == 2);
    assert(cnf->clauses()[0].size() == 10);
    assert(cnf->clauses()[1].size() == 5);
    
    std::cout << " PASSED\n";
}

void test_parser_mixed_format() {
    std::cout << "  - test_parser_mixed_format...";
    
    std::string cnf_str = R"(
c Mixed format test
p cnf 4 4
1 -2 0
c intermediate comment
-1 2 3 0
c another comment
-3 4 0
1 4 0
)";
    
    auto cnf = CNFParser::parse_string(cnf_str);
    assert(cnf != nullptr);
    assert(cnf->num_variables() == 4);
    assert(cnf->num_clauses() == 4);
    
    std::cout << " PASSED\n";
}

void run_cnf_parser_tests() {
    std::cout << "\n=== CNF Parser Tests ===\n";
    test_parser_basic();
    test_parser_comments();
    test_parser_empty_lines();
    test_parser_single_literal_clauses();
    test_parser_large_clauses();
    test_parser_mixed_format();
}
