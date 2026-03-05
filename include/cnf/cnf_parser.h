#ifndef CNF_PARSER_H
#define CNF_PARSER_H

#include "cnf_structure.h"
#include <string>
#include <memory>

namespace sharpsat {

// Parser for DIMACS CNF problems
class CNFParser {
public:
    CNFParser() = default;
    
    // Parse CNF from file
    static std::unique_ptr<CNF> parse_file(const std::string& filename);
    
    // Parse CNF from string
    static std::unique_ptr<CNF> parse_string(const std::string& content);
        
private:
    static bool parse_header_line(const std::string& line, uint32_t& num_vars, uint32_t& num_clauses);
    static bool parse_clause_line(const std::string& line, std::vector<Literal>& literals);
};

} // namespace sharpsat

#endif // CNF_PARSER_H
