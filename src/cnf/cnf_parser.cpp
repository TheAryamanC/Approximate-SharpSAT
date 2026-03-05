#include "cnf/cnf_parser.h"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;
namespace sharpsat {

// Parses a CNF file and returns a CNF object
unique_ptr<CNF> CNFParser::parse_file(const string& filename) {
    // ifstream is used to read the file content
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Invalid file: " << filename << endl;
        return nullptr;
    }
    
    // read entire file content into a stringstream
    stringstream buffer;
    buffer << file.rdbuf();
    return parse_string(buffer.str());
}

// Parses CNF content from a string and returns a CNF object
unique_ptr<CNF> CNFParser::parse_string(const string& content) {
    // create CNF and parse content line by line
    auto cnf = make_unique<CNF>();
    istringstream stream(content);
    string line;
    
    // variables to track header info
    uint32_t declared_vars = 0;
    uint32_t declared_clauses = 0;
    bool header_found = false;
    
    while (getline(stream, line)) {
        if (line.empty()) continue; // skip empty lines
        if (line[0] == 'c') continue; // skip comments
        
        // header
        if (line[0] == 'p') {
            // should be in proper format + only 1 header line allowed
            if (parse_header_line(line, declared_vars, declared_clauses) && !header_found) {
                cnf->set_num_variables(declared_vars);
                header_found = true;
            } else {
                cerr << "Invalid header line: " << line << endl;
                return nullptr;
            }
            continue;
        }
        
        // clauses
        if (header_found) {
            vector<Literal> literals;
            if (parse_clause_line(line, literals)) {
                cnf->add_clause(literals);
            }
        }
    }
    
    // stats
    cnf->compute_variable_occurrences();
    
    return cnf;
}

// Helper function to parse the header line of a CNF file
bool CNFParser::parse_header_line(const string& line, uint32_t& num_vars, uint32_t& num_clauses) {
    istringstream iss(line);
    string p, cnf_str;
    
    iss >> p >> cnf_str >> num_vars >> num_clauses;
    
    // header line format: p cnf <num_vars> <num_clauses>
    return (p == "p" && cnf_str == "cnf" && num_vars > 0 && num_clauses > 0);
}

// Helper function to parse a clause line of a CNF file
bool CNFParser::parse_clause_line(const string& line, vector<Literal>& literals) {
    istringstream iss(line);
    int lit;
    
    // clause line format: <lit1> <lit2> ... 0 (0 denotes end of clause)
    literals.clear(); // pass by reference, so clear any existing literals before parsing new clause
    
    while (iss >> lit) {
        if (lit == 0) { // end of clause
            return true;
        }
        literals.push_back(static_cast<Literal>(lit));
    }
    
    return !literals.empty(); // valid clause must have at least one literal before the terminating 0
}

} // namespace sharpsat
