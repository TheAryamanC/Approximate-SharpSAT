# Makefile for SharpSAT
CXX = g++
NVCC = nvcc
NVCC_HOST_COMPILER = g++-10
PYTHON = python3

# Directories
SRC_DIR = src
CUDA_DIR = cuda
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin
TEST_DIR = tests
ML_DIR = ml_model

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -march=native -pthread -I$(INCLUDE_DIR)
NVCCFLAGS = -ccbin=$(NVCC_HOST_COMPILER) -std=c++14 -O3 --use_fast_math --expt-relaxed-constexpr -Xcompiler -march=native -I$(INCLUDE_DIR)
CUDAFLAGS = -lcudart -lcuda

# CUDA architectures
CUDA_ARCH = -arch=sm_75 -gencode=arch=compute_75,code=sm_75 \
            -gencode=arch=compute_80,code=sm_80 \
            -gencode=arch=compute_86,code=sm_86

# Source files
CPP_SOURCES = $(shell find $(SRC_DIR) -name "*.cpp")
CUDA_SOURCES = $(shell find $(CUDA_DIR) -name "*.cu")
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))
CUDA_OBJECTS = $(patsubst $(CUDA_DIR)/%.cu,$(BUILD_DIR)/cuda_%.o,$(CUDA_SOURCES))

# Test files
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJECTS = $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_DIR)/test_%.o,$(TEST_SOURCES))

# Targets
TARGET = $(BIN_DIR)/sharp_sat
TEST_TARGET = $(BIN_DIR)/sharp_sat_tests

.PHONY: all clean test directories cmake ml_deps

all: directories $(TARGET)

cmake: directories
	@mkdir -p build_cmake
	cd build_cmake && cmake .. && $(MAKE)
	@cp build_cmake/sharp_sat $(BIN_DIR)/
	@echo "Build complete with CMake"

directories:
	@mkdir -p $(BUILD_DIR)/cnf $(BUILD_DIR)/solver $(BUILD_DIR)/utils $(BUILD_DIR)/xor
	@mkdir -p $(BIN_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/cuda_%.o: $(CUDA_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(BUILD_DIR)/test_%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(CPP_OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(CUDAFLAGS)
	@echo "Build complete: $(TARGET)"

$(TEST_TARGET): $(filter-out $(BUILD_DIR)/main.o, $(CPP_OBJECTS)) $(CUDA_OBJECTS) $(TEST_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(CUDAFLAGS)
	@echo "Test build complete: $(TEST_TARGET)"

test: $(TEST_TARGET)
	./$(TEST_TARGET)

ml_deps:
	cd $(ML_DIR) && $(PYTHON) -m pip install -r requirements.txt

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) build_cmake
	find . -name "*.o" -delete
	find . -name "*.so" -delete

run: $(TARGET)
	./$(TARGET) --help

.DEFAULT_GOAL := all
