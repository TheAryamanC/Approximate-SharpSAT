# Makefile for SharpSAT
CXX = g++
NVCC = nvcc
NVCC_HOST_COMPILER = g++-10

# Directories
SRC_DIR = src
CUDA_DIR = src/cuda
INCLUDE_DIR = include
BIN_DIR = bin
TEST_DIR = tests

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

# Targets
TARGET = $(BIN_DIR)/sharp_sat

.PHONY: all clean directories run help

all: directories $(TARGET)

directories:
	@mkdir -p $(BIN_DIR)

$(TARGET): $(CPP_SOURCES) $(CUDA_SOURCES)
	@echo "Compiling SharpSAT..."
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $(CPP_SOURCES) $(CUDA_SOURCES) -o $@ $(CUDAFLAGS)
	@echo "Build complete: $(TARGET)"

clean:
	rm -rf $(BIN_DIR)
	find . -name "*.o" -delete
	find . -name "*.so" -delete

run: $(TARGET)
	./$(TARGET) --help

help:
	@echo "SharpSAT Makefile"
	@echo "================="
	@echo "Targets:"
	@echo "  all      - Build sharp_sat executable (default)"
	@echo "  clean    - Remove all build artifacts"
	@echo "  run      - Build and show help message"
	@echo ""
	@echo "Output:"
	@echo "  $(TARGET)"

.DEFAULT_GOAL := all
