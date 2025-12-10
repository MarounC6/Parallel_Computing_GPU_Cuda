# Compiler and flags
CXX = /usr/local/cuda/bin/nvcc
CXXFLAGS = -O3

# Directories
BIN_DIR = exe_bin
PART1_DIR = Part_1
PART2_DIR = Part_2
PART3_DIR = Part_3

# Create bin directory if it doesn't exist
$(shell mkdir -p $(BIN_DIR))

# All targets
.PHONY: all part1 part2 part3 clean help
.PHONY: pi_sequential pi_cuda_gpu pi_cuda_shared_memory pi_cuda_2_level_reduction pi_multistage_reduction pi_cuda_gpu_tableau pi_cuda_tableau_2_level_reduction
.PHONY: matrix_sequential matrix_cuda_gpu matrix_cuda_shared_memory matrix_cuda_shared_memory_optimized matrix_cuda_2_level_reduction
.PHONY: matrix_mult_sequential

all: part1 part2 part3 part4

# ========== PART 1: Pi Calculation ==========
part1: $(BIN_DIR)/tp_cuda_part_1_pi_sequential \
       $(BIN_DIR)/tp_cuda_part_1_pi_cuda_gpu \
       $(BIN_DIR)/tp_cuda_part_1_pi_cuda_shared_memory \
       $(BIN_DIR)/tp_cuda_part_1_pi_cuda_2_level_reduction \
       $(BIN_DIR)/tp_cuda_part_1_pi_multistage_reduction \
       $(BIN_DIR)/tp_cuda_part_1_pi_cuda_gpu_tableau \
       $(BIN_DIR)/tp_cuda_part_1_pi_cuda_tableau_2_level_reduction
	@echo "Running Part 1 benchmarks..."
	python3 part1_build_csv.py
	@echo "Generating Part 1 performance analysis..."
	python3 part1_perf_analysis.py
	@echo "Part 1 complete!"

# Individual targets for Part 1
pi_sequential: $(BIN_DIR)/tp_cuda_part_1_pi_sequential
pi_cuda_gpu: $(BIN_DIR)/tp_cuda_part_1_pi_cuda_gpu
pi_cuda_shared_memory: $(BIN_DIR)/tp_cuda_part_1_pi_cuda_shared_memory
pi_cuda_2_level_reduction: $(BIN_DIR)/tp_cuda_part_1_pi_cuda_2_level_reduction
pi_multistage_reduction: $(BIN_DIR)/tp_cuda_part_1_pi_multistage_reduction
pi_cuda_gpu_tableau: $(BIN_DIR)/tp_cuda_part_1_pi_cuda_gpu_tableau
pi_cuda_tableau_2_level_reduction: $(BIN_DIR)/tp_cuda_part_1_pi_cuda_tableau_2_level_reduction

$(BIN_DIR)/tp_cuda_part_1_pi_sequential: $(PART1_DIR)/tp_cuda_part_1_pi_sequential.cu
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/tp_cuda_part_1_pi_cuda_gpu: $(PART1_DIR)/tp_cuda_part_1_pi_cuda_gpu.cu
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/tp_cuda_part_1_pi_cuda_shared_memory: $(PART1_DIR)/tp_cuda_part_1_pi_cuda_shared_memory.cu
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/tp_cuda_part_1_pi_cuda_2_level_reduction: $(PART1_DIR)/tp_cuda_part_1_pi_cuda_2_level_reduction.cu
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/tp_cuda_part_1_pi_multistage_reduction: $(PART1_DIR)/tp_cuda_part_1_pi_multistage_reduction.cu
	$(CXX) $(CXXFLAGS) $< -o $@	

$(BIN_DIR)/tp_cuda_part_1_pi_cuda_gpu_tableau: $(PART1_DIR)/tp_cuda_part_1_pi_cuda_gpu_tableau.cu
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/tp_cuda_part_1_pi_cuda_tableau_2_level_reduction: $(PART1_DIR)/tp_cuda_part_1_pi_cuda_tableau_2_level_reduction.cu
	$(CXX) $(CXXFLAGS) $< -o $@

# ========== PART 2: Vector Operations ==========
part2: $(BIN_DIR)/tp_cuda_part_2_matrix_sequential \
       $(BIN_DIR)/tp_cuda_part_2_matrix_cuda_gpu \
       $(BIN_DIR)/tp_cuda_part_2_matrix_cuda_shared_memory \
	   $(BIN_DIR)/tp_cuda_part_2_matrix_cuda_shared_memory_optimized \
	   $(BIN_DIR)/tp_cuda_part_2_matrix_cuda_2_level_reduction
	@echo "Running Part 2 benchmarks..."
	python3 part2_build_csv.py
	@echo "Generating Part 2 performance analysis..."
	python3 part2_perf_analysis.py
	@echo "Part 2 complete!"

# Individual targets for Part 2
matrix_sequential: $(BIN_DIR)/tp_cuda_part_2_matrix_sequential
matrix_cuda_gpu: $(BIN_DIR)/tp_cuda_part_2_matrix_cuda_gpu
matrix_cuda_shared_memory: $(BIN_DIR)/tp_cuda_part_2_matrix_cuda_shared_memory
matrix_cuda_shared_memory_optimized: $(BIN_DIR)/tp_cuda_part_2_matrix_cuda_shared_memory_optimized
matrix_cuda_2_level_reduction: $(BIN_DIR)/tp_cuda_part_2_matrix_cuda_2_level_reduction

$(BIN_DIR)/tp_cuda_part_2_matrix_sequential: $(PART2_DIR)/tp_cuda_part_2_matrix_sequential.cu
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/tp_cuda_part_2_matrix_cuda_gpu: $(PART2_DIR)/tp_cuda_part_2_matrix_cuda_gpu.cu
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/tp_cuda_part_2_matrix_cuda_shared_memory: $(PART2_DIR)/tp_cuda_part_2_matrix_cuda_shared_memory.cu
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/tp_cuda_part_2_matrix_cuda_shared_memory_optimized: $(PART2_DIR)/tp_cuda_part_2_matrix_cuda_shared_memory_optimized.cu
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/tp_cuda_part_2_matrix_cuda_2_level_reduction: $(PART2_DIR)/tp_cuda_part_2_matrix_cuda_2_level_reduction.cu
	$(CXX) $(CXXFLAGS) $< -o $@

# ========== PART 3: Matrix Multiplication ==========
part3: $(BIN_DIR)/tp_cuda_part_3_matrix_mult_sequential

# Individual targets for Part 3
matrix_mult_sequential: $(BIN_DIR)/tp_cuda_part_3_matrix_mult_sequential_sequential

$(BIN_DIR)/tp_cuda_part_3_matrix_mult_sequential_sequential: $(PART3_DIR)/tp_cuda_part_3_matrix_mult_sequential.cu
	$(CXX) $(CXXFLAGS) $< -o $@

# ========== CLEAN ==========
clean:
	rm -rf $(BIN_DIR)/*

# Help target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Build by Part:"
	@echo "  all     - Compile all parts"
	@echo "  part1   - Compile Part 1 (Pi calculation)"
	@echo "  part2   - Compile Part 2 (Vector operations)"
	@echo "  part3   - Compile Part 3 (Matrix multiplication)"
	@echo ""
	@echo "Part 1 - Individual Files:"
	@echo "  pi_sequential                           - Compile sequential Pi implementation"
	@echo "  pi_cuda_gpu                             - Compile basic GPU Pi implementation"
	@echo "  pi_cuda_shared_memory                   - Compile shared memory Pi implementation"
	@echo "  pi_cuda_gpu_2_level_reduction           - Compile 2 level reduction Pi implementation"
	@echo "  pi_cuda_multistage_reduction            - Compile multi-stage reduction Pi implementation"
	@echo "  pi_cuda_gpu_tableau                     - Compile tableau Pi implementation"
	@echo "  pi_cuda_gpu_tableau_2_level_reduction   - Compile tableau 2 level reduction Pi implementation"
	@echo ""
	@echo "Part 2 - Individual Files:"
	@echo "  matrix_sequential                       - Compile sequential vector operations"
	@echo "  matrix_cuda_gpu                         - Compile cuda vector operations"
	@echo "  matrix_cuda_shared_memory               - Compile cuda shared memory vector operations"
	@echo "  matrix_cuda_shared_memory_optimized     - Compile cuda shared memory optimized vector operations"
	@echo "  matrix_cuda_2_level_reduction           - Compile cuda 2 level reduction vector operations"
	@echo ""
	@echo "Part 3 - Individual Files:"
	@echo "  matrix_mult_sequential - Compile sequential matrix multiplication"
	@echo ""
	@echo "Utilities:"
	@echo "  clean  - Remove all compiled binaries"
	@echo "  help   - Show this help message"