# Makefile for enumDevices project

# Variables
BUILD_DIR := build
TARGET := enumDevices
TARGET_PATH := $(BUILD_DIR)/$(TARGET)
SRC_DIR := src
FORMAT_FILES := $(shell find $(SRC_DIR) -name '*.cpp' -o -name '*.h' -o -name '*.hpp')

# Default target
all: build

# Build the project using CMake
build:
	@echo "Building project..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ..
	@cd $(BUILD_DIR) && make
	@echo "Build completed: $(TARGET_PATH)"

# Run the compiled program
run: build
	@echo "Running $(TARGET)..."
	@./$(TARGET_PATH)

# Format source code using clang-format
format:
	@echo "Formatting source files..."
	@if [ -z "$(FORMAT_FILES)" ]; then \
		echo "No source files found in $(SRC_DIR)"; \
	else \
		echo "Formatting files:"; \
		for file in $(FORMAT_FILES); do \
			echo "  $$file"; \
		done; \
		clang-format -i $(FORMAT_FILES) && echo "Formatting completed successfully"; \
	fi

# Check formatting without applying changes
format-check:
	@echo "Checking code formatting..."
	@if [ -z "$(FORMAT_FILES)" ]; then \
		echo "No source files found in $(SRC_DIR)"; \
	else \
		clang-format --dry-run --Werror $(FORMAT_FILES) && echo "Formatting is correct"; \
	fi

# Clean build files
clean:
	@echo "Cleaning build files..."
	@rm -rf $(BUILD_DIR)
	@echo "Clean completed"

# Rebuild from scratch
rebuild: clean build

# Show help
help:
	@echo "Available targets:"
	@echo "  make build        - Build project using CMake"
	@echo "  make run          - Build and run the program"
	@echo "  make format       - Format source code with clang-format"
	@echo "  make format-check - Check formatting without applying changes"
	@echo "  make clean        - Remove build directory"
	@echo "  make rebuild      - Clean and rebuild"
	@echo "  make help         - Show this help message"

# Phony targets (not real files)
.PHONY: all build run format format-check clean rebuild help