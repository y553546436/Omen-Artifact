# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -std=c++20 -Wall -Wextra -pedantic

# Directories
SRCDIR := src
INCDIR := include
BUILDDIR := build
TARGET := bin/performance

# Source files
SRCS := $(wildcard $(SRCDIR)/*.cpp)
# Object files
OBJS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SRCS:.cpp=.o))
# Include directories
INC := -I$(INCDIR)

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p bin
	@echo " Linking..."
	@echo " $(CXX) $^ -o $(TARGET)"; $(CXX) $^ -o $(TARGET)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	@echo " Compiling $<..."; $(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<

clean:
	@echo " Cleaning..."; $(RM) -r $(BUILDDIR) $(TARGET)

run: $(TARGET)
	@echo " Running..."; ./$(TARGET)

eval: $(TARGET)
	@echo " Evaluating..."
	sudo chrt -f 99 ./$(TARGET)

.PHONY: clean
