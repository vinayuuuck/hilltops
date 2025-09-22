# Simple Makefile for Hilltops Server

CXX      := clang++
CXXFLAGS := -std=c++17 -O2 -Wall
SRC      := src/hilltops_server.cpp
TARGET   := hilltops_server

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
