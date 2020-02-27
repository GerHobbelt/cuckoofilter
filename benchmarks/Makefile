# Uncomment one of the following to switch between debug and opt mode
OPT = -O3 -DNDEBUG
#OPT = -O0 -ggdb3 #-fsanitize=address -fsanitize=undefined # -fsanitize=memory -fsanitize-memory-track-origins -DNDEBUG -stdlib=libc++ -nostdinc++ -I/usr/lib/llvm-9/include/c++/v1/ -L/usr/lib/llvm-9/lib -Wl,-rpath,/usr/lib/llvm-9/lib

CXXFLAGS += -fno-strict-aliasing -Wall -std=c++17 -I. -I../src/ $(OPT) -march=native

LDFLAGS+= -Wall -lpthread -lssl -lcrypto

HEADERS = $(wildcard ../src/*.h) *.h

SRC = ../src/hashutil.cc

.PHONY: all

BINS = conext-table3.exe conext-figure5.exe bulk-insert-and-query.exe

all: $(BINS)

clean:
	/bin/rm -f $(BINS)

%.exe: %.cc ${HEADERS} ${SRC} Makefile
	$(CXX) $(CXXFLAGS) $< -o $@ $(SRC) $(LDFLAGS)
