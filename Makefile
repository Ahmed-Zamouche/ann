#Project settings
PROJECT = libann
LIB =
SRC = src
INC = -Isrc
SRC_TST = test

CROSS ?= 

#Compiler settings
CPP = $(CROSS)gcc
CC = $(CROSS)gcc
AS = $(CROSS)gcc
LD = $(CROSS)gcc
AR = $(CROSS)ar

CPP_FLAGS = -g -ggdb -Og $(INC) -Wall -Wextra
CC_FLAGS  = -g -ggdb -Og $(INC) -Wall -Wextra
AS_FLAGS  = $(CC_FLAGS) -D_ASSEMBLER_
LD_FLAGS = -Lbuild/lib -lfixmath -lann -lm

# Find all source files
SRC_CPP = $(foreach dir, $(SRC), $(wildcard $(dir)/*.cpp))
SRC_C   = $(foreach dir, $(SRC), $(wildcard $(dir)/*.c))
SRC_S   = $(foreach dir, $(SRC), $(wildcard $(dir)/*.S))
OBJ_CPP = $(patsubst %.cpp, %.o, $(SRC_CPP))
OBJ_C   = $(patsubst %.c, %.o, $(SRC_C))
OBJ_S   = $(patsubst %.S, %.o, $(SRC_S))

SRC_TST_CPP   = $(foreach dir, $(SRC_TST), $(wildcard $(dir)/*.cpp))
SRC_TST_C     = $(foreach dir, $(SRC_TST), $(wildcard $(dir)/*.c))

OBJ_C        := $(OBJ_C) $(patsubst %.c, %.o, $(SRC_TST_C))
OBJ_CPP      := $(OBJ_CPP) $(patsubst %.c, %.o, $(SRC_TST_CPP))

OBJ          = $(OBJ_CPP) $(OBJ_C) $(OBJ_S)

# Compile rules.
.PHONY : all
all: build

.PHONY : build
build: pre_build libfixmath $(PROJECT).a post_build


.PHONY : clang_format
clang_format:
	clang-format -i src/*.c src/*.h
	clang-format -i test/*.c

.PHONY : pre_build
pre_build: clang_format
	mkdir -p build/lib

.PHONY : libfixmath
libfixmath:
	cd libfixmath/libfixmath &&	make CROSS=$(CROSS)

.PHONY : post_build
post_build:
	mv libfixmath/libfixmath/libfixmath.a build/lib/
	mv libann.a build/lib/

.PHONY : test
test: build
	mkdir -p  build/bin
	$(CC) $(CC_FLAGS) test/TEST_main.c -o build/bin/TEST_ann $(LD_FLAGS)
#./build/bin/TEST_ann

$(PROJECT).a: $(OBJ)
	$(AR) rcs $(PROJECT).a $(OBJ)

$(OBJ_CPP) : %.o : %.cpp
	$(CPP) $(CPP_FLAGS) -o $@ -c $<

$(OBJ_C) : %.o : %.c
	$(CC) $(CC_FLAGS) -o $@ -c $<

$(OBJ_S) : %.o : %.S
	$(AS) $(AS_FLAGS) -o $@ $<

# Clean rules
.PHONY : clean
clean:
	rm -f $(PROJECT).a $(OBJ)

.PHONY : clean_libfixmath
clean_libfixmath:
	cd libfixmath/libfixmath && make clean

