NVCC ?= nvcc
CC = $(NVCC)
CFLAGS=-std=c++17 -O3 kernel.cu
# -flto
default: all

all: kernel.cu
	$(CC) $(CFLAGS) -o movegen_gpu
