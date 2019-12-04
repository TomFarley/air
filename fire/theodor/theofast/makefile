# Detect x64 machines
x64 := $(shell uname -i | grep -qv x86_64; echo $$?)
ifeq ($(x64), 1)
bit=x64
else
bit=x32
endif

CC = gcc
FLAGS = -O3 -fPIC -Wall -std=c99 -ffast-math -fopenmp
LIBS = -lm
INC = 

all: libtheofast_$(bit).so

libtheofast_$(bit).so: theofast_$(bit).o
	$(CC) -shared -o libtheofast_$(bit).so theofast_$(bit).o $(FLAGS) $(LIBS)

theofast_$(bit).o: theofast.c
	$(CC) -o theofast_$(bit).o -c theofast.c $(FLAGS) $(INC)

clean:
	rm -f *_$(bit).so
	rm -f *_$(bit).o
	rm -f *~
