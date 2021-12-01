

# Location of the OpenCL library, where libOpenCL.so resides
CL_LIB=/usr/local/cuda/lib64
# Location of the OpenCL CL directory where cl.h and cl.hpp reside
CL_INCLUDE=/usr/local/cuda/include

# C++ compiler and flags
CXX=g++

ifeq ($(OS),Windows_NT)
	CXXFLAGS=-g -O2 -fPIC -fopenmp -I$(CL_INCLUDE) -std=c++11
	LFLAGS=-g -L$(CL_LIB) -lstdc++ -fopenmp    
else
	uname_s := $(shell uname -s)
	ifeq ($(uname_s),Linux)
		CXXFLAGS=-g -O2 -fPIC -fopenmp -I$(CL_INCLUDE) -std=c++11
		LFLAGS=-g -L$(CL_LIB) -lstdc++ -fopenmp         
	endif
	ifeq ($(uname_s),Darwin)
		CXXFLAGS=-g -O2 -fPIC -fopenmp -std=c++11
		LFLAGS=-g -lstdc++ -framework OpenCL -fopenmp       
	endif
endif

# Matrix multiplication
all:	xcorr \
		template	

xcorr:	xcorr.o
	$(CXX) $(LFLAGS) -o $@ $< -lOpenCL

template:	template.o
	$(CXX) $(LFLAGS) -o $@ $< -lOpenCL

%.o:	%.cpp cl_helper.hpp kernels.cl
	$(CXX) -c $(CXXFLAGS) -o $@ $<

clean:
	rm -rf *.o *.mod \
    template \
    xcorr \
