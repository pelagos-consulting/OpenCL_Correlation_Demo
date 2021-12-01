# OpenCL Correlation Demo
Demo for the computation of 2D cross correlations using OpenCL to farm compute across multiple devices

Example constructed by Dr Toby Potter from [Pelagos Consulting and Education](https://www.pelagos-consulting.com)

# Installation procedure

1. You will need to have an OpenCL environment installed and ready to use with known locations for "CL/cl.hpp" and "libOpenCL.so".
2. In order to use the Jupyter notebook for writing data, you need a jupyter-lab python environment installed with the following packages:
    * Ipyml
    * Matplotlib
    * Scipy
    * Numpy
    * Jupyter 
    * You will also need a C++ compiler
3.  ipyml package. Construction and testing of the demo has been done on Linux but it may also work for MacOS.
4. After downloading the demo, edit the makefile to suit the location of your OpenCL and Library files. Then use the notebook to run the code.
