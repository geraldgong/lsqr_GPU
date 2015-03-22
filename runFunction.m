clear all; close all; clc
system('nvcc -c lsqr_Cuda.cu  -lcublas -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin" -I"C:\Program Files\MATLAB\R2015a\extern\include" ');
mex lsqr_Cuda.cpp lsqr_Cuda.obj -lcudart -lcublas -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64" ...
 -v -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\include";
% 
%  system('nvcc -c lsqr_version1.cu -lcublas -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin" -I"C:\Program Files\MATLAB\R2015a\extern\include" ');
%  mex lsqr_version1.cpp lsqr_version1.obj  -lcudart -lcublas -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64" ...
%       -v -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\include";