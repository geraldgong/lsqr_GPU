clear all; close all; clc

% system('nvcc -c test.cu  -lcudart -lcublas -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin" -I"C:\Program Files\MATLAB\R2015a\extern\include" ');
% mex test.obj -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\lib\x64" ...
%  -v -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\include" -lcudart -lcublas;
% 
system('nvcc -c lsqr_sv_Cuda.cu  -lcudart -lcublas -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin" -I"C:\Program Files\MATLAB\R2015a\extern\include" ');
mex lsqr_sv_Cuda.obj -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\lib\x64" ...
 -v -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\include" -lcudart -lcublas;