#include "mex.h"
#include "LSQR_1.h"
#include "cublas_v2.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
	if (nrhs != 6)
	mexErrMsgTxt("Invaid number of input arguments");

	if (nlhs != 1)
	mexErrMsgTxt("Invalid number of outputs");

	if (!mxIsSingle(prhs[0]))

	mexErrMsgTxt("input vector data type must be single");

	float *A = (float*)mxGetData(prhs[0]);
	int numRowsA = (int)mxGetM(prhs[0]);
	int numColsA = (int)mxGetN(prhs[0]);

	float *b = (float*)mxGetData(prhs[1]);

	int *posx = (int*)mxGetData(prhs[4]);
	int *posy = (int*)mxGetData(prhs[5]);
	double *n_d = (double*)mxGetData(prhs[2]); //_d means double
	double *nz_d = (double*)mxGetData(prhs[3]);

	int n = (int)(*n_d);
	int nz = (int)(*nz_d);
	// int posx = (int)(*posx_d);
	// int posy = (int)(*posy_d);


	// plhs[0] = mxCreateNumericMatrix(numRowsA, numColsA, mxSINGLE_CLASS, mxREAL);
	// float *As = (float*)mxGetData(plhs[0]);

	plhs[0] = mxCreateNumericMatrix(1, numColsA, mxSINGLE_CLASS, mxREAL);
	float* V = (float*)mxGetData(plhs[0]); // _f means float


	// plhs[1] = mxCreateNumericMatrix(1, n*n*nz , mxINT32_CLASS, mxREAL);
	// int *Ps = (int*)mxGetData(plhs[1]);

	// test(A, posx, posy, n, nz, numColsA, numRowsA);
	LSQR_1(A, b, V, posx, posy, n, nz, numColsA, numRowsA);
	return;
}


