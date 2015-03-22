#include "mex.h"


extern void test_cublas(float *A_mat, float *b, int *posx, int *posy, float *vector, int n, int nz, int k, int w, int h);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
	if (nrhs != 7)
		mexErrMsgTxt("Invaid number of input arguments");

	if (nlhs != 1)
		mexErrMsgTxt("Invalid number of outputs");

	if (!mxIsSingle(prhs[0]))
		mexErrMsgTxt("input matrix data type must be single");

  	float *A = (float*)mxGetData(prhs[0]);

	int numRowsA = (int)mxGetM(prhs[0]);
	int numColsA = (int)mxGetN(prhs[0]);

	float *b = (float*)mxGetData(prhs[1]);
  	int *posx = (int*)mxGetData(prhs[2]);
	int *posy = (int*)mxGetData(prhs[3]);
	double *n_d = (double*)mxGetData(prhs[4]); //_d means double
	double *nz_d = (double*)mxGetData(prhs[5]);
	double *k_d = (double*)mxGetData(prhs[6]);
	
	int n = (int)(*n_d);
	int nz = (int)(*nz_d);
	int k = (int)(*k_d);

	plhs[0] = mxCreateNumericMatrix(numColsA, 1, mxSINGLE_CLASS, mxREAL);
	float *v = (float*)mxGetData(plhs[0]);

	// plhs[1] = mxCreateNumericMatrix(numRowsA, numColsA, mxSINGLE_CLASS, mxREAL);
	// float *As = (float*)mxGetData(plhs[1]);


	test_cublas(A, b, posx, posy, v, n, nz, k, numColsA, numRowsA);
	return;
}
