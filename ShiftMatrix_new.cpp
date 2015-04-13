#include "ShiftMatrix_new.h"
#include "C:\Program Files\MATLAB\R2014b\extern\include\mex.h"

extern void shiftmatrix(float *A, float *As, int *ps, int posx, int posy, int n, int nz, int w, int h);

void shift(int shift, int n, int *d_pos_new) // Dim [n, 1]
{

	for (int i = 0; i < shift; i++)
	{
		d_pos_new[i] = shift + 2 - (i + 1);
	}
	for (int i = shift; i < n; i++)
	{
		d_pos_new[i] = (i + 1) - shift;
	}
}

void positions_new_x(int n, int *d_pos_newx, int *d_pos_newx_tot) // Dim [1,n*n]
{
//	shift(shift_x, n, d_pos_newx);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			d_pos_newx_tot[j + i * n] =  d_pos_newx[j] + i * n;
		}
	}
}

void positions_new_y(int n, int *d_pos_newy, int *d_pos_newy_tot) // Dim [1,n*n]
{
//	shift(shift_y, n, d_pos_newy);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			d_pos_newy_tot[j + i * n] =  (d_pos_newy[i] - 1) * n + (j + 1);
		}
	}
}	

// set original postions as [n*n, nz] matrix

void init_positions(int n, int nz, int *positions)
{
	for (int i = 0; i < n*n*nz; i++)
	{
		positions[i] = i + 1;
	}
}

void postions_shifted(int* po, int*d_positions, int *d_pos_newx_tot, int *d_pos_newy_tot,  int *pos_shift_x, int *positions_shifted_matrix, int n, int nz)
{
	
	int w = n*n; //width
	int h = nz;  //hight


	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
    	{
    		po[j] = d_positions[i * w + j]; // Dim [1,n*n]
		} // get po after this loop

		for (int k = 0; k < w; k++)
		{
			int val1 = d_pos_newx_tot[k];
			pos_shift_x[k] = po[val1 - 1];  //[1,n*n]
		} // load po, get pos_shift_x 
		//po = pos_shift_x;

		for (int l = 0; l < w; l++)
		{
			int val2 = d_pos_newy_tot[l];
			int val3 = pos_shift_x[val2 - 1];
			positions_shifted_matrix[i * w + l] = val3; //Dim [nz*n*n]
		} // store the final shifted matix 
    }

}


// void shiftmatrix(float *d_A_mat, float *d_A_mat_shift, int *po, int *d_positions, int shift_x, 
// 							int shift_y, int *d_pos_newx, int *d_pos_newy, int *d_pos_newx_tot, 
// 							int *d_pos_newy_tot, int *pos_shift_x, int *positions_shifted_matrix, 
// 							int n, int nz, int w, int h)
void shiftmatrix(float *A_mat, float *A_mat_shift, int n, int nz, int posx, int posy, int w, int h)
{

// 	int col = threadIdx.x + blockDim.x * blockIdx.x;__
//  int row = threadIdx.y + blockDim.y * blockIdx.y;
// call __device___ kernels to get shifted matrix

    // if (col >= w || row >= h) {
    //     return;
    // }
	int *positions;
	positions = (int *)malloc(sizeof(int) * w);
	init_positions(n, nz, positions);
	// function "shift"
	int shift_x = posx - 1;
    int shift_y = posy - 1;
    int *d_pos_newx; 
    int *d_pos_newy;
    d_pos_newx = (int *)malloc(sizeof(int) * n);
    d_pos_newy = (int *)malloc(sizeof(int) * n);
    shift(shift_x, n, d_pos_newx);
    shift(shift_y, n, d_pos_newy);

    // function "postion_new_x" "position_new_y"
    int *d_pos_newx_tot;
    int *d_pos_newy_tot;
    d_pos_newx_tot = (int *)malloc(sizeof(int) * n * n);
    d_pos_newy_tot = (int *)malloc(sizeof(int) * n * n);
    positions_new_x(n, d_pos_newx, d_pos_newx_tot);
	positions_new_y(n, d_pos_newy, d_pos_newy_tot);

	// function "positions_shifted"
	int *positions_shifted_matrix;
	positions_shifted_matrix = (int *)malloc(sizeof(int) * w);
	int *po;
	po= (int *)malloc(sizeof(int) * n * n);
	int *pos_shift_x;
	pos_shift_x = (int *)malloc(sizeof(int) * n * n);
	postions_shifted(po, positions, d_pos_newx_tot, d_pos_newy_tot, pos_shift_x, positions_shifted_matrix, n, nz);
   

    for (int i = 0; i < w; i++)
    {
    	
    	int index = positions_shifted_matrix[i];

    	for (int j = 0; j < h; j++)
    	{
    		 A_mat_shift[i * h + j] = A_mat[(index - 1) * h + j];
    	}
    }
    
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
	if (nrhs != 5)
	mexErrMsgTxt("Invaid number of input arguments");

	if (nlhs != 1)
	mexErrMsgTxt("Invalid number of outputs");

	if (!mxIsSingle(prhs[0]))
	mexErrMsgTxt("input vector data type must be single");

	int numRowsA = (int)mxGetM(prhs[0]);
	int numColsA = (int)mxGetN(prhs[0]);

	float *A = (float*)mxGetData(prhs[0]); 

	double *n_d = (double*)mxGetData(prhs[1]); //_d means double
	double *nz_d = (double*)mxGetData(prhs[2]);
	double *posx_d = (double*)mxGetData(prhs[3]);
	double *posy_d = (double*)mxGetData(prhs[4]);
	int n = (int)(*n_d); // convert double to int
	int nz = (int)(*nz_d);
	int posx = (int)(*posx_d);
	int posy = (int)(*posy_d);
	
	plhs[0] = mxCreateNumericMatrix(numRowsA, numColsA, mxSINGLE_CLASS, mxREAL);
	float* As = (float*)mxGetData(plhs[0]); // _f means float


	shiftmatrix(A, As, n, nz, posx, posy, numColsA, numRowsA);
	return;
}



// void shiftmatrix(float *A_mat, float *A_mat_shift, int n, int nz, int posx, int posy, int w, int h)
// {	
	
// //	int posx = 3;
// //	int posy = 3;
// 	int size_mat = w * h;
// 	int shift_x = posx - 1;
//     int shift_y = posy - 1;

//     int *positions = new int[(size_t)w];

// 	dim3 threadsPerBlock(16, 16);
// 	dim3 numBlocks(w/threadsPerBlock.x, h/threadsPerBlock.y);
// // input matrix
// 	float *d_A_mat = NULL;
// 	cudaMalloc(&d_A_mat, sizeof(float) * size_mat);
//     cudaMemcpy(d_A_mat, A_mat, sizeof(float) * size_mat, cudaMemcpyHostToDevice);

//     float *d_A_mat_shift = NULL;
//     cudaMalloc(&d_A_mat_shift, sizeof(float) * size_mat);
//     cudaMemcpy(d_A_mat_shift, A_mat_shift, sizeof(float) * size_mat, cudaMemcpyHostToDevice);

//     int *po = NULL;
//     cudaMalloc(&po, sizeof(int) * n * n);
//     cudaMemset(po, 0, sizeof(int) * n * n);

//     int *d_positions = NULL;
//     cudaMalloc(&d_positions, sizeof(int) * w);
//     cudaMemset(&d_positions, 0, sizeof(int) * w);

//     int *d_pos_newx = NULL;
//     cudaMalloc(&d_pos_newx, sizeof(int) * n);
//     cudaMemset(&d_pos_newx, 0, sizeof(int) * n);
   
//     int *d_pos_newy = NULL;
//     cudaMalloc(&d_pos_newy, sizeof(int) * n);
//     cudaMemset(&d_pos_newy, 0, sizeof(int) * n);
   
//     int *d_pos_newx_tot = NULL;
//     cudaMalloc(&d_pos_newx_tot, sizeof(int) * n * n);
//     cudaMemset(&d_pos_newx_tot, 0, sizeof(int) * n * n);   

//     int *d_pos_newy_tot = NULL;
//     cudaMalloc(&d_pos_newy_tot, sizeof(int) * n * n);
//     cudaMemset(&d_pos_newy_tot, 0, sizeof(int) * n * n);
	
// 	int *pos_shift_x = NULL;
//     cudaMalloc(&pos_shift_x, sizeof(int) * n * n);
//     cudaMemset(&pos_shift_x, 0, sizeof(int) * n * n);

//    	int *positions_shifted_matrix = NULL;
//     cudaMalloc(&positions_shifted_matrix, sizeof(int) * w);
//     cudaMemset(&positions_shifted_matrix, 0, sizeof(int) * w);

//     init_positions(n, nz, positions);
//     cudaMemcpy(d_positions, positions, sizeof(int) * w, cudaMemcpyHostToDevice);
    
// 	shiftmatrix<<<numBlocks, threadsPerBlock>>>(d_A_mat, d_A_mat_shift, po, d_positions, shift_x, 
// 							shift_y, d_pos_newx, d_pos_newy, d_pos_newx_tot, d_pos_newy_tot, 
// 							pos_shift_x, positions_shifted_matrix, n, nz, w, h);
	
//     cudaMemcpy(A_mat_shift, d_A_mat_shift, sizeof(float) * size_mat, cudaMemcpyDeviceToHost);
    
//     cudaFree(d_A_mat);
//     cudaFree(d_A_mat_shift);
//     cudaFree(po);
//     cudaFree(d_positions);
//     cudaFree(d_pos_newx);
//     cudaFree(d_pos_newy);
//     cudaFree(d_pos_newx_tot);
//     cudaFree(d_pos_newy_tot);
//     cudaFree(pos_shift_x);
//     cudaFree(positions_shifted_matrix);

// }