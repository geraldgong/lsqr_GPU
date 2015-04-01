#include <stdio.h>
#include <stdlib.h>
#include "mex.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

__global__ void kernel_init_positions(int *d_positions, int w, int h) // 1
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= w || y >= h)
	{
		return;
	}
	int idx = x + w * y;	
	d_positions[idx] = idx + 1;
}

__global__ void kernel_shift(int shift, int *d_pos_new, int n)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;

	if (x >= n)
	{
		return;
	}

	if (x < shift)
		d_pos_new[x] = shift + 2 - (x + 1);
	else
		d_pos_new[x] = (x + 1) - shift;
}

__global__ void kernel_new_total(int *d_pos_newx, int *d_pos_newy, int *d_pos_newx_tot, int *d_pos_newy_tot, int n)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int idx1 = threadIdx.x;
	int idx2 = blockIdx.x;
	if (x >= n*n)
	{
		return;
	}

	d_pos_newx_tot[x] = d_pos_newx[idx1] + idx2 * n;
	d_pos_newy_tot[x] = (d_pos_newy[idx2] - 1) * n + (idx1 + 1);
}


__global__ void kernel_positions_shifted(int *d_positions, int *d_pos_newx_tot, int *d_pos_newy_tot, 
										 int *d_layer_positions, int *pos_shift_x, int *d_positions_shifted, int w, int h)
{

	int col = threadIdx.x + blockDim.x * blockIdx.x;
	// int row = threadIdx.y + blockDim.y * blockIdx.y;


	if (col >= w)
	{
		return;
	}

	d_layer_positions[col] = d_positions[col + w * h];
	int temp1 = d_pos_newx_tot[col];
	pos_shift_x[col] = d_layer_positions[temp1 - 1];
	int temp2 = d_pos_newy_tot[col];
	d_positions_shifted[col + w * h] = pos_shift_x[temp2 - 1];
	
}


__global__ void kernel_shift_matrix(float *d_A_mat, float *d_A_mat_shifted, int *d_positions_shifted, int w, int h)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= w || y >=h)
	{
		return;
	}

	int index = d_positions_shifted[x];
	d_A_mat_shifted[y + x * h] = d_A_mat[y + (index - 1) * h];
}

__global__ void kernel_divide_vector(float *d_b_tot, float *d_b, int i)
{
	int x = threadIdx.x;
	d_b[x] = d_b_tot[x + i * blockDim.x];

}


__global__ void kernel_combine_vector(float *d_v, float *d_v_tot, int i)
{
	int x = threadIdx.x;
	d_v_tot[x + i * blockDim.x] = d_v[x];
}

__global__ void kernel_update_vector(float *d_src, float *d_res, float para, int w)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;

	if (x >= w)
	{
		return;
	}

	d_res[x] = d_src[x] - para * d_res[x];
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
	// if (nrhs != 7)
	// 	mexErrMsgTxt("Invaid number of input arguments");

	// if (nlhs != 1)
	// 	mexErrMsgTxt("Invalid number of outputs");

	// if (!mxIsSingle(prhs[0]))
	// 	mexErrMsgTxt("input matrix data type must be single");
//initialization
  	float *A_mat = (float*)mxGetData(prhs[0]);

	int h = (int)mxGetM(prhs[0]);
	int w = (int)mxGetN(prhs[0]);

	float *b = (float*)mxGetData(prhs[1]);
  	int *pos_x = (int*)mxGetData(prhs[2]);
	int *pos_y = (int*)mxGetData(prhs[3]);
	double *n_d = (double*)mxGetData(prhs[4]); //_d means double
	double *nz_d = (double*)mxGetData(prhs[5]);
	double *k_d = (double*)mxGetData(prhs[6]);
	
	int n = (int)(*n_d);
	int nz = (int)(*nz_d);
	int k = (int)(*k_d);

	int w_p = n*n;
	int h_p = nz;
	int h_b = n*n*h;



	plhs[0] = mxCreateNumericMatrix(w, 1, mxSINGLE_CLASS, mxREAL);
	float *Vector = (float*)mxGetData(plhs[0]);

	// plhs[1] = mxCreateNumericMatrix(w, 1, mxSINGLE_CLASS, mxREAL);
	// float *U = (float*)mxGetData(plhs[1]);
	

	cublasStatus_t status;
	cublasHandle_t handle;
    cublasCreate(&handle);

 // Time
	// cudaEvent_t start, stop;  
	// float time;
 // 	cudaEventCreate(&start);  
	// cudaEventCreate(&stop);  

// Memory Allocation	
	// 1
	int *d_positions = NULL;
	cudaMalloc(&d_positions, sizeof(int) * w_p * h_p);
	// 2
	int *d_pos_newx = NULL;
	cudaMalloc(&d_pos_newx, sizeof(int) * n);
	int *d_pos_newy = NULL;
	cudaMalloc(&d_pos_newy, sizeof(int) * n);
	// 3
	int *d_pos_newx_tot = NULL;
	cudaMalloc(&d_pos_newx_tot, sizeof(int) * w_p);
	int *d_pos_newy_tot = NULL;
	cudaMalloc(&d_pos_newy_tot, sizeof(int) * w_p); 
	// 4
	int *d_layer_positions = NULL;
	cudaMalloc(&d_layer_positions, sizeof(int) * w_p);
	int *pos_shift_x = NULL;
	cudaMalloc(&pos_shift_x, sizeof(int) * w_p);
	int *d_positions_shifted = NULL;
	cudaMalloc(&d_positions_shifted, sizeof(int) * w_p * h_p);
	// 5
	float *d_A_mat = NULL;
	cudaMalloc(&d_A_mat, sizeof(float) * w * h);
	cudaMemcpy(d_A_mat, A_mat, sizeof(float) * w * h, cudaMemcpyHostToDevice);

	float *d_A_mat_shifted = NULL;
	cudaMalloc(&d_A_mat_shifted, sizeof(float) * w * h);
	// cudaMemcpy(d_A_mat_shifted, A_mat_shifted, sizeof(float) * w * h, cudaMemcpyHostToDevice);
	//6
	float *d_b = NULL;
	cudaMalloc(&d_b, sizeof(float) * h_b);
	cudaMemcpy(d_b, b, sizeof(float) * h_b, cudaMemcpyHostToDevice);
	float *d_u = NULL;
	cudaMalloc(&d_u, sizeof(float) * h_b);
	float *d_u_small = NULL;
	cudaMalloc(&d_u_small, sizeof(float) * h);
	float *d_Vector0 = NULL;
	cudaMalloc(&d_Vector0, sizeof(float) * w);
	thrust :: device_ptr<float> dev_ptr_v0(d_Vector0); 
    thrust :: fill(dev_ptr_v0, dev_ptr_v0 + w, 0);
    float *d_r = NULL;
    cudaMalloc(&d_r, sizeof(float) * w);
	float *d_v = NULL;
	cudaMalloc(&d_v, sizeof(float) * w);
	float *d_w = NULL;
	cudaMalloc(&d_w, sizeof(float) * w);
	//7
	float *d_Vector1 = NULL;
	cudaMalloc(&d_Vector1, sizeof(float) * h_b);
	float *d_aux1 = NULL;
	cudaMalloc(&d_aux1, sizeof(float) * h);
	float *d_p = NULL;
	cudaMalloc(&d_p, sizeof(float) * h_b);
	//8
	float *d_aux2 = NULL;
	cudaMalloc(&d_aux2, sizeof(float) * h);
	float *d_Vector2 = NULL;
	cudaMalloc(&d_Vector2, sizeof(float) * w);
	thrust :: device_ptr<float> dev_ptr_v2(d_Vector2); 
    thrust :: fill(dev_ptr_v2, dev_ptr_v2 + w, 0);
	float *d_x = NULL;
	cudaMalloc(&d_x, sizeof(float) * w);
	thrust :: device_ptr<float> dev_ptr_x(d_x); 
    thrust :: fill(dev_ptr_x, dev_ptr_x + w, 0);


// Kernels
	dim3 block1 = dim3(32, 16, 1);
	dim3 grid1 = dim3((w_p + block1.x -1)/block1.x, (h_p + block1.y - 1)/block1.y, 1);

	dim3 block2 = dim3(32, 16, 1);
	dim3 grid2 = dim3((n + block1.x -1)/block1.x, 1, 1);

	dim3 block3 = dim3(n, 1, 1);
	dim3 grid3 = dim3(n, 1, 1);

	dim3 block4 = dim3(32, 8, 1); // test as perfect choice? don't know why...
	dim3 grid4 = dim3((w_p + block4.x - 1)/block4.x, (h_p + block4.y - 1)/block4.y, 1);

	dim3 block5 = dim3(32, 16, 1);
	dim3 grid5 = dim3((w + block5.x - 1)/block5.x, (h + block5.y - 1)/block5.y, 1);

	dim3 block6 = dim3(h, 1, 1);
	dim3 grid6 = dim3(n*n, 1, 1);

	dim3 block7(256, 1);
	dim3 grid7((w + block4.x - 1)/block4.x, 1);

	float alpha = 0.0f;
	float beta = 0.0f;
	float beta_norm = 0.0f;
	float alpha_norm = 0.0f;

	double phi_bar = 0.0;
	double rho_bar = 0.0;
	double rrho = 0.0;
	double c1 = 0.0;
	double s1 = 0.0;
	double theta = 0.0;
	double phi = 0.0;

	float para1 = 0.0f;
	float para2 = 0.0f;

	float alpha1 = 1;
	float beta1 = 1;
	float beta0 = 0;


	status = cublasSnrm2(handle, h_b, d_b, 1, &beta);
   	beta_norm = 1.0 / beta;
   	// normalization
   	status = cublasScopy(handle, h_b, d_b, 1, d_u ,1);
   	status = cublasSscal(handle, h_b, &beta_norm, d_u, 1);

   	kernel_init_positions<<<grid1, block1>>> (d_positions, w_p, h_p);
   	
	for (int i = 0; i < n*n; i++)
	{
		int shift_x = pos_x[i] - 1;
		int shift_y = pos_y[i] - 1;
		// shift matrix
	
		kernel_shift<<<grid2, block2>>> (shift_x, d_pos_newx, n);
		kernel_shift<<<grid2, block2>>> (shift_y, d_pos_newy, n);
		kernel_new_total<<<grid3, block3>>> (d_pos_newx, d_pos_newy, d_pos_newx_tot, d_pos_newy_tot ,n);
		for (int j = 0; j < h_p; j++)
		{
			kernel_positions_shifted<<<grid4, block4>>> (d_positions,  d_pos_newx_tot, 
								    	 				d_pos_newy_tot, d_layer_positions, pos_shift_x, d_positions_shifted, w_p, j);
		}
		kernel_shift_matrix<<<grid5, block5>>> (d_A_mat, d_A_mat_shifted, d_positions_shifted, w, h);
		// matrix-vector multiplication
		kernel_divide_vector<<<grid6, block6>>> (d_u, d_u_small, i);
		status = cublasSgemv(handle, CUBLAS_OP_T, h, w, &alpha1, d_A_mat_shifted, h, d_u_small, 1, &beta1, d_Vector0, 1);
	}

	status = cublasScopy(handle, w, d_Vector0, 1, d_r ,1);
	status = cublasSnrm2(handle, w, d_r, 1, &alpha);
	
	alpha_norm = 1 / alpha;
	phi_bar = beta;
	rho_bar = alpha;
	float alpha_minus = -alpha;
   	float beta_minus;

   	status = cublasScopy(handle, w, d_r, 1, d_v ,1);
	status = cublasSscal(handle, w, &alpha_norm, d_v, 1);
	status = cublasScopy(handle, w, d_v, 1, d_w ,1);

	for(int j = 0; j < k; j++)
	{
		for (int i = 0; i < n*n; i++)
   		{
   			int shift_x = pos_x[i] - 1;
			int shift_y = pos_y[i] - 1;
			// shift matrix

			kernel_shift<<<grid2, block2>>> (shift_x, d_pos_newx, n);
			kernel_shift<<<grid2, block2>>> (shift_y, d_pos_newy, n);
			kernel_new_total<<<grid3, block3>>> (d_pos_newx, d_pos_newy, d_pos_newx_tot, d_pos_newy_tot ,n);
			for (int j = 0; j < h_p; j++)
			{
				kernel_positions_shifted<<<grid4, block4>>> (d_positions,  d_pos_newx_tot, 
							    	 					 	 d_pos_newy_tot, d_layer_positions, pos_shift_x, d_positions_shifted, w_p, j);
			}
			kernel_shift_matrix<<<grid5, block5>>> (d_A_mat, d_A_mat_shifted, d_positions_shifted, w, h);

   			status = cublasSgemv(handle, CUBLAS_OP_N, h, w, &alpha1, d_A_mat_shifted, h, d_v, 1, &beta0, d_aux1, 1);
   			kernel_combine_vector<<<grid6, block6>>> (d_aux1, d_Vector1, i);
    	}

		status = cublasSaxpy(handle, h_b, &alpha_minus, d_u, 1, d_Vector1, 1);
		status = cublasScopy(handle, h_b, d_Vector1, 1, d_p ,1);
		status = cublasSnrm2(handle, h_b, d_p, 1, &beta);
		beta_norm = 1 / beta;
		status = cublasScopy(handle, h_b, d_p, 1, d_u ,1);
		status = cublasSscal(handle, h_b, &beta_norm, d_u, 1);

		for (int i = 0; i < n*n; i++)
		{
			int shift_x = pos_x[i] - 1;
			int shift_y = pos_y[i] - 1;
		// shift matrix
	
			kernel_shift<<<grid2, block2>>> (shift_x, d_pos_newx, n);
			kernel_shift<<<grid2, block2>>> (shift_y, d_pos_newy, n);
			kernel_new_total<<<grid3, block3>>> (d_pos_newx, d_pos_newy, d_pos_newx_tot, d_pos_newy_tot ,n);
			for (int j = 0; j < h_p; j++)
			{
				kernel_positions_shifted<<<grid4, block4>>> (d_positions,  d_pos_newx_tot, 
								    	 					d_pos_newy_tot, d_layer_positions, pos_shift_x, d_positions_shifted, w_p, j);
			}
			kernel_shift_matrix<<<grid5, block5>>> (d_A_mat, d_A_mat_shifted, d_positions_shifted, w, h);
			// matrix-vector multiplication
			kernel_divide_vector<<<grid6, block6>>> (d_u, d_aux2, i);

			status = cublasSgemv(handle, CUBLAS_OP_T, h, w, &alpha1, d_A_mat_shifted, h, d_aux2, 1, &beta1, d_Vector2, 1);
		}
		beta_minus = -beta;

		status = cublasSaxpy(handle, w, &beta_minus, d_v, 1, d_Vector2, 1);
		status = cublasScopy(handle, w, d_Vector2, 1, d_r ,1);
		status = cublasSnrm2(handle, w, d_r, 1, &alpha);
		status = cublasScopy(handle, w, d_r, 1, d_v ,1);
		alpha_norm = 1 / alpha;
		status = cublasSscal(handle, w, &alpha_norm, d_v, 1);

		rrho = sqrt(rho_bar * rho_bar + beta * beta);
   		c1 = rho_bar/rrho;
   		s1 = beta/rrho;
   		theta = s1 * alpha;
   		rho_bar = -c1 * alpha;
   		phi = c1 * phi_bar;
   		phi_bar = s1 * phi_bar;
   		
   	// update solutions	
   		para1 = phi/rrho;
   		
		status = cublasSaxpy(handle, w, &para1, d_w, 1, d_x, 1);
    	para2 = theta/rrho;
    	kernel_update_vector<<<block7, grid7>>> (d_v, d_w, para2, w);

	}


	cublasGetVector(w, sizeof(float), d_x, 1, Vector, 1);
	// cudaMemcpy(U, d_u, sizeof(float) * w, cudaMemcpyDeviceToHost);
	

	cublasDestroy(handle);
	//1
	cudaFree(d_positions);
	//2
	cudaFree(d_pos_newx);
	cudaFree(d_pos_newy);
	//3
	cudaFree(d_pos_newx_tot);
	cudaFree(d_pos_newy_tot);
	//4
	cudaFree(d_layer_positions);
	cudaFree(pos_shift_x);
	cudaFree(d_positions_shifted);
	//5
	cudaFree(d_A_mat);
	cudaFree(d_A_mat_shifted);
	//6
	cudaFree(d_b);
	cudaFree(d_u);
	cudaFree(d_u_small);
	cudaFree(d_Vector0);
	cudaFree(d_v);
	cudaFree(d_w);
	cudaFree(d_aux1);
	cudaFree(d_Vector1);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_aux2);
	cudaFree(d_Vector2);
	cudaFree(d_x);


/*************************************************TEST**************************************************/
// for (int i = 0; i < n*n; i++)
// 	printf("%d, %d\n", i + 1, positions_shifted[i]);

/*************************************************TEST**************************************************/
	// cudaEventRecord(start, 0);

	// cudaEventRecord(stop, 0);  
	// cudaEventSynchronize(stop);  
	// cudaEventElapsedTime(&time, start, stop);  
	// cudaEventDestroy(start);  
	// cudaEventDestroy(stop);

 	
}