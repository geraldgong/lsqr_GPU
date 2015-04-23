
// Yuxiang Gong 02.04.2015
// Master Thesis, LSQR, shift vector, version 2.0

#include <stdio.h>
#include <stdlib.h>
#include "mex.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cmath>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

__device__ void shift_positions(int shift, int n, int *pos_shift)
{
	int i = 0;
	while(i < n)
	{
		if (i < shift)
			pos_shift[i] = shift + 2 - (i + 1);
		else 
			pos_shift[i] = (i + 1) - shift;
		i++;
	}
}

__global__ void kernel_positions_new(int *d_pos_newx, int *d_pos_newy, int *d_pos_newx_tot, int *d_pos_newy_tot, 
									 int shift_x, int shift_y, int n)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int idx1 = threadIdx.x;
	int idx2 = blockIdx.x;
	if (x >= n*n)
	{
		return;
	}

	shift_positions(shift_x, n, d_pos_newx);
	shift_positions(shift_y, n, d_pos_newy);

	d_pos_newx_tot[x] = d_pos_newx[idx1] + idx2 * n;
	d_pos_newy_tot[x] = (d_pos_newy[idx2] - 1) * n + (idx1 + 1);
}


__global__ void kernel_shift_positions(int *d_positions, int *d_pos_newx_tot, int *d_pos_newy_tot, int *d_xtot_large, 
									   int *d_ytot_large, int *d_positions_temp, int *d_positions_shifted, int w, int h)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = x + y * w;
    if (x >= w || y >= h){
        return;
    }

    for (int i = 0; i < h; i++)
    {
    	d_xtot_large[x + i * w] = d_pos_newx_tot[x] + i * w;
    	d_ytot_large[x + i * w] = d_pos_newy_tot[x] + i * w;
	}

    int temp1 = d_xtot_large[idx];
    d_positions_temp[idx] = d_positions[temp1 - 1];
    int temp2 = d_ytot_large[idx];
    d_positions_shifted[idx] = d_positions_temp[temp2 - 1];

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

__global__ void kernel_shift_vector(float *d_vector, float *d_vector_shifted, int *d_positions_shifted, int w)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;

	if (x >= w)
	{
		return;
	}

	int index = d_positions_shifted[x];
	d_vector_shifted[x] = d_vector[index - 1];
}


__global__ void kernel_divide_vector(float *d_src, float *d_res, int w, int i)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;

	if (x < w)
	{
		d_res[x] = d_src[x + i * w];
	}

}

__global__ void kernel_combine_vector(float *d_src, float *d_res, int w, int i)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;

	if (x < w)
	{
		d_res[x + i * w] = d_src[x];
	}

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

__global__ void kernel_store_vectors(float *d_vector, float *d_matrix, int w, int i)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	if (x >= w)
	{
		return;
	}

	d_matrix[x + i * w] = d_vector[x];
}

// __global__ void kernel_extra_matrix(float *d_R, int *d_positions_shifted, int w)
// {
// 	int x = threadIdx.x + blockDim.x * blockIdx.x;
// 	int y = threadIdx.y + blockDim.y * blockIdx.y;

// 	if (x >= w || y >= w)
// 	{
// 		return;
// 	}

// 	int index = d_positions_shifted[x];
// 	if (y == index)
// 		d_R[y + w * x] = 1;
// 	else
// 		d_R[y + w * x] = 0;
// }

__global__ void kernel_extra_matrix(float *d_R, float lambda, int w)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= w || y >= w)
	{
		return;
	}

	if (x == y)
		d_R[x + y * w] = lambda * lambda;
	else
		d_R[x + y * w] = 0;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
	if (nrhs != 8)
		mexErrMsgTxt("Invaid number of input arguments");

	if (nlhs != 1)
		mexErrMsgTxt("Invalid number of outputs");

	if (!mxIsSingle(prhs[0]))
		mexErrMsgTxt("input matrix data type must be single");

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
	double *lambda_d = (double*)mxGetData(prhs[7]);
	
	int n = (int)(*n_d);
	int nz = (int)(*nz_d);
	int k = (int)(*k_d);
	float lambda = (float)(*lambda_d);

	int w_p = n*n;
	int h_p = nz;
	int h_b = n*n*h;

	// plhs[0] = mxCreateNumericMatrix(w, 1, mxSINGLE_CLASS, mxREAL);
	// float *Vector = (float*)mxGetData(plhs[0]);
	plhs[0] = mxCreateNumericMatrix(w, k, mxSINGLE_CLASS, mxREAL);
	float *Matrix = (float*)mxGetData(plhs[0]);
	

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
	thrust :: device_ptr<int> dev_ptr(d_positions); 
	thrust :: sequence(dev_ptr, dev_ptr + w_p*h_p, 1);
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
	int *d_xtot_large = NULL;
    cudaMalloc(&d_xtot_large, sizeof(int) * w_p * h_p);
    int *d_ytot_large = NULL;
    cudaMalloc(&d_ytot_large, sizeof(int) * w_p * h_p);
    int *d_positions_temp = NULL;
    cudaMalloc(&d_positions_temp, sizeof(int) * w_p * h_p);
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
	float *d_aux0 = NULL;
	cudaMalloc(&d_aux0, sizeof(float) * w);
	float *d_aux0_shifted = NULL;
	cudaMalloc(&d_aux0_shifted, sizeof(float) * w);
	//7
	float *d_Vector1 = NULL;
	cudaMalloc(&d_Vector1, sizeof(float) * h_b);
	float *d_aux1 = NULL;
	cudaMalloc(&d_aux1, sizeof(float) * h);
	float *d_p = NULL;
	cudaMalloc(&d_p, sizeof(float) * h_b);
	//8
	float *d_aux2 = NULL;
	cudaMalloc(&d_aux2, sizeof(float) * w);
	float *d_aux2_shifted = NULL;
	cudaMalloc(&d_aux2_shifted, sizeof(float) * w);
	float *d_Vector2 = NULL;
	cudaMalloc(&d_Vector2, sizeof(float) * w);
	thrust :: device_ptr<float> dev_ptr_v2(d_Vector2); 
    thrust :: fill(dev_ptr_v2, dev_ptr_v2 + w, 0);
	float *d_x = NULL;
	cudaMalloc(&d_x, sizeof(float) * w);
	thrust :: device_ptr<float> dev_ptr_x(d_x); 
    thrust :: fill(dev_ptr_x, dev_ptr_x + w, 0);
	float *d_X = NULL;
	cudaMalloc(&d_X, sizeof(float) * w * k);
	//9 
	float *d_R = NULL;
	cudaMalloc(&d_R, sizeof(float) * w * w);
	float *d_Vector1_R = NULL;
	cudaMalloc(&d_Vector1_R, sizeof(float) * w);
	float *d_aux2_R = NULL;
	cudaMalloc(&d_aux2_R, sizeof(float) * w);
	float *d_aux2_R_shifted = NULL;
	cudaMalloc(&d_aux2_R_shifted, sizeof(float) * w);
	float *d_Vector2_R = NULL;
	cudaMalloc(&d_Vector2_R, sizeof(float) * w);


// Kernels
	dim3 block1 = dim3(16, 16, 1);
	dim3 grid1 = dim3((w_p + block1.x -1)/block1.x, (h_p + block1.y - 1)/block1.y, 1);

	dim3 block3 = dim3(n, 1, 1);
	dim3 grid3 = dim3(n, 1, 1);

	dim3 block4 = dim3(16, 16, 1);
	dim3 grid4 = dim3((w + block4.x - 1)/block4.x, (w + block4.y - 1)/block4.y, 1);

	dim3 block5 = dim3(16, 16, 1);
	dim3 grid5 = dim3((w + block5.x - 1)/block5.x, (h + block5.y - 1)/block5.y, 1);

	dim3 block6 = dim3(16, 16);
	dim3 grid6 = dim3((h + block6.x - 1)/block6.x, (n*n + block6.y - 1)/block6.y);

	dim3 block7 = dim3(16, 1);
	dim3 grid7 = dim3((w + block7.x - 1)/block7.x, 1);

	dim3 block8 = dim3(16, 16);
	dim3 grid8 = dim3((w + block8.x - 1)/block8.x, (k + block8.y - 1)/block8.y);


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

	// float alpha0 = 0.0f;
	float alpha1 = 1.0f;
	// float beta1 = 1.0f;
	float beta0 = 0.0f;


	cublasSnrm2(handle, h_b, d_b, 1, &beta);
   	beta_norm = 1.0 / beta;
   	// normalization
   	cublasScopy(handle, h_b, d_b, 1, d_u ,1);
   	cublasSscal(handle, h_b, &beta_norm, d_u, 1);

// Vector0   	
	for (int i = 0; i < n*n; i++)
	{
		int shift_x = pos_x[i] - 1;
		int shift_y = pos_y[i] - 1;

		kernel_positions_new<<<grid3, block3>>> (d_pos_newx, d_pos_newy, d_pos_newx_tot, d_pos_newy_tot, shift_x, shift_y ,n);
	
		kernel_shift_positions<<<grid1, block1>>> (d_positions, d_pos_newx_tot, d_pos_newy_tot, d_xtot_large, 
												   d_ytot_large, d_positions_temp, d_positions_shifted, w_p, h_p);

		kernel_divide_vector<<<grid6, block6>>> (d_u, d_u_small, h, i);
		cublasSgemv(handle, CUBLAS_OP_T, h, w, &alpha1, d_A_mat, h, d_u_small, 1, &beta0, d_aux0, 1);
		kernel_shift_vector<<<grid7, block7>>> (d_aux0, d_aux0_shifted, d_positions_shifted, w);
		cublasSaxpy(handle, w, &alpha1, d_aux0_shifted, 1, d_Vector0, 1);
	}

	cublasScopy(handle, w, d_Vector0, 1, d_r ,1);
	cublasSnrm2(handle, w, d_r, 1, &alpha);
	
	alpha_norm = 1.0 / alpha;
	phi_bar = beta;
	rho_bar = alpha;
	float alpha_minus = -alpha;
   	float beta_minus;

   	cublasScopy(handle, w, d_r, 1, d_v ,1);
	cublasSscal(handle, w, &alpha_norm, d_v, 1);
	cublasScopy(handle, w, d_v, 1, d_w ,1);

	kernel_extra_matrix<<<grid4, block4>>> (d_R, lambda, w);
	float norm_R = 0.0f;
	float beta_nR = 0.0f; //non-regularization
	
// Vector1
	for(int j = 0; j < k; j++)
	{
		for (int i = 0; i < n*n; i++)
   		{
   			int shift_x = pos_x[i] - 1;
			int shift_y = pos_y[i] - 1;

			kernel_positions_new<<<grid3, block3>>> (d_pos_newx, d_pos_newy, d_pos_newx_tot, d_pos_newy_tot, shift_x, shift_y ,n);
			kernel_shift_positions<<<grid1, block1>>> (d_positions, d_pos_newx_tot, d_pos_newy_tot, d_xtot_large, 
												   d_ytot_large, d_positions_temp, d_positions_shifted, w_p, h_p);
			kernel_shift_matrix<<<grid5, block5>>> (d_A_mat, d_A_mat_shifted, d_positions_shifted, w, h);
			
   			cublasSgemv(handle, CUBLAS_OP_N, h, w, &alpha1, d_A_mat_shifted, h, d_v, 1, &beta0, d_aux1, 1);
   			
   			kernel_combine_vector<<<grid6, block6>>> (d_aux1, d_Vector1, h, i);
    	}
    	// *** regularization part //
    	cublasSgemv(handle, CUBLAS_OP_N, w, w, &alpha1, d_R, w, d_v, 1, &beta0, d_Vector1_R, 1);
    	cublasSnrm2(handle, w, d_Vector1_R, 1, &norm_R);
    	// *** //
		cublasSaxpy(handle, h_b, &alpha_minus, d_u, 1, d_Vector1, 1);
		cublasScopy(handle, h_b, d_Vector1, 1, d_p ,1);
		cublasSnrm2(handle, h_b, d_p, 1, &beta_nR);
		// *** //
		beta = sqrt(pow(beta_nR, 2) + n * n * pow(norm_R, 2));
		// *** //
		beta_norm = 1.0/ beta;
		cublasScopy(handle, h_b, d_p, 1, d_u ,1);
		cublasSscal(handle, h_b, &beta_norm, d_u, 1);
		// *** //
		cublasSscal(handle, w, &beta_norm, d_Vector1_R, 1);
		// *** //

		// Vector2
		for (int i = 0; i < n*n; i++)
		{
			int shift_x = pos_x[i] - 1;
			int shift_y = pos_y[i] - 1;
	
			kernel_positions_new<<<grid3, block3>>> (d_pos_newx, d_pos_newy, d_pos_newx_tot, d_pos_newy_tot, shift_x, shift_y ,n);
			kernel_shift_positions<<<grid1, block1>>> (d_positions, d_pos_newx_tot, d_pos_newy_tot, d_xtot_large, 
												   d_ytot_large, d_positions_temp, d_positions_shifted, w_p, h_p);

			kernel_divide_vector<<<grid6, block6>>> (d_u, d_u_small, h, i);

			cublasSgemv(handle, CUBLAS_OP_T, h, w, &alpha1, d_A_mat, h, d_u_small, 1, &beta0, d_aux2, 1);
			kernel_shift_vector<<<grid7, block7>>> (d_aux2, d_aux2_shifted, d_positions_shifted, w);
			cublasSaxpy(handle, w, &alpha1, d_aux2_shifted, 1, d_Vector2, 1);
			// *** //
			cublasSgemv(handle, CUBLAS_OP_T, w, w, &alpha1, d_R, w, d_Vector1_R, 1, &beta0, d_aux2_R, 1);
			kernel_shift_vector<<<grid7, block7>>> (d_aux2_R, d_aux2_R_shifted, d_positions_shifted, w);
			cublasSaxpy(handle, w, &alpha1, d_aux2_R_shifted, 1, d_Vector2_R, 1);
			// *** //
		}
		// *** //
		cublasSaxpy(handle, w, &alpha1, d_Vector2_R, 1, d_Vector2, 1);
		// *** //
		beta_minus = -beta;

		cublasSaxpy(handle, w, &beta_minus, d_v, 1, d_Vector2, 1);
		cublasScopy(handle, w, d_Vector2, 1, d_r ,1);
		cublasSnrm2(handle, w, d_r, 1, &alpha);
		cublasScopy(handle, w, d_r, 1, d_v ,1);
		alpha_norm = 1 / alpha;
		cublasSscal(handle, w, &alpha_norm, d_v, 1);

		rrho = sqrt(rho_bar * rho_bar + beta * beta);
   		c1 = rho_bar/rrho;
   		s1 = beta/rrho;
   		theta = s1 * alpha;
   		rho_bar = -c1 * alpha;
   		phi = c1 * phi_bar;
   		phi_bar = s1 * phi_bar;
   		
   	// update solutions	
   		para1 = phi/rrho;
   		
		cublasSaxpy(handle, w, &para1, d_w, 1, d_x, 1);
    	para2 = theta/rrho;
    	kernel_update_vector<<<grid7, block7>>> (d_v, d_w, para2, w);
    	kernel_store_vectors<<<grid8, block8>>> (d_x, d_X, w, j);

	}


	// cublasGetVector(w, sizeof(float), d_x, 1, Vector, 1);
	cudaMemcpy(Matrix, d_X, sizeof(float) * w * k, cudaMemcpyDeviceToHost);

	cublasDestroy(handle);
	//1
	cudaFree(d_positions);
	//2
	cudaFree(d_pos_newx);
	cudaFree(d_pos_newy);
	//3
	cudaFree(d_pos_newx_tot);
	cudaFree(d_pos_newy_tot);
	cudaFree(d_xtot_large);
	cudaFree(d_ytot_large);
	cudaFree(d_positions_temp);
	cudaFree(d_positions_shifted);
	//5
	cudaFree(d_A_mat);
	cudaFree(d_A_mat_shifted);
	//6
	cudaFree(d_b);
	cudaFree(d_u);
	cudaFree(d_u_small);
	cudaFree(d_aux0);
	cudaFree(d_aux0_shifted);
	cudaFree(d_Vector0);
	cudaFree(d_v);
	cudaFree(d_w);
	cudaFree(d_aux1);
	cudaFree(d_Vector1);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_aux2);
	cudaFree(d_aux2_shifted);
	cudaFree(d_Vector2);
	cudaFree(d_x);
	cudaFree(d_X);
	//9
	cudaFree(d_R);
	cudaFree(d_Vector1_R);
	cudaFree(d_aux2_R);
	cudaFree(d_aux2_R_shifted);
	cudaFree(d_Vector2_R);



/*************************************************TEST**************************************************/
	// cudaEventRecord(start, 0);

	// cudaEventRecord(stop, 0);  
	// cudaEventSynchronize(stop);  
	// cudaEventElapsedTime(&time, start, stop);  
	// cudaEventDestroy(start);  
	// cudaEventDestroy(stop);

 	return;
}
