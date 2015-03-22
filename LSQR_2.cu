#include "LSQR_2.h"
#include "mex.h"
#include "cublas_v2.h"

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

__global__ void kernel_combine_vector(float *d_v, float *d_v_tot, int i)
{
	int x = threadIdx.x;
	d_v_tot[x + i * blockDim.x] = d_v[x];
}

void LSQR_2(float *A_mat, float *v, float *Vector, int *pos_x, int *pos_y, int n, int nz, int w, int h)
{
	int w_p = n*n;
	int h_p = nz;
	int h_b = n*n*h;

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
	//7
	float *d_v = NULL;
	cudaMalloc(&d_v, sizeof(float) * w);
	cublasSetVector(w, sizeof(float), v, 1, d_v, 1);
	float *d_Vector1 = NULL;
	cudaMalloc(&d_Vector1, sizeof(float) * h_b);
	// cudaMemcpy(d_Vector1, Vector, sizeof(float) * h_b, cudaMemcpyHostToDevice);
	// cublasSetVector(h_b, sizeof(float), Vector, 1, d_Vector1, 1);
	float *d_aux1 = NULL;
	cudaMalloc(&d_aux1, sizeof(float) * h);

	// Kernels
	dim3 block1 = dim3(32, 8, 1);
	dim3 grid1 = dim3((w_p + block1.x -1)/block1.x, (h_p + block1.y - 1)/block1.y, 1);

	dim3 block2 = dim3(32, 8, 1);
	dim3 grid2 = dim3((n + block1.x -1)/block1.x, 1, 1);

	dim3 block3 = dim3(n, 1, 1);
	dim3 grid3 = dim3(n, 1, 1);

	dim3 block4 = dim3(32, 8, 1); // test as perfect choice? don't know why...
	dim3 grid4 = dim3((w_p + block4.x - 1)/block4.x, (h_p + block4.y - 1)/block4.y, 1);

	dim3 block5 = dim3(32, 8, 1);
	dim3 grid5 = dim3((w + block5.x - 1)/block5.x, (h + block5.y - 1)/block5.y, 1);

	dim3 block6 = dim3(h, 1, 1);
	dim3 grid6 = dim3(n*n, 1, 1);
	// kernel for vectors
	dim3 block7 = dim3(32, 8, 1);
	dim3 grid7 = dim3((n*n + block7.x - 1)/block7.x, (h + block7.y - 1)/block7.y, 1);

	float alpha_c = 1;
	float beta_c = 1;
	float beta_c1 = 0;
	float beta, alpha;

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

   		status = cublasSgemv(handle, CUBLAS_OP_N, h, w, &alpha_c, d_A_mat_shifted, h, d_v, 1, &beta_c1, d_aux1, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
       	printf("cublasSgemv failed");
   		}
   		cudaThreadSynchronize();
   		kernel_combine_vector<<<grid6, block6>>> (d_aux1, d_Vector1, i);
    }

   	cudaMemcpy(Vector, d_Vector1, sizeof(float) * h_b, cudaMemcpyDeviceToHost);



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
	//7
	cudaFree(d_v);
	cudaFree(d_Vector1);
	cudaFree(d_aux1);

	cublasDestroy(handle);

}