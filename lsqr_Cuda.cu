#include "mex.h"
#include "cublas_v2.h"
#include <cmath>


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

	extern __shared__ int sx[];
	extern __shared__ int sy[];

	shift_positions(shift_x, n, d_pos_newx);
	shift_positions(shift_y, n, d_pos_newy);

	sx[idx1] = d_pos_newx[idx1];
	sy[idx1] = d_pos_newy[idx1];
	__syncthreads();

	d_pos_newx_tot[x] = sx[idx1] + idx2 * n;
	d_pos_newy_tot[x] = (sy[idx2] - 1) * n + (idx1 + 1);

}

__global__ void kernel_init_positions(int *d_positions, int w, int h) 
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = x + w * y;	
	if (x >= w || y >= h)
	{
		return;
	}
	
	d_positions[idx] = idx + 1;
}
// get the shifted positions 
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

__global__ void kernel_shift_vector(float *d_vector, float *d_vector_shifted, int *d_positions_shifted, int w, int h)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = x + y * w;

	if (x >= w || y >=h)
	{
		return;
	}

	int index = d_positions_shifted[idx];
	d_vector_shifted[idx] = d_vector[index - 1];
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

__global__ void kernel_normalization(float *d_vector, float norm, float *d_res, int w, int h)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= w || y > h)
	{
		return;
	}
	int i = x + w * y;

	d_res[i] =  d_vector[i] / norm;
}

__global__ void kernel_divide_vector(float *d_src, float *d_res, int w, int i)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;

	if(x < w)
	{
		d_res[x] = d_src[x + i * w];	
	}
}

__global__ void kernel_combine_vector(float *d_src, float *d_res, int w, int i)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;

	if(x < w)
	{
		d_res[x + i * w] = d_src[x];	
	}
}


// __global__ void kernel_sum_vector(float *d_src, float *d_res, int w)
// {
// 	int x = threadIdx.x + blockDim.x * blockIdx.x;

// 	if(x < w)
// 	{
// 		d_res[x] += d_src[x];	
// 	}
// }

__global__ void kernel_datacopy(float *d_src, float *d_res, int w)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;

	if (x >= w)
	{
		return;
	}

	d_res[x] = d_src[x];
}

__global__ void kernel_update_vector(float *a, float *b, float *c, int w)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;

	if (x >= w)
	{
		return;
	}

	a[x] = b[x] + c[x];
}

void test_cublas(float *A_mat, float *b, int *posx, int *posy, float *vector, int n, int nz, int k, int w, int h)
{

	cublasStatus_t status;
	cublasHandle_t handle;
    cublasCreate(&handle);

	int *d_pos_newx = NULL;
	cudaMalloc(&d_pos_newx, sizeof(int) * n);
	int *d_pos_newy = NULL;
	cudaMalloc(&d_pos_newy, sizeof(int) * n);
	int *d_pos_newx_tot = NULL; 
	cudaMalloc(&d_pos_newx_tot, sizeof(int) * n * n);
	int *d_pos_newy_tot = NULL;
	cudaMalloc(&d_pos_newy_tot, sizeof(int) * n * n);

	int *d_positions = NULL;
	cudaMalloc(&d_positions, sizeof(float) * n * n * nz);
	int *d_positions_shifted = NULL;
	cudaMalloc(&d_positions_shifted, sizeof(int) * n * n * nz);

	int *d_xtot_large = NULL;
    cudaMalloc(&d_xtot_large, sizeof(int) * n * n * nz);
    int *d_ytot_large = NULL;
    cudaMalloc(&d_ytot_large, sizeof(int) * n * n * nz);
    int *d_positions_temp = NULL;
    cudaMalloc(&d_positions_temp, sizeof(int) * n * n * nz);

    float *d_A_mat = NULL;
    cudaMalloc(&d_A_mat, sizeof(float) * w * h);
    cudaMemcpy(d_A_mat, A_mat, sizeof(float) * w * h, cudaMemcpyHostToDevice);
    float *d_A_mat_shifted = NULL;
    cudaMalloc(&d_A_mat_shifted, sizeof(float) * w * h);

    float *d_b = NULL;
    cudaMalloc(&d_b, sizeof(float) * n * n * h);
    cudaMemcpy(d_b, b, sizeof(float) * n * n * h, cudaMemcpyHostToDevice);
    float *d_b_norm = NULL;
    cudaMalloc(&d_b_norm, sizeof(float) * n * n * h);
    float *d_u = NULL;
    cudaMalloc(&d_u, sizeof(float) * h);

    float *d_aux0 = NULL;
    cudaMalloc(&d_aux0, sizeof(float) * w);
    float *d_aux0_shifted = NULL;
    cudaMalloc(&d_aux0_shifted, sizeof(float) * w);
    float *d_vector0 = NULL;
    cudaMalloc(&d_vector0, sizeof(float) * w);

    float *d_vector1 = NULL;
    cudaMalloc(&d_vector1, sizeof(float) * n * n * h);
    float *d_aux1 = NULL;
    cudaMalloc(&d_aux1, sizeof(float) * h);
    float *d_v = NULL;
    cudaMalloc(&d_v, sizeof(float) * w);
    float *d_w = NULL;
    cudaMalloc(&d_w, sizeof(float) * w);

    float *d_u_large = NULL;
    cudaMalloc(&d_u_large, sizeof(float) * n * n * h);
    float *d_aux2 = NULL;
    cudaMalloc(&d_aux2, sizeof(float) * w);
    float *d_aux2_shifted = NULL;
    cudaMalloc(&d_aux2_shifted, sizeof(float) * w);
    float *d_vector2 = NULL;
    cudaMalloc(&d_vector2, sizeof(float) * w);

    float *d_w_scale1 = NULL;
    cudaMalloc(&d_w_scale1, sizeof(float) * w);
    float *d_x = NULL;
    cudaMalloc(&d_x, sizeof(float) * w);
    float *d_w_scale2 = NULL;
    cudaMalloc(&d_w_scale2, sizeof(float) * w);


	dim3 block1(16, 16); 
	dim3 grid1((n * n + block1.x -1)/block1.x, (nz + block1.y - 1)/block1.y);

	dim3 block2(16, 16);
	dim3 grid2((h + block2.x - 1)/block2.x, (n * n + block2.y - 1)/block2.y);

	dim3 block3(n, 1);
	dim3 grid3(n, 1);

	dim3 block4(256, 1);
	dim3 grid4((w + block4.x - 1)/block4.x, 1);

	dim3 block5(16, 16);
	dim3 grid5((w + block5.x - 1)/block5.x, (h + block5.y - 1)/block5.y);


	int shift_x, shift_y;
	float alpha = 0.0f;
	float beta = 0.0f;
	float phi_bar = 0.0f;
	float rho_bar = 0.0f;
	float rrho = 0.0f;
	float c1 = 0.0f;
	float s1 = 0.0f;
	float theta = 0.0f;
	float phi = 0.0f;
	
	const float alpha0 = 0.0f;
	const float alpha1 = 1.0f;
	const float beta0 = 0.0f;
	const float beta1 = 1.0f;

	

	// initial positions
	kernel_init_positions<<<grid1, block1>>> (d_positions, n*n, nz);
	// normalization
	status = cublasSnrm2(handle, n*n*h, d_b, 1, &beta);
	if (status != CUBLAS_STATUS_SUCCESS) {
    printf("cublasSnrm2 failed");
   	}
   	cudaThreadSynchronize();

   	kernel_normalization<<<grid2, block2>>> (d_b, beta, d_b_norm, h, n*n);

	// martrix-vector multiplication in each position 
	// Vector0 shift vector
	for (int i = 0; i < n*n; i++)
	{
		shift_x = posx[i] - 1;
		shift_y = posy[i] - 1;
		kernel_positions_new<<<grid3, block3, n*sizeof(int)>>> (d_pos_newx, d_pos_newy,  d_pos_newx_tot, d_pos_newy_tot, shift_x, shift_y, n);
		kernel_shift_positions<<<grid1, block1>>> (d_positions, d_pos_newx_tot, d_pos_newy_tot, d_xtot_large, 
												   d_ytot_large, d_positions_temp, d_positions_shifted, n*n, nz);
		kernel_divide_vector<<<grid2, block2>>> (d_b_norm, d_u, h, i);

		status = cublasSgemv(handle, CUBLAS_OP_T, h, w, &alpha1, d_A_mat, h, d_u, 1, &beta0, d_aux0, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
        	printf("cublasSgemv failed");
    	}
    	cudaThreadSynchronize();

    	kernel_shift_vector<<<grid1, block1>>> (d_aux0, d_aux0_shifted, d_positions_shifted, n*n, nz);
    	// kernel_sum_vector<<<grid4, block4>>> (d_aux0_shifted, d_vector0, w);

    	status = cublasSaxpy(handle, w, &alpha1, d_aux0_shifted, 1, d_vector0, 1);
    	if (status != CUBLAS_STATUS_SUCCESS) {
        	printf("cublasSaxpy failed");
    	}
    	// cudaThreadSynchronize();

	}

	status = cublasSnrm2(handle, w, d_vector0, 1, &alpha);
	if (status != CUBLAS_STATUS_SUCCESS) {
    printf("cublasSnrm2 failed");
   	}
   	cudaThreadSynchronize();

   	kernel_normalization<<<grid1, block1>>> (d_vector0, alpha, d_v, n*n, nz);
   	phi_bar = beta;
   	rho_bar = alpha;
   	float alpha_minus = -alpha;

   	status = cublasScopy(handle, w, d_v, 1, d_w, 1);
   	if (status != CUBLAS_STATUS_SUCCESS) {
        	printf("cublasScopy failed");
   	}
    cudaThreadSynchronize();

   	// kernel_datacopy<<<grid4, block4>>> (d_v, d_w, w);
   	// Vector1 shift matrix
   	for(int j = 0; j < k; j++)
   	{
   		for (int i = 0; i < n*n; i++)
		{
			shift_x = posx[i] - 1;
			shift_y = posy[i] - 1;
			kernel_positions_new<<<n, n, n*sizeof(int)>>> (d_pos_newx, d_pos_newy,  d_pos_newx_tot, d_pos_newy_tot, shift_x, shift_y, n);
			kernel_shift_positions<<<grid1, block1>>> (d_positions, d_pos_newx_tot, d_pos_newy_tot, d_xtot_large, 
												   d_ytot_large, d_positions_temp, d_positions_shifted, n*n, nz);
			kernel_shift_matrix<<<grid5, block5>>> (d_A_mat, d_A_mat_shifted, d_positions_shifted, w, h);

   			status = cublasSgemv(handle, CUBLAS_OP_N, h, w, &alpha1, d_A_mat_shifted, h, d_v, 1, &beta0, d_aux1, 1);
			if (status != CUBLAS_STATUS_SUCCESS) {
        	printf("cublasSgemv failed");
    		}
    		cudaThreadSynchronize();

    		kernel_combine_vector<<<grid2, block2>>> (d_aux1, d_vector1, h, i);
		}
		// p = vector1 - alpha * u;
		status = cublasSaxpy(handle, h*n*n, &alpha_minus, d_b_norm, 1, d_vector1, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
        	printf("cublasSaxpy failed");
    	}
    	cudaThreadSynchronize();

		status = cublasSnrm2(handle, h*n*n, d_vector1, 1, &beta);
		if (status != CUBLAS_STATUS_SUCCESS) {
    	printf("cublasSnrm2 failed");
   		}
   		cudaThreadSynchronize();

   		kernel_normalization<<<grid2, block2>>> (d_vector1, beta, d_u_large, h, n*n);
   		float beta_minus = -beta;

   	// Vector2 shift vector
   		for (int i = 0; i < n*n; i++)
		{
			shift_x = posx[i] - 1;
			shift_y = posy[i] - 1;
			kernel_positions_new<<<n, n, n*sizeof(int)>>> (d_pos_newx, d_pos_newy,  d_pos_newx_tot, d_pos_newy_tot, shift_x, shift_y, n);
			kernel_shift_positions<<<grid1, block1>>> (d_positions, d_pos_newx_tot, d_pos_newy_tot, d_xtot_large, 
												   d_ytot_large, d_positions_temp, d_positions_shifted, n*n, nz);
			kernel_divide_vector<<<grid2, block2>>> (d_u_large, d_u, h, i);

			status = cublasSgemv(handle, CUBLAS_OP_T, h, w, &alpha1, d_A_mat, h, d_u, 1, &beta0, d_aux2, 1);
			if (status != CUBLAS_STATUS_SUCCESS) {
        		printf("cublasSgemv failed");
    		}
    		cudaThreadSynchronize();

    		kernel_shift_vector<<<grid1, block1>>> (d_aux2, d_aux2_shifted, d_positions_shifted, n*n, nz);
    		kernel_sum_vector<<<grid4, block4>>> (d_aux2_shifted, d_vector2, w);
		}

		status = cublasSaxpy(handle, w, &beta_minus, d_v, 1, d_vector2, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
        	printf("cublasSaxpy failed");
    	}
    	cudaThreadSynchronize();

    	status = cublasSnrm2(handle, w, d_vector2, 1, &alpha);
		if (status != CUBLAS_STATUS_SUCCESS) {
    		printf("cublasSnrm2 failed");
   		}
   		cudaThreadSynchronize();

   		kernel_normalization<<<grid1, block1>>>(d_vector2, alpha, d_v, n*n, nz);

   		rrho = sqrt(pow(rho_bar, 2.0f) + pow(beta, 2.0f));
   		c1 = rho_bar/rrho;
   		s1 = beta/rrho;
   		theta = s1 * alpha;
   		rho_bar = -c1 * alpha;
   		phi = c1 * phi_bar;
   		phi_bar = s1 * phi_bar;
   		float para1 = phi/rrho;

   		status = cublasScopy(handle, w, d_w, 1, d_w_scale1, 1);
   		if (status != CUBLAS_STATUS_SUCCESS) {
        	printf("cublasScopy failed");
   		}
    	cudaThreadSynchronize();

   		status = cublasSscal(handle, w, &para1, d_w_scale1, 1);
   		if (status != CUBLAS_STATUS_SUCCESS) {
        	printf("cublasSscopy failed");
   		}
    	cudaThreadSynchronize();

    	kernel_sum_vector<<<grid4, block4>>>(d_w_scale1, d_x, w);
    	float para2 = -theta/rrho;

    	status = cublasScopy(handle, w, d_w, 1, d_w_scale2, 1);
   		if (status != CUBLAS_STATUS_SUCCESS) {
        	printf("cublasScopy failed");
   		}
    	cudaThreadSynchronize();

    	status = cublasSscal(handle, w, &para2, d_w_scale2, 1);
   		if (status != CUBLAS_STATUS_SUCCESS) {
        	printf("cublasSscopy failed");
   		}
    	cudaThreadSynchronize();

    	kernel_update_vector<<<grid4, block4>>> (d_w, d_v, d_w_scale2, w);
   	}


	cudaMemcpy(vector, d_vector0, sizeof(float) * w, cudaMemcpyDeviceToHost);

	cublasDestroy(handle);
	cudaFree(d_pos_newx);
	cudaFree(d_pos_newy);
	cudaFree(d_pos_newx_tot);
	cudaFree(d_pos_newy_tot);
	cudaFree(d_positions);
	cudaFree(d_positions_shifted);
	cudaFree(d_positions_temp);
	cudaFree(d_xtot_large);
	cudaFree(d_ytot_large);
	cudaFree(d_A_mat);
	cudaFree(d_A_mat_shifted);
	cudaFree(d_b);
	cudaFree(d_b_norm);
	cudaFree(d_u);
	cudaFree(d_aux0);
	cudaFree(d_vector0);
	cudaFree(d_v);
	cudaFree(d_w);
	cudaFree(d_aux1);
	cudaFree(d_vector1);
	cudaFree(d_u_large);
	cudaFree(d_aux2);
	cudaFree(d_aux2_shifted);
	cudaFree(d_vector2);
	cudaFree(d_x);
	cudaFree(d_w_scale1);
	cudaFree(d_w_scale2);

}