#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
//using namespace std;
#include "tron.h"

#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern double dnrm2_(int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern int dscal_(int *, double *, double *, int *);

#ifdef __cplusplus
}
#endif

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: library error reporting.
const char* cublasGetErrorString(cublasStatus_t error);

// CUDA: use 512 threads per block
const int CUDA_NUM_THREADS = 512;
const int SUBMEMALL_NUM_THREADS = 32;
const int SUBMEMA_NUM_THREADS = 128;
const int SUBMEMB_NUM_THREADS = 64;
const int SUBMEMC_NUM_THREADS = 32;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// CUDA: number of blocks for threads.
inline int GET_BLOCKS_VAR(const int N, const int M) {
  return (N + M - 1) / M;
}

__global__ void sub_mem_copy_all(double* X, double* X_sub, double* C, double* C_sub, double* z, double* z_sub, double* Y,
				 // thrust::device_vector<int>& Id, 
				 int* Id, 
				 int sizeI, int n, int m)
{
  #ifdef GRIDSTRIDELOOP
  // use grid-stride loop
  CUDA_KERNEL_LOOP(r, n) {
    int Idr = Id[r];
    double c = C[Idr];
    // note that Idr >= r by construction
    C_sub[r] = c;
    z_sub[r] = c * Y[Idr] * (z[Idr] -1);
    for(int k = 0; k < m; k++)
      X_sub[r*m + k] = X[Idr*m + k];
  }
  #else
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if within image bounds.
  if (r >= sizeI)
    return;
  int Idr = Id[r];
  double c = C[Idr];
  // note that Idr >= r by construction
  C_sub[r] = c;
  z_sub[r] = c * Y[Idr] * (z[Idr] -1);
  #pragma unroll
  for(int k = 0; k < m; k++)
    X_sub[r*m + k] = X[Idr*m + k];
  #endif
}

__global__ void sub_mem_copyA(double* C, double* C_sub,
			      int* Id, 
			      int sizeI, int n, int m)
{
  #ifdef GRIDSTRIDELOOP
  // use grid-stride loop
  CUDA_KERNEL_LOOP(r, n) {
    int Idr = Id[r];
    // note that Idr >= r by construction
    C_sub[r] = C[Idr];
  }
  #else
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if within image bounds.
  if (r >= sizeI)
    return;
  int Idr = Id[r];
  // note that Idr >= r by construction
  C_sub[r] = C[Idr];
  #endif
}

__global__ void sub_mem_copyB(double* z, double* z_sub, double* Y,
			      double* C, int* Id, 
			      int sizeI, int n, int m)
{
  #ifdef GRIDSTRIDELOOP
  // use grid-stride loop
  CUDA_KERNEL_LOOP(r, n) {
    int Idr = Id[r];
    // note that Idr >= r by construction
    z_sub[r] = C[Idr] * Y[Idr] * (z[Idr] -1);
  }
  #else
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if within image bounds.
  if (r >= sizeI)
    return;
  int Idr = Id[r];
  // note that Idr >= r by construction
  z_sub[r] = C[Idr] * Y[Idr] * (z[Idr] -1);
  #endif
}

__global__ void sub_mem_copyC(double* X, double* X_sub,
			      int* Id, 
			      int sizeI, int n, int m)
{
  #ifdef GRIDSTRIDELOOP
  // use grid-stride loop
  CUDA_KERNEL_LOOP(r, n) {
    int Idr = Id[r];
    for(int k = 0; k < m; k++)
      X_sub[r*m + k] = X[Idr*m + k];
  }
  #else
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if within image bounds.
  if (r >= sizeI)
    return;
  int Idr = Id[r];
  #pragma unroll
  for(int k = 0; k < m; k++)
    X_sub[r*m + k] = X[Idr*m + k];
  #endif
}

__global__ void sub_mem_copy2d(double* X, double* X_sub, int* Id, int sizeI, int n, int m)
{
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  const int i = r * m + c; // 1D flat index

  // Check if within image bounds.
  if ((c >= m) || (r >= n))
    return;
  X_sub[i] = X[Id[r] * m + c];
}

__global__ void dgemv_simple(double* A, double* x, double* y, double* C, int n, int m)
{
  // calculate y = C.* A*x, where .* is element-by-element matrix multiplication
  #ifdef GRIDSTRIDELOOP
  CUDA_KERNEL_LOOP(row, n) {
    double sum = 0.0;
    for (int k = 0; k < m; k++) {
      sum += A[row*m+k] * x[k];
    }
    y[row] = C[row] * sum;
  }
  #else
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row >= n)
    return;

  double sum = 0.0;
  #pragma unroll
  for (int k = 0; k < m; k++) {
    sum += A[row*m+k] * x[k];
  }
  y[row] = C[row] * sum;
  #endif
}

__global__ void dgemv_sub_grad(double* A, double* x, double* y, int* Id, int sizeI, int n, int m)
{
  // calculate y = C.* A*x, where .* is element-by-element matrix multiplication
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row >= n)
    return;
  int rowSubId = Id[row];
  double sum = 0.0;
  for (int k = 0; k < m; k++) {
    sum += A[rowSubId*m+k] * x[k];
  }
  y[row] = sum;
}

static void default_print(const char *buf)
{
	fputs(buf,stdout);
	fflush(stdout);
}

static double uTMv(int n, double *u, double *M, double *v)
{
	const int m = n-4;
	double res = 0;
	int i;
	for (i=0; i<m; i+=5)
		res += u[i]*M[i]*v[i]+u[i+1]*M[i+1]*v[i+1]+u[i+2]*M[i+2]*v[i+2]+
			u[i+3]*M[i+3]*v[i+3]+u[i+4]*M[i+4]*v[i+4];
	for (; i<n; i++)
		res += u[i]*M[i]*v[i];
	return res;
}

void TRON::info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*tron_print_string)(buf);
}

TRON::TRON(const function *fun_obj, double eps, double eps_cg, int max_iter)
{
	this->fun_obj=const_cast<function *>(fun_obj);
	this->eps=eps;
	this->eps_cg=eps_cg;
	this->max_iter=max_iter;
	tron_print_string = default_print;
}

TRON::~TRON()
{
}

void TRON::tron(double *w)
{
	// Start next transfer w
        fun_obj->transfer_w(w);
	// Parameters for updating the iterates.
	double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

	// Parameters for updating the trust region size delta.
	double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

	int n = fun_obj->get_nr_variable();
	int i, cg_iter;
	double delta=0, sMnorm, one=1.0;
	double alpha, f, fnew, prered, actred, gs;
	int search = 1, iter = 1, inc = 1;
	double *s = new double[n];
	double *r = new double[n];
	double *g = new double[n];

	double *dev_M;
	double *dev_s;
	double *dev_r;
	// checkCudaErrors(cudaMalloc((void** )&dev_M, n * sizeof(double)));
	// checkCudaErrors(cudaMalloc((void** )&dev_s, n * sizeof(double)));
	// checkCudaErrors(cudaMalloc((void** )&dev_r, n * sizeof(double)));

	const double alpha_pcg = 0.01;
	double *M = new double[n];

	// // calculate gradient norm at w=0 for stopping condition.
	// double *w0 = new double[n];
	// for (i=0; i<n; i++)
	// 	w0[i] = 0;
	// fun_obj->transfer_w(w0);

	// calculate gradient norm at w=0 for stopping condition.
	fun_obj->fun0(g);
	// fun_obj->fun(w0, g);
	// fun_obj->grad(w0, g);
	// Sync gradient stream
	// fun_obj->grad_sync(w0, g);



	double gnorm0 = dnrm2_(&n, g, &inc);
	// delete [] w0;

	fun_obj->sync_deStreams();
	fun_obj->sync_csrStreams();

	f = fun_obj->fun(w, g);
	fun_obj->grad(w, g);
	fun_obj->grad_sync(w, g);
	double gnorm = dnrm2_(&n, g, &inc);

	if (gnorm <= eps*gnorm0)
		search = 0;

	fun_obj->get_diag_preconditioner(M);
	for(i=0; i<n; i++)
		M[i] = (1-alpha_pcg) + alpha_pcg*M[i];
	delta = sqrt(uTMv(n, g, M, g));

	double *w_new = new double[n];
	bool reach_boundary;
	bool delta_adjusted = false;
	while (iter <= max_iter && search)
	{
	  // cg_iter = trpcg(delta, g, M, s, r, &reach_boundary, 
	  // 		  dev_M, dev_s, dev_r);
	  cg_iter = trpcg(delta, g, M, s, r, &reach_boundary); 
			  // dev_M, dev_s, dev_r);

		memcpy(w_new, w, sizeof(double)*n);
		daxpy_(&n, &one, s, &inc, w_new, &inc);

		// Start next transfer w
		fun_obj->transfer_w(w_new);

		gs = ddot_(&n, g, &inc, s, &inc);
		prered = -0.5*(gs-ddot_(&n, s, &inc, r, &inc));
		fnew = fun_obj->fun(w_new, g);

		// Compute the actual reduction.
		actred = f - fnew;

		// On the first iteration, adjust the initial step bound.
		sMnorm = sqrt(uTMv(n, s, M, s));
		if (iter == 1 && !delta_adjusted)
		{
			delta = min(delta, sMnorm);
			delta_adjusted = true;
		}

		// Compute prediction alpha*sMnorm of the step.
		if (fnew - f - gs <= 0)
			alpha = sigma3;
		else
			alpha = max(sigma1, -0.5*(gs/(fnew - f - gs)));

		// Update the trust region bound according to the ratio of actual to predicted reduction.
		if (actred < eta0*prered)
			delta = min(alpha*sMnorm, sigma2*delta);
		else if (actred < eta1*prered)
			delta = max(sigma1*delta, min(alpha*sMnorm, sigma2*delta));
		else if (actred < eta2*prered)
			delta = max(sigma1*delta, min(alpha*sMnorm, sigma3*delta));
		else
		{
			if (reach_boundary)
				delta = sigma3*delta;
			else
				delta = max(delta, min(alpha*sMnorm, sigma3*delta));
		}

		info("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d\n", iter, actred, prered, delta, f, gnorm, cg_iter);

		if (actred > eta0*prered)
		{
			iter++;
			memcpy(w, w_new, sizeof(double)*n);
			f = fnew;
			fun_obj->grad(w, g);
			fun_obj->get_diag_preconditioner(M);
			for(i=0; i<n; i++)
				M[i] = (1-alpha_pcg) + alpha_pcg*M[i];
			
			fun_obj->grad_sync(w, g);
			gnorm = dnrm2_(&n, g, &inc);
			if (gnorm <= eps*gnorm0)
				break;
		}
		if (f < -1.0e+32)
		{
			info("WARNING: f < -1.0e+32\n");
			break;
		}
		if (prered <= 0)
		{
			info("WARNING: prered <= 0\n");
			break;
		}
		if (fabs(actred) <= 1.0e-12*fabs(f) &&
		    fabs(prered) <= 1.0e-12*fabs(f))
		{
			info("WARNING: actred and prered too small\n");
			break;
		}
	}

	delete[] g;
	delete[] r;
	delete[] w_new;
	delete[] s;
	delete[] M;
}

// int TRON::trpcg(double delta, double *g, double *M, double *s, double *r, bool *reach_boundary)
// {
// 	int i, inc = 1;
// 	int n = fun_obj->get_nr_variable();
// 	double one = 1;
// 	double zTr, znewTrnew, alpha, beta, cgtol;
// 	double *d = new double[n];
// 	double *Hd = new double[n];
// 	double *z = new double[n];
// 	double *dev_d;
// 	double *dev_Hd;
// 	double *dev_z;

// 	*reach_boundary = false;
// 	for (i=0; i<n; i++)
// 	{
// 		s[i] = 0;
// 		r[i] = -g[i];
// 		z[i] = r[i] / M[i];
// 		d[i] = z[i];
// 	}

// 	zTr = ddot_(&n, z, &inc, r, &inc);
// 	cgtol = eps_cg*sqrt(zTr);
// 	int cg_iter = 0;
// 	int max_cg_iter = max(n, 5);
// 	double dsq = delta*delta;

// 	while (cg_iter < max_cg_iter)
// 	{
// 		if (sqrt(zTr) <= cgtol)
// 			break;
// 		cg_iter++;
// 		fun_obj->Hv(d, Hd);

// 		alpha = zTr/ddot_(&n, d, &inc, Hd, &inc);
// 		daxpy_(&n, &alpha, d, &inc, s, &inc);

// 		double sMnorm = sqrt(uTMv(n, s, M, s));
// 		if (sMnorm > delta)
// 		{
// 			info("cg reaches trust region boundary\n");
// 			*reach_boundary = true;
// 			alpha = -alpha;
// 			daxpy_(&n, &alpha, d, &inc, s, &inc);

// 			double sTMd = uTMv(n, s, M, d);
// 			double sTMs = uTMv(n, s, M, s);
// 			double dTMd = uTMv(n, d, M, d);
// 			double rad = sqrt(sTMd*sTMd + dTMd*(dsq-sTMs));
// 			if (sTMd >= 0)
// 				alpha = (dsq - sTMs)/(sTMd + rad);
// 			else
// 				alpha = (rad - sTMd)/dTMd;
// 			daxpy_(&n, &alpha, d, &inc, s, &inc);
// 			alpha = -alpha;
// 			daxpy_(&n, &alpha, Hd, &inc, r, &inc);
// 			break;
// 		}
// 		alpha = -alpha;
// 		daxpy_(&n, &alpha, Hd, &inc, r, &inc);

// 		for (i=0; i<n; i++)
// 			z[i] = r[i] / M[i];
// 		znewTrnew = ddot_(&n, z, &inc, r, &inc);
// 		beta = znewTrnew/zTr;
// 		dscal_(&n, &beta, d, &inc);
// 		daxpy_(&n, &one, z, &inc, d, &inc);
// 		zTr = znewTrnew;
// 	}

// 	if (cg_iter == max_cg_iter)
// 		info("WARNING: reaching maximal number of CG steps\n");
	
// 	delete[] d;
// 	delete[] Hd;
// 	delete[] z;

// 	return(cg_iter);
// }

__global__ void init_vectors(double* s, double* r, double* z, double* d, int n, 
			     double* g, double* M)
{
  // use grid-stride loop
  CUDA_KERNEL_LOOP(i, n) {
    s[i] = 0;
    r[i] = -g[i];
    z[i] = r[i] / M[i];
    d[i] = z[i];
  }
}

__global__ void dev_daxpy(int n, double alpha, double* x, double* y)
{
  // use grid-stride loop
  CUDA_KERNEL_LOOP(r, n) {
    y[r] += alpha * x[r];
  }
}

__global__ void dev_uTMv(int n, double *u, double *M, double *v, double* acc)
{
  // use grid-stride loop
  CUDA_KERNEL_LOOP(r, n) {
    acc[r] = u[r] * M[r] * v[r];
  }
}

__global__ void dev_div(int n, double *z, double *r, double *M)
{
  // use grid-stride loop
  CUDA_KERNEL_LOOP(i, n) {
    z[i] = r[i] / M[i];
  }
}

__global__ void dev_dscal_daxpy(int n, double beta, double* z, double* d)
{
  // use grid-stride loop
  CUDA_KERNEL_LOOP(i, n) {
    d[i] = z[i] + beta * d[i];
  }
}

int TRON::trpcg(double delta, double *g, double *M, double *s, double *r, bool *reach_boundary)
// , 
// 		double* dev_M, double* dev_s, double* dev_r)
{
	int i, inc = 1;
	int n = fun_obj->get_nr_variable();
	double one = 1;
	double zTr, znewTrnew, alpha, beta, cgtol, negAlpha;
	double *d = new double[n];
	double *Hd = new double[n];
	double *z = new double[n];
	double *dev_d, *dev_Hd, *dev_z, *dev_g, *dev_M, *dev_s, *dev_r, *acc1;
	cudaStream_t *stream;

	double sMnorm = 0;
	double sTMd = 0;
	double sTMs = 0;
	double dTMd = 0;


	stream = new cudaStream_t;
	checkCudaErrors(cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking));

	checkCudaErrors(cudaMalloc((void** )&dev_d, n * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** )&dev_Hd, n * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** )&dev_z, n * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** )&acc1, n * sizeof(double)));
	// 
	checkCudaErrors(cudaMalloc((void** )&dev_g, n * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** )&dev_M, n * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** )&dev_s, n * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** )&dev_r, n * sizeof(double)));

	checkCudaErrors(cudaMemcpyAsync(dev_g, g, n * sizeof(double), cudaMemcpyHostToDevice, *stream));
	checkCudaErrors(cudaMemcpyAsync(dev_M, M, n * sizeof(double), cudaMemcpyHostToDevice, *stream));
	checkCudaErrors(cudaMemcpyAsync(dev_s, s, n * sizeof(double), cudaMemcpyHostToDevice, *stream));
	checkCudaErrors(cudaMemcpyAsync(dev_r, r, n * sizeof(double), cudaMemcpyHostToDevice, *stream));

	*reach_boundary = false;
	// for (i=0; i<n; i++)
	// {
	// 	s[i] = 0;
	// 	r[i] = -g[i];
	// 	z[i] = r[i] / M[i];
	// 	d[i] = z[i];
	// }

	init_vectors <<< GET_BLOCKS_VAR(n, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, *stream >>> (dev_s, dev_r, dev_z, dev_d, n, dev_g, dev_M);

	// zTr = ddot_(&n, z, &inc, r, &inc);
	zTr = thrust::inner_product(thrust::cuda::par.on(*stream), dev_z, dev_z + n, dev_r, (double) 0);
	cgtol = eps_cg*sqrt(zTr);
	int cg_iter = 0;
	int max_cg_iter = max(n, 5);
	double dsq = delta*delta;

	checkCudaErrors(cudaMemcpyAsync(s, dev_s, n * sizeof(double), cudaMemcpyDeviceToHost, *stream));
	checkCudaErrors(cudaMemcpyAsync(r, dev_r, n * sizeof(double), cudaMemcpyDeviceToHost, *stream));
	checkCudaErrors(cudaMemcpyAsync(z, dev_z, n * sizeof(double), cudaMemcpyDeviceToHost, *stream));
	checkCudaErrors(cudaMemcpyAsync(d, dev_d, n * sizeof(double), cudaMemcpyDeviceToHost, *stream));
	checkCudaErrors(cudaStreamSynchronize(*stream));

	while (cg_iter < max_cg_iter)
	{
		if (sqrt(zTr) <= cgtol)
			break;
		cg_iter++;
		fun_obj->Hv(d, Hd);

		// alpha = zTr/ddot_(&n, d, &inc, Hd, &inc);
		// daxpy_(&n, &alpha, d, &inc, s, &inc);
		// double sMnorm = sqrt(uTMv(n, s, M, s));

		checkCudaErrors(cudaMemcpyAsync(dev_d, d, n * sizeof(double), cudaMemcpyHostToDevice, *stream));
		checkCudaErrors(cudaMemcpyAsync(dev_Hd, Hd, n * sizeof(double), cudaMemcpyHostToDevice, *stream));
		checkCudaErrors(cudaMemcpyAsync(dev_s, s, n * sizeof(double), cudaMemcpyHostToDevice, *stream));

		alpha = zTr / thrust::inner_product(thrust::cuda::par.on(*stream), dev_d, dev_d + n, dev_Hd, (double) 0);
		dev_daxpy <<< GET_BLOCKS_VAR(n, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, *stream >>> 
		  (n, alpha, dev_d, dev_s);

		dev_uTMv <<< GET_BLOCKS_VAR(n, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, *stream >>>
		  (n, dev_s, dev_M, dev_s, acc1);
		sMnorm = sqrt(thrust::reduce(thrust::cuda::par.on(*stream), acc1, acc1 + n, (double) 0,  thrust::plus<double>()));

		// checkCudaErrors(cudaMemcpyAsync(s, dev_s, n * sizeof(double), cudaMemcpyDeviceToHost, *stream));
		checkCudaErrors(cudaStreamSynchronize(*stream));
		
		// sMnorm = sqrt(sMnorm);
		// sMnorm = sqrt(uTMv(n, s, M, s));
		if (sMnorm > delta)
		{
			info("cg reaches trust region boundary\n");
			*reach_boundary = true;
			alpha = -alpha;
			// daxpy_(&n, &alpha, d, &inc, s, &inc);
			// double sTMd = uTMv(n, s, M, d);
			// double sTMs = uTMv(n, s, M, s);
			// double dTMd = uTMv(n, d, M, d);

			dev_daxpy <<< GET_BLOCKS_VAR(n, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, *stream >>> 
			  (n, alpha, dev_d, dev_s);
			checkCudaErrors(cudaMemcpyAsync(s, dev_s, n * sizeof(double), cudaMemcpyDeviceToHost, *stream));

			dev_uTMv <<< GET_BLOCKS_VAR(n, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, *stream >>>
			  (n, dev_d, dev_M, dev_d, acc1);
			dTMd = thrust::reduce(thrust::cuda::par.on(*stream), acc1, acc1 + n, (double) 0,  thrust::plus<double>());

			dev_uTMv <<< GET_BLOCKS_VAR(n, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, *stream >>>
			  (n, dev_s, dev_M, dev_s, acc1);
			sTMs = thrust::reduce(thrust::cuda::par.on(*stream), acc1, acc1 + n, (double) 0,  thrust::plus<double>());

			dev_uTMv <<< GET_BLOCKS_VAR(n, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, *stream >>>
			  (n, dev_s, dev_M, dev_d, acc1);
			sTMd = thrust::reduce(thrust::cuda::par.on(*stream), acc1, acc1 + n, (double) 0,  thrust::plus<double>());

			// checkCudaErrors(cudaStreamSynchronize(*stream));

			double rad = sqrt(sTMd*sTMd + dTMd*(dsq-sTMs));

			if (sTMd >= 0){
				alpha = (dsq - sTMs)/(sTMd + rad);
				negAlpha = -alpha;
			} else {
				alpha = (rad - sTMd)/dTMd;
				negAlpha = -alpha;
			}

			// daxpy_(&n, &alpha, d, &inc, s, &inc);
			// alpha = -alpha;
			// daxpy_(&n, &alpha, Hd, &inc, r, &inc);

			dev_daxpy <<< GET_BLOCKS_VAR(n, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, *stream >>> 
			  (n, alpha, dev_d, dev_s);
			checkCudaErrors(cudaMemcpyAsync(s, dev_s, n * sizeof(double), cudaMemcpyDeviceToHost, *stream));
			dev_daxpy <<< GET_BLOCKS_VAR(n, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, *stream >>> 
			  (n, negAlpha, dev_Hd, dev_r);
			checkCudaErrors(cudaMemcpyAsync(r, dev_r, n * sizeof(double), cudaMemcpyDeviceToHost, *stream));

			alpha = -alpha;

			checkCudaErrors(cudaStreamSynchronize(*stream));
			// checkCudaErrors(cudaMemcpyAsync(dev_s, s, n * sizeof(double), cudaMemcpyHostToDevice, *stream));
			// checkCudaErrors(cudaStreamSynchronize(*stream));
			break;
		} else {
		  checkCudaErrors(cudaMemcpyAsync(s, dev_s, n * sizeof(double), cudaMemcpyDeviceToHost, *stream));
		  checkCudaErrors(cudaStreamSynchronize(*stream));
		}
		alpha = -alpha;
		daxpy_(&n, &alpha, Hd, &inc, r, &inc);

		for (i=0; i<n; i++)
			z[i] = r[i] / M[i];
		znewTrnew = ddot_(&n, z, &inc, r, &inc);
		beta = znewTrnew/zTr;
		dscal_(&n, &beta, d, &inc);
		daxpy_(&n, &one, z, &inc, d, &inc);
		zTr = znewTrnew;
	}

	if (cg_iter == max_cg_iter)
		info("WARNING: reaching maximal number of CG steps\n");
	
 	checkCudaErrors(cudaFree(dev_d));
	checkCudaErrors(cudaFree(dev_Hd));
	checkCudaErrors(cudaFree(dev_z));
 	checkCudaErrors(cudaFree(acc1));
 	checkCudaErrors(cudaFree(dev_g));
 	checkCudaErrors(cudaFree(dev_M));
 	checkCudaErrors(cudaFree(dev_s));
 	checkCudaErrors(cudaFree(dev_r));
	checkCudaErrors(cudaStreamDestroy(*stream));
	delete stream;

	delete[] d;
	delete[] Hd;
	delete[] z;

	return(cg_iter);
}

double TRON::norm_inf(int n, double *x)
{
	double dmax = fabs(x[0]);
	for (int i=1; i<n; i++)
		if (fabs(x[i]) >= dmax)
			dmax = fabs(x[i]);
	return(dmax);
}

void TRON::set_print_string(void (*print_string) (const char *buf))
{
	tron_print_string = print_string;
}
