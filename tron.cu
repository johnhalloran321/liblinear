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

static double uTMv(int n, double *u, double *M, double *v) //, Reduce_Vectors *reduce_vectors)
{
	const int m = n-4;
	double res = 0;
	int i;
#pragma omp parallel for private(i) reduction(+:res) schedule(static)
	for (i=0; i<n; i++)
		res += u[i]*M[i]*v[i];
	// for (i=0; i<m; i+=5)
	// 	res += u[i]*M[i]*v[i]+u[i+1]*M[i+1]*v[i+1]+u[i+2]*M[i+2]*v[i+2]+
	// 		u[i+3]*M[i+3]*v[i+3]+u[i+4]*M[i+4]*v[i+4];
	// for (; i<n; i++)
	// 	res += u[i]*M[i]*v[i];
	return res;
}

Reduce_Vectors::Reduce_Vectors(int size)
{
	nr_thread = omp_get_max_threads();
	this->size = size;
	tmp_array = new double*[nr_thread];
	for(int i = 0; i < nr_thread; i++)
		tmp_array[i] = new double[size];
}

Reduce_Vectors::~Reduce_Vectors(void)
{
	for(int i = 0; i < nr_thread; i++)
		delete[] tmp_array[i];
	delete[] tmp_array;
}

void Reduce_Vectors::init(void)
{
#pragma omp parallel for schedule(static)
	for(int i = 0; i < size; i++)
		for(int j = 0; j < nr_thread; j++)
			tmp_array[j][i] = 0.0;
}

void Reduce_Vectors::sum_scale_x(double scalar, feature_node *x)
{
	int thread_id = omp_get_thread_num();

	sparse_operator::axpy(scalar, x, tmp_array[thread_id]);
}

void Reduce_Vectors::sum_scale_square_x(double scalar, feature_node *x)
{
	int thread_id = omp_get_thread_num();

	sparse_operator::square_axpy(scalar, x, tmp_array[thread_id]);
}

void Reduce_Vectors::reduce_sum(double* v)
{
#pragma omp parallel for schedule(static)
	for(int i = 0; i < size; i++)
	{
		v[i] = 1;
		for(int j = 0; j < nr_thread; j++)
			v[i] += tmp_array[j][i];
	}
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

	const double alpha_pcg = 0.01;
	double *M = new double[n];

	// Reduce_Vectors *reduce_vectors;
	// reduce_vectors = new Reduce_Vectors(get_nr_variable());

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

	// fun_obj->sync_deStreams();
	// fun_obj->sync_csrStreams();

	f = fun_obj->fun(w, g);
	fun_obj->grad(w, g);

	fun_obj->get_diag_preconditioner(M);
	for(i=0; i<n; i++)
		M[i] = (1-alpha_pcg) + alpha_pcg*M[i];
	delta = sqrt(uTMv(n, g, M, g));

	double *w_new = new double[n];
	bool reach_boundary;
	bool delta_adjusted = false;

	fun_obj->grad_sync(w, g);
	double gnorm = dnrm2_(&n, g, &inc);

	if (gnorm <= eps*gnorm0)
		search = 0;

	while (iter <= max_iter && search)
	{
		cg_iter = trpcg(delta, g, M, s, r, &reach_boundary);

		memcpy(w_new, w, sizeof(double)*n);
		daxpy_(&n, &one, s, &inc, w_new, &inc);

		// Start next transfer of w
		fun_obj->transfer_w(w_new);

		gs = ddot_(&n, g, &inc, s, &inc);
		prered = -0.5*(gs-ddot_(&n, s, &inc, r, &inc));
		fnew = fun_obj->fun(w_new, g);

		// On the first iteration, adjust the initial step bound.
		sMnorm = sqrt(uTMv(n, s, M, s));
		if (iter == 1 && !delta_adjusted)
		{
			delta = min(delta, sMnorm);
			delta_adjusted = true;
		}

		fun_obj->sync_stream();
		// Compute the actual reduction.
		actred = f - fnew;

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

	// delete reduce_vectors;
	delete[] g;
	delete[] r;
	delete[] w_new;
	delete[] s;
	delete[] M;
}

int TRON::trpcg(double delta, double *g, double *M, double *s, double *r, bool *reach_boundary)
{
	int i, inc = 1;
	int n = fun_obj->get_nr_variable();
	double one = 1;
	double *d = new double[n];
	double *Hd = new double[n];
	double zTr, znewTrnew, alpha, beta, cgtol;
	double *z = new double[n];

	double *dev_Hs;
	cusparseDnVecDescr_t *vecHs = new cusparseDnVecDescr_t;
	// Allocate device-side storage and create vector
	checkCudaErrors(cudaMalloc((void** )&dev_Hs, n * sizeof(double)));
	cusparseCreateDnVec(vecHs, n, dev_Hs, CUDA_R_64F);

	*reach_boundary = false;
	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -g[i];
		z[i] = r[i] / M[i];
		d[i] = z[i];
	}

	fun_obj->transfer_w(d);
	zTr = ddot_(&n, z, &inc, r, &inc);
	cgtol = eps_cg*sqrt(zTr);
	int cg_iter = 0;
	int max_cg_iter = max(n, 5);
	double dsq = delta*delta;

	while (cg_iter < max_cg_iter)
	{
		if (sqrt(zTr) <= cgtol)
			break;
		cg_iter++;

		alpha = fun_obj->Hv(d, Hd, dev_Hs, vecHs);

		alpha = zTr/alpha;
		daxpy_(&n, &alpha, d, &inc, s, &inc);

		double sMnorm = sqrt(uTMv(n, s, M, s));
		if (sMnorm > delta)
		{
			info("cg reaches trust region boundary\n");
			*reach_boundary = true;
			alpha = -alpha;
			daxpy_(&n, &alpha, d, &inc, s, &inc);

			double sTMd = uTMv(n, s, M, d);
			double sTMs = uTMv(n, s, M, s);
			double dTMd = uTMv(n, d, M, d);
			double rad = sqrt(sTMd*sTMd + dTMd*(dsq-sTMs));
			if (sTMd >= 0)
				alpha = (dsq - sTMs)/(sTMd + rad);
			else
				alpha = (rad - sTMd)/dTMd;
			daxpy_(&n, &alpha, d, &inc, s, &inc);
			alpha = -alpha;
			fun_obj->sync_stream();
			daxpy_(&n, &alpha, Hd, &inc, r, &inc);
			break;
		} else {
		  fun_obj->sync_stream();
		}

		alpha = -alpha;
		daxpy_(&n, &alpha, Hd, &inc, r, &inc);

		for (i=0; i<n; i++)
			z[i] = r[i] / M[i];
		znewTrnew = ddot_(&n, z, &inc, r, &inc);
		beta = znewTrnew/zTr;
		dscal_(&n, &beta, d, &inc);
		daxpy_(&n, &one, z, &inc, d, &inc);
		fun_obj->transfer_w(d);
		zTr = znewTrnew;
	}

	if (cg_iter == max_cg_iter)
		info("WARNING: reaching maximal number of CG steps\n");
	
	delete[] d;
	delete[] Hd;
	delete[] z;

	cusparseDestroyDnVec(*vecHs);
	checkCudaErrors(cudaFree(dev_Hs));
	delete vecHs;
	// checkCudaErrors(cudaStreamDestroy(*stream));
	// delete stream;

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
