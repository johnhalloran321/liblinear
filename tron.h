#ifndef _TRON_H
#define _TRON_H

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <helper_cuda.h>  // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include <iterator>
#include <thrust/iterator/counting_iterator.h>

typedef thrust::device_vector<int>::iterator IndexIterator;

#include "linear.h"
#include <omp.h>

// using namespace std;
class sparse_operator
{
public:
	static double nrm2_sq(const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += x->value*x->value;
			x++;
		}
		return (ret);
	}

	static double dot(const double *s, const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += s[x->index-1]*x->value;
			x++;
		}
		return (ret);
	}

	static void axpy(const double a, const feature_node *x, double *y)
	{
		while(x->index != -1)
		{
			y[x->index-1] += a*x->value;
			x++;
		}
	}

  static void square_axpy(const double a, const feature_node *x, double *y)
	{
		while(x->index != -1)
		{
			y[x->index-1] += x->value*x->value*a;
			x++;
		}
	}

	static void axpy_omp(const double a, const feature_node *x, double *y, int nnz)
	{
#pragma omp parallel for schedule(static)
		for(int k = 0; k < nnz; k++)
		{
			const feature_node *xk = x + k;
			y[xk->index-1] += a*xk->value;
		}
	}
};

class Reduce_Vectors
{
public:
	Reduce_Vectors(int size);
	~Reduce_Vectors();

	void init(void);
	void sum_scale_x(double scalar, feature_node *x);
	void sum_scale_square_x(double scalar, feature_node *x);
	void reduce_sum(double* v);

private:
	int nr_thread;
	int size;
	double **tmp_array;
};

class function
{
public:
        virtual void sync_stream() = 0 ;
	virtual void sync_csrStreams() = 0 ;
	virtual void sync_deStreams() = 0 ;
	virtual void transfer_w(double *w) = 0 ;
	virtual double fun0(double* g) = 0 ;
	virtual double fun(double* w, double* g) = 0 ;
	virtual void grad(double *w, double *g) = 0 ;
	virtual void grad_sync(double *w, double *g) = 0 ;
	virtual double Hv(double *s, double *Hs, double *dev_s, cusparseDnVecDescr_t *vecS) = 0 ;

	virtual int get_nr_variable(void) = 0 ;
	virtual void get_diag_preconditioner(double *M) = 0 ;
	virtual ~function(void){}
};

class TRON
{
public:
	TRON(const function *fun_obj, double eps = 0.1, double eps_cg = 0.1, int max_iter = 1000);
	~TRON();

	void tron(double *w);
	void set_print_string(void (*i_print) (const char *buf));

private:
	int trpcg(double delta, double *g, double *M, double *s, double *r, bool *reach_boundary);
	double norm_inf(int n, double *x);

	double eps;
	double eps_cg;
	int max_iter;
	function *fun_obj;
	void info(const char *fmt,...);
	void (*tron_print_string)(const char *buf);
};
#endif
