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

// using namespace std;

class function
{
public:
	virtual double fun(double *w) = 0 ;
	virtual void grad(double *w, double *g) = 0 ;
	virtual void Hv(double *s, double *Hs) = 0 ;

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
