#pragma once

#include "oclobject.hpp"
#include "opencv2/opencv.hpp"
#include "Helper.h"
#include <errno.h>

typedef OpenCLProgramMultipleKernels* Program;
//convolve() kernel
struct ConvPack
{
private:
	OpenCLBasic *oclobjects;
	cl_kernel convKernel;
	cl_mem d_ker, d_in;
	int w, h, r, nc_;
	bool copyToHost_;
	bool deallocateIn_;
public:
	// must be deallocated by caller if copyToHost_==true
	cl_mem d_out;

	ConvPack(OpenCLBasic *oclobjects, cl_kernel convKernel,
				const float *h_in, float *h_ker, int w, int h, int nc,
				int r, size_t kerbytes, bool copyToHost);
	~ConvPack(){}
	float *exec(cl_mem nd_in = NULL);
};

//pointwise_product() kernel
struct PointPack
{
private:
	OpenCLBasic *oclobjects;
	cl_kernel kernel_;
	cl_mem d_inA, d_inB;
	int w, h, nc;
	bool copyToHost_;
	bool deallocateIn_;
public:
	// must be deallocated by caller if copyToHost_==true
	cl_mem d_out;

	PointPack(OpenCLBasic *oclobjects, cl_kernel pointKernel,
		const float *h_in[2], int w, int h, int nc, bool copyToHost);
	~PointPack(){}
	float *exec(cl_mem nd_inA = NULL, cl_mem nd_inB = NULL);
};


//gradient() kernel
struct GradientPack
{
private:
	OpenCLBasic *oclobjects;
	cl_kernel kernel_;
	cl_mem d_in;
	int w, h, nc;
	bool copyToHost_;
	bool deallocateIn_;
	GradientPack(const GradientPack&);
public:
	// must be deallocated by caller if copyToHost_==true
	cl_mem d_outX, d_outY;
	// must be deallocated by caller if copyToHost_==false
	float *h_outX, *h_outY;

	GradientPack(OpenCLBasic *oclobjects, cl_kernel gradKernel,
		const float *h_in, int w, int h, int nc, bool copyToHost);
	~GradientPack(){}
	void exec(cl_mem nd_in = NULL);
};

//feature_detect() kernel
struct FeatPack
{
private:
	OpenCLBasic *oclobjects;
	cl_kernel kernel_;
	cl_mem d_in, d_in11, d_in12, d_in22;
	int w, h, nc_;
	bool copyToHost_;
	bool deallocateIn_;
	float alpha, beta;
public:
	// must be deallocated by caller if copyToHost_==true
	cl_mem d_out;

	FeatPack(OpenCLBasic *oclobjects, cl_kernel kernel, const float *h_in[4],
				int w, int h, int nc, float alpha, float beta, bool copyToHost);
	~FeatPack(){} // todo for all kernels deallocate d_outs'
	float *exec(cl_mem nd_in = NULL, cl_mem nd_in11 = NULL,
				cl_mem nd_in12 = NULL, cl_mem nd_in22 = NULL);
};