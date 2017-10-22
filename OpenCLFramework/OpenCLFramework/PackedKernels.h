#pragma once

#include "oclobject.hpp"
#include "opencv2/opencv.hpp"
#include "Helper.h"

typedef OpenCLProgramMultipleKernels* Program;
//convolve() kernel
struct ConvPack
{
private:
	OpenCLBasic *oclobjects;
	cl_kernel convKernel;
	cl_mem d_ker, d_in;
	int w, h, r, nc;
	bool copyToHost_;
	bool deallocateIn_;
public:
	// must be deallocated by caller if copyToHost_==true
	cl_mem d_out;

	ConvPack(OpenCLBasic *oclobjects, cl_kernel convKernel,
				const float *h_in, float *h_ker, int w, int h, int nc,
				int r, size_t kerbytes, bool copyToHost);
	~ConvPack(){}
	float *conv(cl_mem nd_in = NULL);
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

