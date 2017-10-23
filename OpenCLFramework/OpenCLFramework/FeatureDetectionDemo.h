#pragma once

#include "Demo.h"
#include "Parameters.h"
#include <stdexcept>

// defines
#define RHO_DEFAULT 0.2f
#define SIGMA_DEFAULT 0.4f
#define F_DEFAULT 10.f
#define ALPHA_DEFAULT 0.005f
#define BETA_DEFAULT 0.0005f


class FeatureDetectionDemo : public Demo
{
	Parameters *params_;
	float sigma_ = SIGMA_DEFAULT, rho_ = RHO_DEFAULT, f_ = F_DEFAULT,
			alpha_ = ALPHA_DEFAULT, beta_ = BETA_DEFAULT;
	size_t numberOfValues_;
	size_t nbytesO_;
	cl_mem d_out, d_in, d_kerS, d_kerR;
	const float* h_in;
	float *h_out, *h_out11, *h_out12, *h_out22, *h_kerS, *h_kerR;
	int w_, h_, nc_;
	OpenCLProgramMultipleKernels *oprogram_;
	cl_kernel gradKernel_, convKernel_, pointKernel_, featKernel_;
	OpenCLBasic *oclobjects_;
	ConvPack *convS_, *convR11_, *convR12_, *convR22_;
	GradientPack *gradPack_;
	PointPack *pPack11_, *pPack12_, *pPack22_;
	FeatPack *fPack_;
public:
	void load_parameters(const Parameters &params);
	FeatureDetectionDemo(Parameters &params)
	{
		init_parameters(params);
		load_parameters(params);
	}
	~FeatureDetectionDemo();
	void init_parameters(Parameters &params);

	void compile_program(OpenCLBasic *oclobjects);
	//e.g allocate input to device, calculate gaussian kernel
	void init_program_args(const float *input, int width,
		int height, int nchannels, size_t nbytesI);
	void execute_program();
	void display_output();
	void deinit_program_args();
	void deinit_parameters();
};

