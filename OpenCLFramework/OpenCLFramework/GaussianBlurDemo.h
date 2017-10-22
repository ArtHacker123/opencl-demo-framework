#pragma once
#include "Demo.h"
#include "Parameters.h"
#include <stdexcept>

#define DEFAULT_SIGMA 2.f

class GaussianBlurDemo : public Demo
{
	Parameters *params_;
	float sigma_ = DEFAULT_SIGMA;
	size_t numberOfValues_;
	size_t nbytesO_;
	cl_mem d_out, d_in, d_ker;
	const float* h_in;
	float *h_out, *h_ker;
	int w_, h_, nc_;
	OpenCLProgramMultipleKernels *oprogram_;
	cl_kernel kernel_;
	OpenCLBasic *oclobjects_;
public:
	void load_parameters(const Parameters &params);
	GaussianBlurDemo(Parameters &params)
	{
		init_parameters(params);
		load_parameters(params);
	}
	~GaussianBlurDemo();
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

