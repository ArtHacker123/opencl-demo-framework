#pragma once


#include "Demo.h"
#include "Parameters.h"
#include <stdexcept>



class GradientDemo :
	public Demo
{
	Parameters *params_;
	size_t numberOfValues_;
	size_t nbytesO_;
	cl_mem d_outX, d_outY, d_in;
	const float* h_in;
	float* h_outX;
	float* h_outY;
	int w_, h_, nc_;
	OpenCLProgramMultipleKernels *oprogram_;
	cl_kernel kernel_;
	OpenCLBasic *oclobjects_;
public:

	void load_parameters(const Parameters &params);
	GradientDemo(Parameters &params)
	{
		init_parameters(params);
		load_parameters(params);
	}
	~GradientDemo();
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

