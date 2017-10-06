#ifndef _GAMMA_CORRECTION_
#define _GAMMA_CORRECTION_
#include "Demo.h"
#include "Parameters.h"
#include <stdexcept>

// defines
#define GAMMA_DEFAULT 0.5f


class GammaCorrection : public Demo
{
	float gamma = GAMMA_DEFAULT;
	size_t numberOfValues_;
	size_t nbytesO_;
	cl_mem d_out, d_in;
	const float* h_in;
	float* h_out;
	int w_, h_, nc_;
	OpenCLProgramMultipleKernels *oprogram_;
	cl_kernel kernel_;
	OpenCLBasic *oclobjects_;
public:
	void load_parameters(const Parameters &params);
	GammaCorrection(const Parameters &params)
	{
		load_parameters(params);
	}
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



#endif