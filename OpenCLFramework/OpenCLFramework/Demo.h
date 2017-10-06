#ifndef _DEMO_INTERFACE_
#define _DEMO_INTERFACE_
// includes
#include "Parameters.h"
#include "oclobject.hpp"
#include "opencv2/opencv.hpp"
#include "Helper.h"
// defines
#define SUCCESS 1

using namespace cv;

class Demo
{
protected:
	Demo() {}
	Demo(const Parameters &params) {}
public:
	virtual void load_parameters(const Parameters &params) = 0;
	//const Parameters &params;
	virtual ~Demo() {}
	//e.g sigma, kernelSize
	virtual void init_parameters(Parameters &params)=0;
	//from .cl file?
	virtual void compile_program(OpenCLBasic *oclobjects)=0;
	//e.g allocate input to device, calculate gaussian kernel
	virtual void init_program_args(const float *input, int width,
									int height, int nchannels, size_t nbytesI)=0;
	virtual void execute_program() = 0;
	virtual void display_output() = 0;
	virtual void deinit_program_args() = 0;
	virtual void deinit_parameters() = 0;
	
};



#endif