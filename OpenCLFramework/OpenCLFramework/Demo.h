#ifndef _DEMO_INTERFACE_
#define _DEMO_INTERFACE_
#define _CRTDBG_MAP_ALLOC
// includes
#include "Parameters.h"
#include "oclobject.hpp"
#include "opencv2/opencv.hpp"
#include "Helper.h"
#include "PackedKernelS.h"

#ifdef _DEBUG
#define MYDEBUG_NEW   new( _NORMAL_BLOCK, __FILE__, __LINE__)
#define new MYDEBUG_NEW
// Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
//allocations to be of _CLIENT_BLOCK type
#else
#define MYDEBUG_NEW
#endif // _DEBUG

// defines
#define SUCCESS 1
#define GET_TYPE(GRAY) ((GRAY) ? (CV_32FC1) : (CV_32FC3))

using namespace cv;

class Demo
{
protected:
	Demo() {}
	Demo(const Parameters &params) {}
	bool gray_;
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