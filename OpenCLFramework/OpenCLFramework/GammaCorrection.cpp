#include "GammaCorrection.h"

const char *getErrorString(cl_int error);

void GammaCorrection::load_parameters(const Parameters &params)
{
	try
	{
		gamma = params.get_float("gamma");
	}
	catch (const std::invalid_argument &e)
	{
		std::cerr << "arg does not exist: " << e.what() << std::endl;
	}
}

void GammaCorrection::init_parameters(Parameters &params)
{
	Parameter<float> *p_gamma = new Parameter<float>("gamma", gamma, "g");
	params.push(p_gamma);
}

void GammaCorrection::compile_program(OpenCLBasic *oclobjects)
{
	oclobjects_ = oclobjects;
	// create program
	oprogram_ = new OpenCLProgramMultipleKernels(*oclobjects_, L"BasicKernels.cl", "");
	kernel_ = (*oprogram_)["gamma_correction"];
}

void GammaCorrection::init_program_args(const float *input, int width,
										int height, int nchannels, size_t nbytesI)
{
	h_in = input;
	cl_int result;
	//Creation and allocation of the input data for the kernel
	cl_mem inputBuffer = clCreateBuffer(oclobjects_->context, CL_MEM_READ_ONLY |
										CL_MEM_COPY_HOST_PTR, nbytesI, (void *)input, &result);
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing input data:" << getErrorString(result) << endl;
		exit(1);
	}

	//Here, we are just allocating a buffer onto the OpenCL device. After the execution, we
	//will use it to copy data on the host memory.
	numberOfValues_ = width*height*nchannels;
	nbytesO_ = sizeof(float)*numberOfValues_;
	d_out = clCreateBuffer(oclobjects_->context, CL_MEM_WRITE_ONLY,
		nbytesO_, NULL, &result);
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing output data:" << getErrorString(result) << endl;
		exit(1);
	}

	//We will tell OpenCL what are the arguments for the kernel using their index the
	//declaration of the kernel (see kernel sources) : input at index 0 and output at index 1
	result = CL_SUCCESS;
	result |= clSetKernelArg(kernel_, 0, sizeof(cl_mem), &inputBuffer);
	result |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &d_out);
	result |= clSetKernelArg(kernel_, 2, sizeof(float), &gamma);
	result |= clSetKernelArg(kernel_, 3, sizeof(size_t), &numberOfValues_);
	if (result != CL_SUCCESS)
	{
		cout << "Error while setting kernel arguments: " << getErrorString(result) << endl;
		exit(1);
	}
}

void GammaCorrection::execute_program()
{
	//Declarations
	cl_uint   workDim = nc_;                      //We can use dimensions to organize data. Here
	//we only got 1 dimension in our array, so we
	//use 1.
	size_t*   globalWorkOffset = NULL;          //Offsets used for each dimension to give
	//identfication values to work-items
	size_t    globalWorkSize = numberOfValues_; //Number of values for each dimension we use
	size_t    localWorkSize;                    //Size of a work-group in each dimension
	cl_event  kernelExecEvent;                  //The event for the execution of the kernel

	//If have to set a correct work group size. The total number of work-items we want to run
	//(in our case, numberOfValues work-items) should be divisable by the work-group size.
	//I will not make to muche test here, just the case where numberOfValues is less than
	//the maximum work-group size.
	cl_int result = clGetKernelWorkGroupInfo(kernel_, oclobjects_->device,
												CL_KERNEL_WORK_GROUP_SIZE, sizeof(localWorkSize),
												&localWorkSize, NULL);
	if (localWorkSize > numberOfValues_) localWorkSize = numberOfValues_;
	if (result != CL_SUCCESS)
	{
		cout << "Error while getting maximum work group size: " << getErrorString(result) << endl;
		exit(1);
	}

	//Execution
	result = clEnqueueNDRangeKernel(oclobjects_->queue, kernel_, workDim,
									globalWorkOffset, &globalWorkSize,
									&localWorkSize, 0, NULL, &kernelExecEvent);
	if (result != CL_SUCCESS)
	{
		cout << "Error while executing the kernel: " << getErrorString(result) << endl;
		exit(1);
	}

	//Declarations
	cl_bool     blockingRead = CL_TRUE;
	size_t offset = 0;
	cl_event    readResultsEvent;           //The event for the execution of the kernel
	

	//Allocations
	h_out = (float*)malloc(numberOfValues_*sizeof(float));

	//Waiting for all commands to end. Note we coul have use the kernelExecEvent as an event
	//to wait the end. But the clFinish function is simplier in this case.
	clFinish(oclobjects_->queue);

	//Execution
	clEnqueueReadBuffer(oclobjects_->queue, d_out, blockingRead, offset, nbytesO_,
		h_out, 0, NULL, &readResultsEvent);
}

void GammaCorrection::display_output()
{
	// host results
	cout << "host:" << endl;
	for (unsigned int i = 0; i<numberOfValues_; i++)
	{
		cout << pow(h_in[i], gamma) << endl;

	}
	cout << endl;
	// device results
	cout << "device:" << endl;
	for (unsigned int i = 0; i<numberOfValues_; i++)
	{
		cout << h_out[i] << endl;

	}
}

void GammaCorrection::deinit_program_args()
{
	free(h_out);
}


const char *getErrorString(cl_int error)
{
	switch (error){
		// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
	}
}