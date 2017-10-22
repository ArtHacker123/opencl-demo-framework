#include "GradientDemo.h"


void GradientDemo::load_parameters(const Parameters &params)
{
	try
	{
		gray_ = params.get_bool("gray");
	}
	catch (const std::invalid_argument &e)
	{
		std::cerr << "arg does not exist: " << e.what() << std::endl;
		exit(1);
	}
}

void GradientDemo::init_parameters(Parameters &params)
{
	params_ = &params;
	/*shared_ptr<Parameter<float>> p_gamma(new Parameter<float>("gamma", gamma, "g"));
	params.push(std::move(p_gamma));*/
}

void GradientDemo::compile_program(OpenCLBasic *oclobjects)
{
	oclobjects_ = oclobjects;
	// create program
	oprogram_ = new OpenCLProgramMultipleKernels(*oclobjects_, L"BasicKernels.cl", "");
	kernel_ = (*oprogram_)["gradient"];
}

void GradientDemo::init_program_args(const float *input, int width,
	int height, int nchannels, size_t nbytesI)
{
	w_ = width;
	h_ = height;
	nc_ = nchannels;
	h_in = input;
	cl_int result;
	//Creation and allocation of the input data for the kernel
	d_in = clCreateBuffer(oclobjects_->context, CL_MEM_READ_ONLY |
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
	d_outX = clCreateBuffer(oclobjects_->context, CL_MEM_WRITE_ONLY,
		nbytesO_, NULL, &result);
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing output data:" << getErrorString(result) << endl;
		exit(1);
	}
	d_outY = clCreateBuffer(oclobjects_->context, CL_MEM_WRITE_ONLY,
		nbytesO_, NULL, &result);
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing output data:" << getErrorString(result) << endl;
		exit(1);
	}

	//We will tell OpenCL what are the arguments for the kernel using their index the
	//declaration of the kernel (see kernel sources) : input at index 0 and output at index 1
	result = CL_SUCCESS;
	result |= clSetKernelArg(kernel_, 0, sizeof(cl_mem), &d_in);
	result |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &d_outX);
	result |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &d_outY);
	result |= clSetKernelArg(kernel_, 3, sizeof(int), &w_);
	result |= clSetKernelArg(kernel_, 4, sizeof(int), &h_);
	result |= clSetKernelArg(kernel_, 5, sizeof(int), &nc_);
	if (result != CL_SUCCESS)
	{
		cout << "Error while setting kernel arguments: " << getErrorString(result) << endl;
		exit(1);
	}
}

void GradientDemo::execute_program()
{
	//Declarations
	cl_uint   workDim = 2;
	//identfication values to work-items
	size_t    localWorkSize[3] = { 32, 8, 0 };
	size_t    globalWorkSize[3] = { ((w_ + localWorkSize[0] - 1) / localWorkSize[0])*localWorkSize[0],
		(((h_*nc_) + localWorkSize[1] - 1) / localWorkSize[1])*localWorkSize[1], 0 }; //Number of values for each dimension we use
	cl_event  kernelExecEvent;                  //The event for the execution of the kernel
	
	//Execution
	cl_int result = clEnqueueNDRangeKernel(oclobjects_->queue, kernel_, workDim,
		NULL, globalWorkSize,
		localWorkSize, 0, NULL, &kernelExecEvent);
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
	h_outX = (float*)malloc(nbytesO_);
	h_outY= (float*)malloc(nbytesO_);

	//Waiting for all commands to end. Note we coul have use the kernelExecEvent as an event
	//to wait the end. But the clFinish function is simplier in this case.
	clFinish(oclobjects_->queue);

	// read outputs from device
	clEnqueueReadBuffer(oclobjects_->queue, d_outX, blockingRead, offset, nbytesO_,
		h_outX, 0, NULL, &readResultsEvent);
	clEnqueueReadBuffer(oclobjects_->queue, d_outY, blockingRead, offset, nbytesO_,
		h_outY, 0, NULL, &readResultsEvent);
}

void GradientDemo::display_output()
{
	Mat mOutX(h_, w_, GET_TYPE(gray_)), mOutY(h_, w_, GET_TYPE(gray_));
	convert_layered_to_mat(mOutX, h_outX);
	convert_layered_to_mat(mOutY, h_outY);
	int nz = 0;
	for (unsigned int i = 0; i<numberOfValues_; i++)
	{
		//nz += (h_out[i]) ? 1 : 0;
		//if (h_out[i]) cout << h_out[i] << endl;
	}
	//cout << "total elem:" << numberOfValues_ << "\nnon zero:" << nz << endl;
	showImage("GradientX", mOutX, 100, 100 );
	showImage("GradientY", mOutY, 100 + w_ + 40, 100);
	//cv::waitKey(0);
}

void GradientDemo::deinit_program_args()
{
	free(h_outX);
	free(h_outY);
	cl_int result = clReleaseMemObject(d_outX);
	result |= clReleaseMemObject(d_outY);
	result |= clReleaseMemObject(d_in);
	if (result != CL_SUCCESS)
	{
		cout << "Error while deallocating device resources: " << getErrorString(result) << endl;
		exit(1);
	}
}

void GradientDemo::deinit_parameters()
{
	delete oprogram_;
	//params_->clear();
}
GradientDemo::~GradientDemo()
{
	deinit_parameters();
}