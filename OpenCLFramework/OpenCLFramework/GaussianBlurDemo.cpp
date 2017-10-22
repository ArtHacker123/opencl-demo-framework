#include "GaussianBlurDemo.h"

void GaussianBlurDemo::load_parameters(const Parameters &params)
{
	try
	{
		sigma_ = params.get_float("sigma");
		gray_ = params.get_bool("gray");
	}
	catch (const std::invalid_argument &e)
	{
		std::cerr << "arg does not exist: " << e.what() << std::endl;
		exit(1);
	}
	catch (const exception &e)
	{
		std::cerr << "exception: " << e.what() << std::endl;
		exit(1);
	}
}

void GaussianBlurDemo::init_parameters(Parameters &params)
{
	params_ = &params;
	shared_ptr<Parameter<float>> p_sigma(new Parameter<float>("sigma", sigma_, "s"));
	params.push(std::move(p_sigma));
}

void GaussianBlurDemo::compile_program(OpenCLBasic *oclobjects)
{
	oclobjects_ = oclobjects;
	// create program
	oprogram_ = new OpenCLProgramMultipleKernels(*oclobjects_, L"BasicKernels.cl", "");
	kernel_ = (*oprogram_)["convolve"];
}

void GaussianBlurDemo::init_program_args(const float *input, int width,
	int height, int nchannels, size_t nbytesI)
{
	w_ = width;
	h_ = height;
	nc_ = nchannels;
	h_in = input;

	int r = ceil(sigma_ * 3);
	int d = (2 * r) + 1;
	h_ker = new float[(size_t)(d * d)];
	kernel(h_ker, r, sigma_);
	size_t kerbytes = d*d*sizeof(float);

	/*w_ = 4;
	h_ = 1;
	nc_ = 1;
	int ni = nc_*w_*h_;
	h_in = new float[ni]{1, 3, 5, 4};
	size_t nbytesn = ni*sizeof(float);
	Mat  mIn(h_, w_, CV_32FC1);
	convert_layered_to_mat(mIn, h_in);
	cout << "Input arr:\n" << mIn << endl;*/



	cl_int result;
	//Creation and allocation of the input data for the kernel
	d_ker = clCreateBuffer(oclobjects_->context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, kerbytes, (void *)h_ker, &result);
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing kernel data:" << getErrorString(result) << endl;
		exit(1);
	}

	d_in = clCreateBuffer(oclobjects_->context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, nbytesI, (void *)input, &result);
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing input data:" << getErrorString(result) << endl;
		exit(1);
	}

	//Here, we are just allocating a buffer onto the OpenCL device. After the execution, we
	//will use it to copy data on the host memory.
	numberOfValues_ = w_*h_*nc_;
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
	result |= clSetKernelArg(kernel_, 0, sizeof(cl_mem), &d_in);
	result |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &d_out);
	result |= clSetKernelArg(kernel_, 2, sizeof(float), &d_ker);
	result |= clSetKernelArg(kernel_, 3, sizeof(int), &w_);
	result |= clSetKernelArg(kernel_, 4, sizeof(int), &h_);
	result |= clSetKernelArg(kernel_, 5, sizeof(int), &r);
	if (result != CL_SUCCESS)
	{
		cout << "Error while setting kernel arguments: " << getErrorString(result) << endl;
		exit(1);
	}
}

void GaussianBlurDemo::execute_program()
{
	//Declarations
	cl_uint   workDim = 2;                      //We can use dimensions to organize data. Here
	//we only got 1 dimension in our array, so we
	//use 1.
	//identfication values to work-items
	size_t    localWorkSize[3] = { 32, 8, 0 };
	size_t    globalWorkSize[3] = { ((w_ + localWorkSize[0] - 1) / localWorkSize[0])*localWorkSize[0],
		(((h_*nc_) + localWorkSize[1] - 1) / localWorkSize[1])*localWorkSize[1], 0 }; //Number of values for each dimension we use
	cl_event  kernelExecEvent;                  //The event for the execution of the kernel

	//If have to set a correct work group size. The total number of work-items we want to run
	//(in our case, numberOfValues work-items) should be divisable by the work-group size.
	//I will not make to muche test here, just the case where numberOfValues is less than
	//the maximum work-group size.


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
	h_out = (float*)malloc(nbytesO_);

	//Waiting for all commands to end. Note we coul have use the kernelExecEvent as an event
	//to wait the end. But the clFinish function is simplier in this case.
	clFinish(oclobjects_->queue);

	//Execution
	clEnqueueReadBuffer(oclobjects_->queue, d_out, blockingRead, offset, nbytesO_,
		h_out, 0, NULL, &readResultsEvent);
}

void GaussianBlurDemo::display_output()
{
	int r = ceil(sigma_ * 3);
	int d = (2 * r) + 1;
	Mat mOut(h_, w_, GET_TYPE(gray_)), mIn(h_, w_, GET_TYPE(gray_)), mKer(d, d, CV_32FC1);
	float *kt = new float[(size_t)(d * d)];
	scale(h_ker, kt, d*d);
	convert_layered_to_mat(mOut, h_out);
	convert_layered_to_mat(mKer, kt);
	convert_layered_to_mat(mIn, h_in);

	//convert_layered_to_mat(mKerOrg, h_ker);
	//convert_layered_to_mat(mIn, h_in);
	//int nz = 0;
	//for (unsigned int i = 0; i<numberOfValues_; i++)
	//{
	//	nz += (h_out[i]) ? 1 : 0;
	//	//if (h_out[i]) cout << h_out[i] << endl;
	//}
	////cout << "total elem:" << numberOfValues_ << "\nnon zero:" << nz << endl;
	//cout << "Kernel:\n" << mKer << endl;
	//cout << "OrgKernel:\n" << mKerOrg << endl;
	//cout << "Input arr:\n" << mIn << endl;
	//cout << "Result arr:\n" << mOut << endl;
	showImage("Input", mIn, 100, 100);
	showImage("Blurred", mOut, 100 + w_ + 40, 100);
	showSizeableImage("Kernel", mKer, 100+80+(2*w_), 100);
	//cv::waitKey(0);
}

void GaussianBlurDemo::deinit_program_args()
{
	free(h_out);
	cl_int result = clReleaseMemObject(d_out);
	result |= clReleaseMemObject(d_in);
	if (result != CL_SUCCESS)
	{
		cout << "Error while deallocating device resources: " << getErrorString(result) << endl;
		exit(1);
	}
}

void GaussianBlurDemo::deinit_parameters()
{
	delete oprogram_;
	//params_->clear();
	params_->rem_float("s");
}
GaussianBlurDemo::~GaussianBlurDemo()
{
	deinit_parameters();
}