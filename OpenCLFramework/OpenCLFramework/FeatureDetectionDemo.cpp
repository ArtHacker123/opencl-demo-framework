#include "FeatureDetectionDemo.h"



void FeatureDetectionDemo::load_parameters(const Parameters &params)
{
	try
	{
		sigma_ = params.get_float("sigma");
		rho_ = params.get_float("rho");
		f_ = params.get_float("f");
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

void FeatureDetectionDemo::init_parameters(Parameters &params)
{
	params_ = &params;
	shared_ptr<Parameter<float>> p_rho(new Parameter<float>("rho", rho_, "r"));
	shared_ptr<Parameter<float>> p_sigma(new Parameter<float>("sigma", sigma_, "s"));
	shared_ptr<Parameter<float>> p_f(new Parameter<float>("f", f_, "f"));
	params.push(std::move(p_rho));
	params.push(std::move(p_sigma));
	params.push(std::move(p_f));
}

void FeatureDetectionDemo::compile_program(OpenCLBasic *oclobjects)
{
	oclobjects_ = oclobjects;
	// create program
	oprogram_ = new OpenCLProgramMultipleKernels(*oclobjects_, L"BasicKernels.cl", "");
	gradKernel_ = (*oprogram_)["gradient"];
	convKernel_ = (*oprogram_)["convolve"];
	pointKernel_ = (*oprogram_)["pointwise_product"];
	//featKernel_ = (*oprogram_)["feature_detect"];
}

void FeatureDetectionDemo::init_program_args(const float *input, int width,
	int height, int nchannels, size_t nbytesI)
{
	w_ = width;
	h_ = height;
	nc_ = nchannels;
	numberOfValues_ = w_*h_;
	h_in = input;
	cl_int result;

	int rS = ceil(sigma_ * 3);
	int dS = (2 * rS) + 1;
	h_kerS = new float[(size_t)(dS * dS)];
	kernel(h_kerS, rS, sigma_);
	size_t kerbytesS = dS*dS*sizeof(float);

	int rR = ceil(rho_ * 3);
	int dR = (2 * rR) + 1;
	h_kerR = new float[(size_t)(dR * dR)];
	kernel(h_kerR, rR, rho_);
	size_t kerbytesR = dR*dR*sizeof(float);

	convS_ = new ConvPack(oclobjects_, convKernel_, h_in, h_kerS, w_,
		h_, nc_, rS, kerbytesS, false);
	convR_ = new ConvPack(oclobjects_, convKernel_, NULL, h_kerR, w_,
		h_, nc_, rR, kerbytesR, true);

	gradPack_ = new GradientPack(oclobjects_, gradKernel_, NULL, w_, h_, nc_, false);
	pPack11_ = new PointPack(oclobjects_, pointKernel_, NULL, w_, h_, nc_, true);
	pPack12_ = new PointPack(oclobjects_, pointKernel_, NULL, w_, h_, nc_, true);
	pPack22_ = new PointPack(oclobjects_, pointKernel_, NULL, w_, h_, nc_, true);
}

void FeatureDetectionDemo::execute_program()
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


	convS_->conv();
	cl_mem d_convOut = convS_->d_out;//todo deallocate
	clFinish(oclobjects_->queue);

	gradPack_->exec(d_convOut);
	clFinish(oclobjects_->queue);
	cl_mem d_gradX = gradPack_->d_outX;
	cl_mem d_gradY = gradPack_->d_outY;

	h_out = pPack11_->exec(d_gradX, d_gradX);
	clFinish(oclobjects_->queue);
	pPack12_->exec(d_gradX, d_gradY);
	clFinish(oclobjects_->queue);
	pPack22_->exec(d_gradY, d_gradY);
	clFinish(oclobjects_->queue);


	//cl_bool     blockingRead = CL_TRUE;
	//size_t offset = 0;
	//cl_event    readResultsEvent;           //The event for the execution of the kernel


	////Allocations
	//size_t nbytesO = w_*h_*nc_*sizeof(float);
	//float *h_out = (float*)malloc(nbytesO);

	////Waiting for all commands to end. Note we coul have use the kernelExecEvent as an event
	////to wait the end. But the clFinish function is simplier in this case.
	//clFinish(oclobjects_->queue);

	////Execution
	//clEnqueueReadBuffer(oclobjects_->queue, d_gradX, blockingRead, offset, nbytesO,
	//	h_out, 0, NULL, &readResultsEvent);

	//h_out = convR_->conv(d_convOut);
	cl_int result = clReleaseMemObject(d_convOut);
	result |= clReleaseMemObject(d_gradX);
	result |= clReleaseMemObject(d_gradY);
	if (result != CL_SUCCESS)
	{
		cout << "Error while deallocating device resources: " << getErrorString(result) << endl;
		exit(1);
	}
}

void FeatureDetectionDemo::display_output()
{
	//Mat mOut(h_, w_, GET_TYPE(gray_));
	Mat mOut(h_, w_, CV_32FC1);
	convert_layered_to_mat(mOut, h_out);
	mOut *= f_;
	int nz = 0;
	for (unsigned int i = 0; i<numberOfValues_; i++)
	{
		nz += (h_out[i]) ? 1 : 0;
		//if (h_out[i]) cout << h_out[i] << endl;
	}
	//cout << "total elem:" << numberOfValues_ << "\nnon zero:" << nz << endl;
	showImage("Output", mOut, 100, 100);
	//cv::waitKey(0);
}

void FeatureDetectionDemo::deinit_program_args()
{
	free(h_out);
	free(h_kerS);
	free(h_kerR);
	delete convR_;
	delete convS_;
	delete gradPack_;
	delete pPack11_;
	delete pPack12_;
	delete pPack22_;
	/*
	cl_int result = clReleaseMemObject(d_out);
	result |= clReleaseMemObject(d_in);
	if (result != CL_SUCCESS)
	{
		cout << "Error while deallocating device resources: " << getErrorString(result) << endl;
		exit(1);
	}*/
}

void FeatureDetectionDemo::deinit_parameters()
{
	delete oprogram_;
	//params_->clear();
	params_->rem_float("s");
	params_->rem_float("r");
	params_->rem_float("f");
}
FeatureDetectionDemo::~FeatureDetectionDemo()
{
	deinit_parameters();
}