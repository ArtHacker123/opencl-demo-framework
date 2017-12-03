#include "FeatureDetectionDemo.h"



void FeatureDetectionDemo::load_parameters(const Parameters &params)
{
	try
	{
		sigma_ = params.get_float("sigma");
		rho_ = params.get_float("rho");
		f_ = params.get_float("f");
		alpha_ = params.get_float("alpha");
		beta_ = params.get_float("beta");
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
	shared_ptr<Parameter<float>> p_alpha(new Parameter<float>("alpha", alpha_, "a"));
	shared_ptr<Parameter<float>> p_beta(new Parameter<float>("beta", beta_, "b"));
	params.push(std::move(p_rho));
	params.push(std::move(p_sigma));
	params.push(std::move(p_f));
	params.push(std::move(p_alpha));
	params.push(std::move(p_beta));
}

void FeatureDetectionDemo::compile_program(OpenCLBasic *oclobjects)
{
	oclobjects_ = oclobjects;
	// create program
	oprogram_ = new OpenCLProgramMultipleKernels(*oclobjects_, L"BasicKernels.cl", "");
	gradKernel_ = (*oprogram_)["gradient"];
	convKernel_ = (*oprogram_)["convolve"];
	pointKernel_ = (*oprogram_)["pointwise_product"];
	featKernel_ = (*oprogram_)["feature_detect"];
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

	int rR = ceil(sigma_ * 3); // todo decide if remove rho_ completely
	int dR = (2 * rR) + 1;
	h_kerR = new float[(size_t)(dR * dR)];
	kernel(h_kerR, rR, sigma_);
	size_t kerbytesR = dR*dR*sizeof(float);
	int ncO = 1;

	d_in = clCreateBuffer(oclobjects_->context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, nbytesI, (void *)input, &result);
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing input data:" << getErrorString(result) << endl;
		exit(1);
	}

	convS_ = new ConvPack(oclobjects_, convKernel_, NULL, h_kerS, w_,
		h_, nc_, rS, kerbytesS, false);
	convR11_ = new ConvPack(oclobjects_, convKernel_, NULL, h_kerR, w_,
		h_, ncO, rR, kerbytesR, false);
	convR12_ = new ConvPack(oclobjects_, convKernel_, NULL, h_kerR, w_,
		h_, ncO, rR, kerbytesR, false);
	convR22_ = new ConvPack(oclobjects_, convKernel_, NULL, h_kerR, w_,
		h_, ncO, rR, kerbytesR, false);

	gradPack_ = new GradientPack(oclobjects_, gradKernel_, NULL, w_, h_, nc_, false);
	pPack11_ = new PointPack(oclobjects_, pointKernel_, NULL, w_, h_, nc_, false);
	pPack12_ = new PointPack(oclobjects_, pointKernel_, NULL, w_, h_, nc_, false);
	pPack22_ = new PointPack(oclobjects_, pointKernel_, NULL, w_, h_, nc_, false);
	fPack_ = new FeatPack(oclobjects_, featKernel_, NULL, w_, h_, nc_, alpha_, beta_, true);
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


	convS_->exec(d_in);
	cl_mem d_convOut = convS_->d_out;//todo deallocate
	clFinish(oclobjects_->queue);

	gradPack_->exec(d_convOut);
	clFinish(oclobjects_->queue);
	cl_mem d_gradX = gradPack_->d_outX;
	cl_mem d_gradY = gradPack_->d_outY;

	pPack11_->exec(d_gradX, d_gradX);
	clFinish(oclobjects_->queue);
	pPack12_->exec(d_gradX, d_gradY);
	clFinish(oclobjects_->queue);
	pPack22_->exec(d_gradY, d_gradY);
	clFinish(oclobjects_->queue);

	cl_mem d_p11 = pPack11_->d_out;
	cl_mem d_p12 = pPack12_->d_out;
	cl_mem d_p22 = pPack22_->d_out;
	convR11_->exec(d_p11);
	clFinish(oclobjects_->queue);
	convR12_->exec(d_p12);
	clFinish(oclobjects_->queue);
	convR22_->exec(d_p22);
	clFinish(oclobjects_->queue);

	cl_mem d_p11C = convR11_->d_out;
	cl_mem d_p12C = convR12_->d_out;
	cl_mem d_p22C = convR22_->d_out;
	h_out = fPack_->exec(d_in, d_p11C, d_p12C, d_p22C);
	clFinish(oclobjects_->queue);

	//h_out = convR_->exec(d_convOut);
	cl_int result = clReleaseMemObject(d_convOut);
	result |= clReleaseMemObject(d_gradX);
	result |= clReleaseMemObject(d_gradY);
	result |= clReleaseMemObject(d_p11);
	result |= clReleaseMemObject(d_p12);
	result |= clReleaseMemObject(d_p22);
	result |= clReleaseMemObject(d_p11C);
	result |= clReleaseMemObject(d_p12C);
	result |= clReleaseMemObject(d_p22C);
	result |= clReleaseMemObject(d_in);
	if (result != CL_SUCCESS)
	{
		cout << "Error while deallocating device resources: " << getErrorString(result) << endl;
		exit(1);
	}
}

void FeatureDetectionDemo::display_output()
{
	Mat mOut(h_, w_, GET_TYPE(gray_)), mIn(h_, w_, GET_TYPE(gray_));
	//Mat mOut(h_, w_, CV_32FC1), mOut11(h_, w_, CV_32FC1), mOut12(h_, w_, CV_32FC1), mOut22(h_, w_, CV_32FC1);
	convert_layered_to_mat(mOut, h_out);
	convert_layered_to_mat(mIn, h_in);
	/*convert_layered_to_mat(mOut11, h_out11);
	convert_layered_to_mat(mOut12, h_out12);
	convert_layered_to_mat(mOut22, h_out22);
	mOut *= f_;
	mOut11 *= f_;
	mOut12 *= f_;
	mOut22 *= f_;*/
	int nz = 0;
	for (unsigned int i = 0; i<numberOfValues_; i++)
	{
		nz += (h_out[i]) ? 1 : 0;
		//if (h_out[i]) cout << h_out[i] << endl;
	}
	//cout << "total elem:" << numberOfValues_ << "\nnon zero:" << nz << endl;
	showImage("Input", mIn, 100, 100);
	showImage("Feature detection", mOut, 100+w_+40, 100);
	/*showImage("Output11", mOut11, 100+w_+40, 100);
	showImage("Output12", mOut12, 100, 100 + h_ + 40);
	showImage("Output22", mOut22, 100 + w_ + 40, 100 + h_ + 40);*/
	//cv::waitKey(0);
}

void FeatureDetectionDemo::deinit_program_args()
{
	delete[] h_out;
	/*free(h_out11);
	free(h_out12);
	free(h_out22);*/
	delete[] h_kerS;
	delete[] h_kerR;
	delete convR11_;
	delete convR12_;
	delete convR22_;
	delete convS_;
	delete gradPack_;
	delete pPack11_;
	delete pPack12_;
	delete pPack22_;
	delete fPack_;
	
	/*cl_int result = clReleaseMemObject(d_out);
	cl_int result = clReleaseMemObject(d_in);
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
	params_->rem_float("a");
	params_->rem_float("b");
}
FeatureDetectionDemo::~FeatureDetectionDemo()
{
	deinit_parameters();
}