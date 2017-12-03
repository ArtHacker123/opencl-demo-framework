#include "PackedKernels.h"

FeatPack::FeatPack(OpenCLBasic *oclobjects, cl_kernel kernel, const float *h_in[4],
	int w, int h, int nc, float alpha, float beta, bool copyToHost) :
	oclobjects(oclobjects), kernel_(kernel),
	w(w), h(h), alpha(alpha), beta(beta), nc_(nc), copyToHost_(copyToHost)
{
	size_t nbytesI = w*h*nc_*sizeof(float);
	size_t nbytesT = w*h*sizeof(float);
	//Creation and allocation of the input data for the kernel
	cl_int result = CL_SUCCESS;
	deallocateIn_ = (bool)h_in;
	d_in = (h_in) ? clCreateBuffer(oclobjects->context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, nbytesI, (void *)h_in[0], &result) : NULL;
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing input data:" << getErrorString(result) << endl;
		exit(1);
	}
	d_in11 = (h_in) ? clCreateBuffer(oclobjects->context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, nbytesT, (void *)h_in[1], &result) : NULL;
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing input data:" << getErrorString(result) << endl;
		exit(1);
	}
	d_in12 = (h_in) ? clCreateBuffer(oclobjects->context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, nbytesT, (void *)h_in[2], &result) : NULL;
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing input data:" << getErrorString(result) << endl;
		exit(1);
	}
	d_in22 = (h_in) ? clCreateBuffer(oclobjects->context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, nbytesT, (void *)h_in[3], &result) : NULL;
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing input data:" << getErrorString(result) << endl;
		exit(1);
	}

	int type = (copyToHost_) ? CL_MEM_WRITE_ONLY : CL_MEM_READ_WRITE;
	d_out = clCreateBuffer(oclobjects->context, type,
		nbytesI, NULL, &result);
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing output data:" << getErrorString(result) << endl;
		exit(1);
	}
}

float *FeatPack::exec(cl_mem nd_in, cl_mem nd_in11, cl_mem nd_in12, cl_mem nd_in22)
{
	if (nd_in) d_in = nd_in;
	if (nd_in11) d_in11 = nd_in11;
	if (nd_in12) d_in12 = nd_in12;
	if (nd_in22) d_in22 = nd_in22;
	cl_int result = CL_SUCCESS;
	result |= clSetKernelArg(kernel_, 0, sizeof(cl_mem), &d_in);
	result |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &d_in11);
	result |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &d_in12);
	result |= clSetKernelArg(kernel_, 3, sizeof(cl_mem), &d_in22);
	result |= clSetKernelArg(kernel_, 4, sizeof(cl_mem), &d_out);
	result |= clSetKernelArg(kernel_, 5, sizeof(int), &w);
	result |= clSetKernelArg(kernel_, 6, sizeof(int), &h);
	result |= clSetKernelArg(kernel_, 7, sizeof(float), &alpha);
	result |= clSetKernelArg(kernel_, 8, sizeof(float), &beta);
	if (result != CL_SUCCESS)
	{
		cout << "FeatPack::Error while setting kernel arguments: " << getErrorString(result) << endl;
		exit(1);
	}

	cl_uint   workDim = 2;                      //We can use dimensions to organize data. Here
	//we only got 1 dimension in our array, so we
	//use 1.
	//identfication values to work-items
	size_t    localWorkSize[3] = { 32, 8, 0 };
	size_t    globalWorkSize[3] = { ((w + localWorkSize[0] - 1) / localWorkSize[0])*localWorkSize[0],
		(((h*nc_) + localWorkSize[1] - 1) / localWorkSize[1])*localWorkSize[1], 0 }; //Number of values for each dimension we use
	cl_event  kernelExecEvent;                  //The event for the execution of the kernel

	//Execution
	result = clEnqueueNDRangeKernel(oclobjects->queue, kernel_, workDim,
		NULL, globalWorkSize,
		localWorkSize, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		cout << "FeatPack::Error while executing the kernel: " << getErrorString(result) << endl;
		exit(1);
	}
	clFinish(oclobjects->queue);

	if (!copyToHost_) return NULL;
	//Read output
	cl_bool     blockingRead = CL_TRUE;
	size_t offset = 0;
	cl_event    readResultsEvent;           //The event for the execution of the kernel


	//Allocations
	size_t nbytesO = w*h*nc_*sizeof(float);
	float *h_out;
	try
	{
		h_out = new float[nbytesO];
	}
	catch (std::bad_alloc& exc)
	{
		cout << "FeatPack::Error new[] h_out: " << exc.what() << endl;
		exit(1);
	}

	//Waiting for all commands to end. Note we coul have use the kernelExecEvent as an event
	//to wait the end. But the clFinish function is simplier in this case.
	clFinish(oclobjects->queue);

	//Execution
	result = clEnqueueReadBuffer(oclobjects->queue, d_out, blockingRead, offset, nbytesO,
		h_out, 0, NULL, &readResultsEvent);
	if (result != CL_SUCCESS)
	{
		cout << "FeatPack::Error while reading d_out resources: " << getErrorString(result) << endl;
		exit(1);
	}

	result = clReleaseMemObject(d_out);
	if (result != CL_SUCCESS)
	{
		cout << "FeatPack::Error while deallocating d_out resources: " << getErrorString(result) << endl;
		exit(1);
	}
	if (deallocateIn_)
	{
		result |= clReleaseMemObject(d_in);
		result |= clReleaseMemObject(d_in11);
		result |= clReleaseMemObject(d_in12);
		result |= clReleaseMemObject(d_in22);
		if (result != CL_SUCCESS)
		{
			cout << "Error while deallocating d_in resources: " << getErrorString(result) << endl;
			exit(1);
		}
	}

	return h_out;
}


ConvPack::ConvPack(OpenCLBasic *oclobjects, cl_kernel convKernel,
	const float *h_in, float *h_ker, int w, int h,
	int nc, int r, size_t kerbytes, bool copyToHost) :
	oclobjects(oclobjects), convKernel(convKernel),
	w(w), h(h), r(r), nc_(nc), copyToHost_(copyToHost)
{
	size_t nbytesI = w*h*nc_*sizeof(float);
	//Creation and allocation of the input data for the kernel
	cl_int result = CL_SUCCESS;
	deallocateIn_ = (bool)h_in;
	d_in = (h_in) ? clCreateBuffer(oclobjects->context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, nbytesI, (void *)h_in, &result) : NULL;
	if (result != CL_SUCCESS)
	{
		cout << "ConvPack::Error while initializing input data:" << getErrorString(result) << endl;
		exit(1);
	}
	d_ker = clCreateBuffer(oclobjects->context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, kerbytes, (void *)h_ker, &result);
	if (result != CL_SUCCESS)
	{
		cout << "ConvPack::Error while initializing kernel data:" << getErrorString(result) << endl;
		exit(1);
	}

	int type = (copyToHost_) ? CL_MEM_WRITE_ONLY : CL_MEM_READ_WRITE;
	d_out = clCreateBuffer(oclobjects->context, type,
		nbytesI, NULL, &result);
	if (result != CL_SUCCESS)
	{
		cout << "ConvPack::Error while initializing output data:" << getErrorString(result) << endl;
		exit(1);
	}
}


float *ConvPack::exec(cl_mem nd_in)
{
	if (nd_in) d_in = nd_in;
	cl_int result = CL_SUCCESS;
	result |= clSetKernelArg(convKernel, 0, sizeof(cl_mem), &d_in);
	result |= clSetKernelArg(convKernel, 1, sizeof(cl_mem), &d_out);
	result |= clSetKernelArg(convKernel, 2, sizeof(cl_mem), &d_ker);
	result |= clSetKernelArg(convKernel, 3, sizeof(int), &w);
	result |= clSetKernelArg(convKernel, 4, sizeof(int), &h);
	result |= clSetKernelArg(convKernel, 5, sizeof(int), &r);
	if (result != CL_SUCCESS)
	{
		cout << "ConvPack::Error while setting kernel arguments: " << getErrorString(result) << endl;
		exit(1);
	}

	cl_uint   workDim = 2;                      //We can use dimensions to organize data. Here
	//we only got 1 dimension in our array, so we
	//use 1.
	//identfication values to work-items
	size_t    localWorkSize[3] = { 32, 8, 0 };
	size_t    globalWorkSize[3] = { ((w + localWorkSize[0] - 1) / localWorkSize[0])*localWorkSize[0],
		(((h*nc_) + localWorkSize[1] - 1) / localWorkSize[1])*localWorkSize[1], 0 }; //Number of values for each dimension we use
	cl_event  kernelExecEvent;                  //The event for the execution of the kernel

	//Execution
	result = clEnqueueNDRangeKernel(oclobjects->queue, convKernel, workDim,
		NULL, globalWorkSize,
		localWorkSize, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		cout << "ConvPack::Error while executing the kernel: " << getErrorString(result) << endl;
		exit(1);
	}
	clFinish(oclobjects->queue);

	if (!copyToHost_) return NULL;
	//Read output
	cl_bool     blockingRead = CL_TRUE;
	size_t offset = 0;
	cl_event    readResultsEvent;           //The event for the execution of the kernel


	//Allocations
	size_t nbytesO = w*h*nc_*sizeof(float);
	float *h_out = new float[nbytesO];

	//Waiting for all commands to end. Note we coul have use the kernelExecEvent as an event
	//to wait the end. But the clFinish function is simplier in this case.
	clFinish(oclobjects->queue);

	//Execution
	result = clEnqueueReadBuffer(oclobjects->queue, d_out, blockingRead, offset, nbytesO,
		h_out, 0, NULL, &readResultsEvent);

	result |= clReleaseMemObject(d_out);
	if (result != CL_SUCCESS)
	{
		cout << "ConvPack::Error while deallocating d_out resources: " << getErrorString(result) << endl;
		exit(1);
	}
	if (deallocateIn_)
	{
		result |= clReleaseMemObject(d_in);
		if (result != CL_SUCCESS)
		{
			cout << "ConvPack::Error while deallocating d_in resources: " << getErrorString(result) << endl;
			exit(1);
		}
	}

	result |= clReleaseMemObject(d_ker);
	if (result != CL_SUCCESS)
	{
		cout << "ConvPack::Error while deallocating d_ker resources: " << getErrorString(result) << endl;
		exit(1);
	}
	return h_out;
}



PointPack::PointPack(OpenCLBasic *oclobjects, cl_kernel pointKernel,
						const float *h_in[2], int w, int h, int nc, bool copyToHost) : 
						oclobjects(oclobjects), kernel_(pointKernel),
						w(w), h(h), nc(nc), copyToHost_(copyToHost)
{
	size_t nbytesI = w*h*nc*sizeof(float);
	size_t nbytesO = w*h*sizeof(float);
	//Creation and allocation of the input data for the kernel
	cl_int result = CL_SUCCESS;
	deallocateIn_ = (bool)h_in;
	d_inA = (h_in) ? clCreateBuffer(oclobjects->context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, nbytesI, (void *)h_in[0], &result) : NULL;
	if (result != CL_SUCCESS)
	{
		cout << "PointPack::Error while initializing input data:" << getErrorString(result) << endl;
		exit(1);
	}
	d_inB = (h_in) ? clCreateBuffer(oclobjects->context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, nbytesI, (void *)h_in[1], &result) : NULL;
	if (result != CL_SUCCESS)
	{
		cout << "PointPack::Error while initializing input data:" << getErrorString(result) << endl;
		exit(1);
	}

	int type = (copyToHost_) ? CL_MEM_WRITE_ONLY : CL_MEM_READ_WRITE;
	d_out = clCreateBuffer(oclobjects->context, type,
		nbytesO, NULL, &result);
	if (result != CL_SUCCESS)
	{
		cout << "PointPack::Error while initializing output data:" << getErrorString(result) << endl;
		exit(1);
	}
}

float *PointPack::exec(cl_mem nd_inA, cl_mem nd_inB)
{
	if (nd_inA) d_inA = nd_inA;
	if (nd_inB) d_inB = nd_inB;
	cl_int result = CL_SUCCESS;
	result |= clSetKernelArg(kernel_, 0, sizeof(cl_mem), &d_inA);
	result |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &d_inB);
	result |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &d_out);
	result |= clSetKernelArg(kernel_, 3, sizeof(int), &w);
	result |= clSetKernelArg(kernel_, 4, sizeof(int), &h);
	result |= clSetKernelArg(kernel_, 5, sizeof(int), &nc);
	if (result != CL_SUCCESS)
	{
		cout << "PointPack::Error while setting kernel arguments: " << getErrorString(result) << endl;
		exit(1);
	}

	cl_uint   workDim = 2;                      //We can use dimensions to organize data. Here
	//we only got 1 dimension in our array, so we
	//use 1.
	//identfication values to work-items
	size_t    localWorkSize[3] = { 32, 8, 0 };
	size_t    globalWorkSize[3] = { ((w + localWorkSize[0] - 1) / localWorkSize[0])*localWorkSize[0],
		(((h) + localWorkSize[1] - 1) / localWorkSize[1])*localWorkSize[1], 0 }; //Number of values for each dimension we use
	cl_event  kernelExecEvent;                  //The event for the execution of the kernel

	//Execution
	result = clEnqueueNDRangeKernel(oclobjects->queue, kernel_, workDim,
		NULL, globalWorkSize,
		localWorkSize, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		cout << "PointPack::Error while executing the kernel: " << getErrorString(result) << endl;
		exit(1);
	}
	clFinish(oclobjects->queue);

	if (!copyToHost_) return NULL;
	//Read output
	cl_bool     blockingRead = CL_TRUE;
	size_t offset = 0;
	cl_event    readResultsEvent;           //The event for the execution of the kernel


	//Allocations
	size_t nbytesO = w*h*sizeof(float);
	float *h_out = new float[nbytesO];

	//Waiting for all commands to end. Note we coul have use the kernelExecEvent as an event
	//to wait the end. But the clFinish function is simplier in this case.
	clFinish(oclobjects->queue);

	//Execution
	clEnqueueReadBuffer(oclobjects->queue, d_out, blockingRead, offset, nbytesO,
		h_out, 0, NULL, &readResultsEvent);

	result = clReleaseMemObject(d_out);
	if (result != CL_SUCCESS)
	{
		cout << "PointPack::Error while deallocating d_out resources: " << getErrorString(result) << endl;
		exit(1);
	}
	if (deallocateIn_)
	{
		result |= clReleaseMemObject(d_inA);
		if (result != CL_SUCCESS)
		{
			cout << "PointPack::Error while deallocating d_in resources: " << getErrorString(result) << endl;
			exit(1);
		}
		result |= clReleaseMemObject(d_inB);
		if (result != CL_SUCCESS)
		{
			cout << "PointPack::Error while deallocating d_in resources: " << getErrorString(result) << endl;
			exit(1);
		}
	}
	return h_out;
}

GradientPack::GradientPack(OpenCLBasic *oclobjects, cl_kernel pointKernel,
	const float *h_in, int w, int h, int nc, bool copyToHost) :
	oclobjects(oclobjects), kernel_(pointKernel),
	w(w), h(h), nc(nc), copyToHost_(copyToHost)
{
	size_t nbytesI = w*h*nc*sizeof(float);
	//Creation and allocation of the input data for the kernel
	cl_int result = CL_SUCCESS;
	deallocateIn_ = (bool)h_in;
	d_in = (h_in) ? clCreateBuffer(oclobjects->context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, nbytesI, (void *)h_in, &result) : NULL;
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing input data:" << getErrorString(result) << endl;
		exit(1);
	}

	int type = (copyToHost_) ? CL_MEM_WRITE_ONLY : CL_MEM_READ_WRITE;
	d_outX = clCreateBuffer(oclobjects->context, type,
		nbytesI, NULL, &result);
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing output data:" << getErrorString(result) << endl;
		exit(1);
	}
	d_outY = clCreateBuffer(oclobjects->context, type,
		nbytesI, NULL, &result);
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing output data:" << getErrorString(result) << endl;
		exit(1);
	}
}


void GradientPack::exec(cl_mem nd_in)
{
	if (nd_in) d_in = nd_in;
	cl_int result = CL_SUCCESS;
	result |= clSetKernelArg(kernel_, 0, sizeof(cl_mem), &d_in);
	result |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &d_outX);
	result |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &d_outY);
	result |= clSetKernelArg(kernel_, 3, sizeof(int), &w);
	result |= clSetKernelArg(kernel_, 4, sizeof(int), &h);
	result |= clSetKernelArg(kernel_, 5, sizeof(int), &nc);
	if (result != CL_SUCCESS)
	{
		cout << "GradientPack::Error while setting kernel arguments: " << getErrorString(result) << endl;
		exit(1);
	}

	cl_uint   workDim = 2;                      //We can use dimensions to organize data. Here
	//we only got 1 dimension in our array, so we
	//use 1.
	//identfication values to work-items
	size_t    localWorkSize[3] = { 32, 8, 0 };
	size_t    globalWorkSize[3] = { ((w + localWorkSize[0] - 1) / localWorkSize[0])*localWorkSize[0],
		(((h*nc)+localWorkSize[1] - 1) / localWorkSize[1])*localWorkSize[1], 0 }; //Number of values for each dimension we use
	cl_event  kernelExecEvent;                  //The event for the execution of the kernel

	//Execution
	result = clEnqueueNDRangeKernel(oclobjects->queue, kernel_, workDim,
		NULL, globalWorkSize,
		localWorkSize, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		cout << "GradientPack::Error while executing the kernel: " << getErrorString(result) << endl;
		exit(1);
	}
	clFinish(oclobjects->queue);

	if (!copyToHost_) return;
	//Read output
	cl_bool     blockingRead = CL_TRUE;
	size_t offset = 0;
	cl_event    readResultsEvent;           //The event for the execution of the kernel


	//Allocations
	size_t nbytesO = w*h*nc*sizeof(float);
	h_outX = new float[nbytesO];
	h_outY = new float[nbytesO];

	//Waiting for all commands to end. Note we coul have use the kernelExecEvent as an event
	//to wait the end. But the clFinish function is simplier in this case.
	clFinish(oclobjects->queue);

	//Execution
	result = clEnqueueReadBuffer(oclobjects->queue, d_outX, blockingRead, offset, nbytesO,
		h_outX, 0, NULL, &readResultsEvent);
	result |= clEnqueueReadBuffer(oclobjects->queue, d_outY, blockingRead, offset, nbytesO,
		h_outY, 0, NULL, &readResultsEvent);
	if (result != CL_SUCCESS)
	{
		cout << "GradientPack::Error while reading d_outX,Y resources: " << getErrorString(result) << endl;
		exit(1);
	}

	result = clReleaseMemObject(d_outX);
	result |= clReleaseMemObject(d_outY);
	if (result != CL_SUCCESS)
	{
		cout << "GradientPack::Error while deallocating d_outX,Y resources: " << getErrorString(result) << endl;
		exit(1);
	}
	if (deallocateIn_)
	{
		result |= clReleaseMemObject(d_in);
		if (result != CL_SUCCESS)
		{
			cout << "Error while deallocating d_in resources: " << getErrorString(result) << endl;
			exit(1);
		}
	}
}