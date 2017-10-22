#include "PackedKernels.h"

ConvPack::ConvPack(OpenCLBasic *oclobjects, cl_kernel convKernel,
					const float *h_in, float *h_ker, int w, int h,
					int nc, int r, size_t kerbytes, bool copyToHost) :
					oclobjects(oclobjects), convKernel(convKernel),
					w(w), h(h), r(r), nc(nc), copyToHost_(copyToHost)
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
	d_ker = clCreateBuffer(oclobjects->context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, kerbytes, (void *)h_ker, &result);
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing kernel data:" << getErrorString(result) << endl;
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


float *ConvPack::conv(cl_mem nd_in)
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
		cout << "Error while setting kernel arguments: " << getErrorString(result) << endl;
		exit(1);
	}

	cl_uint   workDim = 2;                      //We can use dimensions to organize data. Here
	//we only got 1 dimension in our array, so we
	//use 1.
	//identfication values to work-items
	size_t    localWorkSize[3] = { 32, 8, 0 };
	size_t    globalWorkSize[3] = { ((w + localWorkSize[0] - 1) / localWorkSize[0])*localWorkSize[0],
		(((h*nc) + localWorkSize[1] - 1) / localWorkSize[1])*localWorkSize[1], 0 }; //Number of values for each dimension we use
	cl_event  kernelExecEvent;                  //The event for the execution of the kernel

	//Execution
	result = clEnqueueNDRangeKernel(oclobjects->queue, convKernel, workDim,
		NULL, globalWorkSize,
		localWorkSize, 0, NULL, &kernelExecEvent);
	if (result != CL_SUCCESS)
	{
		cout << "Error while executing the kernel: " << getErrorString(result) << endl;
		exit(1);
	}

	if (!copyToHost_) return NULL;
	//Read output
	cl_bool     blockingRead = CL_TRUE;
	size_t offset = 0;
	cl_event    readResultsEvent;           //The event for the execution of the kernel


	//Allocations
	size_t nbytesO = w*h*nc*sizeof(float);
	float *h_out = (float*)malloc(nbytesO);

	//Waiting for all commands to end. Note we coul have use the kernelExecEvent as an event
	//to wait the end. But the clFinish function is simplier in this case.
	clFinish(oclobjects->queue);

	//Execution
	clEnqueueReadBuffer(oclobjects->queue, d_out, blockingRead, offset, nbytesO,
		h_out, 0, NULL, &readResultsEvent);

	result = clReleaseMemObject(d_out);
	if (result != CL_SUCCESS)
	{
		cout << "Error while deallocating d_out resources: " << getErrorString(result) << endl;
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

	result |= clReleaseMemObject(d_ker);
	if (result != CL_SUCCESS)
	{
		cout << "Error while deallocating d_ker resources: " << getErrorString(result) << endl;
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
		cout << "Error while initializing input data:" << getErrorString(result) << endl;
		exit(1);
	}
	d_inB = (h_in) ? clCreateBuffer(oclobjects->context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, nbytesI, (void *)h_in[1], &result) : NULL;
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing input data:" << getErrorString(result) << endl;
		exit(1);
	}

	int type = (copyToHost_) ? CL_MEM_WRITE_ONLY : CL_MEM_READ_WRITE;
	d_out = clCreateBuffer(oclobjects->context, type,
		nbytesO, NULL, &result);
	if (result != CL_SUCCESS)
	{
		cout << "Error while initializing output data:" << getErrorString(result) << endl;
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
		cout << "Error while setting kernel arguments: " << getErrorString(result) << endl;
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
		localWorkSize, 0, NULL, &kernelExecEvent);
	if (result != CL_SUCCESS)
	{
		cout << "Error while executing the kernel: " << getErrorString(result) << endl;
		exit(1);
	}

	if (!copyToHost_) return NULL;
	//Read output
	cl_bool     blockingRead = CL_TRUE;
	size_t offset = 0;
	cl_event    readResultsEvent;           //The event for the execution of the kernel


	//Allocations
	size_t nbytesO = w*h*sizeof(float);
	float *h_out = (float*)malloc(nbytesO);

	//Waiting for all commands to end. Note we coul have use the kernelExecEvent as an event
	//to wait the end. But the clFinish function is simplier in this case.
	clFinish(oclobjects->queue);

	//Execution
	clEnqueueReadBuffer(oclobjects->queue, d_out, blockingRead, offset, nbytesO,
		h_out, 0, NULL, &readResultsEvent);

	result = clReleaseMemObject(d_out);
	if (result != CL_SUCCESS)
	{
		cout << "Error while deallocating d_out resources: " << getErrorString(result) << endl;
		exit(1);
	}
	if (deallocateIn_)
	{
		result |= clReleaseMemObject(d_inA);
		if (result != CL_SUCCESS)
		{
			cout << "Error while deallocating d_in resources: " << getErrorString(result) << endl;
			exit(1);
		}
		result |= clReleaseMemObject(d_inB);
		if (result != CL_SUCCESS)
		{
			cout << "Error while deallocating d_in resources: " << getErrorString(result) << endl;
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
		cout << "Error while setting kernel arguments: " << getErrorString(result) << endl;
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
		localWorkSize, 0, NULL, &kernelExecEvent);
	if (result != CL_SUCCESS)
	{
		cout << "Error while executing the kernel: " << getErrorString(result) << endl;
		exit(1);
	}

	if (!copyToHost_) return;
	//Read output
	cl_bool     blockingRead = CL_TRUE;
	size_t offset = 0;
	cl_event    readResultsEvent;           //The event for the execution of the kernel


	//Allocations
	size_t nbytesO = w*h*nc*sizeof(float);
	h_outX = (float*)malloc(nbytesO);
	h_outY = (float*)malloc(nbytesO);

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
		cout << "Error while reading d_out resources: " << getErrorString(result) << endl;
		exit(1);
	}

	result = clReleaseMemObject(d_outX);
	result |= clReleaseMemObject(d_outY);
	if (result != CL_SUCCESS)
	{
		cout << "Error while deallocating d_out resources: " << getErrorString(result) << endl;
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