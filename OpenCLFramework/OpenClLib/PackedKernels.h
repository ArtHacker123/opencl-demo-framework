#pragma once

#include "oclobject.hpp"
#include "opencv2/opencv.hpp"
#include "Helper.h"

typedef OpenCLProgramMultipleKernels* Program;

struct ConvPack
{
private:
	Program oprogram_;
	OpenCLBasic *oclobjects;
	cl_kernel convKernel;
	cl_mem h_in, d_out, d_ker, d_in;
	int w, h, r, nc;
public:
	ConvPack(Program oprogram_, OpenCLBasic *oclobjects, cl_kernel convKernel,
		float *h_in, float *h_ker, int w, int h, int nc, int r, size_t kerbytes) : oprogram_(oprogram_),
		oclobjects(oclobjects), convKernel(convKernel), 
		w(w), h(h), r(r), nc(nc)
	{
		size_t nbytesI = w*h*nc*sizeof(float);
		//Creation and allocation of the input data for the kernel
		cl_int result = CL_SUCCESS;
		d_in = clCreateBuffer(oclobjects->context, CL_MEM_READ_ONLY |
			CL_MEM_COPY_HOST_PTR, nbytesI, (void *)h_in, &result);
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

		d_out = clCreateBuffer(oclobjects->context, CL_MEM_WRITE_ONLY,
			nbytesI, NULL, &result);
		if (result != CL_SUCCESS)
		{
			cout << "Error while initializing output data:" << getErrorString(result) << endl;
			exit(1);
		}
	}
	~ConvPack(){}
	float *conv()
	{
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
		cl_int result = clEnqueueNDRangeKernel(oclobjects->queue, convKernel, workDim,
			NULL, globalWorkSize,
			localWorkSize, 0, NULL, &kernelExecEvent);
		if (result != CL_SUCCESS)
		{
			cout << "Error while executing the kernel: " << getErrorString(result) << endl;
			exit(1);
		}

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
		return h_out;
	}
};

