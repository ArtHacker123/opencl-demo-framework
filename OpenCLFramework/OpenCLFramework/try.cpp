
// debug memleaks
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#ifdef _DEBUG
#define MYDEBUG_NEW   new( _NORMAL_BLOCK, __FILE__, __LINE__)
#define new MYDEBUG_NEW
// Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
//allocations to be of _CLIENT_BLOCK type
#else
#define MYDEBUG_NEW
#endif // _DEBUG
// includes
#include "basic.hpp"
#include <iostream>
#include <sstream>
#include <CL/cl.h>
#include <cstdlib>
#include <cstdio>

#include <Windows.h>
#include <process.h>
#include "opencv2/opencv.hpp"

#include "Parameters.h"
#include "Demo.h"
#include "GammaCorrection.h"
#include "GradientDemo.h"
#include "DivergenceDemo.h"
#include "SharpeningDemo.h"
#include "LaplacianDemo.h"
#include "GaussianBlurDemo.h"
#include "FeatureDetectionDemo.h"
#include "FaceTrackingDemo.h"

#include <memory> //unique_ptr

#include "oclobject.hpp"
#include "cmdparser.hpp"
#include "Helper.h"



// usings
using namespace std;
using namespace cv;

// defines
#define WINDOW_WIDTH 500
#define MAX_WINDOW_HEIGHT 500

#define CAMERA_DEFAULT true
#define GRAY_DEFAULT false
#define DEMO_DEFAULT 5
#define GAMMA_CORRECTION_DEMO 1
#define GRADIENT_DEMO 2
#define DIVERGENCE_DEMO 3
#define SHARPENING_DEMO 4
#define LAPLACIAN_DEMO 5
#define GAUSSIAN_BLUR_DEMO 6
#define FEATURE_DETECTION_DEMO 7
#define FACE_TRACKING_DEMO 8

// consts
const char *DEFAULT_IMAGE_PATH = "Desert.jpg";

// globals
HANDLE g_paramHandler;
int g_demo = DEMO_DEFAULT;
bool g_quit = false;
bool g_paramsChanged = true;
bool g_camera = CAMERA_DEFAULT;
bool g_cameraOpen = false;
bool g_gray = GRAY_DEFAULT;
Parameters g_params;

// declarations
void deinit_parameters();
void init_parameters();
void reload_parameters(Demo **demo, OpenCLBasic *oclobjects);
void CL_CALLBACK onOpenCLError(const char *errinfo, const void *private_info,
	size_t cb, void *user_data);
unsigned int __stdcall parametersLoop(void*);



int main(int argc, char** argv)
{
	//_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);


	init_parameters();
	
	Demo *demo;
	switch (g_params.get_int("demo"))
	{
	case GAMMA_CORRECTION_DEMO:
		demo = new GammaCorrection(g_params);
		break;
	case GRADIENT_DEMO:
		demo = new GradientDemo(g_params);
		break;
	case DIVERGENCE_DEMO:
		demo = new DivergenceDemo(g_params);
		break;
	case SHARPENING_DEMO:
		demo = new SharpeningDemo(g_params);
		break;
	case LAPLACIAN_DEMO:
		demo = new LaplacianDemo(g_params);
		break;
	case GAUSSIAN_BLUR_DEMO:
		demo = new GaussianBlurDemo(g_params);
		break;
	case FEATURE_DETECTION_DEMO:
		demo = new FeatureDetectionDemo(g_params);
		break;
	case FACE_TRACKING_DEMO:
		demo = new FaceTrackingDemo(g_params);
		break;
	default:
		break;
	}
	
	// Create the necessary OpenCL objects up to device queue.
	OpenCLBasic oclobjects("0", "gpu");

	// create program
	demo->compile_program(&oclobjects);

	string image = "";
	Mat mIn;
	int camW = 640;
	int camH = 480;
	int w, h, nc, nI;
	size_t nbytesI, numberOfValues, sizeOfBuffers, sizeOfBuffersO;
	float *h_in;
	cv::VideoCapture camera;

	
	while (!g_quit)
	{
		reload_parameters(&demo, &oclobjects);
		g_paramsChanged = false;

		if (g_camera)
		{
			if (!g_cameraOpen)
			{
				camera.open(0); // open the default camera
				if (!camera.isOpened())  // check if we succeeded
				{
					cerr << "Couldn't open camera" << endl;
					return -1;
				}
				camera.set(CV_CAP_PROP_FRAME_WIDTH, camW); //todo decide on proper size
				camera.set(CV_CAP_PROP_FRAME_HEIGHT, camH);
				g_cameraOpen = true;
			}
			// read in first frame to get the dimensions
			while (true)
			{
				camera >> mIn;
				if (!mIn.empty()) break;
			}
			if (g_gray) cvtColor(mIn, mIn, cv::COLOR_RGB2GRAY);
		}
		else
		{ 
			if (g_cameraOpen)
			{
				camera.release();
				g_cameraOpen = false;
			}
			bool ret = getParam("i", image, argc, argv);
			if (!ret) if (argc > 1) { cerr << "ERROR: illegal arguments" << endl; exit(1); }			if (argc <= 1) image = string(DEFAULT_IMAGE_PATH);
			//image = "C:\\Wasted.jpg";
			// Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
			mIn = cv::imread(image.c_str(), (g_params.get_bool("gray") ? CV_LOAD_IMAGE_GRAYSCALE : -1));
			//cout << "image:" << image.c_str() << endl;
			// check
			if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
		}
		Size resz(WINDOW_WIDTH, WINDOW_WIDTH * mIn.rows / mIn.cols);
		if (resz.height > MAX_WINDOW_HEIGHT)
		{
			resz.height = MAX_WINDOW_HEIGHT;
			resz.width = resz.height * mIn.cols / mIn.rows;
		}
		resize(mIn, mIn, resz);
		//flip(mIn, mIn, 1);
		// convert to float representation (opencv loads image values as single bytes by default)
		mIn.convertTo(mIn, CV_32F);
		// convert range of each channel to [0,1] (opencv default is [0,255])
		mIn /= 255.f;
		// get image dimensions
		w = mIn.cols;         // width
		h = mIn.rows;         // height
		nc = mIn.channels();  // number of channels
		nI = w*h*nc;
		nbytesI = (size_t)nI*sizeof(float);
		h_in = new float[(size_t)nI];
		//cout << "image- w:" << w << " h:" << h << " nc:" << nc << endl;
		/*showImage("Input", mIn, 100, 100);
		cv::waitKey(0);*/
		convert_mat_to_layered(h_in, mIn);


		demo->load_parameters(g_params);

		demo->init_program_args(h_in, w, h, nc, nbytesI);
		demo->execute_program();
		demo->display_output();
		
		if (!g_camera)
		{
			cv::waitKey(1000);
		}
		else
		{
			cv::waitKey(20);
		}

		// deallocate resources
		delete[] h_in;
		demo->deinit_program_args();

		//std::cout << "Done. waiting..." << std::endl;

		//while (!g_camera && !g_paramsChanged);
	}
    
	delete demo;
	deinit_parameters();
	g_params.complete_clear();
	_CrtDumpMemoryLeaks();
	//_CrtDumpMemoryLeaks();
}




void deinit_parameters()
{
	CloseHandle(g_paramHandler);
}

void CL_CALLBACK onOpenCLError(const char *errinfo, const void *private_info,
	size_t cb, void *user_data)
{
	printf("Error while creating context or working in this context : %s", errinfo);
}

unsigned int __stdcall parametersLoop(void*)
{
	string line, key, val;
	stringstream ss;
	cout << "getline:" << endl;
	getline(cin, line);
	ss = stringstream(line);
	ss >> key >> val;
	while (key != "exit")
	{
		cout << "key:" << key << ", val:" << val << endl;
		if (val != "") g_params.change(key, val);
		/*if (key == "demo")
		{
			delete g_demo;
		}*/
		g_paramsChanged = true;
		getline(cin, line);
		ss = stringstream(line);
		ss >> key >> val;

	}
	g_quit = true;
	return 0;
}

void reload_parameters(Demo **demo, OpenCLBasic *oclobjects)
{
	try
	{
		g_camera = g_params.get_bool("camera");
		g_gray = g_params.get_bool("gray");
		int ndemo = g_params.get_int("demo");
		if (ndemo != g_demo)
		{
			g_demo = ndemo;
			delete *demo;
			destroyAllWindows();
			switch (g_demo)
			{
			case GAMMA_CORRECTION_DEMO:
				*demo = new GammaCorrection(g_params);
				break;
			case GRADIENT_DEMO:
				*demo = new GradientDemo(g_params);
				break;
			case DIVERGENCE_DEMO:
				*demo = new DivergenceDemo(g_params);
				break;
			case SHARPENING_DEMO:
				*demo = new SharpeningDemo(g_params);
				break;
			case LAPLACIAN_DEMO:
				*demo = new LaplacianDemo(g_params);
				break;
			case GAUSSIAN_BLUR_DEMO:
				*demo = new GaussianBlurDemo(g_params);
				break;
			case FEATURE_DETECTION_DEMO:
				*demo = new FeatureDetectionDemo(g_params);
				break;
			case FACE_TRACKING_DEMO:
				*demo = new FaceTrackingDemo(g_params);
				break;
			default:
				std::cerr << "illegal demo value" << std::endl;
				exit(1);
			}
			(*demo)->compile_program(oclobjects);
		}

		if (g_paramsChanged) cout << "params:\n" << g_params << endl;
	}
	catch (const std::invalid_argument &e)
	{
		std::cerr << "arg does not exist: " << e.what() << std::endl;
	}
}

void init_parameters()
{
	cout << "before parametersInit()" << endl;
	g_paramHandler = (HANDLE)_beginthreadex(0, 0, parametersLoop, 0, 0, 0);
	cout << "after parametersInit()" << endl;

	//unique_ptr<Parameter<int>> demo_(new Parameter<int>("demo", g_demo, "demo"));
	std::shared_ptr<Parameter<int>> demo(new Parameter<int>("demo", g_demo, "demo"));
	std::shared_ptr<Parameter<bool>> cam(new Parameter<bool>("camera", g_camera, "c"));
	std::shared_ptr<Parameter<bool>> gray(new Parameter<bool>("gray", g_gray, "gr"));
	g_params.push(std::move(demo));
	g_params.push(std::move(cam));
	g_params.push(std::move(gray));
	cout << "params:" << endl;
	cout << g_params << endl;
}