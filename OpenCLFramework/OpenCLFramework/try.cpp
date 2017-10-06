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

#include "oclobject.hpp"
#include "cmdparser.hpp"
#include "Helper.h"

// usings
using namespace std;
using namespace cv;

// defines
#define CAMERA_DEFAULT false
#define GRAY_DEFAULT false
#define DEMO_DEFAULT 0
#define GAMMA_CORRECTION_DEMO 1

// globals
HANDLE g_paramHandler;
int g_demo = GAMMA_CORRECTION_DEMO;
bool g_quit = false;
bool g_paramsChanged = true;
bool g_camera = CAMERA_DEFAULT;
bool g_gray = GRAY_DEFAULT;
Parameters g_params;

// declarations
void deinit_parameters();
void init_parameters();
void CL_CALLBACK onOpenCLError(const char *errinfo, const void *private_info,
	size_t cb, void *user_data);
unsigned int __stdcall parametersLoop(void*);




int main(int argc, char** argv)
{
	//main init main parameters(demo/camera)
	//demo init parameters
	//demo compile program(kernel(s)) and args
	//LOOP
	//main reload parameters(if demo changed reinit demo)
	//main set input(image/camera)
	//demo process input
	//demo set program(kernel(s)) and args
	//demo execute program
	//demo extract and display output(might be multiple outputs)
	//demo deallocate resources
	//main deallocate resources
	//ENDLOOP
	//demo deinit parameters
	//main deinit parameters

	init_parameters();

	Demo *demo;
	switch (g_params.get_int("demo"))
	{
	case DEMO_DEFAULT:
		cout << "in demo default" << endl;
	case GAMMA_CORRECTION_DEMO:
		demo = new GammaCorrection(g_params);
		demo->init_parameters(g_params);
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
	while (!g_quit)
	{
		g_paramsChanged = false;

		if (g_camera)
		{
			VideoCapture camera(0); // open the default camera
			if (!camera.isOpened())  // check if we succeeded
			{
				cerr << "Couldn't open camera" << endl;
				return -1;
			}
			camera.set(CV_CAP_PROP_FRAME_WIDTH, camW);
			camera.set(CV_CAP_PROP_FRAME_HEIGHT, camH);
			// read in first frame to get the dimensions
			while (true)
			{
				camera >> mIn;
				if (!mIn.empty()) break;
			}
		}
		else
		{
			bool ret = getParam("i", image, argc, argv);
			if (!ret) cerr << "ERROR: no image specified" << endl;			if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-gray]" << endl; return 1; }
			//image = "C:\\Wasted.jpg";
			// Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
			mIn = cv::imread(image.c_str(), (g_params.get_bool("gray") ? CV_LOAD_IMAGE_GRAYSCALE : -1));
			cout << "image:" << image.c_str() << endl;
			// check
			if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
		}
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
		cout << "image- w:" << w << " h:" << h << " nc:" << nc << endl;
		/*showImage("Input", mIn, 100, 100);
		cv::waitKey(0);*/
		convert_mat_to_layered(h_in, mIn);

		////Initialization
		//numberOfValues = 50;
		//sizeOfBuffers = numberOfValues*sizeof(float);
		//sizeOfBuffersO = numberOfValues*sizeof(float);
		//h_in = (float*)malloc(sizeOfBuffers);  //Our input data array
		//for (unsigned int i = 0; i<numberOfValues; i++)                           //We put some numbers in it
		//{
		//	h_in[i] = i;       //I know, I'm very lazy.....
		//}

		demo->load_parameters(g_params);

		demo->init_program_args(h_in, w, h, nc, nbytesI);
		demo->execute_program();
		demo->display_output();


		// deallocate resources
		free(h_in);
		demo->deinit_program_args();

		std::cout << "Done. waiting..." << std::endl;

		while (!g_camera && !g_paramsChanged);
	}
    

	deinit_parameters();
}



void init_parameters()
{
	cout << "before parametersInit()" << endl;
	g_paramHandler = (HANDLE)_beginthreadex(0, 0, parametersLoop, 0, 0, 0);
	cout << "after parametersInit()" << endl;
	
	Parameter<int> *demo_ = new Parameter<int>("demo", g_demo, "demo");
	Parameter<bool> *cam = new Parameter<bool>("camera", g_camera, "c");
	Parameter<bool> *gray = new Parameter<bool>("gray", g_gray, "gr");
	g_params.push(demo_);
	g_params.push(cam);
	g_params.push(gray);
	cout << "params:" << endl;
	cout << g_params << endl;
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
		g_paramsChanged = true;
		cout << "params:" << endl;
		cout << g_params << endl;
		cout << "getline:" << endl;
		getline(cin, line);
		ss = stringstream(line);
		ss >> key >> val;

	}
	g_quit = true;
	return 0;
}