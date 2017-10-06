
#ifndef _HELPER_FOR_CL_CV_STUFF
#define _HELPER_FOR_CL_CV_STUFF

#include <string>
#include <sstream>
#include <ctime>
#include <CL/cl.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;

// parameter processing
template<typename T>
bool getParam(std::string param, T &var, int argc, char **argv);

// parameter processing: template specialization for T=bool
template<>
bool getParam<bool>(std::string param, bool &var, int argc, char **argv);

// opencv helpers
void convert_layered_to_interleaved(float *aOut, const float *aIn, int w, int h, int nc);
void convert_layered_to_mat(cv::Mat &mOut, const float *aIn);
void convert_interleaved_to_layered(float *aOut, const float *aIn, int w, int h, int nc);
void convert_mat_to_layered(float *aOut, const cv::Mat &mIn);
void showImage(string title, const cv::Mat &mat, int x, int y);


// parameter processing
template<typename T>
bool getParam(std::string param, T &var, int argc, char **argv)
{
	const char *c_param = param.c_str();
	for (int i = argc - 1; i >= 1; i--)
	{
		if (argv[i][0] != '-') continue;
		if (strcmp(argv[i] + 1, c_param) == 0)
		{
			if (!(i + 1<argc)) continue;
			std::stringstream ss;
			ss << argv[i + 1];
			ss >> var;
			return (bool)ss;
		}
	}
	return false;
}

const char *getErrorString(cl_int error);

#endif  // end of include guard
