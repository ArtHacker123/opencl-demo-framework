
#ifndef _HELPER_FOR_CL_CV_STUFF
#define _HELPER_FOR_CL_CV_STUFF

#include <string>
#include <sstream>

using namespace std;

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

// parameter processing: template specialization for T=bool
template<>
bool getParam<bool>(std::string param, bool &var, int argc, char **argv)
{
	const char *c_param = param.c_str();
	for (int i = argc - 1; i >= 1; i--)
	{
		if (argv[i][0] != '-') continue;
		if (strcmp(argv[i] + 1, c_param) == 0)
		{
			if (!(i + 1<argc) || argv[i + 1][0] == '-') { var = true; return true; }
			std::stringstream ss;
			ss << argv[i + 1];
			ss >> var;
			return (bool)ss;
		}
	}
	return false;
}

// opencv helpers
void convert_layered_to_interleaved(float *aOut, const float *aIn, int w, int h, int nc)
{
	if (nc == 1) { memcpy(aOut, aIn, w*h*sizeof(float)); return; }
	size_t nOmega = (size_t)w*h;
	for (int y = 0; y<h; y++)
	{
		for (int x = 0; x<w; x++)
		{
			for (int c = 0; c<nc; c++)
			{
				aOut[(nc - 1 - c) + nc*(x + (size_t)w*y)] = aIn[x + (size_t)w*y + nOmega*c];
			}
		}
	}
}
void convert_layered_to_mat(cv::Mat &mOut, const float *aIn)
{
	convert_layered_to_interleaved((float*)mOut.data, aIn, mOut.cols, mOut.rows, mOut.channels());
}
void convert_interleaved_to_layered(float *aOut, const float *aIn, int w, int h, int nc)
{
	if (nc == 1) { memcpy(aOut, aIn, w*h*sizeof(float)); return; }
	size_t nOmega = (size_t)w*h;
	for (int y = 0; y<h; y++)
	{		for (int x = 0; x<w; x++)
		{
			for (int c = 0; c<nc; c++)
			{
				aOut[x + (size_t)w*y + nOmega*c] = aIn[(nc - 1 - c) + nc*(x + (size_t)w*y)];
			}
		}
	}
}
void convert_mat_to_layered(float *aOut, const cv::Mat &mIn)
{
	convert_interleaved_to_layered(aOut, (float*)mIn.data, mIn.cols, mIn.rows, mIn.channels());
}
void showImage(string title, const cv::Mat &mat, int x, int y)
{
	const char *wTitle = title.c_str();
	cv::namedWindow(wTitle, 1);
	cvMoveWindow(wTitle, x, y);
	cv::imshow(wTitle, mat);
}


#endif  // end of include guard
