#include "FaceTrackingDemo.h"



void FaceTrackingDemo::load_parameters(const Parameters &params)
{
	try
	{
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

void FaceTrackingDemo::init_parameters(Parameters &params)
{
	params_ = &params;


	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	if (!face_cascade_.load(face_cascade_name))
	{
		cout << ("--(!)Error loading face cascade\n") << endl; exit(1);
	}
}

void FaceTrackingDemo::compile_program(OpenCLBasic *oclobjects)
{
	oclobjects_ = oclobjects;
	// create program
	oprogram_ = new OpenCLProgramMultipleKernels(*oclobjects_, L"BasicKernels.cl", "");
}

void FaceTrackingDemo::init_program_args(const float *input, int width,
	int height, int nchannels, size_t nbytesI)
{
	w_ = width;
	h_ = height;
	nc_ = nchannels;
	numberOfValues_ = w_*h_;
	h_in = input;
	cl_int result;
	
}

void FaceTrackingDemo::execute_program()
{
	Mat mOut(h_, w_, GET_TYPE(gray_));
	convert_layered_to_mat(mOut, h_in);
	vector<Rect> faces;
	Mat frame_gray;
	Scalar col;
	if (!gray_)
	{
		cvtColor(mOut, frame_gray, cv::COLOR_RGB2GRAY);
		frame_gray *= 255.f;
		col = Scalar(0, 255, 255);
	}
	else
	{
		frame_gray = Mat(mOut);
		col = Scalar(255);
		//frame_gray *= 255.f;
	}
	frame_gray.convertTo(frame_gray, CV_8UC1);
	equalizeHist(frame_gray, frame_gray);
	face_cascade_.detectMultiScale(frame_gray, faces, 1.1, 2, 0, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(mOut, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, col, 4, 8, 0);
	}

	h_out = new float[w_*h_*nc_];
	convert_mat_to_layered(h_out, mOut);

}

void FaceTrackingDemo::display_output()
{
	Mat mOut(h_, w_, GET_TYPE(gray_)), mIn(h_, w_, GET_TYPE(gray_));
	convert_layered_to_mat(mOut, h_out);
	convert_layered_to_mat(mIn, h_in);
	showImage("Input", mIn, 100, 100);
	showImage("Face Tracking", mOut, 100 + w_ + 40, 100);
}

void FaceTrackingDemo::deinit_program_args()
{
	delete[] h_out;

}

void FaceTrackingDemo::deinit_parameters()
{

	delete oprogram_;
}
FaceTrackingDemo::~FaceTrackingDemo()
{
	deinit_parameters();
}