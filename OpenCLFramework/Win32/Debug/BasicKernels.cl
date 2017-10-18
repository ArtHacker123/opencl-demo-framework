
#define IDX3(x, y, c, w, h) (x)+((y)*w)+((c)*w*h)



__kernel void gamma_correction(
   __global float* input, __global float* output,
   float gamma, int w,
   int h, int nc)
{
	int x = get_global_id(0);
	int yt = get_global_id(1);
	int y = yt%h;
	int c = yt/h;
	int i = IDX3(x,y,c,w,h);
	//int i = get_global_id(0);
	if ((x < w) && (y<h)) output[i] = pow(input[i], gamma);
	//if ((x < w) && (y<h)) output[i] = input[i];
	//if (i<1000) output[i] = pow(input[i], gamma);
    //output[i] = input[i];
}

__kernel void gradient(
   __global float* input, __global float* outputX, __global float* outputY,
   int w, int h, int nc)
{
	int x = get_global_id(0);
	int yt = get_global_id(1);
	int y = yt%h;
	int c = yt/h;
	int i = IDX3(x,y,c,w,h);
	//todo add if (x+1 = w && y < h) output[i] = 0;
	if (x < w && y < h)	{		outputX[i] = (x+1<w) ? (input[IDX3(x+1,y,c,w,h)] - input[i]) : 0;		outputY[i] = (y+1<h) ? (input[IDX3(x,y+1,c,w,h)] - input[i]) : 0;	} 
}

__kernel void divergence(
   __global float* inputX, __global float* inputY, __global float* output,
   int w, int h, int nc)
{
	int x = get_global_id(0);
	int yt = get_global_id(1);
	int y = yt%h;
	int c = yt/h;
	int i = IDX3(x,y,c,w,h);
	if (x < w && y < h)
	{
		output[i] = (x>0) ? (inputX[i] - inputX[IDX3(x-1,y,c,w,h)]) : 0;
		output[i] += (y>0) ? (inputY[i] - inputY[IDX3(x,y-1,c,w,h)]) : 0;
	}

}