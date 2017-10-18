
#define IDX3(x, y, c, w, h) (x)+((y)*w)+((c)*w*h)

float get_mat_val(const float *src, int x, int y, int c, int w, int h);

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
	if ((x < w) && (y<h)) output[i] = pow(input[i], gamma);
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

__kernel void l2_norm(
   __global float* input, __global float* output,
   int w, int h, int nc)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int i;
	if (x < w && y < h)
	{
		float res = 0;
		for (int c=0; c<nc; ++c)
		{
			i = IDX3(x,y,c,w,h);
			res += input[i]*input[i];
		}
		output[i] = sqrt(res);
	}
}


__kernel void convolve(
   __global float* input, __global float* output,
   float gamma, int w,
   int h, int nc)
{
	int x = get_global_id(0);
	int yt = get_global_id(1);
	int y = yt%h;
	int c = yt/h;
	int i = IDX3(x,y,c,w,h);
	if ((x < w) && (y<h)) output[i] = pow(input[i], gamma);
}



float get_mat_val(const float *src, int x, int y, int c, int w, int h)
{
    int xt, yt;
    if (x<0) 
    {
        xt = 0;
    }
    else if (x>=w)
    {
        xt = w-1;
    }
    else
    {
        xt = x;
    }

    if (y<0) 
    {
        yt = 0;
    }
    else if (y>=h)
    {
        yt = h-1;
    }
    else
    {
        yt = y;
    }
    return src[IDX3(xt, yt, c, w, h)];
}