
#define IDX3(x, y, c, w, h) (x)+((y)*w)+((c)*w*h)
#define IDX2(x, y, w) (x)+((y)*(w))
#define COLOR_MIN 0.f
#define COLOR_MAX 255.f
#define DARKENING_FACTOR .5f

float get_mat_val(__global float *src, int x, int y, int c, int w, int h);
void eigen_values(const float *src, float *res);

__kernel void feature_detect(__global const float *src, __global const float *src11,
								__global const float *src12, __global const float *src22,
								__global float *dst, int w, int h, float alpha, float beta)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = IDX2(x, y, w);
    float eigenvals[2], tensor[4];
    if (x<w && y<h)
    {
        tensor[0] = src11[idx];
        tensor[1] = tensor[2] = src12[idx];
        tensor[3] = src22[idx];
        eigen_values(tensor, eigenvals);
        if (alpha<=eigenvals[0])//red= 255 - ((c+2)/3 * 255)
        {
            dst[IDX3(x, y, 0, w, h)] = COLOR_MAX;
            dst[IDX3(x, y, 1, w, h)] = COLOR_MIN;
            dst[IDX3(x, y, 2, w, h)] = COLOR_MIN;
        }
        else if (beta>=eigenvals[0] && alpha<=eigenvals[1])//yellow = ((4-c)/3)*255 assuming beta<alpha
        {
            dst[IDX3(x, y, 0, w, h)] = COLOR_MAX;
            dst[IDX3(x, y, 1, w, h)] = COLOR_MAX;
            dst[IDX3(x, y, 2, w, h)] = COLOR_MIN;
        }
        else
        {
            dst[IDX3(x, y, 0, w, h)] = src[IDX3(x, y, 0, w, h)]*DARKENING_FACTOR;
            dst[IDX3(x, y, 1, w, h)] = src[IDX3(x, y, 1, w, h)]*DARKENING_FACTOR;
            dst[IDX3(x, y, 2, w, h)] = src[IDX3(x, y, 2, w, h)]*DARKENING_FACTOR;
        }
    }
}

__kernel void pointwise_product(__global const float *srcA, __global const float *srcB,								__global float *dst, int w, int h, int nc)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    size_t idx = IDX2(x,y,w);//x+y*w
	dst[5]=7.8;
    if (x < w && y < h)
    {
		dst[idx] = 0;
        for (int c=0; c<nc; ++c) dst[idx] += srcA[IDX3(x,y,c,w,h)]*srcB[IDX3(x,y,c,w,h)];
        //dst[idx] = 0.3f;
    }
}

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
	if (x < w && y < h)	{		// naive version		//outputX[i] = (x+1<w) ? (input[IDX3(x+1,y,c,w,h)] - input[i]) : 0;		//outputY[i] = (y+1<h) ? (input[IDX3(x,y+1,c,w,h)] - input[i]) : 0;		// rotationally robust version		float p1, p2;		if ((x+1<w) && (y+1<h))		{			p1 = 3*input[IDX3(x+1,y+1,c,w,h)] + 10*input[IDX3(x+1,y,c,w,h)] + 3*input[IDX3(x+1,y-1,c,w,h)];
			p2 = 3*input[IDX3(x-1,y+1,c,w,h)] + 10*input[IDX3(x-1,y,c,w,h)] + 3*input[IDX3(x-1,y-1,c,w,h)];
			outputX[i] = 0.03125f*(p1 - p2);
			p1 = 3*input[IDX3(x+1,y+1,c,w,h)] + 10*input[IDX3(x,y+1,c,w,h)] + 3*input[IDX3(x-1,y+1,c,w,h)];
			p2 = 3*input[IDX3(x+1,y-1,c,w,h)] + 10*input[IDX3(x,y-1,c,w,h)] + 3*input[IDX3(x-1,y-1,c,w,h)];
			outputY[i] = 0.03125f*(p1 - p2);		}		else		{			outputX[i] = outputY[i] = 0;		}	} 
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
		output[IDX2(x,y,w)] = sqrt(res);
	}
}


__kernel void convolve(__global float* input, __global float* output,
						__global float *ker, int w, int h, int r)
{
	int x = get_global_id(0);
    int yt = get_global_id(1);
    int y = yt%h;
    int c = yt/h;
    size_t idx = IDX3(x,y,c,w,h);
    int kerW = (2*r) + 1;

	if (x < w && y < h)
	{
		output[idx] = 0;
		for (int a = -r; a <= r; ++a)
		{
			for (int b = -r; b <= r; ++b)
			{
				output[idx] += ker[(a+r)+((b+r)*kerW)]*get_mat_val(input, x-a, y-b, c, w, h);
			}
		}
	}
    
}



float get_mat_val(__global float *src, int x, int y, int c, int w, int h)
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


void eigen_values(const float *src, float *res)
{
    float t = src[0]+src[3];
    float d = (src[0]*src[3]) - (src[1]*src[2]);
    float p = sqrt(((t*t)/4.f)-d);
    res[0] = 0.5*t-p;
    res[1] = 0.5*t+p;
}