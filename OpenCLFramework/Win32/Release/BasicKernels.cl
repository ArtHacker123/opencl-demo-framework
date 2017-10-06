



__kernel void gamma_correction(
   __global float* input,
   __global float* output,
   float gamma,
   const unsigned int count)
{
   int i = get_global_id(0);
   if(i < count)
       output[i] = pow(input[i], gamma);
       //output[i] = input[i];
}