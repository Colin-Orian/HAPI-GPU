#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <curand_kernel.h> //created uniform numbers https://stackoverflow.com/questions/24537112/uniformly-distributed-pseudorandom-integers-inside-cuda-kernel
rtBuffer<float4, 2> result_buffer;

rtDeclareVariable(uint2, theLaunchDim, rtLaunchDim, );
rtDeclareVariable(uint2, theLaunchIndex, rtLaunchIndex, );

rtDeclareVariable(rtObject, sysTopObject, , );

rtDeclareVariable(float, left, ,);
rtDeclareVariable(float, top, ,);
rtDeclareVariable(float, sizex, ,);
rtDeclareVariable(float, sizey, ,);
rtDeclareVariable(float, camDist, , );
rtDeclareVariable(int, diamond, ,);

void sampleJitter(int i, int j) {
//	curand_init();
}

RT_PROGRAM void rayGeneration() {
	double x = left + theLaunchIndex.x * sizex;
	//If diamond is true and the j th index is even, add sizex / 2.0f to x. Turned it into a equation for better performance
	x += (diamond) * (1 - theLaunchIndex.y % 2) * sizex / 2.0f; //modoluo arithimitic is slow on GPUs? Check later

	double y = top - theLaunchIndex.y * sizey;
	optix::Ray currentRay;
	currentRay.origin.x;
	currentRay.origin.y = y;
	currentRay.origin.z = -camDist;

	result_buffer[theLaunchIndex] = make_float4(theLaunchIndex.x, theLaunchIndex.y, 0.0f, 1.0f);
}