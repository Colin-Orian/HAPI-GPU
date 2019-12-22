#include <optix.h>
#include <optixu/optixu_math_namespace.h>
struct Angle {
	double x;
	double y;
};
rtBuffer<float4, 2> result_buffer;
rtBuffer<Angle, 1> jitter_buffer;
rtDeclareVariable(uint2, theLaunchDim, rtLaunchDim, );
rtDeclareVariable(uint2, theLaunchIndex, rtLaunchIndex, );

rtDeclareVariable(rtObject, sysTopObject, , );

rtDeclareVariable(float, left, ,);
rtDeclareVariable(float, top, ,);
rtDeclareVariable(float, sizex, ,);
rtDeclareVariable(float, sizey, ,);
rtDeclareVariable(float, camDist, , );
rtDeclareVariable(int, diamond, ,);


RT_PROGRAM void rayGeneration() {
	double x = left + theLaunchIndex.x * sizex;
	//If diamond is true and the j th index is even, add sizex / 2.0f to x. Turned it into a equation for better performance
	x += (diamond) * (1 - theLaunchIndex.y % 2) * sizex / 2.0f; //modoluo arithimitic is slow on GPUs? Check later

	double y = top - theLaunchIndex.y * sizey;
	optix::Ray currentRay;
	currentRay.origin.x;
	currentRay.origin.y = y;
	currentRay.origin.z = -camDist;
	
	result_buffer[theLaunchIndex] = make_float4(1.0f, 0.0f, 1.0f, 0.0f, 1.0f);
}