#include <optix.h>
#include <optixu/optixu_math_namespace.h>

rtBuffer<float4, 2> result_buffer;

rtDeclareVariable(uint2 theLaunchIndex, rtLaunchIndex);
//On the closest hit, make the pixel red
RT_PROGRAM void closestHit() {
	result_buffer[theLaunchIndex] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
}