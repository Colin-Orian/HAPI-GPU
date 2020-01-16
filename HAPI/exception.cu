#include <optix.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <stdio.h>

rtBuffer<float4, 2> result_buffer;
rtDeclareVariable(uint2, theLaunchIndex, rtLaunchIndex, );

RT_PROGRAM void exception() {
	printf("Exception called on thread x = %u, y = %u! \n", theLaunchIndex.x, theLaunchIndex.y);
	rtPrintExceptionDetails();
	result_buffer[theLaunchIndex] = make_float4(0.0f, 1.0f, 0.0f, 1.0f);
}