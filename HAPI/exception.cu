#include <optix.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <stdio.h>

struct Angle {
	double x;
	double y;
};

rtBuffer<Angle, 2> result_buffer;
rtDeclareVariable(uint2, theLaunchIndex, rtLaunchIndex, );

RT_PROGRAM void exception() {
	printf("Exception called on thread x = %u, y = %u! \n", theLaunchIndex.x, theLaunchIndex.y);
	rtPrintExceptionDetails();
	result_buffer[theLaunchIndex].x = 1.0;
	result_buffer[theLaunchIndex].y = 1.0;
}