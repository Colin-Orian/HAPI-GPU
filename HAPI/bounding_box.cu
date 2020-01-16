#include <optix.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <stdio.h>
struct Point {
	double x;
	double y;
	double z;
};
rtBuffer<Point> pointBuffer;
rtBuffer<double> radiusBuffer;

RT_PROGRAM void boundbox_sphere(int primiativeIndex, float result[6]) {	
	
	optix::Aabb *aabb = (optix::Aabb *) result;
	Point point = pointBuffer[primiativeIndex];
	double radius = radiusBuffer[primiativeIndex];
	const float size = 1000000.0f;
	if (radius > 0.0 && !isinf(radius)) {
		//printf("x = %f, y = %f, z = %f, r = %f\n", point.x, point.y, point.z, radius);
		//aabb->m_min = make_float3(point.x - radius, point.y - radius, point.z - radius);
		//aabb->m_max = make_float3(point.x + radius, point.y + radius, point.z + radius);
		aabb->m_min = make_float3(-size, -size, -size);
		aabb->m_max = make_float3(size, size, size);
	}
	else {
		aabb->invalidate();
	}
	
}