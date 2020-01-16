#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <stdio.h>
#include <complex>
struct Angle {
	double x;
	double y;
};
struct Point {
	double x;
	double y;
	double z;
};

struct Payload {
	float3 colour;
	double diff;
	double t;

};

rtBuffer<float4, 2> result_buffer;
rtBuffer<Angle, 2> jitter_buffer;

rtDeclareVariable(uint2, theLaunchDim, rtLaunchDim, );
rtDeclareVariable(uint2, theLaunchIndex, rtLaunchIndex, );

rtDeclareVariable(rtObject, sysTopObject, , );

rtDeclareVariable(float, left, ,);
rtDeclareVariable(float, top, ,);
rtDeclareVariable(float, sizex, ,);
rtDeclareVariable(float, sizey, ,);
rtDeclareVariable(float, camDist, , );
rtDeclareVariable(int, diamond, ,);
rtDeclareVariable(int, xMax, , );
rtDeclareVariable(int, yMax, , );
rtDeclareVariable(int, checkX, , );
rtDeclareVariable(int, checkY, , );
rtDeclareVariable(float, k, , );
static __device__ __inline__
int twoDtoOne(int i, int j) {
	return j + yMax * i;
}
static __device__ __inline__
void printRay(optix::Ray ray) {
	printf("ox = %f, oy = %f, oz = %f, dx = %f, dy = %f, dz = %f\n",ray.origin.x, ray.origin.y, ray.origin.z,  ray.direction.x, ray.direction.y, ray.direction.z);
	
}

RT_PROGRAM void rayGeneration() {
	double x = left + theLaunchIndex.x * sizex;
	if (diamond && (theLaunchIndex.y % 2) == 0) {
		x += sizex / 2.0f;
	}
	//If diamond is true and the j th index is even, add sizex / 2.0f to x. Turned it into a equation for better performance
	//x += (diamond) * (1 - (theLaunchIndex.y % 2)) * sizex / 2.0f; 

	double y = top - theLaunchIndex.y * sizey;

	Payload payload;
	payload.colour = make_float3(0.0f);
	int checkIndex = twoDtoOne(checkX, checkY);	
	float3 resultColour;
	for (int i = 0; i < xMax; i++) {
		for (int j = 0; j < yMax; j++) {
			uint2 index;
			index.x = i;
			index.y = j;
			//int index = twoDtoOne(i, j);
			double dx = jitter_buffer[index].x;
			double dy = jitter_buffer[index].y;
			double dz = 1.0;
			double len = sqrt(dx*dx + dy * dy + dz * dz);

			optix::Ray currentRay;
			
			currentRay.origin.x = x;
			currentRay.origin.y = y;
			currentRay.origin.z = -camDist;
			currentRay.direction.x = dx / len;
			currentRay.direction.y = dy / len;
			currentRay.direction.z = dz / len;
			currentRay.tmax = RT_DEFAULT_MAX;
			rtTrace(sysTopObject, currentRay, payload);
			
			resultColour = payload.colour * exp(1 * k * payload.t) / payload.t;
			//payload.colour = make_float3(0.0f);
		}
	}
	result_buffer[theLaunchIndex] = make_float4(resultColour.x,resultColour.y,resultColour.z, 1.0f);
}