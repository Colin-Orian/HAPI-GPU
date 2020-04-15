#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <stdio.h>
#include <thrust/complex.h>
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

//output buffers
rtBuffer<Angle, 2> result_buffer;
rtBuffer<int, 2> intersect_count;


//input buffers
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

RT_PROGRAM void rayGeneration() {
	double x = left + theLaunchIndex.x * sizex;
	if (diamond && ((theLaunchIndex.y % 2) == 0)) {
		x += sizex / 2.0f;
	}
	//If diamond is true and the j th index is even, add sizex / 2.0f to x. Turned it into a equation for better performance
	//x += (diamond) * (1 - (theLaunchIndex.y % 2)) * sizex / 2.0f; 

	double y = top - theLaunchIndex.y * sizey;

	Payload payload;
	payload.colour = make_float3(0.0f);	
	//Compute the reference of the sphere
	//NOTE: thrust::complex example(realNumber, ImagNumber);
	thrust::complex<double> finalComplex(0.0, 0.0);

	optix::Ray currentRay;
	currentRay.origin.x = x;
	currentRay.origin.y = y;
	currentRay.origin.z = -camDist;
	currentRay.tmax = RT_DEFAULT_MAX;
	float3 dirTotal = make_float3(0.0f);

	int numIntersect = 0;
	for (int i = 0; i < xMax; i++) { //xMax
		for (int j = 0; j < yMax; j++) { //yMax
			uint2 index;
			index.x = i;
			index.y = j;
			//int index = twoDtoOne(i, j);
			double dx = jitter_buffer[index].x;
			double dy = jitter_buffer[index].y;
			double dz = 1.0;
			double len = sqrt(dx*dx + dy * dy + dz * dz);

			currentRay.direction.x = dx / len;
			currentRay.direction.y = dy / len;
			currentRay.direction.z = dz / len;
			
			rtTrace(sysTopObject, currentRay, payload);
			if (payload.t >= 0.0) {
				
				payload.t -= camDist;
				
				thrust::complex<double> complexNum(0.0, (double)(k * payload.t));
				complexNum = thrust::exp<float>(complexNum);

				thrust::complex<double> complexColour = thrust::complex<double>((double)payload.colour.x, 0.0);
				complexNum = (complexColour * complexNum) / thrust::complex<double>((double) payload.t, 0.0);
				finalComplex += complexNum;
				
				numIntersect+=1;
			}
		}
	}
	
	result_buffer[theLaunchIndex].x = finalComplex.real();
	result_buffer[theLaunchIndex].y = finalComplex.imag();

	intersect_count[theLaunchIndex] = numIntersect;
}