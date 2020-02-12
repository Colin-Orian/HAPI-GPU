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

rtBuffer<float4, 2> result_buffer;
rtBuffer<Angle, 2> jitter_buffer;
rtBuffer<Angle, 2 > reference_buffer;

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

//static __device__ __inline__
//int twoDtoOne(int i, int j) {
	//return j + yMax * i;
//}

/*
int i, j;
double x, y;
double xx;

for (i = 0; i < sx; i++) {
	x = i * dx + ox;
	xx = x * nx;
	for (j = 0; j < sy; j++) {
		y = j * dy + oy;
		reference[i][j] = exp(1i*k*(xx + y * ny + nz * z));
	}
}
*/
static __device__ __inline__
thrust::complex<float> computeReference() {
	double nx = 0.1;
	double ny = 0.1;
	double nz = 0.95;
	double z = 100000.0;
	float3 pos;
	pos.x = theLaunchIndex.x * sizex + left;
	pos.y = theLaunchIndex.y * sizey - top;
	pos.z = z;
	float3 nVal;
	nVal.x = nx;
	nVal.y = ny;
	nVal.z = nz;
	thrust::complex<float> number(0.0f,k * optix::dot(pos, nVal));
	return thrust::exp<float>(number);
}

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
	float3 resultColour;
	
	//Compute the reference of the sphere
	thrust::complex<double> reference((float)reference_buffer[theLaunchIndex].x, (float)reference_buffer[theLaunchIndex].y);
	//NOTE: thrust::complex example(realNumber, ImagNumber);
	thrust::complex<double> finalComplex(0.0, 0.0);

	optix::Ray currentRay;
	currentRay.origin.x = x;
	currentRay.origin.y = y;
	currentRay.origin.z = -camDist;
	currentRay.tmax = RT_DEFAULT_MAX;
	float3 dirTotal = make_float3(0.0f);

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

			
			
			currentRay.direction.x = dx / len;
			currentRay.direction.y = dy / len;
			currentRay.direction.z = dz / len;
			
			rtTrace(sysTopObject, currentRay, payload);
			if (payload.t >= 0.0) {
				
				payload.t -= camDist;
				thrust::complex<float> complexNum(0.0, (float)(k * payload.t));
				complexNum = thrust::exp<float>(complexNum);
				complexNum *= thrust::complex<float>((float)payload.colour.x,0.0);
				complexNum /= thrust::complex <float> ((float)payload.t, 0.0);
				
				finalComplex += complexNum;
				
			}
		}
	}
	
	finalComplex *= thrust::conj<float>(reference);
	resultColour = make_float3((float)thrust::abs(finalComplex));
	result_buffer[theLaunchIndex] = make_float4(resultColour.x,resultColour.y,resultColour.z, 1.0f);
	
}