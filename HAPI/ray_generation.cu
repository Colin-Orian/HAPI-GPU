#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <stdio.h>
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
rtDeclareVariable(int, xMax, , );
rtDeclareVariable(int, yMax, , );

/*
CODE I WANT TO PUT ON GPU
ray.ox = x;
ray.oy = y;
ray.oz = -d;
for (m = 0; m < xMax; m++) {
	for (n = 0; n < yMax; n++) {
		dx = samples[n][m].x;
		dy = samples[n][m].y;
		dz = 1.0;
		len = sqrt(dx*dx + dy*dy + dz*dz);
		ray.dx = dx / len;
		ray.dy = dy / len;
		ray.dz = dz / len;
		t = trace(node, &ray, colour); //Does the ray hit?

		if (t < 0) {
			continue;
		}
		intersections++;
		t -= d;
		object[i][j] += colour*exp(1i*k*t)/t;
	}
}
*/

RT_PROGRAM void rayGeneration() {
	double x = left + theLaunchIndex.x * sizex;
	//If diamond is true and the j th index is even, add sizex / 2.0f to x. Turned it into a equation for better performance
	x += (diamond) * (1 - theLaunchIndex.y % 2) * sizex / 2.0f; //modoluo arithimitic is slow on GPUs? Check later

	double y = top - theLaunchIndex.y * sizey;
	optix::Ray currentRay;
	currentRay.origin.x = x;
	currentRay.origin.y = y;
	currentRay.origin.z = -camDist;
	double len;
	for (int i = 0; i < xMax; i++) {
		for (int j = 0; j < yMax; j++) {
			int index = i + yMax * j;
			
			double dx = jitter_buffer[index].x;
			double dy = jitter_buffer[index].y;
			double dz = 1.0f;
			len = sqrt(dx*dx + dy * dy + dz * dz);
			currentRay.direction.x = dx / len;
			currentRay.direction.y = dy / len;
			currentRay.direction.z = dz / len;
			if (theLaunchIndex.x == 1 && theLaunchIndex.y == 1) {
				rtPrintf("index = [%i][%i] dx =%f dy =%f\n", i, j, dx, dy);
			}
			
		}
	}
	result_buffer[theLaunchIndex] = make_float4(abs(currentRay.origin.x), abs(currentRay.origin.y), currentRay.origin.z, 1.0f);
}