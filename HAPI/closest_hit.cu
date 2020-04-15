#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <stdio.h>
struct Payload {
	float3 colour;
	double diff;
	double t;
};

rtDeclareVariable(Payload, payload, rtPayload, );
rtDeclareVariable(double, diff, attribute diff, ); //Where on the object does the ray intersect?
rtDeclareVariable(uint2, theLaunchIndex, rtLaunchIndex, );
rtDeclareVariable(double, colour, attribute colour, );
//On the closest hit, make the pixel red
RT_PROGRAM void closestHit() {
	double ambient = 0.2;
	double diffuse = 0.8;
	double result = diff * diffuse * colour + ambient;


	payload.colour = make_float3(result);
}