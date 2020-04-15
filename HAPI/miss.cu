#include <optix.h>
#include <optixu/optixu_math_namespace.h>
struct Payload {
	float3 colour;
	double diff;
	double t;
};

rtDeclareVariable(Payload, payload, rtPayload, );
//On a miss, make the pixel green
RT_PROGRAM void miss() {
	payload.colour += make_float3(0.0f, 0.0f, 0.0f);
}