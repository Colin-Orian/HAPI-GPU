#include <optix.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <stdio.h>
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

rtBuffer<Point> pointBuffer;
rtBuffer<double> radiusBuffer;
rtDeclareVariable(Payload, payload, rtPayload, );
rtDeclareVariable(optix::Ray, currentRay, rtCurrentRay, );
rtDeclareVariable(double, diff, attribute diff, ); //Where on the object does the ray intersect?
rtDeclareVariable(uint2, theLaunchIndex, rtLaunchIndex, );

RT_PROGRAM void intersect_sphere(int primitiveIndex) {
	//Sphere intersection
	Point point = pointBuffer[primitiveIndex];
	double radius = radiusBuffer[primitiveIndex];
	//Quadratic equation
	double A = 1.0;
	double B;
	double C;
	float3 ec;
	
	//Calculate the discriminant of the quadratic equation
	ec.x = currentRay.origin.x - point.x;
	ec.y = currentRay.origin.y - point.y;
	ec.z = currentRay.origin.z - point.z;
	B = 2 * (currentRay.direction.x * ec.x + currentRay.direction.y * ec.y + currentRay.direction.z * ec.z);
	C = ec.x * ec.x + ec.y * ec.y + ec.z * ec.z - radius * radius;
	double discriminant = B * B - 4.0 * A * C;
	//FIX ME. GET RID OF THE IF STATEMENTS
	double finalT = -1.0;
	if(discriminant >= 0) { //The ray hit the sphere
		discriminant = sqrt(discriminant);
		//two roots of the quadratic equation
		double t1 = (-B - discriminant) / (2 * A);
		double t2 = (-B + discriminant) / (2 * A);
		if (t1 > 0.0) {
			finalT = t1;
		}
		else if (t2 > 0.0) {
			finalT = t2;
		}
		if (rtPotentialIntersection(finalT)) {
			payload.t = finalT;
			double lx = 0.0;
			double ly = 0.0;
			double lz = -0.5;
			double len = sqrt(lx*lx + ly * ly + lz * lz);
			lx /= len;
			ly /= len;
			lz /= len;
			double nx = currentRay.origin.x + finalT * currentRay.direction.x - point.x;
			double ny = currentRay.origin.y + finalT * currentRay.direction.y - point.y;
			double nz = currentRay.origin.z + finalT * currentRay.direction.z - point.z;
			len = sqrt(nx * nx + ny * ny + nz * nz);
			nx /= len;
			ny /= len;
			nz /= len;
			diff = nx * lx + ny * ly + nz * lz;
			if (diff < 0.0) {
				diff = 0.0;
			}
			rtReportIntersection(0);
		}
	}

	/*
	nx = ray->ox + t * ray->dx - plist[current]->x; //Where on the ray is the intersect?
	ny = ray->oy + t * ray->dy - plist[current]->y;
	nz = ray->oz + t * ray->dz - plist[current]->z;
	len = sqrt(nx*nx + ny * ny + nz * nz);
	nx /= len;
	ny /= len;
	nz /= len;
	*/
	
}	