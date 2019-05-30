/**************************************************
 *
 *                 ComputeInterference.cpp
 *
 *  Compute the interference pattern based on the
 *  simple superposition of a set of point light
 *  sources.
 *
 ***************************************************/

#include "HAPI.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include "helper.h"

extern double pointDistance;

double sizex;	// horizontal distance between mirrors
double sizey;		// vertical distance between mirrors
int width;
int height;

#define KCOS 1000
double cosTable[KCOS + 1];
static int cosFlag = 0;
double cosScale;

void buildCos() {
	double theta;
	double dtheta;
	int i;

	printf("building cos table\n");

	theta = 0.0;
	dtheta = 2 * M_PI / KCOS;
	for (i = 0; i < KCOS; i++) {
		theta = i*dtheta;
		cosTable[i] = cos(theta);
	}
	cosFlag = 1;
	cosScale = KCOS / (2.0 * M_PI);
}

inline double myCos(double theta) {
	int i;

	if (theta < 0.0)
		theta += 2.0*M_PI;
	i = (int)(theta * cosScale);
	return(cosTable[i]);
}

double minSqrt = 10000000000.0;
double maxSqrt = 0.0;

double mySqrt(double x) {
	if (x < minSqrt)
		minSqrt = x;
	if (x > maxSqrt)
		maxSqrt = x;
	return(sqrt(x));
}

void computeInterferenceSP(GeometryNode *node, InterferencePattern *pattern, double lambda, double xr, double yr, double zr);
void computeInterferenceRS(GeometryNode *node, InterferencePattern *pattern, double lambda, double xr, double yr, double zr);
void computeInterferenceRay(GeometryNode *node, InterferencePattern *pattern, double lambda, double xr, double yr, double zr);
void computeInterferenceP(GeometryNode *node, InterferencePattern *pattern, double lambda, double xr, double yr, double zr);

void computeInterference(GeometryNode *node, InterferencePattern *pattern, double lambda, double xr, double yr, double zr) {
	int alg;

	alg = getAlgorithm();
	switch(alg) {
	case SIMPLE_POINT:
		computeInterferenceSP(node, pattern, lambda, xr, yr, zr);
		break;
	case RAINBOW_SLIT:
		computeInterferenceRS(node, pattern, lambda, xr, yr, zr);
		break;
	case RAY_TRACE:
		computeInterferenceRay(node, pattern, lambda, xr, yr, zr);
		break;
	case COMPLEX_POINT:
		computeInterferenceP(node, pattern, lambda, xr, yr, zr);
		break;
	default:
		printf("unrecognized algorithm: %d\n", alg);
		abort();
	}
}


void computeInterferenceSP(GeometryNode *node, InterferencePattern *pattern, double lambda, double xr, double yr, double zr) {
	double left;
	double top;
	int i,j;
	double x, y;
	double k;
	double intens;
	int p;
	double dx, dy;
	double theta;
	int k1;
	std::vector<Point *> plist;
	size_t n;
	Point *point;
	double etaMax;
	double thetaMax;
	double tanEta;
	double tanTheta;
	double fz;
	int count;
	Device *device;
	int diamond;

	printf("\nCompute Interference\n");

	device = pattern->getDevice();
	sizex = device->sizeWide();
	sizey = device->sizeHeight();
	width = device->pixelsWidth();
	height = device->pixelsHeight();
	diamond = (int) device->diamond();

	if(cosFlag == 0)
		buildCos();

	count = 0;

	etaMax = asin(lambda/(2*sizex));
	thetaMax = asin(lambda/(2*sizey));
	tanEta = tan(etaMax/2);
	tanTheta = tan(thetaMax/2);
	printf("angles: %f %f\n",etaMax/M_PI*180.0, thetaMax/M_PI*180.0);
	printf("tan: %f %f\n",tanEta, tanTheta);
	printf("Lambda: %f\n", lambda);

	left = -width/2*sizex;
	top = height/2*sizey;
	printf("%f %f\n", left, top);
	k = 2*M_PI/lambda;
	printf("k: %f\n", k);
	plist = node->getLightPoints();
	n = plist.size();
	printf("n: %zd\n", n);
	for(i=0; i<width; i++ ) {
		/*
		if(i % 10 == 0)
			printf("Line %d\n",i);
			*/
		for(j=0; j<height; j++) {
			x = left + i*sizex;
			if(diamond && ((j % 2) == 0))
				x += sizex/2.0;;
			y = top - j*sizey;
			/*
			 *  diffraction pattern computation
			 */

			intens = 0;
			for(p=0; p<n; p++) {
				point = plist[p];
				fz = fabs(point->z);
				
				if(x < point->x - fz*tanEta) {
					count++;
					continue;
				}
				if(x > point->x + fz*tanEta) {
					count++;
					continue;
				}
				if(y < point->y - fz*tanTheta) {
					count++;
					continue;
				}
				if(y > point->y + fz*tanTheta) {
					count++;
					continue;
				}
				
				dx = x - point->x;
				dy = y - point->y;
				theta = k*mySqrt(dx*dx + dy*dy + point->z*point->z);
				dx = x - xr;
				dy = y - yr;
				theta -= k*mySqrt(dx*dx + dy*dy +zr*zr);
				if((theta > 2*M_PI) || (theta < -2*M_PI)) {
					k1 = (int) (theta/(2*M_PI));
					theta -= k1*2*M_PI;
				}
//				intens += myCos(theta);
				intens += cos(theta);
			}

			pattern->add(i, j, intens);
		}
	}
	printf("count: %d %zd\n",count, height*width*n);
	printf("sqrt: %f %f\n", minSqrt, maxSqrt);
}

void computeInterferenceRS(GeometryNode *node, InterferencePattern *pattern, double lambda, double xr, double yr, double zr) {
	double left;
	double top;
	int i,j;
	double x, y;
	double k;
	double intens;
	size_t n;
	int p;
	std::vector<Point *> plist;
	Point *point;
	double thetaRef;
	double RRef, IRef;
	double RO, IO;
	double r;
	double dx;
	double dy;

	left = -width/2*sizex;
	top = height/2*sizey;
	k = 2*M_PI/lambda;
	plist = node->getLightPoints();
	n = plist.size();
	thetaRef = 0.8;  // approximately 45 degrees
	dy = (pointDistance + sizey)/2.0;

	for(j=0; j<height; j++) {
		y = top - j*sizey;
		RRef = cos(k*y*sin(thetaRef));
		IRef = sin(k*y*sin(thetaRef));
		for(i=0; i<width; i++) {
			x = left + i*sizex;
			if((j % 2) == 0)
				x += sizex/2.0;

			intens = 0;
			for(p=0; p<n; p++) {
				point = plist[p];
				if((point->y < y-dy) || (point->y > y+dy))
					continue;
				dx = point->x - x;
				r = sqrt(dx*dx + point->z*point->z);
				RO = cos(k*r)/r;
				IO = sin(k*r)/r;
				intens += RO*RRef + IO*IRef;
			}
			pattern->add(i, j, intens);
		}
	}
}

#include <complex>
using namespace std::complex_literals;

#define nMax 2028
#define mMax 2048
std::complex<double> reference[nMax][mMax];
std::complex<double> object[nMax][mMax];

void computeReference(int sx, int sy, double dx, double dy, double ox, double oy, double nx, double ny, double nz, double z, double k) {
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
}

void computeInterferenceP(GeometryNode *node, InterferencePattern *pattern, double lambda, double xr, double yr, double zr) {
	double left;
	double top;
	int i, j;
	double x, y;
	double k;
	double intens;
	int p;
	double dx, dy;
	double theta;
	int k1;
	std::vector<Point *> plist;
	size_t n;
	Point *point;
	double etaMax;
	double thetaMax;
	double tanEta;
	double tanTheta;
	double fz;
	int count;
	Device *device;
	int diamond;
	double rr;

	printf("\nCompute Interference\n");

	device = pattern->getDevice();
	sizex = device->sizeWide();
	sizey = device->sizeHeight();
	width = device->pixelsWidth();
	height = device->pixelsHeight();
	diamond = (int)device->diamond();

	count = 0;

	etaMax = asin(lambda / (2 * sizex));
	thetaMax = asin(lambda / (2 * sizey));
	tanEta = tan(etaMax / 2);
	tanTheta = tan(thetaMax / 2);
	printf("angles: %f %f\n", etaMax / M_PI * 180.0, thetaMax / M_PI * 180.0);
	printf("tan: %f %f\n", tanEta, tanTheta);
	printf("Lambda: %f\n", lambda);

	left = -width / 2 * sizex;
	top = height / 2 * sizey;
	printf("%f %f\n", left, top);
	k = 2 * M_PI / lambda;
	printf("k: %f\n", k);
	plist = node->getLightPoints();
	n = plist.size();

	computeReference(width, height, sizex, sizey, left, -top, 0.1, 0.1, 0.95, 10000.0, k);

	printf("n: %zd\n", n);
	for (i = 0; i < width; i++)
		for (j = 0; j < height; j++)
			object[i][j] = 0.0;

	for (i = 0; i < width; i++) {
		if(i % 10 == 0)
			printf("Line %d\n",i);
		for (j = 0; j < height; j++) {
			x = left + i * sizex;
			if (diamond && ((j % 2) == 0))
				x += sizex / 2.0;;
			y = top - j * sizey;
			/*
			 *  diffraction pattern computation
			 */

			for (p = 0; p < n; p++) {
				point = plist[p];
				fz = fabs(point->z);

				/*
				if (x < point->x - fz * tanEta) {
					count++;
					continue;
				}
				if (x > point->x + fz * tanEta) {
					count++;
					continue;
				}
				if (y < point->y - fz * tanTheta) {
					count++;
					continue;
				}
				if (y > point->y + fz * tanTheta) {
					count++;
					continue;
				}
				*/

				dx = x - point->x;
				dy = y - point->y;
				rr = sqrt(dx*dx + dy * dy + point->z*point->z);
				object[i][j] += exp(1i*k*rr)/rr ;
			}
		}
	}
	for (i = 0; i < width; i++) {
		for (j = 0; j < height; j++) {
			intens = abs(object[i][j] * conj(reference[i][j]));
			pattern->add(i, j, intens);
		}
	}
	printf("count: %d %zd\n", count, height*width*n);
	printf("sqrt: %f %f\n", minSqrt, maxSqrt);
}


struct Ray {
	double ox, oy, oz;
	double dx, dy, dz;
};

double sphereIntersect(double cx, double cy, double cz, double r, struct Ray *ray) {
	double A, B, C;
	double ec[3];
	double disc;
	double t1, t2;

	
//		printf("Sphere: %f %f %f %f\n", cx, cy, cz, r);
//		printf("Ray: %f %f %f %f %f %f\n", ray->ox, ray->oy, ray->oz, ray->dx, ray->dy, ray->dz);

//	A = ray->dx*ray->dx + ray->dy*ray->dy + ray->dz*ray->dz;
	A = 1.0;
	ec[0] = ray->ox - cx;
	ec[1] = ray->oy - cy;
	ec[2] = ray->oz - cz;
	B = 2 * (ray->dx*ec[0] + ray->dy*ec[1] + ray->dz*ec[2]);
	C = ec[0] * ec[0] + ec[1] * ec[1] + ec[2] * ec[2] - r*r;

	disc = B*B - 4.0*A*C;
	
//		printf("%f %f %f %f\n", A, B, C, disc);
	
	if (disc < 0)
		return(-1.0);
	disc = sqrt(disc);
	t1 = (-B - disc) / (2 * A);
	t2 = (-B + disc) / (2 * A);
	/*
		printf("%f %f\n", t1, t2);
		printf("%f %f %f\n", ray->ox + t1*ray->dx, ray->oy + t1*ray->dy, ray->oz + t1*ray->dz);
		printf("Ray: %f %f %f %f %f %f\n\n", ray->ox, ray->oy, ray->oz, ray->dx, ray->dy, ray->dz);
		*/
	if (t1 > 0.0) {
		return(t1);
	}
	else if (t2 > 0.0) {
		return(t2);
	}
	else {
		return(-1.0);
	}
}

double trace(GeometryNode *node, struct Ray *ray, double &colour) {
	int n;
	std::vector<Point*> plist;
	std::vector<double> rlist;
	std::vector<double> clist;
	double t;
	int current;
	int i;
	double d;
	double len;
	double nx, ny, nz;
	double lx, ly, lz;
	double ambient = 0.2;
	double diffuse = 0.8;
	double diff;

	t = 1E12;
	current = -1;
	colour = 0.0;

	lx = 0.0;
	ly = 0.0;
	lz = -0.5;
	len = sqrt(lx*lx + ly * ly + lz * lz);
	lx /= len;
	ly /= len;
	lz /= len;

	plist = node->getLightPoints();
	rlist = node->getTransRadius();
	clist = node->getColour();
	n = plist.size();
	for (i = 0; i < n; i++) {
		d = sphereIntersect(plist[i]->x, plist[i]->y, plist[i]->z, rlist[i], ray);
		if (d > 0.0) {
			if (d < t) {
				t = d;
				current = i;
			}
		}
	}
	if (current < 0) {
		return(-1.0);
	}
	nx = ray->ox + t * ray->dx - plist[current]->x;
	ny = ray->oy + t * ray->dy - plist[current]->y;
	nz = ray->oz + t * ray->dz - plist[current]->z;
	len = sqrt(nx*nx + ny * ny + nz * nz);
	nx /= len;
	ny /= len;
	nz /= len;
	diff = nx * lx + ny * ly + nz * lz;
	if (diff < 0.0) {
		diff = 0.0;
	}
	colour = diffuse * clist[current] * diff + ambient;
	return(t);
}

struct Angles {
	double x;
	double y;
};

struct Angles samples[1024][1024];

void samplePower(int w, int h, double etaMax, double thetaMax, double width, double p) {
	int i, j;
	double deta;
	double dtheta;
	int midx;
	int midy;

	midx = w / 2;
	midy = h / 2;
	deta = 2 * width*etaMax / (w - 1);
	dtheta = 2 * width*thetaMax / (h - 1);

	for (i = 0; i < w; i++) {
		samples[i][midy].x = pow(sin(-width * etaMax + i * deta), p);
		samples[i][midy].y = 0.0;
		samples[midx][i].x = 0.0;
		samples[midx][i].y = pow(sin(-width * thetaMax + i * dtheta), p);
	}

	for (i = midx + 1; i < w; i++) {
		for (j = midy + 1; j < h; j++) {
			samples[i][j].x = pow(sin(i*deta), p);
			samples[i][j].y = pow(sin(j*dtheta), p);
			samples[2 * midx - i][2 * midy - j].x = -samples[i][j].x;
			samples[2 * midx - i][2 * midy - j].y = -samples[i][j].y;
			samples[i][2 * midy - j].x = samples[i][j].x;
			samples[i][2 * midy - j].y = -samples[i][j].y;
			samples[2 * midx - i][j].x = -samples[i][j].x;
			samples[2 * midx - i][j].y = samples[i][i].y;
		}
	}

}

void sampleSimpleLinear(int w, int h, double etaMax, double thetaMax, double width) {
	int i, j;
	double deta;
	double dtheta;
	int midx;
	int midy;

	midx = w / 2;
	midy = h / 2;
//	printf("mid point %d %d\n", midx, midy);
	deta = 2 * width*etaMax / (w - 1);
	dtheta = 2 * width*thetaMax / (h - 1);

	/*
	for (i = 0; i < w; i++) {
		samples[i][midy].x = sin(-width*etaMax + i * deta);
		samples[i][midy].y = 0.0;
		samples[midx][i].x = 0.0;
		samples[midx][i].y = sin(-width * thetaMax + i * dtheta);
	}

	for (i = midx+1; i < w; i++) {
//		printf("%d %d\n", i, 2 * midx - i);
		for (j = midy+1; j < h; j++) {
			samples[i][j].x = sin(i*deta);
			samples[i][j].y = sin(j*dtheta);
			samples[2 * midx - i][2 * midy - j].x = -samples[i][j].x;
			samples[2 * midx - i][2 * midy - j].y = -samples[i][j].y;
			samples[i][2 * midy - j].x = samples[i][j].x;
			samples[i][2 * midy - j].y = -samples[i][j].y;
			samples[2 * midx - i][j].x = -samples[i][j].x;
			samples[2 * midx - i][j].y = samples[i][i].y;
		}
	*/
	etaMax = width * etaMax;
	thetaMax = width * thetaMax;
	for (i = 0; i < w; i++) {
		for (j = 0; j < h; j++) {
			samples[i][j].x = sin(-etaMax + i * deta);
			samples[i][j].y = sin(-thetaMax + j * dtheta);
		}
	}

}

#include <random>


void sampleUniform(int w, int h, double etaMax, double thetaMax, double width) {
	int i, j;
	double fx, fy;
	std::random_device rd;
	std::mt19937 gen(rd());
//	std::uniform_real_distribution<> u1(-width * etaMax, width*etaMax);
//	std::uniform_real_distribution<> u2(-width * thetaMax, width*thetaMax);
	std::uniform_real_distribution<> u1(-1.0, 1.0);
	std::uniform_real_distribution<> u2(-1.0, 1.0);

	fx = width * etaMax;
	fy = width * thetaMax;
	for (i = 0; i < w; i++) {
		for (j = 0; j < h; j++) {
			samples[i][j].x = sin(fx*u1(gen));
			samples[i][j].y = sin(fy*u2(gen));
		}
	}
}

void sampleJitter(int w, int h, double etaMax, double thetaMax, double width) {
	int i, j;
	double deta;
	double dtheta;
	int midx;
	int midy;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> u1(-1.0, 1.0);
	std::uniform_real_distribution<> u2(-1.0, 1.0);
	double scalex, scaley;
	double noise;

	midx = w / 2;
	midy = h / 2;
	deta = 2 * width*etaMax / (w - 1);
	dtheta = 2 * width*thetaMax / (h - 1);

	noise = getnoise();
	scalex = noise*deta;
	scaley = noise*dtheta;

	etaMax = width * etaMax;
	thetaMax = width * thetaMax;
	for (i = 0; i < w; i++) {
		for (j = 0; j < h; j++) {
			samples[i][j].x = sin(-etaMax + i * deta);
			samples[i][j].y = sin(-thetaMax + j * dtheta);
		}
	}

	for (i = 0; i < w; i++) {
		for (j = 0; j < h; j++) {
			samples[i][j].x += scalex*u1(gen);
			samples[i][j].y += scaley*u2(gen);
		}
	}

}

void computeInterferenceRay(GeometryNode *node, InterferencePattern *pattern, double lambda, double xr, double yr, double zr) {
	double left;
	double top;
	int i, j;
	double x, y;
	double k;
	double dx, dy, dz;
	double len;
	double etaMax;
	double thetaMax;
	double deta;
	double dtheta;
	int xMax;
	int yMax;
	Device *device;
	int diamond;
	int m, n;
	struct Ray ray;
	double t;
	long long int intersections = 0;
	long long int maxInter;
//	struct Angles samples[1000][1000];
	double colour;
	double intens;
	double d;
	double rho;
	int rx, ry;
	double hi, lo;

	device = pattern->getDevice();
	sizex = device->sizeWide();
	sizey = device->sizeHeight();
	width = device->pixelsWidth();
	height = device->pixelsHeight();
	diamond = (int) device->diamond();

	etaMax = asin(lambda / (2 * sizex));
	thetaMax = asin(lambda / (2 * sizey));

	rho = getrho();
	d = sizex / (2 * tan(rho*etaMax));
	printf("d: %f\n", d);

	left = -width / 2 * sizex;
	top = height / 2 * sizey;
	k = 2 * M_PI / lambda;

	getres(rx, ry);
	xMax = rx;
	yMax = ry;
	deta = 2*etaMax / (xMax-1);
	dtheta = 2 * thetaMax / (yMax-1);
	printf("Angles: %f %f %f %f\n", etaMax, thetaMax, deta, dtheta);

	/*
	for (i = 0; i < xMax; i++) {
		for (j = 0; j < yMax; j++) {
			samples[i][j].x = sin(-etaMax + i*deta);
			samples[i][j].y = sin(-thetaMax + j*dtheta);
		}
	}
	*/

//	sampleSimpleLinear(xMax, yMax, etaMax, thetaMax, rho);

//	samplePower(xMax, yMax, etaMax, thetaMax, 0.1, rho);

//	sampleUniform(xMax, yMax, etaMax, thetaMax, rho);

	sampleJitter(xMax, yMax, etaMax, thetaMax, rho);

	lo = left + samples[0][0].x*80000;
	hi = -left + samples[xMax - 1][0].x*80000;
	printf("width: %f %f %f\n", lo, hi, hi - lo);

	computeReference(width, height, sizex, sizey, left, -top, 0.1, 0.1, 0.95, 10000.0, k);

	for (i = 0; i < width; i++)
		for (j = 0; j < height; j++)
			object[i][j] = 0.0;

	for (i = 0; i < width; i++) {
		
//		if(i % 10 == 0)
//		printf("Line %d\n",i);
		
		for (j = 0; j < height; j++) {
			x = left + i*sizex;
			if (diamond && ((j % 2) == 0))
				x += sizex / 2.0;;
			y = top - j*sizey;
			/*
			*  diffraction pattern computation
			*/

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
					t = trace(node, &ray, colour);
					if (t < 0) {
						continue;
					}
					intersections++;
					t -= d;
					object[i][j] += colour*exp(1i*k*t)/t;
				}
			}
		}
	}
	for (i = 0; i < width; i++) {
		for (j = 0; j < height; j++) {
			intens = abs(object[i][j] * conj(reference[i][j]));
			pattern->add(i, j, intens);
		}
	}
	maxInter = xMax*yMax;
	maxInter = maxInter * width*height;
	printf("intersections: %ld %ld %f\n", intersections, maxInter, ((double) intersections)/maxInter);
}