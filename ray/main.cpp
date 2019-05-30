/********************************************
*
*            ray
*
*  Test program for the HAPI library. This
*  program has three optional parameters that
*  can be used to move the hologram on the
*  display surface and depth.  The x and y
*  displacements are used to avoid artifacts
*  caused by the interaction between the laser
*  and the DMD pixels.  The z displacement can
*  be used to control the distance between the
*  DMD surface and where the hologram appears.
*
********************************************/
#define _USE_MATH_DEFINES
#include <math.h>
#include "HAPI.h"
#include <stdio.h>
#include "Matrix.h"
#include <vector>
#include "helper.h"

int main(int argc, char **argv) {
	Node *node = new Node(NONE);
	GeometryNode *gnode;
	TransformationNode *tnode;
	int i;
	char buffer[256];
	double t1;
	double dx, dy, dz;

	struct Settings {
		double rho;
		double noise;
		int rx;
		int ry;
	};
	struct Settings settings[] = {
		0.1, 1.0, 25, 25,
		0.4, 1.0, 25, 25,
//		0.2, 1.0, 299, 299
	};
	int NSettings = sizeof(settings) / sizeof(Settings);
//	int NSettings = 1;

	dx = dy = dz = 0.0;
	if (argc == 4) {
		dx = atof(argv[1]);
		dy = atof(argv[2]);
		dz = atof(argv[3]);
		printf("dz: %f\n", dz);
	}

	t1 = getSeconds();
	/*
	*  Compute only the red channel and
	*  use a wavelenght of 650 nm for red
	*/
	setChannels(RED);
	
//	setWavelength(RED, 650.0);

	setWavelength(RED, 532.0);
	setChannelPosition(RED, 0.0, 0.0, 37.5);
	/*
	*  Store the resulting interference
	*  pattern in a file, the base name
	*  for the files constructed is "test"
	*/
	setTarget(FILE);
	setBaseFileName("test");
	/*
	*  The world space defines the coordinate system
	*  used in the application.  Use whatever units
	*  you like.
	*/
	setWorldSpace(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0);
	/*
	*  The device space specifies where the hologram
	*  will appear in the real world.  The units used
	*  are cm.  These values are a good start for our
	*  prototype display
	*/
	//	setDeviceSpace(-0.4+dx, -0.4+dy, 17.75+dz, 0.4+dx, 0.4+dy, 18.25+dz);

	/*
	*  Create the lines for the truncated pyramid
	*/
	gnode = new GeometryNode(SPHERE);
	gnode->addSphere(0.0, 0.0, 0.0, 0.6, 1.0);
//	gnode->addSphere(0.0, 0.0, -0.5, 0.2, 0.0);
//	gnode->addSphere(-0.5, -0.5, -0.5, 0.01, 1.0);

	/*
	*  Set up the transformation for rotating the
	*  truncated pyramid
	*/
	tnode = new TransformationNode();
	tnode->rotateZ(0.0);

	/*
	*  Link up the nodes in the scene graph
	*/
	node->addChild(gnode);
//	tnode->addChild(gnode);
	/*
	*  Generate the interference patterns
	*/
	setPointDistance(250.0);
	setAlgorithm(RAY_TRACE);
//	setres(199, 199);
//	setrho(0.1);
//	setnoise(1.0);
	printf("settings: %d\n", NSettings);
	for (i = 0; i<NSettings; i++) {
		printf("===================================================\n");
		printf("     setting %d\n", i);
		printf("   %f %f %d %d\n", settings[i].rho, settings[i].noise, settings[i].rx, settings[i].ry);
		printf("====================================================\n");
		setres(settings[i].rx, settings[i].ry);
		setrho(settings[i].rho);
		setnoise(settings[i].noise);
		sprintf(buffer, "setting%d", i);
//		sprintf(buffer, "hiddenr");
		setBaseFileName(buffer);
		display(node);
//		tnode->rotateZ(0.1);
	}
	printf("total time: %f\n", getSeconds() - t1);
}