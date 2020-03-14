#pragma once
#define NOMINMAX
#include <optix.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sutil/sutil.h>
#include <fstream>
#include <string>
#include <iostream>
#include <map>

#include "ProgramCreator.h"
class Renderer {
private:
	int WIDTH;
	int HEIGHT;
	optix::Context context;

public:
	Renderer();
	Renderer(int width, int height, int rayCount, int entryPoints, int stackSize);

	/*
	Create a buffer that can be used to read or write from the GPU
	*/
	void createBuffer(std::string bufferName, int elementSize);
	
	/*
	Render the scene
	*/
	void render(int entryPoints);

	/*
	Display the data from the output buffer to the screen
	*/
	void display(int * argc, char * argv[], std::string outputBufferName);
	void cleanUp();
	optix::Context getContext();

	template<typename T>
	void createOutputBuffer(std::string bufferName, int width, int height) {
		optix::Buffer buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER, width, height);
		buffer->setElementSize(sizeof(T));
		buffer->setSize(width, height);
		context[bufferName]->setBuffer(buffer);
	}

	template<typename T>
	void createInputBuffer(std::string bufferName, std::vector<T> data, int width, int height) {
		optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, width, height);
		buffer->setElementSize(sizeof(T));
		buffer->setSize(width, height);
		void* buffLoc = buffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
		memcpy(buffLoc, data.data(), sizeof(T)*width*height);
		buffer->unmap();
		context[bufferName]->setBuffer(buffer);
	}

	template<typename T>
	void createGeoBuffer(optix::Geometry& geo, std::string bufferName, std::vector<T> data, int width, int height) {
		optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
		buffer->setElementSize(sizeof(T));
		buffer->setSize(width);
		void* buffLoc = buffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
		memcpy(buffLoc, data.data(), sizeof(T) * width);
		buffer->unmap();
		geo[bufferName]->setBuffer(buffer);
	}
};