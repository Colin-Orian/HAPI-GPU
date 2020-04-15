#include "Renderer.h"

Renderer::Renderer()
{
}

Renderer::Renderer(int width, int height, int rayCount, int entryPoints, int stackSize) {
	this->WIDTH = width;
	this->HEIGHT = height;
	this->bytesUsed = 0;
	context = optix::Context::create();
	context->setRayTypeCount(rayCount);
	context->setEntryPointCount(entryPoints);
	context->setStackSize(stackSize);
}

void Renderer::createBuffer(std::string bufferName, int elementSize) {
	optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT2, WIDTH, HEIGHT);
	context[bufferName]->set(buffer);
}

void Renderer::render(int entryPoints) {
	try {
		context->validate();
		
	}
	catch (optix::Exception e) {
		std::cout << "Validate failed: " << e.getErrorString() << std::endl;
	}
	unsigned int totalMemory = context->getAvailableDeviceMemory(0);
	std::cout << "Used Memory: " << this->bytesUsed << ",Total Memory: " << totalMemory << " Percent Used: " << (float)this->bytesUsed / (float)totalMemory << std::endl;

	try {
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		context->launch(entryPoints, WIDTH, HEIGHT);
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

		std::cout << "It took me " << time_span.count() << " seconds to render.";
		std::cout << std::endl;
	}
	catch (optix::Exception e) {
		std::cout << e.getErrorString() << std::endl;
	}

}

void Renderer::display(int* argc, char* argv[], std::string outputBufferName) {
	sutil::initGlut(argc, argv);
	sutil::displayBufferGlut("test", context[outputBufferName]->getBuffer());
}


void Renderer::cleanUp() {
	if (context) {
		context->destroy();
	}
}

optix::Context Renderer::getContext() {
	return context;
}