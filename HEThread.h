#ifndef __APP_HETHREAD_H__
#define __APP_HETHREAD_H__

#include <opencv2/opencv.hpp>
#include <UnigineThread.h>
#include <UnigineImage.h>
#include "HandExtractorV2.h"
#include "CVThread.h"

class HEThread : public Unigine::Thread
{

public:
	HEThread();
	~HEThread();

	Unigine::ImagePtr getImage();
	void useCVReader(CVThread &thread);
	bool isReady() { return updated; };
	bool isLoaded() { return loaded; };

protected:
	void process();
	void combine(cv::Mat& source, cv::Mat& mask, cv::Mat& result);
	void opening_operation(const cv::Mat& src, cv::Mat& dst, int kernel_size = 5, int iterations = 2);

private:
	bool loaded = false;
	CVThread* cvthread;
	Unigine::ImagePtr img;
	cv::Mat frame, frameTemp, frameFloat, frameMask, frameOut, result, resultFull;
	cv::Size sizein, sizeout, sizefinal;
	mutable Unigine::Mutex mutex;
	HandExtractorV2 haxtr;
	bool updated = false;

	std::chrono::time_point<std::chrono::steady_clock> start, end;
};

#endif