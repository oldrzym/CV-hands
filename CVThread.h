#ifndef __APP_CVTHREAD_H__
#define __APP_CVTHREAD_H__

#include <opencv2/opencv.hpp>
#include <UnigineThread.h>

class CVThread : public Unigine::Thread
{

public:
	CVThread();
	~CVThread();

	bool getImage(cv::Mat& output);
	bool isReady() { return updated; };

	cv::Size getSize() { return size; };
protected:
	void process();

private:
	cv::Size size;
	cv::VideoCapture cap;
	cv::Mat framein;
	clock_t timer;
	mutable Unigine::Mutex mutex;
	bool updated = false;
	//bool resized = false;
};

#endif