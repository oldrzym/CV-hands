#include "HEThread.h"

HEThread::HEThread() {
	loaded = haxtr.init();
    //sizein = cv::Size(480, 272);
    sizein = cv::Size(640, 480);
	frameMask = cv::Mat::zeros(sizein, CV_32FC1);

	sizefinal = cv::Size(1920, 1080);

	resultFull = cv::Mat::zeros(sizefinal, CV_8UC4);

	img = Unigine::Image::create();
	img->create2D(sizefinal.width, sizefinal.height, Unigine::Image::FORMAT_RGBA8);
}

HEThread::~HEThread() {
	frameTemp.release();
	frameMask.release();
	frameFloat.release();
	frameOut.release();
}

void HEThread::useCVReader(CVThread &thread) {
	cvthread = &thread;
	sizeout = cvthread->getSize();

	result = cv::Mat::zeros(sizeout, CV_8UC4);
}

void HEThread::process() {
	while (isRunning() && cvthread != NULL) {
		//printf("[HEThread] PROCESS START\n");
		if (cvthread->isReady() && cvthread->getImage(frame)) {
			//cv::resize(frame, frameTemp, sizein); // resize to fit the model
			if (sizein != frame.size()) {
				cv::resize(frame, frameTemp, sizein, 0, 0, 0);
				cv::cvtColor(frameTemp, frameTemp, cv::COLOR_BGR2RGB); // set RGB order 
			} else {
				cv::cvtColor(frame, frameTemp, cv::COLOR_BGR2RGB); // set RGB order 
			}

			frameTemp.convertTo(frameFloat, CV_32F); // convert to float to fit the model
			if (haxtr.process(frameFloat, frameMask)) { // make prediction mask (keep frameMask unchanged)
				//forgroundMask.convertTo(gray, CV_8UC1, 255);
				//cv::bitwise_not(gray, gray);
				//frameOut = cv::Scalar(0, 0, 0);
				//frame.copyTo(frameOut, gray);
				//cv::cvtColor(frameOut, frameOut, cv::COLOR_RGB2BGR);
				//cv::resize(frame, frameTemp, sizeout);


                opening_operation(frameMask, frameMask, 3, 1); //for small mask

                if (sizein != frame.size()) {
                    cv::resize(frameMask, frameOut, sizeout, 0, 0, 0); //cv::INTER_CUBIC, cv::INTER_LINEAR, cv::INTER_NEAREST, cv::INTER_AREA, cv::INTER_LANCZOS4
				} else {
					frameMask.copyTo(frameOut);
                } //for small mask

                //cv::resize(frameMask, frameOut, sizefinal, 0, 0, cv::INTER_CUBIC);//for big mask
                //opening_operation(frameOut, frameOut, 3, 1);//for big mask
                mutex.lock();
                //cv::resize(frame, frameTemp, sizefinal, 0, 0, cv::INTER_CUBIC);//for big mask

                //combine(frameTemp, frameOut, resultFull);//for big mask
                combine(frame, frameOut, result); //for small mask
                //std::cout << "sizeFrame " << frameOut.size() << "\n";
                cv::resize(result, resultFull, sizefinal, 0, 0, 0);//for small mask
				updated = true;
				mutex.unlock();

				//printf("[HEThread] PROCESS STEP END\n");
			}
		} else {
			sleep(5);
		}
	}
}

Unigine::ImagePtr HEThread::getImage() {
	//printf("[HEThread] GET IMAGE\n");
	if (!updated) return nullptr;

	if (!mutex.isLocked()) {
		std::lock_guard<Unigine::Mutex> lock(mutex);
		void* buff = img->getPixels();
		memcpy(buff, resultFull.data, resultFull.total() * resultFull.elemSize());
		updated = false;
		//printf("[HEThread] GET IMAGE NOT LOCKED\n");
	}

	//printf("[HEThread] GET IMAGE LOCKED\n");
	return img;
}

// source - 3 channel, mask - 1 channel, result - 4 channel
void HEThread::combine(cv::Mat& source, cv::Mat& mask, cv::Mat& result) {
	int size = source.rows * source.cols;
	uchar* sdata = source.ptr();
	float* mdata = (float*)mask.ptr();
	uchar* rdata = result.ptr();
	unsigned int k, l, m;

	for (l = 0; l < size; l++) {
		m = l * 3;
		k = l * 4;
		if (mdata[l] < 0.1f) {
			rdata[k] = 0;
			rdata[k + 1] = 0;
			rdata[k + 2] = 0;
			rdata[k + 3] = 0;
		}
		else {
			rdata[k] = sdata[m + 2];
			rdata[k + 1] = sdata[m + 1];
			rdata[k + 2] = sdata[m];
			rdata[k + 3] = 255;
		}
	}
}

/*void HEThread::opening_operation(const cv::Mat& src, cv::Mat& dst, int kernel_size, int iterations) {
    start = std::chrono::high_resolution_clock::now();

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(src, dst, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), iterations);
    //cv::morphologyEx(dst, dst, cv::MORPH_DILATE, kernel, cv::Point(-1, -1), iterations);
    cv::morphologyEx(dst, dst, cv::MORPH_ERODE, kernel, cv::Point(-1, -1), iterations);

    end = std::chrono::high_resolution_clock::now();
    std::cout << "[TIME LOG] FLOOD TIME " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "\n";
}*/
void HEThread::opening_operation(const cv::Mat& src, cv::Mat& dst, int kernel_size, int iterations) {

	start = std::chrono::high_resolution_clock::now();
	cv::Mat tmp;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    //cv::morphologyEx(src, tmp, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), iterations);
    cv::morphologyEx(src, tmp, cv::MORPH_ERODE, kernel, cv::Point(-1, -1), iterations);

	// Create a mask for the floodFill operation
    cv::Mat floodMask = cv::Mat::zeros(tmp.rows + 2, tmp.cols + 2, CV_8U);

    for (int col = 0; col < tmp.cols; col++) {
		if (tmp.at<float>(tmp.rows - 1, col) > 0.5) {
			cv::floodFill(tmp, floodMask, cv::Point(col, tmp.rows - 1), 255, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
		}
	}

    floodMask = floodMask(cv::Range(1, floodMask.rows - 1), cv::Range(1, floodMask.cols - 1));

    floodMask.convertTo(dst, CV_32FC1);

	end = std::chrono::high_resolution_clock::now();
	std::cout << "[TIME LOG] FLOOD TIME " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "\n";

}

