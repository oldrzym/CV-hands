#include "CVThread.h"

CVThread::CVThread() {
	//sizein = cv::Size(480, 272);
    //cap.open("C:/Users/PC-1/Documents/Jupyter/workworkwork/Video to frames/видосики/Саша1.mp4");
    cap.open(0);

	size = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	/*int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	int fps = (int)cap.get(cv::CAP_PROP_FPS);
	int f = (int)cap.get(cv::CAP_PROP_FOURCC);
	char EXT[] = { (char)(f & 0XFF) , (char)((f & 0XFF00) >> 8),(char)((f & 0XFF0000) >> 16),(char)((f & 0XFF000000) >> 24), 0 };
	printf("[CVThread] SIZE %d %d\n", width, height);
	printf("[CVThread] FPS %d\n", fps);
	printf("[CVThread] FPS %s\n", EXT);*/

	//resized = size != sizein;

	timer = clock();
}

CVThread::~CVThread() {
	framein.release();
	cap.release();
}

void CVThread::process() {
	while (isRunning() && cap.isOpened()) {
		//printf("[CVThread] PROCESS START\n");
		//clock_t tmp = clock();
		//if (tmp > timer) {
            //timer = tmp + 33;
			/*cap.read(framein);
			if (!framein.empty()) {
				mutex.lock();
				cv::resize(framein, frameout, sizein);
				updated = true;
				mutex.unlock();
				//printf("[CVThread] PROCESS RESIZED\n");
			}*/
		//}
		//printf("[CVThread] PROCESS END\n");

        clock_t tmp = clock();////
        if (tmp > timer) {////
            timer = tmp + 33;////
            std::lock_guard<Unigine::Mutex> lock(mutex);
            cap.read(framein);
            if (!framein.empty()) {
                updated = true;
            } else break;
        }////
	}
}

bool CVThread::getImage(cv::Mat& output) {
	//printf("[CVThread] GET IMAGE\n");
	if (!updated) return false;
	std::lock_guard<Unigine::Mutex> lock(mutex);
	framein.copyTo(output);
	updated = false;
	return true;
}
