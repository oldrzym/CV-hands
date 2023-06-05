#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>
#include <tensorflow/c/c_api_experimental.h>
#include <chrono>

class HandExtractorV2 {
private:

	uint width;
	uint height;
	TF_Graph* bc_graph;
	TF_Graph* he_graph;
	//TF_Graph* er_graph;
	TF_SessionOptions* sessopts;
	TF_Session* bc_session;
	TF_Session* he_session;
	//TF_Session* er_session;

	TF_Output bc_input_tensor;
	TF_Output bc_output_tensor;
	TF_Tensor* bc_input_value = NULL;
	TF_Tensor** bc_output_value = NULL;

	TF_Output he_input_tensor;
	TF_Output he_output_tensor;
	TF_Tensor* he_input_value = NULL;
	TF_Tensor** he_output_value = NULL;

	/*TF_Output* er_input_tensor;
	TF_Output er_output_tensor;
	TF_Tensor** er_input_value = NULL;
	TF_Tensor** er_output_value = NULL;*/

public:
	HandExtractorV2();
	~HandExtractorV2();

    bool init(const char* bc_dir = "C:/Users/PC-1/Documents/Jupyter/workworkwork/binary_classificator_H_v5_640x480", const char* he_dir = "C:/Users/PC-1/Documents/Jupyter/workworkwork/pspnet_3_7");
	bool process(cv::Mat& source, cv::Mat& output);
	TF_Tensor* createTensorFromMat(cv::Mat& input);
	void copyMatToTensor(cv::Mat& input, TF_Tensor* tensor);
	static void deallocator(void* data, size_t length, void* arg);

	std::chrono::time_point<std::chrono::steady_clock> start, end;
};
