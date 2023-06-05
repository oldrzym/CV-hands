#include "HandExtractorV2.h"

#define BINT 0.3f

void HandExtractorV2::deallocator(void* data, size_t length, void* arg) {}

HandExtractorV2::HandExtractorV2() {}

bool HandExtractorV2::init(const char* bc_dir, const char* he_dir) {
	const char* tags[] = { "serve" };
	unsigned num_tags = 1;

	TF_Status* status = TF_NewStatus();

	//uint8_t bytes[5] = { 0x32, 0x3, 0x2a, 0x1, 0x31 };
	uint8_t bytes[5] = { 0x32, 0x3, 0x2a, 0x1, 0x31 };//  2 gpu
	sessopts = TF_NewSessionOptions();
	TF_SetConfig(sessopts, bytes, 5, status);
	if (TF_GetCode(status) != TF_OK)
	{
		printf("[TF] ERROR: TF_SetConfig\n");
		printf("%s", TF_Message(status));
		return false;
	}

	// banary classificator
	bc_graph = TF_NewGraph();

	TF_Buffer* runopts = NULL;
	bc_session = TF_LoadSessionFromSavedModel(sessopts,
		runopts,
		bc_dir,
		tags, num_tags, // tags and number of tags
		bc_graph,
		NULL,
		status);

	if (TF_GetCode(status) != TF_OK)
	{
		printf("[TF] ERROR: TF_LoadSessionFromSavedModel\n");
		printf("%s", TF_Message(status));
		return false;
	}

	bc_input_tensor = { TF_GraphOperationByName(bc_graph, "serving_default_input_1"), 0 };
	bc_output_tensor = { TF_GraphOperationByName(bc_graph, "StatefulPartitionedCall"), 0 };
	bc_output_value = new TF_Tensor * [1]{ NULL }; // do not forget update destructor

	// hand extractor
	he_graph = TF_NewGraph();

	he_session = TF_LoadSessionFromSavedModel(sessopts,
		runopts,
		he_dir,
		tags, num_tags, // tags and number of tags
		he_graph,
		NULL,
		status);

	if (TF_GetCode(status) != TF_OK)
	{
		printf("[TF] ERROR: TF_LoadSessionFromSavedModel\n");
		printf("%s", TF_Message(status));
		return false;
	}

	he_input_tensor = { TF_GraphOperationByName(he_graph, "serving_default_input_1"), 0 };
	he_output_tensor = { TF_GraphOperationByName(he_graph, "StatefulPartitionedCall"), 0 };
	he_output_value = new TF_Tensor * [1]{ NULL }; // do not forget update destructor

	/*er_graph = TF_NewGraph();

	er_session = TF_LoadSessionFromSavedModel(sessopts,
		runopts,
		er_dir,
		tags, num_tags, // tags and number of tags
		er_graph,
		NULL,
		status);

	if (TF_GetCode(status) != TF_OK)
	{
		printf("[TF] ERROR: TF_LoadSessionFromSavedModel\n");
		printf("%s", TF_Message(status));
		return false;
	}
	er_input_tensor = new TF_Output[2];
	er_input_tensor[0] = { TF_GraphOperationByName(er_graph, "serving_default_input_4"), 0 };
	er_input_tensor[1] = { TF_GraphOperationByName(er_graph, "serving_default_input_5"), 0 };
	er_output_tensor = { TF_GraphOperationByName(er_graph, "StatefulPartitionedCall"), 0 };
	er_input_value = new TF_Tensor * [2]{ NULL, NULL }; // do not forget update destructor
	er_output_value = new TF_Tensor * [1]{ NULL }; // do not forget update destructor*/


	TF_DeleteStatus(status);
	return true;
}

HandExtractorV2::~HandExtractorV2() {
	if (bc_input_value != NULL) {
		TF_DeleteTensor(bc_input_value);
		bc_input_value = NULL;
	}

	if (bc_output_value != NULL) {
		if (bc_output_value[0] != NULL) {
			TF_DeleteTensor(bc_output_value[0]);
		}
		delete[] bc_output_value;
		bc_output_value = NULL;
	}

	if (he_input_value != NULL) {
		TF_DeleteTensor(he_input_value);
		he_input_value = NULL;
	}

	if (he_output_value != NULL) {
		if (he_output_value[0] != NULL) {
			TF_DeleteTensor(he_output_value[0]);
		}
		delete[] he_output_value;
		he_output_value = NULL;
	}

	/*if (er_input_value != NULL) {
		delete[] er_input_value;
		er_input_value = NULL;
	}

	if (er_output_value != NULL) {
		if (er_output_value[0] != NULL) {
			TF_DeleteTensor(er_output_value[0]);
		}
		delete[] er_output_value;
		er_output_value = NULL;
	}

	if (er_input_tensor != NULL) {
		delete[] er_input_tensor;
		er_input_tensor = NULL;
	}*/

	TF_Status* status = TF_NewStatus();
	TF_DeleteSession(bc_session, status);
	TF_DeleteGraph(bc_graph);
	TF_DeleteSession(he_session, status);
	TF_DeleteSessionOptions(sessopts);
	TF_DeleteGraph(he_graph);
	//TF_DeleteSession(er_session, status);
	//TF_DeleteGraph(er_graph);
	TF_DeleteStatus(status);
}

bool HandExtractorV2::process(cv::Mat& source, cv::Mat& output) {

	start = std::chrono::high_resolution_clock::now();

	if (he_input_value != NULL) {
		copyMatToTensor(source, he_input_value);
	}
	else {
		he_input_value = createTensorFromMat(source);
	}
	TF_Status* status = TF_NewStatus();

	TF_SessionRun(bc_session,
		NULL,
		&bc_input_tensor, &he_input_value, 1,
		&bc_output_tensor, bc_output_value, 1, // output variable 
		NULL, 0, //target operations
		NULL,
		status);


	if (TF_GetCode(status) != TF_OK)
	{
		printf("[TF] ERROR: Unable to run\n");
		printf("%s", TF_Message(status));

		return false;
	}

	if (((float*)TF_TensorData(bc_output_value[0]))[0] < BINT) {

		TF_DeleteTensor(bc_output_value[0]);
		bc_output_value[0] = NULL;
		output = cv::Scalar(0);
		return true;
	}

	TF_DeleteTensor(bc_output_value[0]);
	bc_output_value[0] = NULL;

	TF_SessionRun(he_session,
		NULL,
		&he_input_tensor, &he_input_value, 1,
		&he_output_tensor, he_output_value, 1, // output variable 
		NULL, 0, //target operations
		NULL,
		status);


	if (TF_GetCode(status) != TF_OK)
	{
		printf("[TF] ERROR: Unable to run\n");
		printf("%s", TF_Message(status));

		return false;
	}

	memcpy((void*)output.data, (void*)TF_TensorData(he_output_value[0]), source.rows * source.cols * sizeof(float));

	TF_DeleteTensor(he_output_value[0]);
	he_output_value[0] = NULL;

	/*er_input_value[0] = he_input_value;
	er_input_value[1] = he_output_value[0];

	TF_SessionRun(er_session,
		NULL,
		er_input_tensor, er_input_value, 2,
		&er_output_tensor, er_output_value, 1, // output variable 
		NULL, 0, //target operations
		NULL,
		status);


	if (TF_GetCode(status) != TF_OK)
	{
		printf("[TF] ERROR: Unable to run\n");
		printf("%s", TF_Message(status));

		return false;
	}

	TF_DeleteStatus(status);
	memcpy((void*)output.data, (void*)TF_TensorData(er_output_value[0]), source.rows * source.cols * sizeof(float));

	TF_DeleteTensor(he_output_value[0]);
	he_output_value[0] = NULL;
	er_input_value[0] = NULL;
	er_input_value[1] = NULL;
	TF_DeleteTensor(er_output_value[0]);
	er_output_value[0] = NULL;*/

	end = std::chrono::high_resolution_clock::now();
	std::cout << "[TIME LOG] EXTARCT TIME " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "\n";

	return true;
}

TF_Tensor* HandExtractorV2::createTensorFromMat(cv::Mat& input) {
	int total_size = input.cols * input.rows * input.channels() * sizeof(float);
	int64_t dims[] = { 1, input.rows, input.cols, input.channels() };
	return TF_NewTensor(TF_FLOAT, dims, 4, (void*)input.ptr(), total_size, &HandExtractorV2::deallocator, 0);
}

void HandExtractorV2::copyMatToTensor(cv::Mat& input, TF_Tensor* tensor) {
	int total_size = input.cols * input.rows * input.channels() * sizeof(float);
	memcpy((void*)TF_TensorData(tensor), (void*)input.data, total_size);
}
