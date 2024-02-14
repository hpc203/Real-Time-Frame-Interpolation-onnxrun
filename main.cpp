#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>  ///如果使用cuda加速，需要取消注释
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

class RIFE
{
public:
	RIFE(string modelpath);
	Mat interpolate(Mat srcimg1, Mat srcimg2);
	vector<Mat> interpolate_4x(Mat srcimg1, Mat srcimg2);
private:
	void preprocess(Mat img, vector<float>& input_img);
	vector<float> input1_image;
	vector<float> input2_image;
	int inpWidth;
	int inpHeight;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Frame Interpolation");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
};

RIFE::RIFE(string model_path)
{
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());  ////windows写法
	///OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   ///如果使用cuda加速，需要取消注释

	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows写法
	////ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法

	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
}

void RIFE::preprocess(Mat img, vector<float>& input_img)
{
	Mat rgbimg;
	cvtColor(img, rgbimg, COLOR_BGR2RGB);
	const int h = rgbimg.rows;
	const int w = rgbimg.cols;
	const int align = 32;
	if (h % align != 0 || w % align != 0)
	{
		const int ph = (int((h - 1) / align) + 1)*align;
		const int pw = (int((w - 1) / align) + 1)*align;
		copyMakeBorder(rgbimg, rgbimg, 0, ph - h, 0, pw - w, BORDER_CONSTANT, 0);
	}
	this->inpHeight = rgbimg.rows;
	this->inpWidth = rgbimg.cols;
	rgbimg.convertTo(rgbimg, CV_32FC3, 1 / 255.);

	const int image_area = rgbimg.rows * rgbimg.cols;
	input_img.resize(3 * image_area);
	size_t single_chn_size = image_area * sizeof(float);
	vector<cv::Mat> rgbChannels(3);
	split(rgbimg, rgbChannels);
	memcpy(input_img.data(), (float*)rgbChannels[0].data, single_chn_size);
	memcpy(input_img.data() + image_area, (float*)rgbChannels[1].data, single_chn_size);
	memcpy(input_img.data() + image_area * 2, (float*)rgbChannels[2].data, single_chn_size);
}

Mat RIFE::interpolate(Mat srcimg1, Mat srcimg2)
{
	const int srch = srcimg1.rows;
	const int srcw = srcimg1.cols;
	this->preprocess(srcimg1, this->input1_image);
	this->preprocess(srcimg2, this->input2_image);

	std::vector<int64_t> input_img_shape = { 1, 3, this->inpHeight, this->inpWidth };
	std::vector<Ort::Value> inputTensors;
	inputTensors.push_back((Ort::Value::CreateTensor<float>(memory_info_handler, this->input1_image.data(), this->input1_image.size(), input_img_shape.data(), input_img_shape.size())));
	inputTensors.push_back((Ort::Value::CreateTensor<float>(memory_info_handler, this->input2_image.data(), this->input2_image.size(), input_img_shape.data(), input_img_shape.size())));

	Ort::RunOptions runOptions;
	vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), inputTensors.data(), inputTensors.size(), this->output_names.data(), output_names.size());

	float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	std::vector<int64_t> outs_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	const int out_h = outs_shape[2];
	const int out_w = outs_shape[3];
	const int channel_step = out_h * out_w;
	Mat rmat(out_h, out_w, CV_32FC1, pdata);
	Mat gmat(out_h, out_w, CV_32FC1, pdata + channel_step);
	Mat bmat(out_h, out_w, CV_32FC1, pdata + 2 * channel_step);
	rmat *= 255.f;
	gmat *= 255.f;
	bmat *= 255.f;
	///output_image = 等价np.clip(output_image, 0, 255)
	rmat.setTo(0, rmat < 0);
	rmat.setTo(255, rmat > 255);
	gmat.setTo(0, gmat < 0);
	gmat.setTo(255, gmat > 255);
	bmat.setTo(0, bmat < 0);
	bmat.setTo(255, bmat > 255);

	vector<Mat> channel_mats(3);
	channel_mats[0] = bmat + 0.5;
	channel_mats[1] = gmat + 0.5;
	channel_mats[2] = rmat + 0.5;

	Mat dstimg;
	merge(channel_mats, dstimg);
	dstimg.convertTo(dstimg, CV_8UC3);
	Mat mid_img = dstimg(Rect{ 0, 0, srcw, srch });
	return mid_img;
}

vector<Mat> RIFE::interpolate_4x(Mat srcimg1, Mat srcimg2)
{
	Mat mid_img = this->interpolate(srcimg1, srcimg2);
	Mat first_half = this->interpolate(srcimg1, mid_img);
	Mat second_half = this->interpolate(mid_img, srcimg2);
	vector<Mat> inter_imgs = { first_half, mid_img, second_half };
	return inter_imgs;
}

int main()
{
	RIFE mynet("RIFE_HDv3.onnx");
	const int use_video = 0;
	if (use_video == 0)
	{
		vector<string> inputs_imgpath = { "testimgs/frame07.png", "testimgs/frame08.png", "testimgs/frame09.png", "testimgs/frame10.png", "testimgs/frame11.png", "testimgs/frame12.png", "testimgs/frame13.png", "testimgs/frame14.png" };
		const int imgnum = inputs_imgpath.size();
		if (imgnum < 2)
		{
			printf("input must be at least two or more images");
			exit(1);
		}

		for (int i = 0; i < imgnum - 1; i++)
		{
			Mat srcimg1 = imread(inputs_imgpath[i]);
			Mat srcimg2 = imread(inputs_imgpath[i + 1]);
			Mat mid_img = mynet.interpolate(srcimg1, srcimg2);

			string save_imgpath = "imgs_results/mid" + to_string(i) + ".jpg";
			imwrite(save_imgpath, mid_img);

			/*namedWindow("srcimg1", WINDOW_NORMAL);
			imshow("srcimg1", srcimg1);
			namedWindow("mid_img", WINDOW_NORMAL);
			imshow("mid_img", mid_img);
			namedWindow("srcimg2", WINDOW_NORMAL);
			imshow("srcimg2", srcimg2);
			waitKey(0);
			destroyAllWindows();*/
		}
	}
	else
	{
		string videopath = "test.mp4";
		string savepath = "result.mp4";
		VideoCapture vcapture(videopath);
		if (!vcapture.isOpened())
		{
			cout << "VideoCapture,open video file failed, " << videopath;
			return -1;
		}
		int height = vcapture.get(cv::CAP_PROP_FRAME_HEIGHT);
		int width = vcapture.get(cv::CAP_PROP_FRAME_WIDTH);
		int fps = vcapture.get(cv::CAP_PROP_FPS);
		int video_length = vcapture.get(cv::CAP_PROP_FRAME_COUNT);
		VideoWriter vwriter;
		vwriter.open(savepath,
			cv::VideoWriter::fourcc('X', '2', '6', '4'),
			fps,
			Size(width, height));

		Mat frame;
		vector<Mat> images;
		while (vcapture.read(frame))
		{
			if (frame.empty())
			{
				cout << "cv::imread source file failed, " << videopath;
				return -1;
			}

			images.push_back(frame);
			if (images.size() < 2)
			{
				continue;
			}
			else if (images.size() > 2)
			{
				images.erase(images.begin());
			}
			
			Mat mid_img = mynet.interpolate(images[0], images[1]);
			
			vwriter.write(images[0]);
			vwriter.write(mid_img);
		}
		destroyAllWindows();
		vwriter.release();
		vcapture.release();
	}


	return 0;
}
