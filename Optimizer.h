#pragma once


#include <opencv2/opencv.hpp>

#include <vector>

class Optimizer
{
public:
	std::vector<cv::Mat> LoadAndPreProcess();
	std::vector<std::vector<cv::Point2f>> OptFlow(std::vector<cv::Mat> images);

	cv::Mat WarpStack(std::vector<cv::Mat> images, std::vector<std::vector<cv::Point2f>> shiftedPoints);
	cv::Mat ShiftStack(std::vector<cv::Mat> images, std::vector<std::vector<cv::Point2f>> shiftedPoints);
	cv::Mat RecursiveMatching(std::vector<cv::Mat> images, std::vector<std::vector<cv::Point2f>> shiftedPoints);

	cv::Mat LucyRichardson(cv::Mat src);
	cv::Mat PostProcess(std::vector<cv::Mat> images);

private:

	cv::Mat Downscale(cv::Mat src, int factor);

	cv::Mat WienerFilter(cv::Mat src);

	cv::Mat UnsharpMasking(cv::Mat src);

	cv::Mat EqualizeHistogram(cv::Mat src);

	cv::Mat LK_one(cv::Mat src, cv::Mat ref);
};

