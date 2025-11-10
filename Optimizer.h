#pragma once


#include <opencv2/opencv.hpp>

#include <vector>

class Optimizer
{
public:

	enum IMAGE_SEQ {TUILES_PROCHES, ARBRES_PROCHES, HORNISGRINDE_HD, JARDIN};

	IMAGE_SEQ imSeq;
	std::string seqName;

	std::vector<cv::Mat> OLD_LoadAndPreProcess(IMAGE_SEQ imSeq);
	std::vector<cv::Mat> LoadAndPreProcess(IMAGE_SEQ imSeq);

	// The returned points are quiet NANs when not found
	std::vector<std::vector<cv::Point2f>> OptFlow(std::vector<cv::Mat> images);

	void DelaunayUnwarping(std::vector<cv::Mat> images, std::vector<std::vector<cv::Point2f>> shiftedPoints);

	void ShowSharpnessRanking(std::vector<cv::Mat> images, std::vector<std::vector<cv::Point2f>> shiftedPoints);
	void ShowBarycentricStabilization(std::vector<cv::Mat> images, std::vector<std::vector<cv::Point2f>> shiftedPoints);

	cv::Mat RecursiveMatching(std::vector<cv::Mat> images, std::vector<std::vector<cv::Point2f>> shiftedPoints);

	cv::Mat FullFrameSequentialMatcher(std::vector<cv::Mat> images);

	//sigmaG is the variance of the gaussian PSF
	cv::Mat LucyRichardson(cv::Mat src, int nIterations, float sigmaG);

	cv::Mat PostProcess(std::vector<cv::Mat> images);

private:
	cv::Mat UnsharpMasking(cv::Mat src);

	cv::Mat EqualizeHistogram(cv::Mat src);

	// TODO do not use incorrect at the end TODO correct
	std::vector<cv::Point2f> computeOffsetsNew(std::vector<std::vector<cv::Point2f>> shiftedPoints);

	std::vector<cv::Point2f> computeOffsetsOld(std::vector<std::vector<cv::Point2f>> shiftedPoints);
};

