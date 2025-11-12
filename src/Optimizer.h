#pragma once


#include <opencv2/opencv.hpp>

#include <vector>

class Optimizer
{
public:

	enum IMAGE_SEQ {
		JARDIN_NO_VIBRATION, JARDIN, HANGAR_1, HANGAR_2, HANGAR_ZOOM, ZI_FLOUE_1, ZI_FLOUE_2, ZI_3,
		PORT_DU_RHIN_1, CATHEDRALE_1, PORT_DU_RHIN_2, MAISON_1, MAISON_2, MAISON_3, GRAVIERE, ZI_4, USINE_1, USINE_2, MAISON_4
	};

	IMAGE_SEQ imSeq;
	std::string seqName;

	void LoadAndPreProcess(IMAGE_SEQ _imSeq, std::vector<cv::Mat>& dstBayerImgs, std::vector<cv::Mat>& dstRGBImgs);

	// The returned points are quiet NANs when not found
	std::vector<std::vector<cv::Point2f>> OptFlow(std::vector<cv::Mat> images);

	void DelaunayUnwarping(std::vector<cv::Mat> images, std::vector<std::vector<cv::Point2f>> shiftedPoints);

	void ShowSharpnessRanking(std::vector<cv::Mat> images, std::vector<std::vector<cv::Point2f>> shiftedPoints);
	void ShowBarycentricStabilization(std::vector<cv::Mat> images, std::vector<std::vector<cv::Point2f>> shiftedPoints);

	cv::Mat FullFrameSequentialMatcher(std::vector<cv::Mat> bayerImages, std::vector<cv::Mat> RGBImages);

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

