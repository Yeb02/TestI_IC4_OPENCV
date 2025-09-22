#include "Optimizer.h"
#include "windows.h"

#include <algorithm>


using namespace cv;
using namespace std;



std::vector<cv::Mat> Optimizer::LoadAndPreProcess()
{
	std::vector<cv::Mat> images;

	for (int i = 0; i < 100; i++)
	{
		//std::string incompleteFilename = "C:\\Users\\alpha\\Desktop\\CSHR\\CSHR_2\\bursts\\img_" + std::to_string(i) +   + "*";

		//img_99_j17_m09_a2025_10h_28m_37s
		//img_0_j17_m09_a2025_10h_28m_34s

		//img_99_j17_m09_a2025_10h_26m_14s
		//img_0_j17_m09_a2025_10h_26m_11s

		//img_99_j17_m09_a2025_10h_25m_33s
		//img_0_j17_m09_a2025_10h_25m_30s
		int sec0 = 30;
		const char* base = "_j17_m09_a2025_10h_25m_";

		//img_99_j17_m09_a2025_10h_23m_57s
		//img_0_j17_m09_a2025_10h_23m_54s

		//img_99_j16_m09_a2025_16h_16m_56s
		//img_0_j16_m09_a2025_16h_16m_53s
		//int sec0 = 53;
		//const char* base = "_j16_m09_a2025_16h_16m_";

		//img_99_j16_m09_a2025_16h_16m_23s
		//img_0_j16_m09_a2025_16h_16m_19s
		//int sec0 = 19;
		//const char* base = "_j16_m09_a2025_16h_16m_";
		

		//img_99_j16_m09_a2025_16h_16m_10s
		//img_0_j16_m09_a2025_16h_16m_06s
		//int sec0 = 6;
		//const char* base = "_j16_m09_a2025_16h_16m_";
		

		//img_99_j16_m09_a2025_16h_15m_40s
		//img_0_j16_m09_a2025_16h_15m_37s


		
		std::string incompleteFilename;
		std::string filename;
		for (int s = 0; s < 5; s++)
		{
			int sec = sec0 + s;
			std::string secs = sec < 10 ? "0" + std::to_string(sec) : std::to_string(sec);
			incompleteFilename = "C:\\Users\\alpha\\Desktop\\CSHR\\CSHR_2\\bursts\\img_" + std::to_string(i) + base + secs + "*";
			WIN32_FIND_DATAA ffd;
			HANDLE hFind = FindFirstFileA(incompleteFilename.c_str(), &ffd);

			if (hFind != INVALID_HANDLE_VALUE)
			{ 
				filename = ffd.cFileName;
				break;
			}
		}

		/*do
		{
			std::string strFile = ffd.cFileName;
		} while (FindNextFileA(hFind, &ffd) != 0);*/

		cv::Mat img = cv::imread("C:\\Users\\alpha\\Desktop\\CSHR\\CSHR_2\\bursts\\" + filename, IMREAD_GRAYSCALE);
		if (img.empty())
		{
			throw std::runtime_error("Failed to load image: " + filename);
		}


		// calcOpticalFlowPyrLK only takes in CV_8U, so we cant use CV_32F yet.

		

		//img = EqualizeHistogram(img); //TODO same transform for all.

		//cv::GaussianBlur(img, img, Size(), 3.f);

		images.push_back(img);
	}

	return images;
}


std::vector<std::vector<cv::Point2f>> Optimizer::OptFlow(std::vector<cv::Mat> images)
{
	cv::Mat reference = images[0];

	vector<vector<Point2f>> points(images.size());

	vector<Point2f>& referencePoints = points[0];

	const int MAX_POINTS_TRACKED = 500;
	const double MIN_DISTANCE = 10.;
	const double QUALITY_LEVEL = .05; // default .3

	// Points that are near the edges are likely to disappear from frame to frame due to distortions/vibrations.
	const int MARGIN_SIZE = 25;
	Rect roi(MARGIN_SIZE, MARGIN_SIZE, reference.cols - 2 * MARGIN_SIZE, reference.rows - 2 * MARGIN_SIZE);

	// Take first frame and find corners in it
	goodFeaturesToTrack(reference(roi), referencePoints, MAX_POINTS_TRACKED, QUALITY_LEVEL, MIN_DISTANCE, Mat(), 7, false, 0.04);

	for (int i = 0; i < referencePoints.size(); i++)
	{
		referencePoints[i] += Point2f((float)MARGIN_SIZE, (float)MARGIN_SIZE);
	}

	vector<Scalar> colors;
	RNG rng;
	for (int i = 0; i < MAX_POINTS_TRACKED; i++)
	{
		int grayLevel = 255;
		//int grayLevel = rng.uniform(150, 256);
		colors.emplace_back(grayLevel);
	}

	

	for (int i = 1; i < images.size(); i++)
	{
		// Create a mask image for drawing purposes
		Mat mask = Mat::zeros(reference.size(), reference.type());

		vector<Point2f> potentialPoints;
		vector<Point2f>& pointsDetectedInCurrentImage = points[i];

		
		
		// calculate optical flow
		vector<uchar> status;
		vector<float> err;
		TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);
		calcOpticalFlowPyrLK(reference, images[i], referencePoints, potentialPoints, status, err, Size(15, 15), 2, criteria);
		
		for (uint j = 0; j < potentialPoints.size(); j++)
		{
			// Select good points
			if (status[j] == 1) {
				pointsDetectedInCurrentImage.push_back(potentialPoints[j]);

				// draw the tracks
				circle(mask, potentialPoints[j], 3, colors[j], -1);
				line(mask, potentialPoints[j], referencePoints[j], Scalar(0), 1);
			}
		}
		std::cout << "nMatches at step " << i << " : " << pointsDetectedInCurrentImage.size() << " / " << referencePoints.size() << std::endl;

		Mat img;
		add(images[i], mask, img);

		/*imshow("Frame", img);
		int keyboard = waitKey(100);*/

		//if (keyboard == 'q' || keyboard == 27)
		//	break;

	}

	return points; 
}


cv::Mat Optimizer::WarpStack(std::vector<cv::Mat> images, std::vector<vector<Point2f>> shiftedPoints)
{
	vector<Point2f> barycentres(images.size());

	for (int i = 0; i < images.size(); i++)
	{
		barycentres[i] = Point2f(.0f, .0f);
		for (int j = 0; j < shiftedPoints[i].size(); j++)
		{
			barycentres[i] += shiftedPoints[i][j];
		}
		barycentres[i] /= (float)shiftedPoints[i].size();
	}

	const int UPSCALE_FACTOR = 1;
	const int MARGIN_SIZE = 30; // px

	int LR_w = images[0].cols;
	int LR_h = images[0].rows;
	Size SR_size(LR_w * UPSCALE_FACTOR + MARGIN_SIZE * 2, LR_h * UPSCALE_FACTOR + MARGIN_SIZE * 2);


	for (int i = 0; i < images.size(); i++)
	{
		images[i].convertTo(images[i], CV_32F); // Can happen as soon as calcOpticalFlowPyrLK has been called on the image, because it is the only reason we start with CV_8U in the first place
	}


	Mat stack = cv::Mat::zeros(SR_size, CV_32F);
	Mat maskStack(SR_size, CV_32F, Scalar(.00001f)); // Initial value to avoid divisions by 0 later.

	
	cv::Mat upsizeBuffer(Size(LR_w * UPSCALE_FACTOR, LR_h * UPSCALE_FACTOR), CV_32F);
	cv::Mat translatedBuffer(SR_size, CV_32F);
	cv::Mat laplacian(images[0].size(), CV_32F);
	cv::Mat mask(SR_size, CV_8U);


	Point2f refBarycentre = barycentres[0];
	for (int i = 0; i < images.size(); i++) 
	{
		// Sharpness mask
		{
			cv::Laplacian(images[i], laplacian, CV_32F);

			const int MASK_DECIMATION = 10;
			const float LAPLACIAN_VARIANCE_THRESHOLD = 3.f;

			cv::Mat lowResMask = Mat::zeros(Size(LR_w / MASK_DECIMATION, LR_h / MASK_DECIMATION), CV_8U); // TODO cover the whole image

			for (int X = 0; X < lowResMask.cols; X++)
			{
				for (int Y = 0; Y < lowResMask.rows; Y++)
				{
					Rect _roi(X * MASK_DECIMATION, Y * MASK_DECIMATION, MASK_DECIMATION, MASK_DECIMATION);
					Scalar var, mean;
					cv::meanStdDev(images[i](_roi), mean, var);
					//lowResMask.at<float>(Y, X) = (float)var[0]*.2f;
					lowResMask.at<char>(Y, X) = (float)var[0] > LAPLACIAN_VARIANCE_THRESHOLD ? 1 : 0;
				}
			}

			cv::resize(lowResMask, mask, mask.size(), 0, 0, INTER_NEAREST);
		}

		

		// nearest neighbor, as per "sharp stack"
		{
			
		}

		// Averaging
		{
			Point2f offset = Point2f((float)MARGIN_SIZE, (float)MARGIN_SIZE) + barycentres[i] - refBarycentre;
			offset.x = std::clamp(offset.x, 0.f, (float)(2 * MARGIN_SIZE));
			offset.y = std::clamp(offset.y, 0.f, (float)(2 * MARGIN_SIZE));

			upsizeBuffer.setTo(.0f);
			cv::resize(images[i], upsizeBuffer, upsizeBuffer.size()); // not in place because we dont want to overwrite the original image

			std::cout << "Image " << i << ", offset x = " << offset.x << ", offset y = " << offset.y << std::endl;

			Mat warpMat = Mat::zeros(Size(3, 2), CV_32F);
			warpMat.at<float>(0, 0) = 1.f;
			warpMat.at<float>(1, 1) = 1.f;
			warpMat.at<float>(0, 2) = offset.x;
			warpMat.at<float>(1, 2) = offset.y;

			translatedBuffer.setTo(.0f);
			cv::warpAffine(upsizeBuffer, translatedBuffer, warpMat, translatedBuffer.size(), INTER_LINEAR); // Cant operate in place

			add(stack, translatedBuffer, stack, mask);

			
			add(maskStack, mask, maskStack, noArray(), CV_32F);
		}
	}

	imwrite("stack.png", stack);

	cv::divide(stack, maskStack, stack, 1.f/255.f);


	imshow("Stack", stack);

	int keyboard = waitKey(0);
		
	return stack;
}

cv::Mat Optimizer::PostProcess(std::vector<cv::Mat> images)
{
	return Mat();
}


cv::Mat Optimizer::Downscale(cv::Mat src, int factor)
{
	int dW = src.cols / factor;
	int dH = src.cols / factor;

	cv::Mat res(Size(dW, dH), src.type());

	return res;
}


cv::Mat Optimizer::UnsharpMasking(cv::Mat src)
{
	cv::Mat blurred;
	cv::GaussianBlur(src, blurred, cv::Size(0, 0), 5.0);

	float strength = 2.f;
	blurred = (1. + strength) * src - strength * blurred;

	cv::imshow("src", src);
	cv::imshow("unsharp masked", blurred);
	cv::waitKey();
	cv::destroyWindow("unsharp masked");
	cv::destroyWindow("src");

	return blurred;
}


cv::Mat Optimizer::EqualizeHistogram(cv::Mat src)
{
	cv::Mat dst0, dst1;

	cv::equalizeHist(src, dst0);

	//auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
	//clahe->apply(src, dst1);

	//cv::imshow("Local Equalized Histogram", dst0);
	//cv::imshow("Global Equalized Histogram", dst1);
	//cv::imshow("src", src);
	//cv::waitKey();
	//cv::destroyAllWindows();

	return dst0;
}






void calcPSF(Mat& outputImg, Size filterSize, int R)
{
	Mat h(filterSize, CV_32F, Scalar(0));
	Point point(filterSize.width / 2, filterSize.height / 2);
	circle(h, point, R, 255, -1, 8);
	Scalar summa = sum(h);
	outputImg = h / summa[0];
}

void fftshift(const Mat& inputImg, Mat& outputImg)
{
	outputImg = inputImg.clone();
	int cx = outputImg.cols / 2;
	int cy = outputImg.rows / 2;
	Mat q0(outputImg, Rect(0, 0, cx, cy));
	Mat q1(outputImg, Rect(cx, 0, cx, cy));
	Mat q2(outputImg, Rect(0, cy, cx, cy));
	Mat q3(outputImg, Rect(cx, cy, cx, cy));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
	Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI, DFT_SCALE);

	Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
	Mat complexH;
	merge(planesH, 2, complexH);
	Mat complexIH;
	mulSpectrums(complexI, complexH, complexIH, 0);

	idft(complexIH, complexIH);
	split(complexIH, planes);
	outputImg = planes[0];
}

void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
{
	Mat h_PSF_shifted;
	fftshift(input_h_PSF, h_PSF_shifted);
	Mat planes[2] = { Mat_<float>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	split(complexI, planes);
	Mat denom;
	pow(abs(planes[0]), 2, denom);
	denom += nsr;
	divide(planes[0], denom, output_G);
}


cv::Mat Optimizer::WienerFilter(cv::Mat src)
{
	cv::Mat Hw, h, dst;

	int R = 5;
	float snr = 100.f;

	calcPSF(h, src.size(), R);

	calcWnrFilter(h, Hw, 1.0 / double(snr));
	//Hw calculation (stop)

	// filtering (start)
	filter2DFreq(src, dst, Hw);
	// filtering (stop)

	dst.convertTo(dst, CV_8U);
	cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX);
	cv::imshow("Original", src);
	cv::imshow("Debluring", dst);
	cv::waitKey(0);


	return dst;
}
