#define NOMINMAX
#include "Optimizer.h"

#include "delaunator.h"

#include "windows.h"

#include <algorithm>

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


#include <time.h> // Time since 1/1/1970



using namespace cv;
using namespace std;




std::vector<cv::Mat> Optimizer::LoadAndPreProcess(IMAGE_SEQ _imSeq)
{
	//TUILES_PROCHES, ARBRES_PROCHES, HORNISGRINDE_HD};
	imSeq = _imSeq;
	switch (imSeq)
	{
	case TUILES_PROCHES:
		seqName = "TUILES_PROCHES";
		break;
	case ARBRES_PROCHES:
		seqName = "ARBRES_PROCHES";
		break;
	case HORNISGRINDE_HD:
		seqName = "HORNISGRINDE_HD";
		break;
	}

	std::vector<cv::Mat> images;

	for (int i = 0; i < 100; i++)
	{
		int sec0 = -1;
		std::string base = "";

		//img_99_j17_m09_a2025_10h_28m_37s
		//img_0_j17_m09_a2025_10h_28m_34s


		// TUILES PROCHES
		//img_99_j17_m09_a2025_10h_26m_14s
		//img_0_j17_m09_a2025_10h_26m_11s
		//int sec0 = 11;
		//const char* base = "_j17_m09_a2025_10h_26m_";

		// ARBRES PROCHES
		//img_99_j17_m09_a2025_10h_25m_33s
		//img_0_j17_m09_a2025_10h_25m_30s
		//int sec0 = 30;
		//const char* base = "_j17_m09_a2025_10h_25m_";

		//img_99_j17_m09_a2025_10h_23m_57s
		//img_0_j17_m09_a2025_10h_23m_54s

		//img_99_j16_m09_a2025_16h_16m_56s
		//img_0_j16_m09_a2025_16h_16m_53s
		//int sec0 = 53;
		//const char* base = "_j16_m09_a2025_16h_16m_";


		// HORNISGRINDE_HD
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


		switch (imSeq)
		{
		case TUILES_PROCHES:
			sec0 = 11;
			base = "_j17_m09_a2025_10h_26m_";
			break;
		case ARBRES_PROCHES:
			sec0 = 30;
			base = "_j17_m09_a2025_10h_25m_";
			break;
		case HORNISGRINDE_HD:
			sec0 = 19;
			base = "_j16_m09_a2025_16h_16m_";
			break;
		}

		
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

		cv::Mat img = cv::imread("C:\\Users\\alpha\\Desktop\\CSHR\\CSHR_2\\bursts\\" + filename);
		if (img.empty())
		{
			throw std::runtime_error("Failed to load image: " + filename);
		}

		Mat channels[3];
		cv::split(img, channels);

		// For visualization only:
		auto c0 = channels[0];
		auto c1 = channels[1];
		auto c2 = channels[2];

		//images.push_back(c1);

		cvtColor(img, img, COLOR_BGR2GRAY);
		images.push_back(img);
	}

	return images;
}


std::vector<std::vector<cv::Point2f>> Optimizer::OptFlow(std::vector<cv::Mat> images)
{
	cv::Mat reference = images[0];

	vector<vector<Point2f>> points(images.size());

	vector<Point2f>& referencePoints = points[0];

	const int MAX_POINTS_TRACKED = 300;
	const double MIN_DISTANCE = 20.;
	const double QUALITY_LEVEL = .3; // default .3
	const int BLOCK_SIZE = 2 * 1 + 1;
	const int GRADIENT_SIZE = 2 * 1 + 1;

	// Points that are near the edges are likely to disappear from frame to frame due to distortions/vibrations. 
	const int MARGIN_SIZE = 25;
	Rect roi(MARGIN_SIZE, MARGIN_SIZE, reference.cols - 2 * MARGIN_SIZE, reference.rows - 2 * MARGIN_SIZE);

	// Take first frame and find corners in it
	auto time1 = high_resolution_clock::now();
	goodFeaturesToTrack(reference(roi), referencePoints, MAX_POINTS_TRACKED, QUALITY_LEVEL, MIN_DISTANCE, Mat(), BLOCK_SIZE, GRADIENT_SIZE, false, 0.04);
	auto time2 = high_resolution_clock::now();
	int ms_int = (int)duration_cast<milliseconds>(time2 - time1).count();
	//std::cout << "\nFinding good features in the first frame took " << ms_int << " ms." << std::endl;



	// Refine the corners to subpixel precision
	time1 = high_resolution_clock::now();
	Size winSize = Size(BLOCK_SIZE / 2, BLOCK_SIZE / 2);
	Size zeroZone = Size(-1, -1);
	TermCriteria _criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001);
	cornerSubPix(reference(roi), referencePoints, winSize, zeroZone, _criteria);
	time2 = high_resolution_clock::now();
	ms_int = (int)duration_cast<milliseconds>(time2 - time1).count();
	//std::cout << "Refining to subpix accuracy took " << ms_int << " ms." << std::endl;

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

		points[i].resize(referencePoints.size());



		// calculate optical flow
		vector<uchar> status;
		vector<float> err;
		TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);

		auto _time1 = high_resolution_clock::now();
		calcOpticalFlowPyrLK(reference, images[i], referencePoints, potentialPoints, status, err, Size(15, 15), 2, criteria, 0, 1.0E-3);
		auto _time2 = high_resolution_clock::now();
		int _ms_int = (int)duration_cast<milliseconds>(_time2 - _time1).count();
		//std::cerr << "Matching features took " << _ms_int << " ms." << std::endl;

		int nMatches = 0;
		for (uint j = 0; j < potentialPoints.size(); j++)
		{
			// Select good points
			if (status[j] == 1) {
				points[i][j] = potentialPoints[j];
				nMatches++;

				// draw the tracks
				circle(mask, potentialPoints[j], 3, colors[j], -1);
				cv::line(mask, potentialPoints[j], referencePoints[j], Scalar(0), 1);
			}
			else
			{
				points[i][j] = Point2f(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
			}
		}
		std::cerr << "nMatches at step " << i << " : " << nMatches << " / " << referencePoints.size() << std::endl;

		Mat img;
		cv::add(images[i], mask, img);

		/*imshow("Frame", img);
		int keyboard = waitKey(100);*/

		//if (keyboard == 'q' || keyboard == 27)
		//	break;

	}

	return points;
}


std::vector<Point2f> Optimizer::computeOffsetsNew(std::vector<std::vector<cv::Point2f>> trackedPoints)
{
	// For each tracked point, the average distance (vector) to its image barycenter.
	// Contains the sum rather than the average, so we divide by (float)i when we need it
	vector<Point2f> averagePointOffsetToImageBarycenter(trackedPoints[0].size());

	// Per tracked point barycentre in the image sequence. If a point is not found in a given image, that skews its barycentre over the sequence, and then the relative offsets. 
	// So we replace this missing point with an estimate, averagePointOffsetToImageBarycenter[j] + temporaryImageBarycentre. It is crucial that all points are found in the 
	// first image, which is guarantedd iff it is the reference frame for all optical flow computations.
	vector<Point2f> pointsBarycentres(trackedPoints[0].size());

	// The offset of each image relative to a virtual common reference frame.
	std::vector<Point2f> imageOffsets(trackedPoints.size());

	for (int j = 0; j < trackedPoints[0].size(); j++)
	{
		averagePointOffsetToImageBarycenter[j] = Point2f(.0f, .0f);
		pointsBarycentres[j] = Point2f(.0f, .0f);
	}
	for (int i = 0; i < trackedPoints.size(); i++)
	{
		Point2f temporaryImageBarycentre(.0f, .0f);

		int nPts = 0;
		for (int j = 0; j < trackedPoints[i].size(); j++)
		{
			if (std::isnan(trackedPoints[i][j].x)) continue;

			pointsBarycentres[j] += trackedPoints[i][j];
			temporaryImageBarycentre += trackedPoints[i][j];
			nPts++;
		}

		temporaryImageBarycentre /= (float) nPts;

		for (int j = 0; j < trackedPoints[i].size(); j++)
		{
			if (std::isnan(trackedPoints[i][j].x)) 
			{ 
				Point2f estimatedPosition = averagePointOffsetToImageBarycenter[j] / (float) i + temporaryImageBarycentre;
				pointsBarycentres[j] += estimatedPosition;
				averagePointOffsetToImageBarycenter[j] += estimatedPosition - temporaryImageBarycentre; // Keeps the average unchanged, but since we store the sum it lets us keep the count at i. (increment it by 1)
			}
			else
			{
				averagePointOffsetToImageBarycenter[j] += trackedPoints[i][j] - temporaryImageBarycentre;
			}
		}
	}

	for (int j = 0; j < trackedPoints[0].size(); j++)
	{
		pointsBarycentres[j] /= (float)trackedPoints.size();
	}


	for (int i = 0; i < trackedPoints.size(); i++)
	{
		int nPts = 0;

		// Points in the i-th image, relative to their respective barycenters over the sequence
		Point2f avgPointsOffsetToTheirBarycentres(.0f,.0f);
		for (int j = 0; j < trackedPoints[0].size(); j++)
		{
			if (std::isnan(trackedPoints[i][j].x)) continue;

			avgPointsOffsetToTheirBarycentres += trackedPoints[i][j] - pointsBarycentres[j];
			nPts++;
		}

		if (nPts != 0) avgPointsOffsetToTheirBarycentres /= (float) nPts;

		imageOffsets[i] = avgPointsOffsetToTheirBarycentres;
	}

	return imageOffsets;
}

std::vector<Point2f> Optimizer::computeOffsetsOld(std::vector<std::vector<cv::Point2f>> trackedPoints)
{
	// The barycenter of all tracked points in each image.
	vector<Point2f> barycentres(trackedPoints[0].size());

	// The offset of each image relative to a virtual common reference frame.
	std::vector<Point2f> imageOffsets(trackedPoints.size());

	for (int j = 0; j < trackedPoints[0].size(); j++)
	{
		barycentres[j] = Point2f(.0f, .0f);
	}
	for (int j = 0; j < trackedPoints.size(); j++)
	{
		imageOffsets[j] = Point2f(.0f, .0f);
	}
	for (int i = 0; i < trackedPoints.size(); i++)
	{
		int nPts = 0;
		for (int j = 0; j < trackedPoints[i].size(); j++)
		{
			if (std::isnan(trackedPoints[i][j].x)) continue;
			barycentres[j] += trackedPoints[i][j];
			imageOffsets[i] += trackedPoints[i][j];
			nPts++;
		}
		imageOffsets[i] /= (float)nPts;
	}


	Point2f stackCenter(.0f, .0f);
	for (int j = 0; j < trackedPoints[0].size(); j++)
	{
		barycentres[j] /= (float)trackedPoints.size();
		stackCenter += barycentres[j];
	}
	stackCenter /= (float) trackedPoints[0].size();
	for (int j = 0; j < trackedPoints.size(); j++)
	{
		imageOffsets[j] -= stackCenter;
	}

	return imageOffsets;
}

void Optimizer::DelaunayUnwarping(std::vector<cv::Mat> images, std::vector<std::vector<cv::Point2f>> shiftedPoints)
{
	// The offset of each image relative to a common reference frame.
	vector<Point2f> globalMovement = computeOffsetsOld(shiftedPoints);

	// The "true" position of each tracked point
	vector<Point2f> perPointBarycenter(shiftedPoints[0].size());

	for (int j = 0; j < perPointBarycenter.size(); j++) perPointBarycenter[j] = Point2f(0.f, 0.f);

	for (int i = 0; i < shiftedPoints.size(); i++)
	{
		for (int j = 0; j < shiftedPoints[i].size(); j++)
		{
			perPointBarycenter[j] += shiftedPoints[i][j];
		}
	}

	for (int j = 0; j < perPointBarycenter.size(); j++) perPointBarycenter[j] /= (float) shiftedPoints[0].size();



	// additional points on the edges.
	int nPointsPerSide = 5;

	std::vector<double> coords(shiftedPoints[0].size() * 2 + nPointsPerSide * 4 * 2);

	int offset = (int)shiftedPoints[0].size() * 2;

	// Vertical edges
	for (int i = 0; i < nPointsPerSide; i++)
	{
		double py0 = 1.;
		double py1 = (double)images[0].rows - 2.;
		double px = (double)i * (double)images[0].cols / (double)nPointsPerSide;

		coords[offset++] = px;
		coords[offset++] = py0;
		
		coords[offset++] = px;
		coords[offset++] = py1;
	}

	// Horizontal edges
	for (int i = 0; i < nPointsPerSide; i++)
	{
		double py = 1. + (double)(i + 1) * (double)(images[0].rows - 2) / (double)nPointsPerSide;
		double px0 = 1.;
		double px1 = (double)images[0].cols - 2.;

		coords[offset++] = px0;
		coords[offset++] = py;
		coords[offset++] = px1;
		coords[offset++] = py;
	}

	int originalSize = (int) perPointBarycenter.size();
	perPointBarycenter.resize(originalSize + nPointsPerSide * 4);
	for (int j = originalSize; j < originalSize + 4 * nPointsPerSide; j++)
	{
		perPointBarycenter[j] = Point2f((float)coords[2 * j], (float)coords[2 * j + 1]);
	}
	
	

	for (int i = 0; i < images.size(); i++)
	{
		Mat scratchpad = Mat::zeros(images[0].size(), images[0].type());
		Mat unwarped = Mat::zeros(images[0].size(), images[0].type());
		//Mat maskSums = Mat::zeros(images[0].size(), images[0].type());

#ifdef _DEBUG
		Mat meshOnImage = images[i].clone();
#endif
		for (int j = 0; j < shiftedPoints[0].size(); j++)
		{
#ifdef _DEBUG
			circle(meshOnImage, shiftedPoints[i][j], 3, Scalar(255), -1);
#endif
			coords[2 * j] = (double)shiftedPoints[i][j].x;
			coords[2 * j + 1] = (double)shiftedPoints[i][j].y;
		}

		//triangulation happens inside the constructor.
		delaunator::Delaunator d(coords);


		for (std::size_t j = 0; j < d.triangles.size(); j += 3)
		{
			//printf(
			//	"Triangle points: [[%f, %f], [%f, %f], [%f, %f]]\n",
			//	d.coords[2 * d.triangles[i]],        //tx0
			//	d.coords[2 * d.triangles[i] + 1],    //ty0
			//	d.coords[2 * d.triangles[i + 1]],    //tx1
			//	d.coords[2 * d.triangles[i + 1] + 1],//ty1
			//	d.coords[2 * d.triangles[i + 2]],    //tx2
			//	d.coords[2 * d.triangles[i + 2] + 1] //ty2
			//);

			int id0 = d.triangles[j];
			int id1 = d.triangles[j+1];
			int id2 = d.triangles[j+2];

			std::vector<Point2f> srcTri(3);
			srcTri[0] = Point2f((float)d.coords[2 * id0], (float)d.coords[2 * id0 + 1]);
			srcTri[1] = Point2f((float)d.coords[2 * id1], (float)d.coords[2 * id1 + 1]);
			srcTri[2] = Point2f((float)d.coords[2 * id2], (float)d.coords[2 * id2 + 1]);

#ifdef _DEBUG
			cv::line(meshOnImage, srcTri[0], srcTri[1], Scalar(255), 2);
			cv::line(meshOnImage, srcTri[1], srcTri[2], Scalar(255), 2);
			cv::line(meshOnImage, srcTri[2], srcTri[0], Scalar(255), 2);
#endif

			std::vector<Point2f> dstTri(3);
			dstTri[0] = perPointBarycenter[id0];
			dstTri[1] = perPointBarycenter[id1];
			dstTri[2] = perPointBarycenter[id2];

			Mat warp_mat = getAffineTransform( srcTri, dstTri );

			cv::Rect srcROI = cv::boundingRect(srcTri);
			cv::Rect dstROI = cv::boundingRect(dstTri);

			std::vector<Point2i> maskTri(3);
			maskTri[0] = (Point2i) srcTri[0] - srcROI.tl();
			maskTri[1] = (Point2i) srcTri[1] - srcROI.tl();
			maskTri[2] = (Point2i) srcTri[2] - srcROI.tl();

			// we must undo the above.
			warp_mat.at<double>(0, 2) += srcROI.x;
			warp_mat.at<double>(1, 2) += srcROI.y;

			cv::Mat srcMask = cv::Mat::zeros(srcROI.size(), images[0].type());
			fillConvexPoly(srcMask, maskTri, 1, cv::LINE_8, 0);

			cv::Mat extractedSrcTriangle;
			cv::multiply(images[i](srcROI), srcMask, extractedSrcTriangle);

			warpAffine(extractedSrcTriangle, scratchpad, warp_mat, scratchpad.size());

			cv::add(scratchpad(dstROI), unwarped(dstROI), unwarped(dstROI));
		}

		cv::imshow("Triangulation", unwarped);
		int keyboard = waitKey(50);
	}
}


const int SQUARE_OVERLAP = 3; // on each side
const int SQUARE_SPACING = 50;
const int SHARPNESS_DOWNSCALE = 2;
const int TRACKING_DOWNSCALE = 2;
const int SQUARE_SIZE = SQUARE_SPACING + 2*SQUARE_OVERLAP; 

void Optimizer::ShowSharpnessRanking(std::vector<cv::Mat> images, std::vector<vector<Point2f>> trackedPoints)
{

	int LR_w = images[0].cols;
	int LR_h = images[0].rows;

	vector<Point2f> globalMovement = computeOffsetsOld(trackedPoints);

	// Is here, but could happen as soon as calcOpticalFlowPyrLK has been called on the image, because it is the only reason images are in CV_8U in the first place
	for (int i = 0; i < images.size(); i++)
	{
		images[i].convertTo(images[i], CV_32F);

#ifdef _DEBUG
		images[i] /= 255.f;
#endif
	}


	// Create a vector of matrices, the i-th matrix contains the sharpness of the squares in the i-th image of the sequence.
	std::vector<cv::Mat> sharpness(images.size());

	// Buffer to avoid reallocs
	cv::Mat laplacian(images[0].size(), CV_32F);

	// Compute sharpnesses
	for (int i = 0; i < images.size(); i++)
	{
		sharpness[i] = Mat(Size(LR_w/SQUARE_SIZE, LR_h/SQUARE_SIZE), CV_32F);

		cv::Laplacian(images[i], laplacian, CV_32F);

		for (int x = 1; x < sharpness[i].cols-1; x++)
		{
			for (int y = 1; y < sharpness[i].rows-1; y++)
			{
				Scalar varLaplacian;
				Scalar meanLaplacian;

				// TODO ints ?
				Rect _roi(x * SQUARE_SIZE - (int)globalMovement[i].x, y * SQUARE_SIZE - (int)globalMovement[i].y, SQUARE_SIZE, SQUARE_SIZE);

				cv::meanStdDev(images[i](_roi), meanLaplacian, varLaplacian);

				sharpness[i].at<float>(y, x) = (float) varLaplacian[0]; // accessors reverse the order for some reason
			}
		}
	}

	// TODO better sharpness


	int nLoops = 1000;

	std::vector<int> ids(images.size());
	std::vector<float> sharpnesses_xy(images.size());

	// Doivent être > 1, < max - 1
	//int x = 8, y = 5; // tuiles proches
	//int x = 9, y = 6; // arbres proches
	int x = 12, y = 1;// hornisgrinde HD

	// sort ids by descending sharpness
	for (int i = 0; i < images.size(); i++) 
	{
		ids[i] = i;
		sharpnesses_xy[i] = sharpness[i].at<float>(y, x);
	}
	std::sort(ids.begin(), ids.end(),
		[&sharpnesses_xy](int id_a, int id_b) {
			return sharpnesses_xy[id_a] > sharpnesses_xy[id_b];
		});

	const int VISUALISATION_UPSCALE = 4;
	cv::Mat native = Mat(Size(SQUARE_SIZE, SQUARE_SIZE), CV_32F);
	cv::Mat _upscaled = Mat(Size(SQUARE_SIZE* VISUALISATION_UPSCALE, SQUARE_SIZE* VISUALISATION_UPSCALE), CV_32F);
	for (int _i = 0; _i < nLoops*images.size(); _i++)
	{
		int i = ids[_i%images.size()];

		std::cout << sharpnesses_xy[i] << std::endl;

		Point2f center{ (float) (SQUARE_SPACING * x - SQUARE_OVERLAP), (float) (SQUARE_SPACING * y - SQUARE_OVERLAP) };
		center += Point2f((float)(SQUARE_SIZE/2),(float)(SQUARE_SIZE/2));
		center += globalMovement[i];
		cv::getRectSubPix(images[i], Size(SQUARE_SIZE, SQUARE_SIZE), center, native);

		cv::resize(native, _upscaled, _upscaled.size());
		cv::imshow("Best to worst", _upscaled);
		int keyboard = waitKey(50);

		if (_i%images.size() == images.size()-1) int _keyboard = waitKey(1000);
	}
}


void Optimizer::ShowBarycentricStabilization(std::vector<cv::Mat> images, std::vector<vector<Point2f>> trackedPoints)
{
	const int PADDING_SIZE = 30; // In pixels 

	int LR_w = images[0].cols;
	int LR_h = images[0].rows;



	bool useOldMethod = true;

	vector<Point2f> globalMovement;
	if (useOldMethod) globalMovement = computeOffsetsOld(trackedPoints);
	else globalMovement = computeOffsetsNew(trackedPoints);


	int nLoops = 1000;
	cv::Mat frame(images[0].size(), CV_8U);
	for (int _i = 0; _i < nLoops*images.size(); _i++)
	{
		int i = _i%images.size();
		Rect _roi;
		if ((_i / images.size())%2 == 1)
		{
			cv::imshow("Raw", images[i]);
			int keyboard = waitKey(50);
		}
		else {
			int w = LR_w - PADDING_SIZE; // PADDING because the subpixels rect is shiftd by the global movement and we cant extract an out of bounds ROI
			int h = LR_h - PADDING_SIZE;

			Point2f center{ (float) (w / 2), (float) (h / 2) };
			center += Point2f((float)(PADDING_SIZE/2),(float)(PADDING_SIZE/2));
			center += globalMovement[i];
			cv::getRectSubPix(images[i], Size(w, h), center, frame);

			cv::imshow("Stabilized", frame);
			int keyboard = waitKey(50);
		}
			
			

		if (_i < images.size()) std::cout << globalMovement[i] << std::endl;

		if (_i % images.size() == 0) 
		{
			cv::destroyAllWindows();
			std::cout << "Changing mode" << std::endl;
		}
	}
	
}



void remapAdd(cv::Mat LR_CHAR_src, cv::Mat SR_FLOAT_dst, float UPSCALE_FACTOR, Point2f inSrcOffset)
{
	for (int x = 0; x < SR_FLOAT_dst.cols; x++)
	{
		for (int y = 0; y < SR_FLOAT_dst.rows; y++)
		{
			// For some reason there is no std round to int, but it should not be a problem as srcX and srcY should not leave the range of correctly represented integers
			// using float values. See https://www.h-schmidt.net/FloatConverter/IEEE754.html, correct up to ~10 000 000
			int srcX = (int) std::round( inSrcOffset.x +  ((float) x) / UPSCALE_FACTOR);  // or static_cast<int> ? Or interpolation if on the edge ? TODO
			int srcY = (int) std::round( inSrcOffset.y +  ((float) y) / UPSCALE_FACTOR);  // or static_cast<int> ? Or interpolation if on the edge ? TODO
			SR_FLOAT_dst.at<float>(y, x) += static_cast<float>(LR_CHAR_src.at<uchar>(srcY, srcX));
		}
	}
}

// AKA "sharp stack", https://www.researchgate.net/publication/269107332_Generation_of_Super-Resolution_Stills_from_Video
void remapNearest(cv::Mat LR_CHAR_src, cv::Mat SR_FLOAT_dst, cv::Mat SR_FLOAT_distanceBuffer, int UPSCALE_FACTOR, Point2f inSrcOffset)
{
	// Implementation optimized so as to minimize branch predictions. Not SIMD friendly anyway.


	// Cant use LR_CHAR_src because it is the full image and not a ROI
	int nSrcCols = SR_FLOAT_dst.cols / UPSCALE_FACTOR;
	int nSrcRows = SR_FLOAT_dst.rows / UPSCALE_FACTOR;

	for (int xx = 0; xx < UPSCALE_FACTOR; xx++)
	{
		for (int yy = 0; yy < UPSCALE_FACTOR; yy++)
		{
			float fracSrcX = inSrcOffset.x +  ((float) xx) / (float)UPSCALE_FACTOR;
			float fracSrcY = inSrcOffset.y +  ((float) yy) / (float)UPSCALE_FACTOR;


			// For some reason there is no std round to int, but it should not be a problem as srcX and srcY should not leave the range of correctly represented integers
			// using float values. See https://www.h-schmidt.net/FloatConverter/IEEE754.html, correct up to ~10 000 000
			int srcX = (int) std::round(fracSrcX); 
			int srcY = (int) std::round(fracSrcY); 

			float dx = srcX - fracSrcX;
			float dy = srcY - fracSrcY;

			float d2 = dx*dx + dy*dy;

			// The buffer contains the previous smallest squared distance to an LR pixel.
			float currentClosest = SR_FLOAT_distanceBuffer.at<float>(yy, xx);
			if (currentClosest > d2)
			{
				SR_FLOAT_distanceBuffer.at<float>(yy, xx) = d2;


				for (int x = 0; x < nSrcCols; x++)
				{
					for (int y = 0; y < nSrcRows; y++)
					{
						SR_FLOAT_dst.at<float>(yy + y * UPSCALE_FACTOR, xx + x * UPSCALE_FACTOR) = static_cast<float>(LR_CHAR_src.at<uchar>(srcY + y, srcX + x));
					}
				}
			}
		}
	}
}


//#define SHARP_STACK

cv::Mat Optimizer::RecursiveMatching(std::vector<cv::Mat> images, std::vector<vector<Point2f>> trackedPoints)
{
	const int UPSCALE_FACTOR = 1;

	// between 0 and 100
	const float STACKED_PERCENTILE = 10.f;

	int LR_w = images[0].cols;
	int LR_h = images[0].rows;


	// The offset of this image in the sequence, relative to a common imaginary reference frame.
	// Approximate, because distorsions and blur induce inaccuracies in the point matching (opt flow).
	// Is used as a first estimate before refining per fragment.
	vector<Point2f> globalMovement = computeOffsetsOld(trackedPoints);

	int nSquareX = LR_w / SQUARE_SPACING;
	int nSquareY = LR_h / SQUARE_SPACING;


	// Create a vector of matrices, the i-th matrix contains the sharpness of the squares in the i-th image of the sequence.
	std::vector<cv::Mat> sharpness(images.size());

	// Buffer to avoid reallocs
	cv::Mat laplacian(images[0].size(), CV_32F);




	// Compute sharpnesses
	auto time1 = high_resolution_clock::now();
	Mat _sharpnessComputationBuffer(Size(SQUARE_SIZE/SHARPNESS_DOWNSCALE, SQUARE_SIZE/SHARPNESS_DOWNSCALE), CV_8U); // Downscaled to negate noise a bit
	for (int i = 0; i < images.size(); i++)
	{
		sharpness[i] = Mat(Size(nSquareX, nSquareY), CV_32F);

		cv::Laplacian(images[i], laplacian, CV_32F);

		for (int x = 1; x < sharpness[i].cols - 1; x++)
		{
			for (int y = 1; y < sharpness[i].rows - 1; y++)
			{
				Scalar varLaplacian;
				Scalar meanLaplacian;

				Rect _roi(
					x * SQUARE_SPACING - SQUARE_OVERLAP + (int)globalMovement[i].x,
					y * SQUARE_SPACING - SQUARE_OVERLAP + (int)globalMovement[i].y,
					SQUARE_SIZE,
					SQUARE_SIZE
				);

				cv::resize(images[i](_roi), _sharpnessComputationBuffer, _sharpnessComputationBuffer.size(), 0, 0, INTER_LINEAR);
				cv::meanStdDev(_sharpnessComputationBuffer, meanLaplacian, varLaplacian);

				sharpness[i].at<float>(y, x) = (float)varLaplacian[0]; 
			}
		}
	}
	auto time2 = high_resolution_clock::now();
	int ms_int = (int) duration_cast<milliseconds>(time2 - time1).count();
	std::cout << "\nSharpness computations took " << ms_int << " ms, i.e. " << (ms_int / (int)images.size()) << " ms per image." << std::endl;

	// TODO better sharpness


	// TODO nearest neighbor, as per "sharp stack"




	// preallocated. The to-be-sorted images ids in the sequence.
	std::vector<int> ids(images.size());
	// preallocated, to simplify sorting
	std::vector<float> sharpnesses_xy(images.size());
	


	// For each image, the number of its fragments that are used in the sub stacks. 
	std::vector<int> nUsedFragsPerImage(images.size());
	for (int i = 0; i < images.size(); i++) nUsedFragsPerImage[i] = 0;


	// The square upscaled stacked fragments. Will be merged into the stacked image later.
	std::vector<std::vector<cv::Mat>> fragments;

	// yayx
	std::vector<std::vector<std::vector<Point2f>>> perFragmentImageOffsets;
	std::vector<std::vector<std::vector<int>>> perFragmentImageIDs;


	fragments.resize(nSquareX);
	perFragmentImageOffsets.resize(nSquareX);
	perFragmentImageIDs.resize(nSquareX);
	for (int x = 1; x < sharpness[0].cols - 1; x++) {
		fragments[x].resize(nSquareY);
		perFragmentImageOffsets[x].resize(nSquareY);
		perFragmentImageIDs[x].resize(nSquareY);
	}
	


	// Just the target number of the sharpest fragments to stack.
	int nStackedSquares = (int)((float)images.size() * STACKED_PERCENTILE / 100.f);

#ifdef _DEBUG
	Mat debugger = Mat::zeros(Size(nSquareX, nSquareY), CV_8U); // to monitor how many fragments were matched to the sharpest
#endif

	time1 = high_resolution_clock::now();
	for (int x = 1; x < sharpness[0].cols - 1; x++)
	{
		for (int y = 1; y < sharpness[0].rows - 1; y++)
		{
			// sort ids by descending sharpness
			for (int i = 0; i < images.size(); i++)
			{
				ids[i] = i;
				sharpnesses_xy[i] = sharpness[i].at<float>(y, x);
			}
			std::sort(ids.begin(), ids.end(),
				[&sharpnesses_xy](int id_a, int id_b) {
					return sharpnesses_xy[id_a] > sharpnesses_xy[id_b];
				});


			

			const int REGION_SIZE = 5;
			const int MAX_POINTS_TRACKED = 20;
			const double MIN_DISTANCE = (double)(SQUARE_SIZE / 8);
			const double QUALITY_LEVEL = .1; // default .3

			// The size of the expected maximum error in barycenter computations. That is, points within the reference square - CROP_SIZE on each side 
			// are expected to be found in all the other sharp fragments. Not a big issue if they arent
			const int CROP_SIZE = 3; 

			int N_REFERENCES = std::min(1, nStackedSquares);

			// features coordinates in the sharpest fragment
			vector<vector<Point2f>> vecReferencePoints(N_REFERENCES);

			for (int i = 0; i < N_REFERENCES; i++)
			{
				int _id0 = ids[i];


				// A cropped ROI is used (-CROP_SIZE pixels at both ends) to avoid tracking pixels that could disappear 
				int ROI0_x_cropped = x * SQUARE_SPACING - SQUARE_OVERLAP + (int)globalMovement[_id0].x + CROP_SIZE; // top left
				int ROI0_y_cropped = y * SQUARE_SPACING - SQUARE_OVERLAP + (int)globalMovement[_id0].y + CROP_SIZE; // top left
				int ROI0_w_cropped = SQUARE_SIZE - 2 * CROP_SIZE;
				int ROI0_h_cropped = SQUARE_SIZE - 2 * CROP_SIZE;
				Rect ROI0_cropped(ROI0_x_cropped, ROI0_y_cropped, ROI0_w_cropped, ROI0_h_cropped); 

				vector<Point2f> localReferencePoints;
				goodFeaturesToTrack(images[_id0](ROI0_cropped), localReferencePoints, MAX_POINTS_TRACKED, QUALITY_LEVEL, MIN_DISTANCE, Mat(), REGION_SIZE, false, 0.04);			


				// If no good features were found, there is no point stacking because we wont get subpixel accuracy, so we keep things as they are and go on to stack the next square sequence.
				// At these coordinates, the stack will only contain the sharpest fragment that was put in above.
				if (localReferencePoints.size() == 0)
				{
					continue;
				}

				
				// TODO perform what follows only on the best frag


				// Refine the corners to subpixel precision
				Size winSize = Size( REGION_SIZE/2, REGION_SIZE/2 );
				Size zeroZone = Size( -1, -1 );
				TermCriteria _criteria = TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001 );
				cornerSubPix( images[_id0](ROI0_cropped), localReferencePoints, winSize, zeroZone, _criteria);

				// We are interested in coordinates relative to the uncropped ROI, ROI0.
				for (int j = 0; j < localReferencePoints.size(); j++) localReferencePoints[j] += Point2f((float)CROP_SIZE, (float)CROP_SIZE);

				vecReferencePoints[i] = std::move(localReferencePoints);
			}
			
			int argmaxVecReferencePoints = 0;
			int _id0 = ids[argmaxVecReferencePoints];

			fragments[x][y] = Mat::zeros(Size(SQUARE_SIZE*UPSCALE_FACTOR, SQUARE_SIZE*UPSCALE_FACTOR), CV_32F);


			// We use the integer-rounded coordinates of globalMovement[_id0], because this simplifies things later on when we compare the i-th fragment's points position.
			// ROI's coordinates are ints, and nothing we can do about it.
			int ROI0_x = x * SQUARE_SPACING - SQUARE_OVERLAP + (int)globalMovement[_id0].x; // top left
			int ROI0_y = y * SQUARE_SPACING - SQUARE_OVERLAP + (int)globalMovement[_id0].y; // top left
			Rect ROI0(ROI0_x, ROI0_y, SQUARE_SIZE, SQUARE_SIZE);

			

			// Initialize the fragment stack with the sharpest fragment:
#ifdef SHARP_STACK
			Mat squaredDistancesBuffer = Mat::ones(Size(UPSCALE_FACTOR,UPSCALE_FACTOR), CV_32F) * UPSCALE_FACTOR*UPSCALE_FACTOR;
			remapNearest(images[_id0], fragments[x][y], squaredDistancesBuffer, UPSCALE_FACTOR, Point2f((float)ROI0_x, (float)ROI0_y));
#else
			remapAdd(images[_id0], fragments[x][y], (float) UPSCALE_FACTOR, Point2f((float)ROI0_x, (float)ROI0_y));
#endif


			// Note that the offsets are whole numbers (floats though), because we used the integer coordinates of the ROI for the remapAdd
			nUsedFragsPerImage[_id0]++;
			perFragmentImageOffsets[x][y].push_back(Point2f(.0f,.0f));
			perFragmentImageIDs[x][y].push_back(_id0);


			if (vecReferencePoints[argmaxVecReferencePoints].size() == 0) continue;


			int nEffectivelyStackedSquares = 1; // To normalize the stack fragment, if not sharpStack.
		

			// match all the other sharpest squares to the original one, and add them to fragments[x][y].
			vector<Point2f> potentialPoints(vecReferencePoints[argmaxVecReferencePoints].size());
			vector<Point2f> pointsDetectedInCurrentSquare(vecReferencePoints[argmaxVecReferencePoints].size());
			vector<uchar> status;
			vector<float> err;
			TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);
			for (int i = 1; i < nStackedSquares; i++)
			{
				int _id = ids[i];


				int ROIi_x = x * SQUARE_SPACING - SQUARE_OVERLAP + (int)globalMovement[_id].x; // top left
				int ROIi_y = y * SQUARE_SPACING - SQUARE_OVERLAP + (int)globalMovement[_id].y; // top left
				int ROIi_w = SQUARE_SIZE;
				int ROIi_h = SQUARE_SIZE;
				Rect ROIi(ROIi_x, ROIi_y, ROIi_w, ROIi_h);

				if (vecReferencePoints[argmaxVecReferencePoints].size() == 0) __debugbreak();

				// calculate optical flow
				pointsDetectedInCurrentSquare.resize(0);
				status.resize(0);
				err.resize(0);
				//TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);
				calcOpticalFlowPyrLK(images[_id0](ROI0), images[_id](ROIi), vecReferencePoints[argmaxVecReferencePoints], potentialPoints, status, err, Size(7, 7), 2, criteria, 0, 1.0E-3);

				// TODO reuse the zero-th pyramid


				Point2f localMovement(.0f, .0f);
				int nPts = 0;
				for (uint j = 0; j < potentialPoints.size(); j++)
				{
					//TODO exclude those with too high a returned error 
					//TODO RANSAC with pointsDetectedInCurrentSquare
					if (status[j] == 1) {
						pointsDetectedInCurrentSquare.push_back(potentialPoints[j]);
						localMovement += (potentialPoints[j] - vecReferencePoints[argmaxVecReferencePoints][j]);
						nPts++;
					}
				}

				if (nPts == 0) continue;


#ifdef _DEBUG
				Mat maskFrag = Mat::ones(fragments[x][y].size(), CV_32F);
				Mat maskCurrent = Mat::ones(ROIi.size(), CV_32F);
				for (uint j = 0; j < potentialPoints.size(); j++) {
					if (status[j] == 1) {
						circle(maskFrag, vecReferencePoints[argmaxVecReferencePoints][j] * UPSCALE_FACTOR, 2, Scalar(.8f), -1);
					}
					else {
						circle(maskFrag, vecReferencePoints[argmaxVecReferencePoints][j] * UPSCALE_FACTOR, 1, Scalar(.8f), -1);
					}
				}
				for (uint j = 0; j < pointsDetectedInCurrentSquare.size(); j++) {
					circle(maskCurrent, pointsDetectedInCurrentSquare[j], 1, Scalar(.8f), -1);
				}
				
				multiply(maskFrag, fragments[x][y], maskFrag, 1.f/255.f);
				multiply(maskCurrent, images[_id](ROIi), maskCurrent, 1.f/255.f, CV_32F);

				int _aaaa = 0;
				//if (abs(x * SQUARE_SPACING - 600) < SQUARE_SPACING)
				//{
				//	//__debugbreak();
				//}
					
#endif

				nEffectivelyStackedSquares++;

				localMovement /= (float)nPts;


				Point2f intROI_i_tl((float)ROIi_x, (float)ROIi_y);

				// Such that frag[ (i,j) ] = Image_id[ (i,j) / UPSCALE_FACTOR + localMovement + intROI_i_tl]. The sub pixel mapping depends on the remap implementation.

#ifdef SHARP_STACK
				remapNearest(images[_id], fragments[x][y], squaredDistancesBuffer, UPSCALE_FACTOR, localMovement + intROI_i_tl);
#else
				remapAdd(images[_id], fragments[x][y], (float) UPSCALE_FACTOR, localMovement + intROI_i_tl);
#endif
				nUsedFragsPerImage[_id]++;
				perFragmentImageOffsets[x][y].push_back(localMovement);
				perFragmentImageIDs[x][y].push_back(_id);


				
			}

#ifdef _DEBUG
			debugger.at<char>(y, x) = nEffectivelyStackedSquares;
#endif


#ifdef SHARP_STACK
			//No normalization needed
#else
			fragments[x][y] *= 1.f / (float)nEffectivelyStackedSquares;
#endif
		}
	}
	time2 = high_resolution_clock::now();
	ms_int = (int) duration_cast<milliseconds>(time2 - time1).count();
	std::cout << "\nFragment substacking computations took " << ms_int << " ms, i.e. more than " << (ms_int / ((int)images.size() * nStackedSquares)) << " ms per sharp fragment." << std::endl;








	// The offset of each fragment in the stack (before upscaling the coordinates)
	std::vector<std::vector<Point2f>> fragmentOffsets(nSquareX); 

	// true (1) if we know an estimate of this frag's offset, false (0) otherwise.
	std::vector<std::vector<int>> knownFragmentsOffsets(nSquareX); 

	for (int x = 1; x < sharpness[0].cols - 1; x++) 
	{
		fragmentOffsets[x].resize(nSquareY); // Need not be 0 initialized
		knownFragmentsOffsets[x].resize(nSquareY);

		std::fill(knownFragmentsOffsets[x].begin(), knownFragmentsOffsets[x].end(), 0);
	}

	// How many frags offsets we have an approximation of
	int nAdjustedFrags = 0;
	int nAdjustableFrags = (nSquareX - 2) * (nSquareY - 2);

	// The offset of each image in the stack (before upscaling the coordinates). Need not be 0 initialized.
	std::vector<Point2f> imagesOffsets(images.size()); 

	// true (1) if we know an estimate of this image's offset, false (0) otherwise.
	std::vector<int> knownImagesOffsets(images.size()); 
	std::fill(knownImagesOffsets.begin(), knownImagesOffsets.end(), 0);


	// Initialize the images offsets with that of the image having the most stacked fragments across the substacks.
	{	
		int id0 = 0;
		float nSharps = 0.f;
		for (int i = 0; i < images.size(); i++)
		{
			if (nSharps < nUsedFragsPerImage[i])
			{
				nSharps = (float) nUsedFragsPerImage[i];
				id0 = i;
			}
		}
		imagesOffsets[id0] = Point2f(.0f, .0f);
		knownImagesOffsets[id0] = 1;
	}

	// The values that the vectors above will take at the next iteration.
	std::vector<Point2f> nextImagesOffsets(images.size()); 
	std::vector<int> nextImagesOffsetsNormalizer(images.size()); 


	int step = 0;
	const int MAX_STEPS = 10;
	while (step < MAX_STEPS) // && non stationarity check with nAdjustedFrags
	{	
		int _initial_nAdjustedFrags = nAdjustedFrags;

		for (int i = 0; i < images.size(); i++)
		{
			nextImagesOffsets[i] = Point2f(.0f,.0f);
			nextImagesOffsetsNormalizer[i] = 0;
		}

		for (int x = 1; x < sharpness[0].cols - 1; x++)
		{
			for (int y = 1; y < sharpness[0].rows - 1; y++)
			{
				//if (knownFragmentsOffsets[x][y]) continue; // Vraiment ? Sans ça, on tient un procédé itératif convergent non ?

				fragmentOffsets[x][y] = Point2f(.0f, .0f);
				int nKnownImageOffsetsInThisFrag = 0;

				for (int j = 0; j < perFragmentImageIDs[x][y].size(); j++)
				{
					int _id = perFragmentImageIDs[x][y][j];

					if (knownImagesOffsets[_id])
					{
						fragmentOffsets[x][y] += imagesOffsets[_id] - perFragmentImageOffsets[x][y][j];
						nKnownImageOffsetsInThisFrag++;
					}
				}

				if (nKnownImageOffsetsInThisFrag == 0) continue;


				fragmentOffsets[x][y] /= (float) nKnownImageOffsetsInThisFrag;
				if (!knownFragmentsOffsets[x][y])
				{
					nAdjustedFrags++;
					knownFragmentsOffsets[x][y] = 1;
				}

				

				for (int j = 0; j < perFragmentImageIDs[x][y].size(); j++)
				{
					int _id = perFragmentImageIDs[x][y][j];

					nextImagesOffsets[_id] += fragmentOffsets[x][y] + perFragmentImageOffsets[x][y][j];
					nextImagesOffsetsNormalizer[_id]++;
				}

				
			}
		}


		for (int i = 0; i < images.size(); i++)
		{
			if (nextImagesOffsetsNormalizer[i] == 0) continue;

			imagesOffsets[i] = nextImagesOffsets[i] / (float)nextImagesOffsetsNormalizer[i];
			knownImagesOffsets[i] = 1;
		}

		step++;

		// We terminate if there has not been any new fragment covered at this step. That means the graph isnt connex, and the optimization wont progress anymore.
		// (well, it could improve on the already known frags, but the unknown frags will never be filled)
		if  (_initial_nAdjustedFrags == nAdjustedFrags && nAdjustedFrags != nAdjustableFrags) break;
	}


	Size SR_size(LR_w * UPSCALE_FACTOR, LR_h * UPSCALE_FACTOR);

	Mat stack = cv::Mat::zeros(SR_size, CV_32F);
	Mat normalizer = cv::Mat::zeros(SR_size, CV_32F);


	// TODO first place all substacks that contain only 1 frag because matching failed (typically the sky), then write the good substacks
	// over this initial background. Not sum, but overwrite. They are potentially poorly offset so we dont want them disrupting the good stacks.
	
	// We can't place the fragments at subpixel locations in the stack directly, we have to go the other way around,
	// we extract a subPixRect from the fragment, offset by -the fractional part of its offset. To make sure the subPixRect
	// is within the fragment, it must be 2 pixel smaller on each dimension. (1px at both ends)
	Mat subPixShiftedSquare = Mat(Size(SQUARE_SIZE * UPSCALE_FACTOR - 2, SQUARE_SIZE * UPSCALE_FACTOR - 2), CV_32F);
	Mat onesBlock = cv::Mat::ones(Size(SQUARE_SIZE * UPSCALE_FACTOR - 2, SQUARE_SIZE * UPSCALE_FACTOR - 2), CV_32F);

	for (int x = 1; x < sharpness[0].cols - 1; x++)
	{
		for (int y = 1; y < sharpness[0].rows - 1; y++)
		{

			Point2f fragOffset = fragmentOffsets[x][y] * (float) UPSCALE_FACTOR;
			
			// Such that frag[ (i,j) + fragOffset ] = Stack[ (x0,y0) + (i,j) ]

			Point2f intFragOffset(std::round(fragOffset.x), std::round(fragOffset.y));


			Point2f center(
				(float)SQUARE_SIZE * UPSCALE_FACTOR / 2.f,
				(float)SQUARE_SIZE * UPSCALE_FACTOR / 2.f);
			center += fragOffset - intFragOffset;

			cv::getRectSubPix(fragments[x][y], subPixShiftedSquare.size(), center, subPixShiftedSquare);


			Rect stackROI(
				(x * SQUARE_SPACING - SQUARE_OVERLAP) * UPSCALE_FACTOR - (int) intFragOffset.x + 1, // +1 for clarity, as this is where it would be without the 2px shrinkage, but
				(y * SQUARE_SPACING - SQUARE_OVERLAP) * UPSCALE_FACTOR - (int) intFragOffset.y + 1, // not really necessary since what matter is the relative position of the frags.
				SQUARE_SIZE * UPSCALE_FACTOR - 2,
				SQUARE_SIZE * UPSCALE_FACTOR - 2
			);

			
			cv::add(subPixShiftedSquare, stack(stackROI), stack(stackROI));
			cv::add(onesBlock, normalizer(stackROI), normalizer(stackROI));
		}
	}





	cv::divide(stack, normalizer, stack);

#ifdef _DEBUG
	// Needs be in [0,1] for examination in image watch
	stack *= 1.f/255.f;

	debugger.convertTo(debugger, CV_32F);
	debugger *= 1.f / (float) nStackedSquares;
	
	

	stack *= 255.f;
#endif


	Mat deconv;
	stack *= 1.f/255.f;
	for (int r = 0; r < 1; r++)
	{
		Rect deconvROI = cv::selectROI(stack);
		std::cout << "\n SELECTED CONVOLUTION ROI DIMENSIONS: " << deconvROI.x << " " << deconvROI.y << " " << deconvROI.width << " " << deconvROI.height << std::endl;
		cv::destroyAllWindows();
		//Rect deconvROI(1659, 422, 509, 375);
		Mat toBeDeconv = stack(deconvROI).clone();

		const int nSigmas = 3;
		float sigmas[nSigmas];
		for (int i = 0; i < nSigmas; i++)
		{
			sigmas[i] = (float) UPSCALE_FACTOR + (float) (i-1) * .5f;
			deconv = LucyRichardson(toBeDeconv, 10, sigmas[i]);
			deconv = LucyRichardson(toBeDeconv, 20, sigmas[i]);
			deconv = LucyRichardson(toBeDeconv, 30, sigmas[i]);

			Mat deconv8U = deconv.clone() * 255.f;
			deconv8U.convertTo(deconv8U, CV_8U);

			std::string name = "SR_samples\\" + seqName + "_" + std::to_string(UPSCALE_FACTOR) + "x_" + std::to_string((int)STACKED_PERCENTILE) + "percentOf"
								+ std::to_string((int)images.size()) + "images_sigma" + std::to_string(sigmas[i]) 
#ifdef SHARP_STACK
				+ "_SHARP_STACK_"
#else
				+ "_AVERAGE_STACK_"
#endif
				+ std::to_string(time(0)) + ".bmp";
			cv::imwrite(name, deconv8U);
		}
	}
	stack *= 255.f;

	//Mat unsharpMasked = UnsharpMasking(deconv);
	//Mat equalHist = EqualizeHistogram(deconv8U);

	




	
	

	
	stack.convertTo(stack, CV_8U);

	cv::imwrite("stack5.bmp", stack);

	cv::imshow("Stack", stack);

	int keyboard = waitKey(0);

	return stack;
}


cv::Mat Optimizer::LucyRichardson(cv::Mat img, int nIterations, float sigmaG)
{
	const float EPSILON = .000001f;


	// Window size of PSF
	//int winSize = 10 * sigmaG + 1 ;
	//int winSize = 6 * sigmaG + 1 ; TODO

	// Initializations
	Mat Y = img.clone();
	Mat J1 = img.clone();
	Mat J2 = img.clone();
	Mat wI = img.clone(); 
	Mat imR = img.clone();  
	Mat reBlurred = img.clone();    

	Mat T1, T2, tmpMat1, tmpMat2;
	T1 = Mat(img.rows,img.cols, CV_64F, 0.0);
	T2 = Mat(img.rows,img.cols, CV_64F, 0.0);

	// Lucy-Rich. Deconvolution CORE

	double lambda = 0;
	for(int j = 0; j < nIterations; j++) 
	{       
		if (j>1) {
			// calculation of lambda
			multiply(T1, T2, tmpMat1);
			multiply(T2, T2, tmpMat2);
			lambda=sum(tmpMat1)[0] / (sum( tmpMat2)[0]+EPSILON);
			// calculation of lambda
		}

		Y = J1 + lambda * (J1-J2);
		Y.setTo(0, Y < 0);

		// 1)
		GaussianBlur( Y, reBlurred, Size(), sigmaG);//applying Gaussian filter 
		reBlurred.setTo(EPSILON , reBlurred <= 0); 

		// 2)
		divide(wI, reBlurred, imR);
		imR = imR + EPSILON;

		// 3)
		GaussianBlur( imR, imR, Size(), sigmaG);//applying Gaussian filter 

		// 4)
		J2 = J1.clone();
		multiply(Y, imR, J1);

		T2 = T1.clone();
		T1 = J1 - Y;
	}

	// mats are shared ptrs
	return J1;
}


cv::Mat Optimizer::PostProcess(std::vector<cv::Mat> images)
{
	return Mat();
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

	//cv::imshow("Global Equalized Histogram", dst0);
	//cv::imshow("Local Equalized Histogram", dst1);
	//cv::imshow("src", src);
	//cv::waitKey();
	//cv::destroyAllWindows();

	return dst0;
}
