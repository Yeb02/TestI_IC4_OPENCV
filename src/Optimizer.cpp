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

void Optimizer::LoadAndPreProcess(IMAGE_SEQ _imSeq, bool showSeq, std::vector<cv::Mat>& dstBayerImgs, std::vector<cv::Mat>& dstRGBImgs)
{
	imSeq = _imSeq;
	std::string folder;

	switch (imSeq)
	{
	case JARDIN_NO_VIBRATION:
		seqName = "JARDIN_NO_VIBRATION";
		folder = "burst_j10_m11_a2025_12h_47m_56s";
		break;
	case JARDIN:
		seqName = "JARDIN";
		folder = "burst_j10_m11_a2025_16h_11m_09s";
		break;
	case HANGAR_1:
		seqName = "HANGAR_1";
		folder = "burst_j11_m11_a2025_12h_05m_14s";
		break;
	case HANGAR_2:
		seqName = "HANGAR_2";
		folder = "burst_j11_m11_a2025_12h_07m_30s";
		break;
	case HANGAR_ZOOM:
		seqName = "HANGAR_ZOOM";
		folder = "burst_j11_m11_a2025_12h_08m_12s";
		break;
	case ZI_FLOUE_1:
		seqName = "ZI_FLOUE_1";
		folder = "burst_j11_m11_a2025_12h_11m_01s";
		break;
	case ZI_FLOUE_2:
		seqName = "ZI_FLOUE_2";
		folder = "burst_j11_m11_a2025_12h_11m_50s";
		break;
	case ZI_3:
		seqName = "ZI_3";
		folder = "burst_j11_m11_a2025_12h_12m_25s";
		break;
	case PORT_DU_RHIN_1:
		seqName = "PORT_DU_RHIN_1";
		folder = "burst_j11_m11_a2025_12h_14m_25s";
		break;
	case CATHEDRALE_1:
		seqName = "CATHEDRALE_1";
		folder = "burst_j11_m11_a2025_12h_15m_31s";
		break;
	case PORT_DU_RHIN_2:
		seqName = "PORT_DU_RHIN_2";
		folder = "burst_j11_m11_a2025_12h_16m_15s";
		break;
	case MAISON_1:
		seqName = "MAISON_1";
		folder = "burst_j11_m11_a2025_12h_18m_13s";
		break;
	case MAISON_2:
		seqName = "MAISON_2";
		folder = "burst_j11_m11_a2025_12h_19m_09s";
		break;
	case MAISON_3:
		seqName = "MAISON_3";
		folder = "burst_j11_m11_a2025_12h_20m_29s";
		break;
	case GRAVIERE:
		seqName = "GRAVIERE";
		folder = "burst_j11_m11_a2025_12h_21m_40s";
		break;
	case ZI_4:
		seqName = "ZI_4";
		folder = "burst_j11_m11_a2025_12h_22m_09s";
		break;
	case USINE_1:
		seqName = "USINE_1";
		folder = "burst_j11_m11_a2025_12h_26m_23s";
		break;
	case USINE_2:
		seqName = "USINE_2";
		folder = "burst_j11_m11_a2025_12h_27m_17s";
		break;
	case MAISON_4:
		seqName = "MAISON_4";
		folder = "burst_j11_m11_a2025_12h_29m_36s";
		break;
	}

	const int N_IMAGES = 200;

	dstRGBImgs.resize(N_IMAGES);
	dstBayerImgs.resize(N_IMAGES);
	
	Rect stackingROI;
	for (int i = 0; i < N_IMAGES; i++)
	{
		string filename = "C:\\Users\\alpha\\Desktop\\CSHR\\CSHR_2\\bursts\\" + folder + "\\im_" + std::to_string(i) + ".bmp";
		cv::Mat img = cv::imread(filename, CV_8U);
		if (img.empty())
		{
			throw std::runtime_error("Failed to load image: " + filename);
		}


		if (i == 0)
		{
			bool alreadyPicked = !true;
			if (alreadyPicked)
			{
				stackingROI = Rect(720, 346, 476, 246);
			}
			else
			{
				stackingROI = cv::selectROI(img);

				auto _m2 = [] (int _v) {return (_v/2)*2;};

				stackingROI.x = _m2(stackingROI.x);
				stackingROI.y = _m2(stackingROI.y);
				stackingROI.width = _m2(stackingROI.width);
				stackingROI.height = _m2(stackingROI.height);


				std::cout << "\n SELECTED IMAGE ROI: " << stackingROI.x << ", " << stackingROI.y << ", " << stackingROI.width << ", " << stackingROI.height << std::endl;
				cv::destroyAllWindows();
			}
		}


		dstBayerImgs[i] = img(stackingROI);
		cvtColor(dstBayerImgs[i], dstRGBImgs[i], COLOR_BayerRGGB2BGR);
		//cvtColor(dstBayerImgs[i], dstRGBImgs[i], COLOR_BayerRGGB2GRAY);

	}

	int i = 0;
	while (showSeq)
	{
		imshow("Raws", dstRGBImgs[i%200]);
		waitKey(50);
		i++;
	}
}

std::vector<std::vector<cv::Point2f>> Optimizer::OptFlow(std::vector<cv::Mat> RGBImages)
{
	vector<Mat> GRAYImages(RGBImages.size());

	for (int i = 0; i < RGBImages.size(); i++)
	{
		cvtColor(RGBImages[i], GRAYImages[i], COLOR_BGR2GRAY);
	}

	cv::Mat reference = GRAYImages[0];

	vector<vector<Point2f>> points(GRAYImages.size());

	vector<Point2f>& referencePoints = points[0];


	// GOOD FEATURES TO TRACK HYPERPARAMETERS
	const int MARGIN_SIZE = 10;  // Points that are near the edges are likely to disappear from frame to frame due to distortions/vibrations. We ignore them.
	const int MAX_POINTS_TRACKED = 100;
	const double MIN_DISTANCE = 15.;
	const double QUALITY_LEVEL = .1; // default .3
	const int BLOCK_SIZE = 2 * 2 + 1;
	const int GRADIENT_SIZE = 2 * 2 + 1;

	// LUCAS KANADE FLOW MATCHING HYPERPARAMETERS
	const float ERR_REFERENCE_THRESHOLD = 500.f; 
	const float ERR_SEQUENCE_THRESHOLD = 500.f;  
	const int LK_WINDOW_SIZE = 15;
	const int LK_PYRAMID_LEVELS = 2; // Starts at 0.


	Rect roi(MARGIN_SIZE, MARGIN_SIZE, reference.cols - 2 * MARGIN_SIZE, reference.rows - 2 * MARGIN_SIZE);

	// Take first frame and find corners in it
	auto time1 = high_resolution_clock::now();
	goodFeaturesToTrack(reference(roi), referencePoints, MAX_POINTS_TRACKED, QUALITY_LEVEL, MIN_DISTANCE, Mat(), BLOCK_SIZE, GRADIENT_SIZE, false, 0.04);
	auto time2 = high_resolution_clock::now();
	int ms_int = (int)duration_cast<milliseconds>(time2 - time1).count();
	std::cout << "\nFinding good features in the first frame took " << ms_int << " ms." << std::endl;



	// Refine the corners to subpixel precision
	time1 = high_resolution_clock::now();
	Size winSize = Size(BLOCK_SIZE / 2, BLOCK_SIZE / 2);
	Size zeroZone = Size(-1, -1);
	TermCriteria _criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001);
	cornerSubPix(reference(roi), referencePoints, winSize, zeroZone, _criteria);
	time2 = high_resolution_clock::now();
	ms_int = (int)duration_cast<milliseconds>(time2 - time1).count();
	std::cout << "Refining to subpix accuracy took " << ms_int << " ms." << std::endl;

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


	for (int i = 1; i < GRAYImages.size(); i++)
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
		calcOpticalFlowPyrLK(reference, GRAYImages[i], referencePoints, potentialPoints, status, err, Size(LK_WINDOW_SIZE, LK_WINDOW_SIZE), LK_PYRAMID_LEVELS, criteria, 0, 1.0E-3);
		auto _time2 = high_resolution_clock::now();
		int _ms_int = (int)duration_cast<milliseconds>(_time2 - _time1).count();
		std::cerr << "Matching features in the " << i << "-th frame took " << _ms_int << " ms." << std::endl;


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
				points[i][j] = Point2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
			}
		}
		std::cerr << "nMatches at step " << i << " : " << nMatches << " / " << referencePoints.size() << std::endl;

		Mat img;
		cv::add(GRAYImages[i], mask, img);

		imshow("Frame", img);
		int keyboard = waitKey(50);

		if (keyboard == 'q' || keyboard == 27)
			break;

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
	std::vector<cv::Mat> results(images.size());

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

	for (int j = 0; j < perPointBarycenter.size(); j++)
	{
		perPointBarycenter[j] /= (float)shiftedPoints.size();
	}



	// additional points on the edges.
	int nPointsPerSide = 5;

	std::vector<double> coords(shiftedPoints[0].size() * 2 + nPointsPerSide * 4 * 2);

	int offset = (int)shiftedPoints[0].size() * 2;

	const double DIST_TO_EDGE = 1.;

	// Horizontal edges
	for (int i = 0; i < nPointsPerSide + 1; i++)
	{
		double py0 = DIST_TO_EDGE;
		double py1 = (double)images[0].rows - 2. * DIST_TO_EDGE;
		double px = DIST_TO_EDGE + (double)i * ((double)images[0].cols  - 2. * DIST_TO_EDGE) / (double)nPointsPerSide;

		coords[offset++] = px;
		coords[offset++] = py0;
		
		coords[offset++] = px;
		coords[offset++] = py1;
	}

	// Vertical edges
	for (int i = 1; i < nPointsPerSide; i++)
	{
		double py = DIST_TO_EDGE + (double)i * (double)(images[0].rows - 2 * DIST_TO_EDGE) / (double)nPointsPerSide;
		double px0 = DIST_TO_EDGE;
		double px1 = (double)images[0].cols - 2. * DIST_TO_EDGE;

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
		Mat maskSums = Mat::zeros(images[0].size(), images[0].type());

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

			int id0 = (int) d.triangles[j];
			int id1 = (int) d.triangles[j+1];
			int id2 = (int) d.triangles[j+2];

			std::vector<Point2f> srcTri(3);
			std::vector<Point2f> dstTri(3);

			for (int p = 0; p < 3; p++)
			{
				int id = (int) d.triangles[j + p];

				srcTri[p] = Point2f((float)d.coords[2 * id], (float)d.coords[2 * id + 1]);
				dstTri[p] = perPointBarycenter[id];
			}

			
			Point2f srcBary = (srcTri[0] + srcTri[1] + srcTri[2]) / 3.F;
			Point2f dstBary = (dstTri[0] + dstTri[1] + dstTri[2]) / 3.F;

			for (int p = 0; p < 3; p++)
			{
				const float mag = .5f; // CANT BE GREATER THAN DIST_TO_EDGE

				float dx = srcBary.x > srcTri[p].x ? mag : -mag;
				float dy = srcBary.y > srcTri[p].y ? mag : -mag;
				srcTri[p].x -= dx;
				srcTri[p].y -= dy;

				// Needs be redone because triangulation changes at each frame.
				dx = dstBary.x > dstTri[p].x ? mag : -mag;
				dy = dstBary.y > dstTri[p].y ? mag : -mag;
				dstTri[p].x -= dx;
				dstTri[p].y -= dy;

#ifdef _DEBUG
				cv::line(meshOnImage, srcTri[p], srcTri[(p+1)%3], Scalar(255), 2);
#endif
			}
			

			cv::Rect srcROI = cv::boundingRect(srcTri);
			cv::Rect dstROI = cv::boundingRect(dstTri);

			srcTri[0] -= (Point2f) srcROI.tl();
			srcTri[1] -= (Point2f) srcROI.tl();
			srcTri[2] -= (Point2f) srcROI.tl();

			Mat warp_mat = getAffineTransform( srcTri, dstTri );


			cv::Mat srcMask = cv::Mat::zeros(srcROI.size(), images[0].type());
			std::vector<Point2i> iSrcTri(3);  // fillConvexPoly requires integer vertices...
			iSrcTri[0] = (Point2i)srcTri[0];
			iSrcTri[1] = (Point2i)srcTri[1];
			iSrcTri[2] = (Point2i)srcTri[2];
			fillConvexPoly(srcMask, iSrcTri, 1, cv::LINE_8, 0);

			cv::Mat extractedSrcTriangle;
			cv::multiply(images[i](srcROI), srcMask, extractedSrcTriangle);

			warpAffine(extractedSrcTriangle, scratchpad, warp_mat, scratchpad.size(), cv::INTER_NEAREST, cv::BORDER_TRANSPARENT); // INTER_AREA

			Mat mask = (unwarped(dstROI) == cv::Mat(cv::Scalar(0)));
			cv::add(scratchpad(dstROI), unwarped(dstROI), unwarped(dstROI), mask);
			
		}

		results[i] = unwarped;
		std::cout << "Delaunay processed "  << i << " th frame, that had " << d.triangles.size() << " triangles." << std::endl;
	}


	while (true)
	{
		for (int i = 0; i < images.size(); i++)
		{
			cv::imshow("Base", images[i]);
			int keyboard = waitKey(50);
		}

		for (int i = 0; i < results.size(); i++)
		{
			cv::imshow("Unwarped", results[i]);
			int keyboard = waitKey(50);
		}
	}
	
}


// The size of the patch is a dilema, balancing between keeping it small to capture the sharp instances, and keeping it large, to minimize performance impact and errors
// in the sharpness computations due to the inaccurate registration of the patch across images of the sequence (+ noise within)
//const int SQUARE_OVERLAP = 0; // on each side.
//const int SQUARE_SIZE = SQUARE_SPACING + 2*SQUARE_OVERLAP; 
const int SQUARE_SIZE = 50; 
// We shift the individual squares before computing the sharpness, to minimize the variation in laplacian variance due to registration errors. 
// The margin size should be bigger than the maximal expected displacement in the sequence, to ensure we dont leave the image.
const int SQUARE_GRID_MARGIN = 20; 




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

		Point2f center{ (float) (SQUARE_SIZE * x), (float) (SQUARE_SIZE * y) };
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



cv::Mat Optimizer::FullFrameSequentialMatcher(std::vector<cv::Mat> bayerImages, std::vector<cv::Mat> RGBImages)
{
	const int N_REFERENCE_FRAMES = 10;

	const int UPSCALE_FACTOR = 2;

	// between 0 and 100, in practice between 5 and 25
	const float STACKED_PERCENTILE = 10.f;

	int LR_w = RGBImages[0].cols; // Initial (low-res) image width
	int LR_h = RGBImages[0].rows; // Initial (low-res) image height
	int nSquareX = (LR_w - 2*SQUARE_GRID_MARGIN) / SQUARE_SIZE;
	int nSquareY = (LR_h - 2*SQUARE_GRID_MARGIN) / SQUARE_SIZE;

	
	// GOOD FEATURES TO TRACK HYPERPARAMETERS
	const int MARGIN_SIZE = 10;  // Points that are near the edges are likely to disappear from frame to frame due to distortions/vibrations. We ignore them.
	const int MAX_POINTS_TRACKED = 100;
	const double MIN_DISTANCE = 15.;
	const double QUALITY_LEVEL = .1; // default .3
	const int BLOCK_SIZE = 2 * 2 + 1;
	const int GRADIENT_SIZE = 2 * 2 + 1;

	// LUCAS KANADE FLOW MATCHING HYPERPARAMETERS
	const float ERR_REFERENCE_THRESHOLD = 500.f; 
	const float ERR_SEQUENCE_THRESHOLD = 500.f;  
	const int LK_WINDOW_SIZE = 15;
	const int LK_PYRAMID_LEVELS = 2; // Starts at 0.

	// TODO think about how to spread out references across the sequence a bit more. Luminosity could vary and invalidate the ref at the start. Hmmmm.

	
	vector<vector<Point2f>> referencePoints(N_REFERENCE_FRAMES);
	

	vector<Mat> GRAYImages(RGBImages.size());

	for (int i = 0; i < RGBImages.size(); i++)
	{
		cvtColor(RGBImages[i], GRAYImages[i], COLOR_BGR2GRAY);
	}

	// In the reference frames, find the features that we will track across the image sequence.
	for (int i = 0; i < N_REFERENCE_FRAMES; i++)
	{
		Rect featureFinderROI(MARGIN_SIZE, MARGIN_SIZE, RGBImages[i].cols - 2 * MARGIN_SIZE, RGBImages[i].rows - 2 * MARGIN_SIZE);

		goodFeaturesToTrack(GRAYImages[i](featureFinderROI), referencePoints[i], MAX_POINTS_TRACKED, QUALITY_LEVEL, MIN_DISTANCE, Mat(), BLOCK_SIZE, GRADIENT_SIZE, false, 0.04);

		//Mat imcl = RGBImages[i](featureFinderROI).clone();
		//for (int j = 0; j < referencePoints[i].size(); j++)
		//{
		//	cv::circle(imcl, referencePoints[i][j], 2, 255);
		//}
		//__debugbreak();

		Size winSize = Size(BLOCK_SIZE / 2, BLOCK_SIZE / 2);
		Size zeroZone = Size(-1, -1);
		TermCriteria _criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001);
		cornerSubPix(GRAYImages[i](featureFinderROI), referencePoints[i], winSize, zeroZone, _criteria); // I did not notice any benefit from this, but the computational cost seems negligible, so...

		for (int j = 0; j < referencePoints[i].size(); j++)
		{
			referencePoints[i][j] += Point2f((float)MARGIN_SIZE, (float)MARGIN_SIZE);
		}

		std::cout << "REF IMAGE " << i << ", N POINTS = " << referencePoints[i].size() << std::endl;
	}





	// Contains the position of each frame relative to the first reference frame, which is considered to be the absolute referential. (therefore absoluteOffsets[0] = (0,0))
	// If p = absoluteOffsets[ref_id], then ref_im[0]((x,y) + p) = ref_im[ref_id]((x,y)). (Coarsest approx. Initially necessary, then we will interpolate tracked points)
	vector<Point2f> absoluteOffsets(N_REFERENCE_FRAMES);

	vector<vector<Point2f>> referencePointsAbsoluteCoordinates(N_REFERENCE_FRAMES);

	// Find the absoluteOffsets of the reference frames and the absolute positions of the reference point. 
	{
		// Holds the barycenter of the points in the ref frame that were matched in this image, so not a constant.
		vector<Point2f> referenceBarycenters(N_REFERENCE_FRAMES);
		// Holds the barycenter of the points that were matched in this image with the ref frame.
		vector<Point2f> currentBarycenters(N_REFERENCE_FRAMES);
		
		// positionOfReferencePointsInOtherReferences[i][j] holds the vector of positions in j of the features originating in i tracked in j.
		vector<vector<vector<Point2f>>> positionOfReferencePointsInOtherReferences(N_REFERENCE_FRAMES);
		for (int i = 0; i < N_REFERENCE_FRAMES; i++) positionOfReferencePointsInOtherReferences[i].resize(N_REFERENCE_FRAMES);

		// A util to compute absoluteOffsets. If p = mutualRelativeOffsetsOfTheReferences[i][ref_id], then im[ref_id](x,y) = im[i](x+px,y+py) 
		vector<vector<Point2f>> mutualRelativeOffsetsOfTheReferences(N_REFERENCE_FRAMES);



		// Match the LK points between reference frames
		for (int i = 0; i < N_REFERENCE_FRAMES; i++)
		{
			mutualRelativeOffsetsOfTheReferences[i].resize(N_REFERENCE_FRAMES);

			// Match the current frame to the reference frames.
			for (int ref_id = 0; ref_id < N_REFERENCE_FRAMES; ref_id++)
			{
				referenceBarycenters[ref_id] = Point2f(.0f, .0f);
				currentBarycenters[ref_id] = Point2f(.0f, .0f);

				if (ref_id == i) continue;

				std::vector<Point2f> potentialPoints;

				// calculate optical flow
				vector<uchar> status;
				vector<float> err;
				TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);

				calcOpticalFlowPyrLK(GRAYImages[ref_id], GRAYImages[i], referencePoints[ref_id], potentialPoints, status, err, Size(LK_WINDOW_SIZE, LK_WINDOW_SIZE), LK_PYRAMID_LEVELS, criteria, 0, 1.0E-3);


				int nMatches = 0;

				positionOfReferencePointsInOtherReferences[ref_id][i].resize(referencePoints[ref_id].size());

				for (uint j = 0; j < referencePoints[ref_id].size(); j++)
				{
					// Select good points
					if (status[j] == 1 && err[j] < ERR_REFERENCE_THRESHOLD) {
						currentBarycenters[ref_id] += potentialPoints[j];
						referenceBarycenters[ref_id] += referencePoints[ref_id][j];

						// Note the order of indices
						positionOfReferencePointsInOtherReferences[ref_id][i][j] = potentialPoints[j];

						nMatches++;
					}
					else
					{
						// Note the order of indices
						positionOfReferencePointsInOtherReferences[ref_id][i][j] = Point2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
					}
				}
				//std::cerr << "nMatches at step " << i << " : " << nMatches << " / " << referencePoints[ref_id].size() << std::endl;

				// TODO ransac

				if (nMatches > 0)
				{
					currentBarycenters[ref_id] /= (float)nMatches;
					referenceBarycenters[ref_id] /= (float)nMatches;
				}
				else
				{
					std::cout << "NO MATCHES BETWEEN 2 REF FRAMES !!!" << std::endl;
				}
				
				// fill the mutualRelativeOffsetsOfTheReferences matrix.
				mutualRelativeOffsetsOfTheReferences[i][ref_id] = currentBarycenters[ref_id] - referenceBarycenters[ref_id];
			}

			
			for (int ref_id = 0; ref_id < N_REFERENCE_FRAMES; ref_id++)
			{
				mutualRelativeOffsetsOfTheReferences[i][ref_id] = currentBarycenters[ref_id] - referenceBarycenters[ref_id];
			}
		}



		// BARYCENTRE MATCHING:
		// The matrix mutualRelativeOffsetsOfTheReferences holds the mutual (potentialy asymmetrical) positions of the reference images. The following iterative algorithm
		// finds an optimum for the absolute position of each of the reference frames, stored in absoluteOffsets. We use the first frame's top left as the (0, 0).
		{
			//For all i, j < N_REFERENCE_FRAMES, for any pixel (x,y),  we want im[0]( (x,y) + currAbsolutePos[i]) = im[i]( (x,y) ) 
			// Trivially equivalent to im[i]( (x,y) - currAbsolutePos[i]) = im[j]( (x,y) - currAbsolutePos[j]) 
			std::vector<Point2f> currAbsolutePos(N_REFERENCE_FRAMES);
			std::vector<Point2f> prevAbsolutePos(N_REFERENCE_FRAMES);
			for (int ref_id = 0; ref_id < N_REFERENCE_FRAMES; ref_id++)
			{
				currAbsolutePos[ref_id] = Point2f(.0f, .0f);
				prevAbsolutePos[ref_id] = Point2f(.0f, .0f);
			}

			const int _maxIter = 10;
			for (int iter = 0; iter < _maxIter; iter++)
			{

				for (int ref_id_i = 0; ref_id_i < N_REFERENCE_FRAMES; ref_id_i++)
				{

					for (int ref_id_j = 0; ref_id_j < N_REFERENCE_FRAMES; ref_id_j++)
					{
						currAbsolutePos[ref_id_i] += prevAbsolutePos[ref_id_j] - mutualRelativeOffsetsOfTheReferences[ref_id_i][ref_id_j];
					}
					currAbsolutePos[ref_id_i] /= (float)N_REFERENCE_FRAMES;
				}


				prevAbsolutePos = currAbsolutePos;

				// We keep the first image at 0,0 for a fixed reference, in case the matrix mutualRelativeOffsetsOfTheReferences is not anti symmetrical. 
				// (it should be ideally, but the error in measurements is why we are here in the first place.)
				for (int ref_id_i = 0; ref_id_i < N_REFERENCE_FRAMES; ref_id_i++) prevAbsolutePos[ref_id_i] -= currAbsolutePos[0];
			}

			for (int ref_id = 0; ref_id < N_REFERENCE_FRAMES; ref_id++)
			{
				absoluteOffsets[ref_id] = currAbsolutePos[ref_id];
			}

			// Visualizing the alignement for monitoring:
			/*while (true)
			{
			for (int ref_id = 0; ref_id < N_REFERENCE_FRAMES; ref_id++)
			{
			Rect alignementTestRoi(MARGIN_SIZE - absoluteOffsets[ref_id].x, MARGIN_SIZE - absoluteOffsets[ref_id].y, featureFinderROI.width, featureFinderROI.height);
			cv::imshow("alignement", images[ref_id](alignementTestRoi));
			int keyboard = waitKey(50);
			}
			}*/
		}



		// Fill referencePointsAbsoluteCoordinates, combining information from positionOfReferencePointsInOtherReferences ant the barycentres.
		for (int ref_id = 0; ref_id < N_REFERENCE_FRAMES; ref_id++)
		{
			referencePointsAbsoluteCoordinates[ref_id].resize(referencePoints[ref_id].size());

			vector<int> nMatchesPerPoint(referencePoints[ref_id].size());
			std::fill(nMatchesPerPoint.begin(), nMatchesPerPoint.end(), 0);

			for (int matched_ref_id = 0; matched_ref_id < N_REFERENCE_FRAMES; matched_ref_id++)
			{
				if (ref_id == matched_ref_id) continue;

				
				for (uint j = 0; j < referencePoints[ref_id].size(); j++)
				{
					if (!std::isnan(positionOfReferencePointsInOtherReferences[ref_id][matched_ref_id][j].x)) {
						referencePointsAbsoluteCoordinates[ref_id][j] += positionOfReferencePointsInOtherReferences[ref_id][matched_ref_id][j] + absoluteOffsets[ref_id];
						nMatchesPerPoint[j]++;
					}
				}

			}

			for (uint j = 0; j < referencePoints[ref_id].size(); j++)
			{
				if (nMatchesPerPoint[j] > 0) {
					referencePointsAbsoluteCoordinates[ref_id][j] /= (float) nMatchesPerPoint[j];
				}
				else
				{
					referencePointsAbsoluteCoordinates[ref_id][j] = absoluteOffsets[ref_id] + referencePoints[ref_id][j];
				}
			}
		}
	}
	
	std::cout << "\nReference frames (" << N_REFERENCE_FRAMES << ") aligned.\n" << std::endl;



	/////////////////// END OF REFERENCE FRAMES PROCESSING //////////////////////////////////



	// Preallocated for efficiency. Reused for each image of the sequence, for each reference frame. Contains the coordinates in the current image of the points tracked from the ref image. No NaNs.
	std::vector<Point2f> foundPointsCurr(MAX_POINTS_TRACKED);

	// Preallocated for efficiency. Reused for each image of the sequence, for each reference frame. Contains the absolute coordinates of the points tracked from the ref image that were found in the current one. No NaNs.
	std::vector<Point2f> foundPointsRefAbsPos(MAX_POINTS_TRACKED);

	// Matrix of vectors, (x,y) containing the vector of images.size() unsorted variances of the laplacians, 
	// (x,y,i) being the variance in the subrect at (x*SQUARE_SPACING, y*SQUARE_SPACING) in absolute coordinates in i-th image of the sequence 
	std::vector<std::vector<std::vector<float>>> sharpnesses(nSquareX);
	for (int _x = 0; _x < nSquareX; _x++)
	{
		sharpnesses[_x].resize(nSquareY);
		for (int _y = 0; _y < nSquareY; _y++)
		{
			sharpnesses[_x][_y].resize(RGBImages.size());
		}
	}

	// Pre allocated for reuse by each image
	Mat laplacianBuffer(RGBImages[0].size(), CV_32F);


	vector<Mat> perSquareAbsoluteOffset(RGBImages.size());
	for (int i = N_REFERENCE_FRAMES; i < RGBImages.size(); i++) perSquareAbsoluteOffset[i] = Mat::zeros(Size(nSquareX, nSquareY), CV_32FC2);

	// Preallocated for efficiency.
	Mat currentPerSquareNormalizer = Mat::zeros(Size(nSquareX, nSquareY), CV_32F);

#ifdef _DEBUG
	vector<Point2f> imOffsets(RGBImages.size());
#endif

	// In each image of the sequence, find the LK features of the references, and compute for each square patch its estimated offset in the absolute referential.
	for (int i = N_REFERENCE_FRAMES; i < RGBImages.size(); i++)
	{
		
		// Barycentre of the offsets of the tracked points across the references. Used to give an offset to pixels in the image that are too far from any tracked point to have any meaningful "nearest neighbour".
		Point2f currentFrameAbsoluteOffset(.0f, .0f);
		int totalPointsMatched = 0;

		currentPerSquareNormalizer = 0;

		for (int ref_id = 0; ref_id < N_REFERENCE_FRAMES; ref_id++)
		{

			std::vector<Point2f> potentialPoints;

			// calculate optical flow
			vector<uchar> status;
			vector<float> err;
			TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);

			calcOpticalFlowPyrLK(GRAYImages[ref_id], GRAYImages[i], referencePoints[ref_id], potentialPoints, status, err, Size(LK_WINDOW_SIZE, LK_WINDOW_SIZE), LK_PYRAMID_LEVELS, criteria, 0, 1.0E-3);

#ifdef _DEBUG
			Mat refIm = GRAYImages[ref_id].clone();
			Mat currIm = GRAYImages[i].clone();

			std::vector<Point2f> deltas(potentialPoints.size());
#endif


			int nMatches = 0;

			for (uint j = 0; j < potentialPoints.size(); j++)
			{
				// Select good points
				if (status[j] == 1 && err[j] < ERR_SEQUENCE_THRESHOLD) {
					foundPointsCurr[nMatches] = potentialPoints[j];
					foundPointsRefAbsPos[nMatches] = referencePointsAbsoluteCoordinates[ref_id][j];

					currentFrameAbsoluteOffset += referencePointsAbsoluteCoordinates[ref_id][j] - potentialPoints[j];

					nMatches++;


					// TODO ransac



					// Fill the coarse grid with "nearest neighbour" offsets, in a 3x3 area around the points.
					int _x = (int) floorf((potentialPoints[j].x - SQUARE_GRID_MARGIN) / (float)SQUARE_SIZE);
					int _y = (int) floorf((potentialPoints[j].y - SQUARE_GRID_MARGIN) / (float) SQUARE_SIZE);

					for (int _xx = -1; _xx <= 1; _xx++)
					{
						for (int _yy = -1; _yy <= 1; _yy++)
						{
							int sx = _x + _xx;
							int sy = _y + _yy;

							if (sx < 0 || sx > nSquareX-1 || sy < 0 || sy > nSquareY-1) continue;

							Point2f squareCenter(
								(float) SQUARE_GRID_MARGIN + (float)SQUARE_SIZE* ((float)sx + .5f),
								(float) SQUARE_GRID_MARGIN + (float)SQUARE_SIZE* ((float)sy + .5f));

							Point2f _d = potentialPoints[j] - squareCenter;

							float _weight = 1.f - sqrtf(_d.x * _d.x + _d.y * _d.y) / ((float) SQUARE_SIZE * 1.5f * 1.4143f); // between 0 and 1

							if (_weight < 0) __debugbreak();

							perSquareAbsoluteOffset[i].at<Point2f>(sy, sx) += _weight * (referencePointsAbsoluteCoordinates[ref_id][j] - potentialPoints[j]);
							currentPerSquareNormalizer.at<float>(sy, sx) += _weight;
						}
					}
					
#ifdef _DEBUG
					cv::circle(refIm, referencePoints[ref_id][j], 2, 255);
					cv::circle(currIm, potentialPoints[j], 2, 255);

					deltas[j] =  potentialPoints[j] - referencePoints[ref_id][j];
#endif
				}
			}

			//std::cerr << "nMatches at step " << i << " : " << nMatches << " / " << referencePoints[ref_id].size() << std::endl;


			totalPointsMatched += nMatches;

#ifdef _DEBUG
			int matHalfSize = 100;
			float multiplier = 40.f;
			Mat visu = Mat::zeros(Size(matHalfSize*2+1,matHalfSize*2+1), CV_8U);
			vector<int> _ids(nMatches);
			{
				int _m = 0;
				for (uint j = 0; j < potentialPoints.size(); j++)
				{
					if (status[j] == 1 && err[j] < ERR_SEQUENCE_THRESHOLD)
					{
						_ids[_m] = j;
						_m++;
					}
				} // ids sorted by ascending err.
				std::sort(_ids.begin(), _ids.end(),
					[&err](int id_a, int id_b) {
						return err[id_a] > err[id_b];
					});
			}


			for (uint _i = 0; _i < _ids.size(); _i++)
			{
				int _j = _ids[_i];

				int _x = (int) (deltas[_j].x * multiplier) + matHalfSize + 1;
				_x = std::clamp(_x, 0, 2*matHalfSize);
				int _y = (int) (deltas[_j].y * multiplier) + matHalfSize + 1;
				_y = std::clamp(_y, 0, 2*matHalfSize);
				visu.at<uchar>(_y, _x) = (uchar)(255.f * (float)(_i+1) / (float)_ids.size());
			}

			
			int a = 0;
			
#endif
		}

		currentFrameAbsoluteOffset /= (float) totalPointsMatched;

#ifdef _DEBUG
		imOffsets[i] = currentFrameAbsoluteOffset;
#endif


		// Compute sharpnesses:

		cv::Laplacian(RGBImages[i], laplacianBuffer, CV_32F);

		for (int x = 0; x < nSquareX; x++)
		{
			for (int y = 0; y < nSquareY; y++)
			{
				// I would have done this cleanly outside of the loops if OpenCV allowed 2channels * 1channel...
				constexpr float globalOffsetWeight = .5f;
				perSquareAbsoluteOffset[i].at<Point2f>(y, x) += globalOffsetWeight * currentFrameAbsoluteOffset;
				perSquareAbsoluteOffset[i].at<Point2f>(y, x) /= currentPerSquareNormalizer.at<float>(y, x) + globalOffsetWeight;


				// Incorrect registration of the patches completely ruins the sorting by sharpness. This solution is okayish.

				Point2f registrationOffset = perSquareAbsoluteOffset[i].at<Point2f>(y, x);

				Rect _roi(x * SQUARE_SIZE + SQUARE_GRID_MARGIN - (int)registrationOffset.x, y * SQUARE_SIZE + SQUARE_GRID_MARGIN - (int)registrationOffset.y, SQUARE_SIZE, SQUARE_SIZE);

				if (_roi.x < 0 || _roi.y < 0 || _roi.x + _roi.width >= RGBImages[i].cols || _roi.y + _roi.height >= RGBImages[i].rows)
				{
					sharpnesses[x][y][i] = .0f;
				}
				else
				{
					Scalar varLaplacian;
					Scalar meanLaplacian;
					cv::meanStdDev(RGBImages[i](_roi), meanLaplacian, varLaplacian);
					sharpnesses[x][y][i] = (float)varLaplacian[0] + .1f; // .1f to distinguish between valid squares that are constant, and squares that are invalid because either outside of the frame or from the ref frames.
				}	
			}
		}

		std::cout << "Sharpness, alignement : " << i  << std::endl;
	}

	std::cout << "\nSharpness computed, square patches aligned across the sequence. \n" << std::endl;







	/////////////////////////// END OF ACQUISITION, ALL THAT HAPPENS NEXT REQUIRES THE WHOLE SEQUENCE ////////////////////////////
	
	Size SR_size(LR_w * UPSCALE_FACTOR, LR_h * UPSCALE_FACTOR);

	vector<Mat> stack(3);			 // 1 channel per color
	vector<Mat> normalizer(3);       // 1 channel per color
	for (int channel = 0; channel < 3; channel++)
	{
		stack[channel] = cv::Mat::zeros(SR_size, CV_32F);
		normalizer[channel] = cv::Mat::zeros(SR_size, CV_32F);
	}

	vector<int> ids(RGBImages.size());

	// Just the target number of the sharpest fragments to stack.
	int nStackedSquares = (int)((float)RGBImages.size() * STACKED_PERCENTILE / 100.f);

	// Preallocated for efficiency
	Mat basePatch = Mat::zeros(Size(UPSCALE_FACTOR+2, UPSCALE_FACTOR+2), CV_32F);
	basePatch(Rect(1,1,UPSCALE_FACTOR,UPSCALE_FACTOR)) = 1.f;
	Mat shiftedPatch(Size(UPSCALE_FACTOR+1, UPSCALE_FACTOR+1), CV_32F);

	// precomputed for efficiency
	Point2f commonPatchCenter(
		1.f + (float)UPSCALE_FACTOR / 2.f,
		1.f + (float)UPSCALE_FACTOR / 2.f
	);

	for (int x = 0; x < nSquareX; x++)
	{
		for (int y = 0; y < nSquareY; y++)
		{
			vector<float>& s = sharpnesses[x][y];

			// As of now, reference frames are not used for stacking. 
			for (int i = 0; i < N_REFERENCE_FRAMES; i++) s[i] = 0.f;
			
			// sort ids by descending sharpness:
			for (int i = 0; i < RGBImages.size(); i++) ids[i] = i;
			std::sort(ids.begin(), ids.end(),
				[&s](int id_a, int id_b) {
					return s[id_a] > s[id_b];
				});


			Point2i patchTopLeft(x * SQUARE_SIZE + SQUARE_GRID_MARGIN, y * SQUARE_SIZE + SQUARE_GRID_MARGIN);
			
#ifdef _DEBUG
			if (true)
			{
				const int upscale = 5;
				Mat smallFrag(Size(SQUARE_SIZE, SQUARE_SIZE), CV_8U);
				Mat bigFrag(Size(SQUARE_SIZE * upscale, SQUARE_SIZE * upscale), CV_8U);

				Point2f commonFragCenter( 
					((float)x + .5f)* (float)SQUARE_SIZE + (float)SQUARE_GRID_MARGIN,
					((float)y + .5f)* (float)SQUARE_SIZE + (float)SQUARE_GRID_MARGIN);

				std::cout << x << " " << y << std::endl;
				std::cout << "Phase correlated (local)" << std::endl;
				Mat srcPatch;
				Mat dstPatch;
				const int S2 = 3;
				int id0 = ids[0];
				Point2f squareOffset0 = imOffsets[id0];
				Rect _ROI0((int) (patchTopLeft.x - squareOffset0.x) - S2, (int) (patchTopLeft.y - squareOffset0.y) - S2, SQUARE_SIZE + 2 *S2, SQUARE_SIZE + 2 *S2);
				GRAYImages[id0](_ROI0).convertTo(srcPatch, CV_32F);
				for (int i = 1; i < 50; i++)
				{
					int id = ids[i];


					Point2f squareOffset = imOffsets[id];

					Rect _ROI( (int) (patchTopLeft.x - squareOffset.x)  - S2, (int) (patchTopLeft.y - squareOffset.y)  - S2, SQUARE_SIZE + 2 *S2, SQUARE_SIZE + 2 *S2);

					
					GRAYImages[id](_ROI).convertTo(dstPatch, CV_32F);

					Point2d relativeOffset = phaseCorrelate(srcPatch, dstPatch);

					Point2f deltaROI(_ROI.x - _ROI0.x, _ROI.y - _ROI0.y);

					Point2f center = commonFragCenter + deltaROI + (Point2f) relativeOffset;
					getRectSubPix(RGBImages[id], Size(SQUARE_SIZE,SQUARE_SIZE), center, smallFrag, CV_8U);

					resize(smallFrag, bigFrag, Size(), upscale, upscale);


					imshow("frag", bigFrag);
					waitKey(50);
				}
				std::cout << "Stabilized (local)" << std::endl;
				for (int i = 0; i < 50; i++)
				{

					int id = ids[i];

					//Point2f squareOffset(.0f, .0f);
					//float _normalizer = .0f;
					//for (int _xx = -1; _xx <= 1; _xx++)
					//{
					//	for (int _yy = -1; _yy <= 1; _yy++)
					//	{
					//		int sx = _xx + x;
					//		int sy = _yy + y;

					//		if (sx < 0 || sx > nSquareX - 1 || sy < 0 || sy > nSquareY - 1) continue;

					//		_normalizer += 1.f;
					//		squareOffset += perSquareAbsoluteOffset[id].at<Point2f>(sy, sx);
					//	}
					//}
					//squareOffset /= _normalizer;


					Point2f squareOffset = perSquareAbsoluteOffset[id].at<Point2f>(y, x);

					Point2f center = commonFragCenter - squareOffset;

					getRectSubPix(RGBImages[id], Size(SQUARE_SIZE,SQUARE_SIZE), center, smallFrag, CV_8U);
					resize(smallFrag, bigFrag, Size(), upscale, upscale);


					imshow("frag", bigFrag);
					waitKey(50);
				}
				std::cout << "Stabilized (global)" << std::endl;
				for (int i = 0; i < 50; i++)
				{

					int id = ids[i];

					Point2f squareOffset = imOffsets[id];

					Point2f center = commonFragCenter - squareOffset;

					getRectSubPix(RGBImages[id], Size(SQUARE_SIZE,SQUARE_SIZE), center, smallFrag, CV_8U);
					resize(smallFrag, bigFrag, Size(), upscale, upscale);


					//std::cout << s[id] << std::endl;
					//std::cout << _roi.x << " " << _roi.y << " " << squareOffset.x << " " << squareOffset.y << std::endl;


					imshow("frag", bigFrag);
					waitKey(50);
				}
				std::cout << "Not stabilized" << std::endl;
				for (int i = 0; i < 50; i++)
				{

					int id = ids[i];

					Point2f center = commonFragCenter;

					getRectSubPix(RGBImages[id], Size(SQUARE_SIZE,SQUARE_SIZE), center, smallFrag, CV_8U);
					resize(smallFrag, bigFrag, Size(), upscale, upscale);
					imshow("frag", bigFrag);
					waitKey(50);
				}
			}
#endif

			for (int i = 0; i < nStackedSquares; i++)
			{
				int id = ids[i];
				
				// The copied patch from the src sequence image is coarsely positioned, rounded to integer coordinates so within 1 px of its absolute position.
				// This whole approach forces us to transform square from the seq ims into non square areas of the stack. We use a normalizer matrix to normalize each pixel
				// of the stack once we are done stacking. Hopefully (more and more likely as nStackedSquares grows), the areas overlap sufficently for the stack to be fully covered.
				// We can always have the squares overlap like previously, but that increases costs (not constrained part ?) and is no guarantee. (see SQUARE_SPACING's definition)

				Point2f squareRegistrationOffset = perSquareAbsoluteOffset[id].at<Point2f>(y, x);
				Point2i seqimTopLeft(
					patchTopLeft.x - (int)squareRegistrationOffset.x,
					patchTopLeft.y - (int)squareRegistrationOffset.y
				);



				Point2f squareOffset(.0f, .0f);
				Point2f squareCenter(
					(float) SQUARE_GRID_MARGIN + (float)SQUARE_SIZE* ((float)x + .5f),
					(float) SQUARE_GRID_MARGIN + (float)SQUARE_SIZE* ((float)y + .5f));
				float _normalizer = 0.f;

				for (int _xx = -1; _xx <= 1; _xx++)
				{
					for (int _yy = -1; _yy <= 1; _yy++)
					{
						int sx = _xx + x;
						int sy = _yy + y;

						if (sx < 0 || sx > nSquareX-1 || sy < 0 || sy > nSquareY-1) continue;

						Point2f currSquareCenter(
							(float) SQUARE_GRID_MARGIN + (float)SQUARE_SIZE* ((float)sx + .5f),
							(float) SQUARE_GRID_MARGIN + (float)SQUARE_SIZE* ((float)sy + .5f));

						Point2f _d = currSquareCenter - squareCenter;

						float _weight = 1.f - sqrtf(_d.x * _d.x + _d.y * _d.y) / ((float) SQUARE_SIZE * 1.5f * 1.4143f); // between 0 and 1

						squareOffset += _weight * perSquareAbsoluteOffset[id].at<Point2f>(sy, sx);
						_normalizer += _weight;
					}
				}
				squareOffset /= _normalizer;
				squareOffset *= (float) UPSCALE_FACTOR;


				//Perform per pixel bilinear upscaling of the square in the id-th image into the stack, warping the id-th image with values from perSquareAbsoluteOffset.
				for (int xx = seqimTopLeft.x; xx < seqimTopLeft.x + SQUARE_SIZE; xx++)
				{
					for (int yy = seqimTopLeft.y; yy < seqimTopLeft.y + SQUARE_SIZE; yy++)
					{
						// The bayer pattern is BG . The result of the following computation makes channel = 1 if green, 2 if blue, 0 if red
					    //					    GR
						int isGreen = (xx % 2) ^ (yy % 2); // Bitwise XOR.
						int channel = isGreen + (1 - isGreen) * 2 * (xx % 2);

						
						Point2f stackxxyy = Point2f((float)(UPSCALE_FACTOR * xx), (float)(UPSCALE_FACTOR * yy));
						Point2f inStackPosition = stackxxyy + squareOffset;

						shiftedPatch = 0.0f;

						float floor_x = floorf(inStackPosition.x);
						float floor_y = floorf(inStackPosition.y);
						Point2f fracCenterOffset = Point2f(inStackPosition.x - floor_x, inStackPosition.y - floor_y);

						cv::getRectSubPix(basePatch, shiftedPatch.size(), commonPatchCenter - fracCenterOffset, shiftedPatch);

						Rect stackROI((int)floor_x, (int)floor_y, UPSCALE_FACTOR + 1, UPSCALE_FACTOR + 1);

						if (stackROI.x < 0 || stackROI.y < 0) __debugbreak();

						add(normalizer[channel](stackROI), shiftedPatch, normalizer[channel](stackROI));

						shiftedPatch *= (float) bayerImages[id].at<uchar>(yy, xx);

						add(stack[channel](stackROI), shiftedPatch, stack[channel](stackROI));
					}
				}
			}
		}
	}

	for (int channel = 0; channel < 3; channel++) cv::divide(stack[channel], normalizer[channel], stack[channel]);

	

	//  Because openCV doesnt normalize its windows...
	for (int channel = 0; channel < 3; channel++) stack[channel] *= 1.f/255.f;

	Mat deconv;
	for (int r = 0; r < 1; r++)
	{
		Rect deconvROI = cv::selectROI(stack[1]);
		std::cout << "\n SELECTED CONVOLUTION ROI DIMENSIONS: " << deconvROI.x << " " << deconvROI.y << " " << deconvROI.width << " " << deconvROI.height << std::endl;
		cv::destroyAllWindows();


		Mat toBeDeconv = stack[1](deconvROI).clone();

		const int nSigmas = 3;
		float sigmas[nSigmas];
		for (int i = 0; i < nSigmas; i++)
		{
			sigmas[i] = (float) UPSCALE_FACTOR * .5f * (1.f + (float) i);
			deconv = LucyRichardson(toBeDeconv, 10, sigmas[i]);
			deconv = LucyRichardson(toBeDeconv, 20, sigmas[i]);
			deconv = LucyRichardson(toBeDeconv, 30, sigmas[i]);

			Mat deconv8U;
			deconv.convertTo(deconv8U, CV_8U);

			std::string name = "SR_samples\\" + seqName + "_" + std::to_string(UPSCALE_FACTOR) + "x_" + std::to_string((int)STACKED_PERCENTILE) + "percentOf"
				+ std::to_string((int)RGBImages.size()) + "images_sigma" + std::to_string(sigmas[i]) 
				+ std::to_string(time(0)) + ".bmp";
			cv::imwrite(name, deconv8U);
		}
	}

	for (int channel = 0; channel < 3; channel++) stack[channel] *= 255.f;

	//Mat unsharpMasked = UnsharpMasking(deconv);
	//Mat equalHist = EqualizeHistogram(deconv8U);

	Mat coloredStack32F, coloredStack8U;

	// Correcting chromatic aberration:
	//int offset = 4;
	//stack[0] = stack[0](Rect(offset,0, stack[0].cols-offset,stack[0].rows-offset)); // blue
	//stack[1] = stack[1](Rect(0,0,	   stack[1].cols-offset,stack[1].rows-offset)); // green
	//stack[2] = stack[2](Rect(0,0,	   stack[2].cols-offset,stack[2].rows-offset));	// red


	cv::merge(stack, coloredStack32F);

	coloredStack32F.convertTo(coloredStack8U, CV_8U);

	cv::imwrite("stack.bmp", coloredStack8U);

	cv::imshow("Stack", coloredStack8U);

	int keyboard = waitKey(0);

	return coloredStack8U;
}



cv::Mat Optimizer::WeightedTotalCombination(std::vector<cv::Mat> bayerImages, std::vector<cv::Mat> RGBImages)
{
	const int N_REFERENCE_FRAMES = 10;

	const int UPSCALE_FACTOR = 3;

	int LR_w = RGBImages[0].cols; // Initial (low-res) image width
	int LR_h = RGBImages[0].rows; // Initial (low-res) image height
	int nSquareX = (LR_w - 2*SQUARE_GRID_MARGIN) / SQUARE_SIZE;
	int nSquareY = (LR_h - 2*SQUARE_GRID_MARGIN) / SQUARE_SIZE;


	// GOOD FEATURES TO TRACK HYPERPARAMETERS
	const int MARGIN_SIZE = 10;  // Points that are near the edges are likely to disappear from frame to frame due to distortions/vibrations. We ignore them.
	const int MAX_POINTS_TRACKED = 100;
	const double MIN_DISTANCE = 15.;
	const double QUALITY_LEVEL = .1; // default .3
	const int BLOCK_SIZE = 2 * 2 + 1;
	const int GRADIENT_SIZE = 2 * 2 + 1;

	// LUCAS KANADE FLOW MATCHING HYPERPARAMETERS
	const float ERR_REFERENCE_THRESHOLD = 500.f; 
	const float ERR_SEQUENCE_THRESHOLD = 500.f;  
	const int LK_WINDOW_SIZE = 15;
	const int LK_PYRAMID_LEVELS = 2; // Starts at 0.

	// TODO think about how to spread out references across the sequence a bit more. Luminosity could vary and invalidate the ref at the start. Hmmmm.


	vector<vector<Point2f>> referencePoints(N_REFERENCE_FRAMES);


	vector<Mat> GRAYImages(RGBImages.size());

	for (int i = 0; i < RGBImages.size(); i++)
	{
		cvtColor(RGBImages[i], GRAYImages[i], COLOR_BGR2GRAY);
	}

	// In the reference frames, find the features that we will track across the image sequence.
	for (int i = 0; i < N_REFERENCE_FRAMES; i++)
	{
		Rect featureFinderROI(MARGIN_SIZE, MARGIN_SIZE, RGBImages[i].cols - 2 * MARGIN_SIZE, RGBImages[i].rows - 2 * MARGIN_SIZE);

		goodFeaturesToTrack(GRAYImages[i](featureFinderROI), referencePoints[i], MAX_POINTS_TRACKED, QUALITY_LEVEL, MIN_DISTANCE, Mat(), BLOCK_SIZE, GRADIENT_SIZE, false, 0.04);

		//Mat imcl = RGBImages[i](featureFinderROI).clone();
		//for (int j = 0; j < referencePoints[i].size(); j++)
		//{
		//	cv::circle(imcl, referencePoints[i][j], 2, 255);
		//}
		//__debugbreak();

		Size winSize = Size(BLOCK_SIZE / 2, BLOCK_SIZE / 2);
		Size zeroZone = Size(-1, -1);
		TermCriteria _criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001);
		cornerSubPix(GRAYImages[i](featureFinderROI), referencePoints[i], winSize, zeroZone, _criteria); // I did not notice any benefit from this, but the computational cost seems negligible, so...

		for (int j = 0; j < referencePoints[i].size(); j++)
		{
			referencePoints[i][j] += Point2f((float)MARGIN_SIZE, (float)MARGIN_SIZE);
		}

		std::cout << "REF IMAGE " << i << ", N POINTS = " << referencePoints[i].size() << std::endl;
	}





	// Contains the position of each frame relative to the first reference frame, which is considered to be the absolute referential. (therefore absoluteOffsets[0] = (0,0))
	// If p = absoluteOffsets[ref_id], then ref_im[0]((x,y) + p) = ref_im[ref_id]((x,y)). (Coarsest approx. Initially necessary, then we will interpolate tracked points)
	vector<Point2f> absoluteOffsets(N_REFERENCE_FRAMES);

	vector<vector<Point2f>> referencePointsAbsoluteCoordinates(N_REFERENCE_FRAMES);

	// Find the absoluteOffsets of the reference frames and the absolute positions of the reference point. 
	{
		// Holds the barycenter of the points in the ref frame that were matched in this image, so not a constant.
		vector<Point2f> referenceBarycenters(N_REFERENCE_FRAMES);
		// Holds the barycenter of the points that were matched in this image with the ref frame.
		vector<Point2f> currentBarycenters(N_REFERENCE_FRAMES);

		// positionOfReferencePointsInOtherReferences[i][j] holds the vector of positions in j of the features originating in i tracked in j.
		vector<vector<vector<Point2f>>> positionOfReferencePointsInOtherReferences(N_REFERENCE_FRAMES);
		for (int i = 0; i < N_REFERENCE_FRAMES; i++) positionOfReferencePointsInOtherReferences[i].resize(N_REFERENCE_FRAMES);

		// A util to compute absoluteOffsets. If p = mutualRelativeOffsetsOfTheReferences[i][ref_id], then im[ref_id](x,y) = im[i](x+px,y+py) 
		vector<vector<Point2f>> mutualRelativeOffsetsOfTheReferences(N_REFERENCE_FRAMES);



		// Match the LK points between reference frames
		for (int i = 0; i < N_REFERENCE_FRAMES; i++)
		{
			mutualRelativeOffsetsOfTheReferences[i].resize(N_REFERENCE_FRAMES);

			// Match the current frame to the reference frames.
			for (int ref_id = 0; ref_id < N_REFERENCE_FRAMES; ref_id++)
			{
				referenceBarycenters[ref_id] = Point2f(.0f, .0f);
				currentBarycenters[ref_id] = Point2f(.0f, .0f);

				if (ref_id == i) continue;

				std::vector<Point2f> potentialPoints;

				// calculate optical flow
				vector<uchar> status;
				vector<float> err;
				TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);

				calcOpticalFlowPyrLK(GRAYImages[ref_id], GRAYImages[i], referencePoints[ref_id], potentialPoints, status, err, Size(LK_WINDOW_SIZE, LK_WINDOW_SIZE), LK_PYRAMID_LEVELS, criteria, 0, 1.0E-3);


				int nMatches = 0;

				positionOfReferencePointsInOtherReferences[ref_id][i].resize(referencePoints[ref_id].size());

				for (uint j = 0; j < referencePoints[ref_id].size(); j++)
				{
					// Select good points
					if (status[j] == 1 && err[j] < ERR_REFERENCE_THRESHOLD) {
						currentBarycenters[ref_id] += potentialPoints[j];
						referenceBarycenters[ref_id] += referencePoints[ref_id][j];

						// Note the order of indices
						positionOfReferencePointsInOtherReferences[ref_id][i][j] = potentialPoints[j];

						nMatches++;
					}
					else
					{
						// Note the order of indices
						positionOfReferencePointsInOtherReferences[ref_id][i][j] = Point2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
					}
				}
				//std::cerr << "nMatches at step " << i << " : " << nMatches << " / " << referencePoints[ref_id].size() << std::endl;

				// TODO ransac

				if (nMatches > 0)
				{
					currentBarycenters[ref_id] /= (float)nMatches;
					referenceBarycenters[ref_id] /= (float)nMatches;
				}
				else
				{
					std::cout << "NO MATCHES BETWEEN 2 REF FRAMES !!!" << std::endl;
				}

				// fill the mutualRelativeOffsetsOfTheReferences matrix.
				mutualRelativeOffsetsOfTheReferences[i][ref_id] = currentBarycenters[ref_id] - referenceBarycenters[ref_id];
			}


			for (int ref_id = 0; ref_id < N_REFERENCE_FRAMES; ref_id++)
			{
				mutualRelativeOffsetsOfTheReferences[i][ref_id] = currentBarycenters[ref_id] - referenceBarycenters[ref_id];
			}
		}



		// BARYCENTRE MATCHING:
		// The matrix mutualRelativeOffsetsOfTheReferences holds the mutual (potentialy asymmetrical) positions of the reference images. The following iterative algorithm
		// finds an optimum for the absolute position of each of the reference frames, stored in absoluteOffsets. We use the first frame's top left as the (0, 0).
		{
			//For all i, j < N_REFERENCE_FRAMES, for any pixel (x,y),  we want im[0]( (x,y) + currAbsolutePos[i]) = im[i]( (x,y) ) 
			// Trivially equivalent to im[i]( (x,y) - currAbsolutePos[i]) = im[j]( (x,y) - currAbsolutePos[j]) 
			std::vector<Point2f> currAbsolutePos(N_REFERENCE_FRAMES);
			std::vector<Point2f> prevAbsolutePos(N_REFERENCE_FRAMES);
			for (int ref_id = 0; ref_id < N_REFERENCE_FRAMES; ref_id++)
			{
				currAbsolutePos[ref_id] = Point2f(.0f, .0f);
				prevAbsolutePos[ref_id] = Point2f(.0f, .0f);
			}

			const int _maxIter = 10;
			for (int iter = 0; iter < _maxIter; iter++)
			{

				for (int ref_id_i = 0; ref_id_i < N_REFERENCE_FRAMES; ref_id_i++)
				{

					for (int ref_id_j = 0; ref_id_j < N_REFERENCE_FRAMES; ref_id_j++)
					{
						currAbsolutePos[ref_id_i] += prevAbsolutePos[ref_id_j] - mutualRelativeOffsetsOfTheReferences[ref_id_i][ref_id_j];
					}
					currAbsolutePos[ref_id_i] /= (float)N_REFERENCE_FRAMES;
				}


				prevAbsolutePos = currAbsolutePos;

				// We keep the first image at 0,0 for a fixed reference, in case the matrix mutualRelativeOffsetsOfTheReferences is not anti symmetrical. 
				// (it should be ideally, but the error in measurements is why we are here in the first place.)
				for (int ref_id_i = 0; ref_id_i < N_REFERENCE_FRAMES; ref_id_i++) prevAbsolutePos[ref_id_i] -= currAbsolutePos[0];
			}

			for (int ref_id = 0; ref_id < N_REFERENCE_FRAMES; ref_id++)
			{
				absoluteOffsets[ref_id] = currAbsolutePos[ref_id];
			}

			// Visualizing the alignement for monitoring:
			/*while (true)
			{
			for (int ref_id = 0; ref_id < N_REFERENCE_FRAMES; ref_id++)
			{
			Rect alignementTestRoi(MARGIN_SIZE - absoluteOffsets[ref_id].x, MARGIN_SIZE - absoluteOffsets[ref_id].y, featureFinderROI.width, featureFinderROI.height);
			cv::imshow("alignement", images[ref_id](alignementTestRoi));
			int keyboard = waitKey(50);
			}
			}*/
		}



		// Fill referencePointsAbsoluteCoordinates, combining information from positionOfReferencePointsInOtherReferences ant the barycentres.
		for (int ref_id = 0; ref_id < N_REFERENCE_FRAMES; ref_id++)
		{
			referencePointsAbsoluteCoordinates[ref_id].resize(referencePoints[ref_id].size());

			vector<int> nMatchesPerPoint(referencePoints[ref_id].size());
			std::fill(nMatchesPerPoint.begin(), nMatchesPerPoint.end(), 0);

			for (int matched_ref_id = 0; matched_ref_id < N_REFERENCE_FRAMES; matched_ref_id++)
			{
				if (ref_id == matched_ref_id) continue;


				for (uint j = 0; j < referencePoints[ref_id].size(); j++)
				{
					if (!std::isnan(positionOfReferencePointsInOtherReferences[ref_id][matched_ref_id][j].x)) {
						referencePointsAbsoluteCoordinates[ref_id][j] += positionOfReferencePointsInOtherReferences[ref_id][matched_ref_id][j] + absoluteOffsets[ref_id];
						nMatchesPerPoint[j]++;
					}
				}

			}

			for (uint j = 0; j < referencePoints[ref_id].size(); j++)
			{
				if (nMatchesPerPoint[j] > 0) {
					referencePointsAbsoluteCoordinates[ref_id][j] /= (float) nMatchesPerPoint[j];
				}
				else
				{
					referencePointsAbsoluteCoordinates[ref_id][j] = absoluteOffsets[ref_id] + referencePoints[ref_id][j];
				}
			}
		}
	}

	std::cout << "\nReference frames (" << N_REFERENCE_FRAMES << ") aligned.\n" << std::endl;



	/////////////////// END OF REFERENCE FRAMES PROCESSING //////////////////////////////////



	// Pre allocated for efficiency.
	Mat laplacianBuffer(RGBImages[0].size(), CV_32F);
	

	Size SR_size(LR_w * UPSCALE_FACTOR, LR_h * UPSCALE_FACTOR);
	Size SR_insideSquareGrid(nSquareX*SQUARE_SIZE*UPSCALE_FACTOR, nSquareY*SQUARE_SIZE*UPSCALE_FACTOR);

	Mat SR_src(SR_size, CV_32F); // A buffer for the upscaled bayer source image (Nearest neighbor).
	Mat extractedChannel(SR_size, CV_32F); // A buffer for a channel of SR_src.

	Mat srcRectSubPix(SR_insideSquareGrid, CV_32F);
	Mat normRectSubPix(SR_insideSquareGrid, CV_32F);

	vector<Mat> stack(3);			 // 1 channel per color
	vector<Mat> normalizer(3);       // 1 channel per color
	vector<Mat> LR_bayerMasks(3);			 // 1 channel per color
	vector<Mat> SR_bayerMasks(3);			 // 1 channel per color
	vector<Mat> SR_localMasks(3);			 // 1 channel per color


	// Initialize the vectors and matrices defined above.
	{
		for (int channel = 0; channel < 3; channel++)
		{
			stack[channel] = cv::Mat::zeros(SR_insideSquareGrid, CV_32F);
			normalizer[channel] = cv::Mat::zeros(SR_insideSquareGrid, CV_32F);
			LR_bayerMasks[channel] = cv::Mat::zeros(Size(LR_w, LR_h), CV_32F);
			SR_bayerMasks[channel] = cv::Mat::zeros(SR_size, CV_32F);
			SR_localMasks[channel] = cv::Mat::zeros(SR_size, CV_32F);
		}

		for (int x = 0; x < LR_w; x++)
		{
			for (int y = 0; y < LR_h; y++)
			{
				// The bayer pattern is BG . The result of the following computation makes channel = 1 if green, 0 if blue, 2 if red (or blue and red reversed)
				//					    GR
				int isGreen = (x % 2) ^ (y % 2); // Bitwise XOR.
				int channel = isGreen + (1 - isGreen) * 2 * (x % 2);

				LR_bayerMasks[channel].at<float>(y, x) = 1.f;
			}
		}

		for (int channel = 0; channel < 3; channel++)
		{
			resize(LR_bayerMasks[channel], SR_bayerMasks[channel], SR_bayerMasks[channel].size(), 0, 0, INTER_NEAREST);
		}
	}


#ifdef _DEBUG
	int resolution = 10; // sqrt(Bins per pixel)
	Mat bins = Mat::zeros(Size(resolution*2, resolution*2), CV_8U);
#endif


	// In each image of the sequence, find the LK features of the references, and compute its estimated offset in the absolute referential.
	for (int i = N_REFERENCE_FRAMES; i < RGBImages.size(); i++)
	{

		// Barycentre of the offsets of the tracked points across the references. 
		Point2f currentFrameAbsoluteOffset(.0f, .0f);

		int totalPointsMatched = 0;

		for (int ref_id = 0; ref_id < N_REFERENCE_FRAMES; ref_id++)
		{

			std::vector<Point2f> potentialPoints;

			// calculate optical flow
			vector<uchar> status;
			vector<float> err;
			TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);

			calcOpticalFlowPyrLK(GRAYImages[ref_id], GRAYImages[i], referencePoints[ref_id], potentialPoints, status, err, Size(LK_WINDOW_SIZE, LK_WINDOW_SIZE), LK_PYRAMID_LEVELS, criteria, 0, 1.0E-3);

#ifdef _DEBUG
			Mat refIm = GRAYImages[ref_id].clone();
			Mat currIm = GRAYImages[i].clone();

			std::vector<Point2f> deltas(potentialPoints.size());
#endif


			int nMatches = 0;

			for (uint j = 0; j < potentialPoints.size(); j++)
			{
				// Select good points
				if (status[j] == 1 && err[j] < ERR_SEQUENCE_THRESHOLD) {
					currentFrameAbsoluteOffset += referencePointsAbsoluteCoordinates[ref_id][j] - potentialPoints[j];

					nMatches++;

					// TODO ransac

#ifdef _DEBUG
					cv::circle(refIm, referencePoints[ref_id][j], 2, 255);
					cv::circle(currIm, potentialPoints[j], 2, 255);

					deltas[j] =  potentialPoints[j] - referencePoints[ref_id][j];
#endif
				}
			}

			//std::cerr << "nMatches at step " << i << " : " << nMatches << " / " << referencePoints[ref_id].size() << std::endl;


			totalPointsMatched += nMatches;

#ifdef _DEBUG
			int matHalfSize = 100;
			float multiplier = 40.f;
			Mat visu = Mat::zeros(Size(matHalfSize*2+1,matHalfSize*2+1), CV_8U);
			vector<int> _ids(nMatches);
			{
				int _m = 0;
				for (uint j = 0; j < potentialPoints.size(); j++)
				{
					if (status[j] == 1 && err[j] < ERR_SEQUENCE_THRESHOLD)
					{
						_ids[_m] = j;
						_m++;
					}
				} // ids sorted by ascending err.
				std::sort(_ids.begin(), _ids.end(),
					[&err](int id_a, int id_b) {
						return err[id_a] > err[id_b];
					});
			}


			for (uint _i = 0; _i < _ids.size(); _i++)
			{
				int _j = _ids[_i];

				int _x = (int) (deltas[_j].x * multiplier) + matHalfSize + 1;
				_x = std::clamp(_x, 0, 2*matHalfSize);
				int _y = (int) (deltas[_j].y * multiplier) + matHalfSize + 1;
				_y = std::clamp(_y, 0, 2*matHalfSize);
				visu.at<uchar>(_y, _x) = (uchar)(255.f * (float)(_i+1) / (float)_ids.size());
			}


			int a = 0;

#endif
		}

		currentFrameAbsoluteOffset /= (float) totalPointsMatched;



		// Compute the weights of each pixel, proportional to their sharpness.

		cv::Laplacian(RGBImages[i], laplacianBuffer, CV_32F);

		for (int x = 0; x < nSquareX; x++)
		{
			for (int y = 0; y < nSquareY; y++)
			{

				Rect srcRoi(x * SQUARE_SIZE + SQUARE_GRID_MARGIN - (int)currentFrameAbsoluteOffset.x, y * SQUARE_SIZE + SQUARE_GRID_MARGIN - (int)currentFrameAbsoluteOffset.y, SQUARE_SIZE, SQUARE_SIZE);

				float sharpness;
				if (srcRoi.x < 0 || srcRoi.y < 0 || srcRoi.x + srcRoi.width >= RGBImages[i].cols || srcRoi.y + srcRoi.height >= RGBImages[i].rows)
				{
					sharpness = .0f;
				}
				else
				{
					Scalar varLaplacian;
					Scalar meanLaplacian;
					cv::meanStdDev(RGBImages[i](srcRoi), meanLaplacian, varLaplacian);
					sharpness = (float)varLaplacian[0]; 
				}	
				
				Rect maskRoi(srcRoi.x * UPSCALE_FACTOR, srcRoi.y * UPSCALE_FACTOR, srcRoi.width * UPSCALE_FACTOR, srcRoi.height * UPSCALE_FACTOR);
				for (int c = 0; c < 3; c++)
				{
					SR_localMasks[c](maskRoi) = SR_bayerMasks[c](maskRoi) * 1.f;
					//SR_localMasks[c](maskRoi) = SR_bayerMasks[c](maskRoi) * (sharpness + .1f);
				}
			}
		}


		// Combine the channels of the current bayer image with the channels of the stack:


		Mat bayerIm_32F;
		bayerImages[i].convertTo(bayerIm_32F, CV_32F);
		resize(bayerIm_32F, SR_src, SR_size, 0,0, INTER_NEAREST);

		Point2f SRgridOffset((float) (SQUARE_GRID_MARGIN * UPSCALE_FACTOR), (float) (SQUARE_GRID_MARGIN * UPSCALE_FACTOR));
		Point2f center = SRgridOffset + .5f * (Point2f) SR_insideSquareGrid - (float) UPSCALE_FACTOR * currentFrameAbsoluteOffset;

		for (int c = 0; c < 3; c++)
		{
			multiply(SR_src, SR_localMasks[c], extractedChannel);

			// Extract the portion of the current SR channel that falls exactly in the square grid of the absolute referential.
			getRectSubPix(extractedChannel, SR_insideSquareGrid, center, srcRectSubPix);
			getRectSubPix(SR_localMasks[c], SR_insideSquareGrid, center, normRectSubPix);

			add(stack[c], srcRectSubPix, stack[c]);
			add(normalizer[c], normRectSubPix, normalizer[c]);
		}


#ifdef _DEBUG
		float fracBayerX = 2.f * (float) resolution * currentFrameAbsoluteOffset.x - 2.f * (float) resolution * floorf(currentFrameAbsoluteOffset.x);
		float fracBayerY = 2.f * (float) resolution * currentFrameAbsoluteOffset.y - 2.f * (float) resolution * floorf(currentFrameAbsoluteOffset.y);

		bins.at<char>((int)fracBayerY, (int)fracBayerX) += 1;
#endif
		std::cout << "Processed image " << i  << ", offset " << currentFrameAbsoluteOffset.x << " " << currentFrameAbsoluteOffset.y << std::endl;
	}

	
	for (int channel = 0; channel < 3; channel++)
	{
		cv::divide(stack[channel], normalizer[channel], stack[channel]);
	}



	//  Because openCV doesnt normalize its windows...
	//for (int channel = 0; channel < 3; channel++) stack[channel] *= 1.f/255.f;

	vector<Mat> deconv8U(3);
	Mat deconvRes(stack[0].size(), CV_32F);
	for (int channel = 0; channel < 3; channel++)
	{
		//Rect deconvROI = cv::selectROI(stack[1]);
		//std::cout << "\n SELECTED CONVOLUTION ROI DIMENSIONS: " << deconvROI.x << " " << deconvROI.y << " " << deconvROI.width << " " << deconvROI.height << std::endl;
		//cv::destroyAllWindows();

		Rect deconvROI(0, 0, stack[channel].cols, stack[channel].rows);

		Mat toBeDeconv = stack[channel](deconvROI);

		const int nSigmas = 4;
		float sigmas[nSigmas];
		for (int i = 3; i < nSigmas; i++)
		{
			sigmas[i] = (float) UPSCALE_FACTOR * .5f * (1.f + (float) i);

			//deconvRes = LucyRichardson(toBeDeconv, 10, sigmas[i]);
			deconvRes = LucyRichardson(toBeDeconv, 20, sigmas[i]); 
			//deconvRes = LucyRichardson(toBeDeconv, 30, sigmas[i]); 

			deconvRes.convertTo(deconv8U[channel], CV_8U);

			//std::string name = "SR_samples\\" + seqName + "_" + std::to_string(UPSCALE_FACTOR) + "x_"
			//	+ std::to_string((int)RGBImages.size()) + "images_sigma" + std::to_string(sigmas[i]) 
			//	+ std::to_string(time(0)) + ".bmp";
			//cv::imwrite(name, deconv8U);
		}
	}

	//for (int channel = 0; channel < 3; channel++) stack[channel] *= 255.f;



	//Mat unsharpMasked = UnsharpMasking(deconv);
	//Mat equalHist = EqualizeHistogram(deconv8U);

	Mat coloredStack32F, coloredStack8U;
	Mat coloredDeconv8U;


	// Correcting chromatic aberration (tests)
	//vector<Mat> comp;
	//for (int offset = 0; offset < 8; offset+= 2)
	//{
	//	stack[0] = stack[0](Rect(offset,0, stack[0].cols-offset,stack[0].rows-offset)); // blue
	//	stack[1] = stack[1](Rect(0,0,	   stack[1].cols-offset,stack[1].rows-offset)); // green
	//	stack[2] = stack[2](Rect(0,0,	   stack[2].cols-offset,stack[2].rows-offset));	// red
	//	deconv8U[0] = deconv8U[0](Rect(offset,0, stack[0].cols-offset,stack[0].rows-offset)); // blue
	//	deconv8U[1] = deconv8U[1](Rect(0,0,	   stack[1].cols-offset,stack[1].rows-offset)); // green
	//	deconv8U[2] = deconv8U[2](Rect(0,0,	   stack[2].cols-offset,stack[2].rows-offset));	// red

	//	cv::merge(stack, coloredStack32F);
	//	cv::merge(deconv8U, coloredDeconv8U);

	//	coloredStack32F.convertTo(coloredStack8U, CV_8U);

	//	cvtColor(coloredStack8U, coloredStack8U, COLOR_BGR2RGB);
	//	cvtColor(coloredDeconv8U, coloredDeconv8U, COLOR_BGR2RGB);

	//	comp.push_back(coloredDeconv8U.clone());
	//}
	//__debugbreak();



	// Correcting chromatic aberration:
	int offset = 2;
	stack[0] = stack[0](Rect(offset,0, stack[0].cols-offset,stack[0].rows-offset)); // blue
	stack[1] = stack[1](Rect(0,0,	   stack[1].cols-offset,stack[1].rows-offset)); // green
	stack[2] = stack[2](Rect(0,0,	   stack[2].cols-offset,stack[2].rows-offset));	// red
	deconv8U[0] = deconv8U[0](Rect(offset,0, stack[0].cols-offset,stack[0].rows-offset)); // blue
	deconv8U[1] = deconv8U[1](Rect(0,0,	   stack[1].cols-offset,stack[1].rows-offset)); // green
	deconv8U[2] = deconv8U[2](Rect(0,0,	   stack[2].cols-offset,stack[2].rows-offset));	// red

	cv::merge(stack, coloredStack32F);
	cv::merge(deconv8U, coloredDeconv8U);

	coloredStack32F.convertTo(coloredStack8U, CV_8U);

	cvtColor(coloredStack8U, coloredStack8U, COLOR_BGR2RGB);
	cvtColor(coloredDeconv8U, coloredDeconv8U, COLOR_BGR2RGB);

	cv::imwrite("stack.bmp", coloredStack8U);
	cv::imwrite("deconv.bmp", coloredDeconv8U);

	cv::imshow("Stack", coloredStack8U);
	cv::imshow("Deconvolution", coloredDeconv8U);

	int i = 0;
	Mat upscaledSrcIm(coloredStack8U.size(), CV_8UC3);
	Rect srcROI(SQUARE_GRID_MARGIN, SQUARE_GRID_MARGIN, coloredDeconv8U.cols/UPSCALE_FACTOR, coloredDeconv8U.rows/UPSCALE_FACTOR);
	while (true)
	{
		Mat srcIm = RGBImages[i%200](srcROI);
		resize(srcIm, upscaledSrcIm, upscaledSrcIm.size(), 0, 0, INTER_LINEAR);
		imshow("Src (bilinear upscale)", upscaledSrcIm);
		waitKey(50);
		i++;
	}

	int keyboard = waitKey(0);

	return coloredStack8U;
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
