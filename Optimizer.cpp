#define NOMINMAX
#include "Optimizer.h"

#include "delaunator.h"

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


		// TUILES PROCHES
		//img_99_j17_m09_a2025_10h_26m_14s
		//img_0_j17_m09_a2025_10h_26m_11s
		int sec0 = 11;
		const char* base = "_j17_m09_a2025_10h_26m_";

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


		// HORNISGRINDE HD
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

		cv::Mat img = cv::imread("C:\\Users\\alpha\\Desktop\\CSHR\\CSHR_2\\bursts\\" + filename);
		if (img.empty())
		{
			throw std::runtime_error("Failed to load image: " + filename);
		}


		// calcOpticalFlowPyrLK only takes in CV_8U, so we cant use CV_32F yet.

		

		//img = EqualizeHistogram(img); //TODO same transform for all.

		//cv::GaussianBlur(img, img, Size(), 3.f);

		Mat channels[3];
		cv::split(img, channels);

		// For visualization only:
		auto c0 = channels[0];
		auto c1 = channels[1];
		auto c2 = channels[2];

		//images.push_back(channels[0]);

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
	const double QUALITY_LEVEL = .2; // default .3

	// Points that are near the edges are likely to disappear from frame to frame due to distortions/vibrations. 
	const int MARGIN_SIZE = 25;
	Rect roi(MARGIN_SIZE, MARGIN_SIZE, reference.cols - 2 * MARGIN_SIZE, reference.rows - 2 * MARGIN_SIZE);

	// Take first frame and find corners in it
	goodFeaturesToTrack(reference(roi), referencePoints, MAX_POINTS_TRACKED, QUALITY_LEVEL, MIN_DISTANCE, Mat(), 7, false, 0.04);

	for (int i = 0; i < referencePoints.size(); i++)
	{
		referencePoints[i] += Point2f((float)MARGIN_SIZE, (float)MARGIN_SIZE);
	}



	constexpr bool TEST_DELAUNAY = false;
	if (TEST_DELAUNAY) {

		Mat mask = Mat::zeros(reference.size(), reference.type());

		// additional points on the edges.
		int nPointsPerSide = 5;

		std::vector<double> coords(referencePoints.size() * 2 + nPointsPerSide * 4 * 2);
		for (int i = 0; i < referencePoints.size(); i++)
		{
			circle(mask, referencePoints[i], 3, Scalar(255), -1);
			coords[2 * i] = (double)referencePoints[i].x;
			coords[2 * i + 1] = (double)referencePoints[i].y;
		}

		int offset = (int)referencePoints.size() * 2;

		// Vertical edges
		for (int i = 0; i < nPointsPerSide; i++)
		{
			double py0 = 0;
			double py1 = (double)reference.rows;
			double px = (double)i * (double)reference.cols / (double)nPointsPerSide;

			coords[offset++] = px;
			coords[offset++] = py0;
			coords[offset++] = px;
			coords[offset++] = py1;
		}

		// Horizontal edges
		for (int i = 0; i < nPointsPerSide; i++)
		{
			double py = (double)(i + 1) * (double)reference.rows / (double)nPointsPerSide;
			double px0 = 0;
			double px1 = (double)reference.cols;

			coords[offset++] = px0;
			coords[offset++] = py;
			coords[offset++] = px1;
			coords[offset++] = py;
		}

		//triangulation happens here
		delaunator::Delaunator d(coords);

		for (std::size_t i = 0; i < d.triangles.size(); i += 3)
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

			Point2f p0((float)d.coords[2 * d.triangles[i]], (float)d.coords[2 * d.triangles[i] + 1]);
			Point2f p1((float)d.coords[2 * d.triangles[i + 1]], (float)d.coords[2 * d.triangles[i + 1] + 1]);
			Point2f p2((float)d.coords[2 * d.triangles[i + 2]], (float)d.coords[2 * d.triangles[i + 2] + 1]);

			cv::line(mask, p0, p1, Scalar(255), 2);
			cv::line(mask, p1, p2, Scalar(255), 2);
			cv::line(mask, p2, p0, Scalar(255), 2);
		}
		cv::Mat result;
		cv::add(reference, mask, result);

		cv::imshow("Triangulation", result);
		int keyboard = waitKey(0);
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
		calcOpticalFlowPyrLK(reference, images[i], referencePoints, potentialPoints, status, err, Size(15, 15), 2, criteria, 0, 1.0E-3);
		
		for (uint j = 0; j < potentialPoints.size(); j++)
		{
			// Select good points
			if (status[j] == 1) {
				pointsDetectedInCurrentImage.push_back(potentialPoints[j]);

				// draw the tracks
				circle(mask, potentialPoints[j], 3, colors[j], -1);
				cv::line(mask, potentialPoints[j], referencePoints[j], Scalar(0), 1);
			}
		}
		std::cout << "nMatches at step " << i << " : " << pointsDetectedInCurrentImage.size() << " / " << referencePoints.size() << std::endl;

		Mat img;
		cv::add(images[i], mask, img);

		/*imshow("Frame", img);
		int keyboard = waitKey(100);*/

		//if (keyboard == 'q' || keyboard == 27)
		//	break;

	}

	return points; 
}


cv::Mat Optimizer::WarpStack(std::vector<cv::Mat> images, std::vector<vector<Point2f>> trackedPoints)
{
	const int UPSCALE_FACTOR = 2;
	const int PADDING_SIZE = 30; // In pixels 

	// additional points on the edges. Per edge, so total is 4x N_EDGE_POINTS
	const int N_EDGE_POINTS = 5;

	// between 0 and 100
	const float STACKED_PERCENTILE = 10.f;

	int LR_w = images[0].cols;
	int LR_h = images[0].rows;
	

	vector<Point2f> barycentres(trackedPoints[0].size());
	vector<Point2f> globalMovement(images.size());

	// Compute the barycenter of each tracked point over the image sequence
	{
		for (int j = 0; j < trackedPoints[0].size(); j++)
		{
			barycentres[j] = Point2f(.0f, .0f);
		}
		for (int j = 0; j < images.size(); j++)
		{
			globalMovement[j] = Point2f(.0f, .0f);
		}
		for (int i = 0; i < images.size(); i++)
		{
			if (trackedPoints[i].size() != trackedPoints[0].size())
			{
				__debugbreak();
			}
			for (int j = 0; j < trackedPoints[i].size(); j++)
			{
				barycentres[j] += trackedPoints[i][j];
				globalMovement[i] += trackedPoints[i][j];
			}
			globalMovement[i] /= (float)trackedPoints[i].size();
		}
		Point2f stackCenter(.0f, .0f);
		for (int j = 0; j < trackedPoints[0].size(); j++)
		{
			barycentres[j] /= (float)images.size();
			stackCenter += barycentres[j];
		}
		stackCenter /= (float)trackedPoints[0].size();
		for (int j = 0; j < images.size(); j++)
		{
			globalMovement[j] -= stackCenter;
		}
	}

	
	std::vector<double> coords(barycentres.size() * 2 + N_EDGE_POINTS * 4 * 2);

	// Fill coords with the barycenters and add the edge points
	{
		for (int i = 0; i < barycentres.size(); i++)
		{
			coords[2 * i] = (double)barycentres[i].x;
			coords[2 * i + 1] = (double)barycentres[i].y;
		}

		int offset = (int)barycentres.size() * 2;


		// The edge points are 2 pixels inwards, to avoid edge cases later, especially when computin the bounding rects of mesh triangles.

		// Horizontal edges
		for (int i = 0; i < N_EDGE_POINTS; i++)
		{
			double py0 = 2;
			double py1 = (double)LR_h - 2;
			double px0 = 2 + (double)i * (double)(LR_w - 4) / (double)N_EDGE_POINTS;
			double px1 = 2 + (double)(i+1) * (double)(LR_w - 4) / (double)N_EDGE_POINTS;

			coords[offset++] = px0;
			coords[offset++] = py0;
			coords[offset++] = px1;
			coords[offset++] = py1;
		}

		// Vertical edges
		for (int i = 0; i < N_EDGE_POINTS; i++)
		{
			double py0 = 2 + (double)(i + 1) * (double)(LR_h - 4) / (double)N_EDGE_POINTS;
			double py1 = 2 + (double)i * (double)(LR_h - 4) / (double)N_EDGE_POINTS;
			double px0 = 2;
			double px1 = (double)LR_w - 2;

			coords[offset++] = px0;
			coords[offset++] = py0;
			coords[offset++] = px1;
			coords[offset++] = py1;
		}
	}


	// Triangulation happens here
	delaunator::Delaunator d(coords);



	bool SHOW_WARPS = true;
	if (SHOW_WARPS) {
		for (int i = 0; i < images.size(); i++)
		{
			// Create a mask image for drawing purposes
			Mat mask = Mat::zeros(images[0].size(), images[0].type());

			for (uint j = 0; j < trackedPoints[i].size(); j++)
			{
				circle(mask, trackedPoints[i][j], 3, Scalar(255), -1);
			}

			for (int p = 0; p < barycentres.size(); p++)
			{
				coords[2 * p] = (double)trackedPoints[i][p].x;
				coords[2 * p + 1] = (double)trackedPoints[i][p].y;
			}

			for (std::size_t t = 0; t < d.triangles.size(); t += 3)
			{
				Point2f p0((float)coords[2 * d.triangles[t]], (float)coords[2 * d.triangles[t] + 1]);
				Point2f p1((float)coords[2 * d.triangles[t + 1]], (float)coords[2 * d.triangles[t + 1] + 1]);
				Point2f p2((float)coords[2 * d.triangles[t + 2]], (float)coords[2 * d.triangles[t + 2] + 1]);

				cv::line(mask, p0, p1, Scalar(255), 1);
				cv::line(mask, p1, p2, Scalar(255), 1);
				cv::line(mask, p2, p0, Scalar(255), 1);
			}

			Mat img;
			cv::add(images[i], mask, img);

			cv::imshow("Frame", img);
			int keyboard = waitKey(1000);

			//if (keyboard == 'q' || keyboard == 27)
			//	cv::destroyWindow("Frame");
			//	break;
		}
	}




	// Create a vector of vectors, containing for each triangle of the mesh the vector of measured sharpness over the image sequence.
	std::vector<std::vector<float>> sharpness(d.triangles.size()/3);
	for (int j = 0; j < d.triangles.size(); j += 3) sharpness[j/3].resize(images.size());
	
	// Buffer to avoid reallocs
	cv::Mat laplacian(images[0].size(), CV_32F);


	// Is here, but could happen as soon as calcOpticalFlowPyrLK has been called on the image, because it is the only reason images are in CV_8U in the first place
	for (int i = 0; i < images.size(); i++)
	{
		images[i].convertTo(images[i], CV_32F);

#ifdef _DEBUG
		images[i] /= 255.f;
#endif
	}
	

	// fill sharpness with the appropriate values.
	for (int i = 0; i < images.size(); i++)
	{
		cv::Laplacian(images[i], laplacian, CV_32F);

		// d.coords is a deep copy of coords. It holds the barycenters, but we want to use the positions of the triangle in the i-th image, 
		// so we reuse (overwrite) coord with those positions in the i-th image. (Avoids us the reallocation, and rewriting the edge points)

		for (int j = 0; j < trackedPoints[i].size(); j++)
		{
			coords[2 * j] = (double)trackedPoints[i][j].x;
			coords[2 * j + 1] = (double)trackedPoints[i][j].y;
		}


		
		for (int j = 0; j < d.triangles.size(); j+=3)
		{
			Scalar varLaplacian;
			Scalar meanLaplacian;

			Point2i p0((int)coords[2 * d.triangles[j]],     (int)coords[2 * d.triangles[j] + 1]);
			Point2i p1((int)coords[2 * d.triangles[j + 1]], (int)coords[2 * d.triangles[j + 1] + 1]);
			Point2i p2((int)coords[2 * d.triangles[j + 2]], (int)coords[2 * d.triangles[j + 2] + 1]);

			std::vector<Point2i> triangle{ p0, p1, p2 };

			// Should not use boundingRect, because it sucks.
			cv::Rect _roi = cv::boundingRect(triangle); 
			cv::Mat _mask = cv::Mat::zeros(Size(_roi.width, _roi.height), CV_8U);

			triangle[0] -= _roi.tl();
			triangle[1] -= _roi.tl();
			triangle[2] -= _roi.tl();

			cv::fillConvexPoly(_mask, triangle, 1);

			cv::meanStdDev(images[i](_roi), meanLaplacian, varLaplacian, _mask);

			sharpness[j/3][i] = (float) varLaplacian[0];
		}
	}
	
	
	
	// TODO move the edge points from the i-th barycenter's offset when unwarping

	// TODO nearest neighbor, as per "sharp stack"


	Size SR_size(LR_w * UPSCALE_FACTOR + PADDING_SIZE * 2, LR_h * UPSCALE_FACTOR + PADDING_SIZE * 2);
	Mat stack = cv::Mat::zeros(SR_size, CV_32F);
	Mat normalizer = cv::Mat::zeros(SR_size, CV_32F);

	// preallocated. The to-be-sorted images ids in the sequence.
	std::vector<int> ids(images.size());

	// For each triangle, sort its warped (i.e. original) versions in all images by sharpness, unwarp and stack the sharpest fraction 
	for (int j = 0; j < d.triangles.size(); j+=3)
	{
		// sort ids by descending sharpness
		for (int i = 0; i < images.size(); i++) ids[i] = i;
		std::sort(ids.begin(), ids.end(),
			[&](int id_a, int id_b) {
				return sharpness[j/3][id_a] > sharpness[j/3][id_b];
			});


		

		cv::Rect _dstROI; // a bounding box containing the triangle in the stack, slighlty wider because the trianlge coordinates are floats and ROI are ints
		std::vector<Point2f> dstTri(3); // the triangle in the stack, using coordinates relative to the top of the _dstROI bounding box

		// Compute the roi then the triangle. The ROI is computed manually to make sure that the triangle (float vertices) is within it.
		{
			double maxX = std::numeric_limits<double>::min();
			double minX = std::numeric_limits<double>::max();
			double maxY = std::numeric_limits<double>::min();
			double minY = std::numeric_limits<double>::max();

			for (int t = 0; t < 3; t++)
			{
				maxX = std::max(maxX, d.coords[2 * d.triangles[j + t]]);
				minX = std::min(minX, d.coords[2 * d.triangles[j + t]]);
				maxY = std::max(maxY, d.coords[2 * d.triangles[j + t] + 1]);
				minY = std::min(minY, d.coords[2 * d.triangles[j + t] + 1]);
			}

			_dstROI.x = (int) std::floor(minX);
			_dstROI.y = (int) std::floor(minY);
			_dstROI.width  = (int) std::ceil(maxX) - _dstROI.x;
			_dstROI.height = (int) std::ceil(maxY) - _dstROI.y;

			_dstROI.x = _dstROI.x * UPSCALE_FACTOR + PADDING_SIZE;
			_dstROI.y = _dstROI.y * UPSCALE_FACTOR + PADDING_SIZE;
			_dstROI.width  *= UPSCALE_FACTOR;
			_dstROI.height *= UPSCALE_FACTOR;


			Point2f dstP0((float)d.coords[2 * d.triangles[j]],     (float)d.coords[2 * d.triangles[j]     + 1]);
			Point2f dstP1((float)d.coords[2 * d.triangles[j + 1]], (float)d.coords[2 * d.triangles[j + 1] + 1]);
			Point2f dstP2((float)d.coords[2 * d.triangles[j + 2]], (float)d.coords[2 * d.triangles[j + 2] + 1]);

			dstP0 = dstP0 * UPSCALE_FACTOR + Point2f(PADDING_SIZE, PADDING_SIZE);
			dstP1 = dstP1 * UPSCALE_FACTOR + Point2f(PADDING_SIZE, PADDING_SIZE);
			dstP2 = dstP2 * UPSCALE_FACTOR + Point2f(PADDING_SIZE, PADDING_SIZE);

			dstTri[0] = dstP0 - (Point2f)_dstROI.tl();
			dstTri[1] = dstP1 - (Point2f)_dstROI.tl();
			dstTri[2] = dstP2 - (Point2f)_dstROI.tl();
		}


		// unwarp and stack
		int nBest = (int)((float)images.size() * STACKED_PERCENTILE / 100.f);
		for (int i = 0; i < nBest; i++)
		{
			int _id = ids[i];

			// We use "coords" as the array holding the position of the points in the _id-th image, and "d.coords"  for the positions of the barycenters.
			// d.coords already holds the correct values, but coords needs be filled for each one of the best images.
			for (int t = 0; t < 3; t++)
			{
				if (d.triangles[j + t] < trackedPoints[_id].size()) // i.e. if this point of the triangle is not one of the manually added edge points
				{
					coords[2 * d.triangles[j + t]]     = trackedPoints[_id][d.triangles[j + t]].x;
					coords[2 * d.triangles[j + t] + 1] = trackedPoints[_id][d.triangles[j + t]].y;
				}
				else
				{
					coords[2 * d.triangles[j + t]]     = d.coords[2 * d.triangles[j + t]]     + globalMovement[_id].x;
					coords[2 * d.triangles[j + t] + 1] = d.coords[2 * d.triangles[j + t] + 1] + globalMovement[_id].y;
				}
			}
			

			cv::Rect _srcROI; // a bounding box containing the triangle in the _id-th image, slighlty wider because the triangle coordinates are floats and ROI are ints
			std::vector<Point2f> srcTri(3); // Triangle with float coordinates, used to determine the affine mapping between the triangle in the _id-th image and in the stack:

			// fill _srcROI and srcTri
			{
				double maxX = std::numeric_limits<double>::min();
				double minX = std::numeric_limits<double>::max();
				double maxY = std::numeric_limits<double>::min();
				double minY = std::numeric_limits<double>::max();

				for (int t = 0; t < 3; t++)
				{
					maxX = std::max(maxX, coords[2 * d.triangles[j + t]]);
					minX = std::min(minX, coords[2 * d.triangles[j + t]]);
					maxY = std::max(maxY, coords[2 * d.triangles[j + t] + 1]);
					minY = std::min(minY, coords[2 * d.triangles[j + t] + 1]);
				}

				_srcROI.x = (int) std::floor(minX);
				_srcROI.y = (int) std::floor(minY);
				_srcROI.width  = (int) std::ceil(maxX) - _srcROI.x;
				_srcROI.height = (int) std::ceil(maxY) - _srcROI.y;

				
				srcTri[0] = Point2f((float)coords[2 * d.triangles[j]],     (float)coords[2 * d.triangles[j]     + 1]) - (Point2f)_srcROI.tl();
				srcTri[1] = Point2f((float)coords[2 * d.triangles[j + 1]], (float)coords[2 * d.triangles[j + 1] + 1]) - (Point2f)_srcROI.tl();
				srcTri[2] = Point2f((float)coords[2 * d.triangles[j + 2]], (float)coords[2 * d.triangles[j + 2] + 1]) - (Point2f)_srcROI.tl();
			}



			// Not really used as a conventional mask: once warped and upscaled by the affine transform, it is used to multiply termwise to carve out
			// an antialiased triangle, and is kept in the "normalizer" matrix for the normalization of the stack at the end.
			cv::Mat triangleMask = cv::Mat::zeros(_srcROI.size(), CV_32F);
			std::vector<Point2i> srcTri_INT {(Point2i)srcTri[0], (Point2i)srcTri[1], (Point2i)srcTri[2]};

			cv::fillConvexPoly(triangleMask, srcTri_INT, 1.f); // Requires int coordinates
			// For some reason, it wont draw antialiased lines unless the type is CV_8U. Since anti aliasing is not strictly necessary, we dont do it.
			//cv::fillConvexPoly(triangleMask, srcTri_INT, 1.f, cv::LINE_AA); 


			Mat warp_mat = getAffineTransform(srcTri, dstTri);

			Mat warpedRoi = Mat::zeros(_dstROI.size(), CV_32F);
			Mat warpedMask = Mat::zeros(_dstROI.size(), CV_32F);

			cv::warpAffine(images[_id](_srcROI), warpedRoi,  warp_mat, warpedRoi.size());
			cv::warpAffine(triangleMask        , warpedMask, warp_mat, warpedMask.size());

			cv::multiply(warpedRoi, warpedMask, warpedRoi);

			cv::add(warpedRoi,  stack(_dstROI),      stack(_dstROI));
			cv::add(warpedMask, normalizer(_dstROI), normalizer(_dstROI));
		}
	}
	


	// Normalize
	cv::divide(stack, normalizer, stack);

#ifdef _DEBUG
	stack *= 255.f; // images were put in [0,1] for visualization
#endif

	stack.convertTo(stack, CV_8U);

	cv::imwrite("stack2.png", stack);

	cv::imshow("Stack", stack);

	int keyboard = waitKey(0);
		
	return stack;
}



cv::Mat Optimizer::ShiftStack(std::vector<cv::Mat> images, std::vector<vector<Point2f>> trackedPoints)
{
	const int UPSCALE_FACTOR = 2;
	const int PADDING_SIZE = 30; // In pixels 

	// between 0 and 100
	const float STACKED_PERCENTILE = 1.f;


	const int SQUARE_SIZE = 20*2; // in px. Must be even, to simplify cv::getRectSubPix's center

	int LR_w = images[0].cols;
	int LR_h = images[0].rows;


	vector<Point2f> barycentres(trackedPoints[0].size());
	vector<Point2f> globalMovement(images.size());

	// Compute the barycenter of each tracked point over the image sequence
	{
		for (int j = 0; j < trackedPoints[0].size(); j++)
		{
			barycentres[j] = Point2f(.0f, .0f);
		}
		for (int j = 0; j < images.size(); j++)
		{
			globalMovement[j] = Point2f(.0f, .0f);
		}
		for (int i = 0; i < images.size(); i++)
		{
			if (trackedPoints[i].size() != trackedPoints[0].size())
			{
				__debugbreak();
			}
			for (int j = 0; j < trackedPoints[i].size(); j++)
			{
				barycentres[j] += trackedPoints[i][j];
				globalMovement[i] += trackedPoints[i][j];
			}
			globalMovement[i] /= (float)trackedPoints[i].size();
		}
		Point2f stackCenter(.0f, .0f);
		for (int j = 0; j < trackedPoints[0].size(); j++)
		{
			barycentres[j] /= (float)images.size();
			stackCenter += barycentres[j];
		}
		stackCenter /= (float) trackedPoints[0].size();
		for (int j = 0; j < images.size(); j++)
		{
			globalMovement[j] -= stackCenter;
		}
	}



	// Is here, but could happen as soon as calcOpticalFlowPyrLK has been called on the image, because it is the only reason images are in CV_8U in the first place
	for (int i = 0; i < images.size(); i++)
	{
		images[i].convertTo(images[i], CV_32F);

#ifdef _DEBUG
		images[i] /= 255.f;
#endif
	}


	bool showStabilizedFootage = false;
	if (showStabilizedFootage){
		int nLoops = 1000;
		cv::Mat frame(images[0].size(), CV_32F);
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
				Rect _roi(x * SQUARE_SIZE - globalMovement[i].x, y * SQUARE_SIZE - globalMovement[i].y, SQUARE_SIZE, SQUARE_SIZE);

				cv::meanStdDev(images[i](_roi), meanLaplacian, varLaplacian);

				sharpness[i].at<float>(y, x) = (float) varLaplacian[0]; // accessors reverse the order for some reason
			}
		}
	}

	// TODO better sharpness


	bool showSharpness = !false;
	if (showSharpness){
		int nLoops = 1000;

		std::vector<int> ids(images.size());
		std::vector<float> sharpnesses_xy(images.size());
		
		int x = 8, y = 5; // tuiles proches
		//int x = 9, y = 6; // arbres proches
		//int x = 12, y = 8;// hornisgrinde HD

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

		int u = 4;
		cv::Mat native = Mat(Size(SQUARE_SIZE, SQUARE_SIZE), CV_32F);
		cv::Mat _upscaled = Mat(Size(SQUARE_SIZE* u, SQUARE_SIZE* u), CV_32F);
		for (int _i = 0; _i < nLoops*images.size(); _i++)
		{
			int i = ids[_i%images.size()];
			
			std::cout << sharpnesses_xy[i] << std::endl;

			Point2f center{ (float) (SQUARE_SIZE * x + SQUARE_SIZE / 2), (float) (SQUARE_SIZE * y + SQUARE_SIZE / 2) };
			center += Point2f((float)(PADDING_SIZE/2),(float)(PADDING_SIZE/2));
			center += globalMovement[i];
			cv::getRectSubPix(images[i], Size((float)SQUARE_SIZE, (float)SQUARE_SIZE), center, native);

			cv::resize(native, _upscaled, _upscaled.size());
			cv::imshow("Best to worst", _upscaled);
			int keyboard = waitKey(50);

			if (_i%images.size() == images.size()-1) int _keyboard = waitKey(1000);
		}
	}

	// TODO nearest neighbor, as per "sharp stack"


	Size SR_size(LR_w * UPSCALE_FACTOR + PADDING_SIZE * 2, LR_h * UPSCALE_FACTOR + PADDING_SIZE * 2);
	Mat stack = cv::Mat::zeros(SR_size, CV_32F);

	// preallocated. The to-be-sorted images ids in the sequence.
	std::vector<int> ids(images.size());

	// preallocated, to simplify sorting
	std::vector<float> sharpnesses_xy(images.size());

	// preallocated, to avoid reallocs
	Mat extractedSquare(Size(SQUARE_SIZE, SQUARE_SIZE), CV_32F);
	Mat upscaledSquare(Size(SQUARE_SIZE * UPSCALE_FACTOR, SQUARE_SIZE * UPSCALE_FACTOR), CV_32F);


	int nStackedSquares = (int)((float)images.size() * STACKED_PERCENTILE / 100.f);

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


			Rect stackROI(PADDING_SIZE + x * SQUARE_SIZE * UPSCALE_FACTOR, PADDING_SIZE + y * SQUARE_SIZE * UPSCALE_FACTOR, SQUARE_SIZE * UPSCALE_FACTOR, SQUARE_SIZE * UPSCALE_FACTOR);

			
			for (int i = 0; i < nStackedSquares; i++)
			{
				int _id = ids[i];

				
				Point2f center{ (float) (x * SQUARE_SIZE + SQUARE_SIZE / 2), (float) (y * SQUARE_SIZE + SQUARE_SIZE / 2) };
				center += globalMovement[_id];
				cv::getRectSubPix(images[_id], Size(SQUARE_SIZE, SQUARE_SIZE), center, extractedSquare);

				cv::resize(extractedSquare, upscaledSquare, upscaledSquare.size(), 0, 0, cv::INTER_LINEAR);

				cv::add(upscaledSquare, stack(stackROI), stack(stackROI));
			}
		}
	}

	stack *= 1.f / (float) nStackedSquares;
	Mat im = images[0];

#ifdef _DEBUG
	stack *= 255.f; // images were put in [0,1] for visualization
#endif

	stack.convertTo(stack, CV_8U);

	cv::imwrite("stack5.bmp", stack);

	cv::imshow("Stack", stack);

	int keyboard = waitKey(0);

	return stack;
}


cv::Mat Optimizer::RecursiveMatching(std::vector<cv::Mat> images, std::vector<vector<Point2f>> trackedPoints)
{
	const int UPSCALE_FACTOR = 2;
	const int PADDING_SIZE = 10; // In pixels 


	// between 0 and 100
	const float STACKED_PERCENTILE = 5.f;


	const int SQUARE_SIZE = 30; // in px. Must be even, to simplify cv::getRectSubPix's center

	int LR_w = images[0].cols;
	int LR_h = images[0].rows;


	vector<Point2f> barycentres(trackedPoints[0].size());
	vector<Point2f> globalMovement(images.size());

	// Compute the barycenter of each tracked point over the image sequence
	{
		for (int j = 0; j < trackedPoints[0].size(); j++)
		{
			barycentres[j] = Point2f(.0f, .0f);
		}
		for (int j = 0; j < images.size(); j++)
		{
			globalMovement[j] = Point2f(.0f, .0f);
		}
		for (int i = 0; i < images.size(); i++)
		{
			if (trackedPoints[i].size() != trackedPoints[0].size())
			{
				__debugbreak();
			}
			for (int j = 0; j < trackedPoints[i].size(); j++)
			{
				barycentres[j] += trackedPoints[i][j];
				globalMovement[i] += trackedPoints[i][j];
			}
			globalMovement[i] /= (float)trackedPoints[i].size();
		}
		Point2f stackCenter(.0f, .0f);
		for (int j = 0; j < trackedPoints[0].size(); j++)
		{
			barycentres[j] /= (float)images.size();
			stackCenter += barycentres[j];
		}
		stackCenter /= (float) trackedPoints[0].size();
		for (int j = 0; j < images.size(); j++)
		{
			globalMovement[j] -= stackCenter;
		}
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
				Rect _roi(x * SQUARE_SIZE - globalMovement[i].x, y * SQUARE_SIZE - globalMovement[i].y, SQUARE_SIZE, SQUARE_SIZE);

				cv::meanStdDev(images[i](_roi), meanLaplacian, varLaplacian);

				sharpness[i].at<float>(y, x) = (float) varLaplacian[0]; // accessors reverse the order for some reason
			}
		}
	}

	// TODO better sharpness


	// TODO nearest neighbor, as per "sharp stack"


	Size SR_size(LR_w * UPSCALE_FACTOR + PADDING_SIZE * 2, LR_h * UPSCALE_FACTOR + PADDING_SIZE * 2);
	Mat stack = cv::Mat::zeros(SR_size, CV_32F);

	// preallocated. The to-be-sorted images ids in the sequence.
	std::vector<int> ids(images.size());

	// preallocated, to simplify sorting
	std::vector<float> sharpnesses_xy(images.size());

	
	// preallocated, to avoid reallocs
	Mat extractedSquare(Size(SQUARE_SIZE, SQUARE_SIZE), CV_32F);
	Mat upscaledSquare(Size(SQUARE_SIZE * UPSCALE_FACTOR, SQUARE_SIZE * UPSCALE_FACTOR), CV_32F);


	int nStackedSquares = (int)((float)images.size() * STACKED_PERCENTILE / 100.f);

	Mat debugger = Mat::zeros(Size(LR_w/SQUARE_SIZE, LR_h/SQUARE_SIZE), CV_8U); // to monitor how many fragments were matched to the sharpest

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

			Rect stackROI(PADDING_SIZE + x * SQUARE_SIZE * UPSCALE_FACTOR, PADDING_SIZE + y * SQUARE_SIZE * UPSCALE_FACTOR, SQUARE_SIZE * UPSCALE_FACTOR, SQUARE_SIZE * UPSCALE_FACTOR);


			// Put the sharpest square in the stack
			int _id0 = ids[0];
			Point2f center0{ (float)(x * SQUARE_SIZE + SQUARE_SIZE / 2), (float)(y * SQUARE_SIZE + SQUARE_SIZE / 2) };
			center0 += globalMovement[_id0];
			cv::getRectSubPix(images[_id0], Size(SQUARE_SIZE, SQUARE_SIZE), center0, extractedSquare);
			cv::resize(extractedSquare, upscaledSquare, upscaledSquare.size(), 0, 0, cv::INTER_LINEAR);
			cv::add(upscaledSquare, stack(stackROI), stack(stackROI));
			

			// Find the features in the sharpest fragment
			vector<Point2f> referencePoints;
			const int MAX_POINTS_TRACKED = 10;
			const double MIN_DISTANCE = (double) (SQUARE_SIZE / 4);
			const double QUALITY_LEVEL = .05; // default .3
			const int MARGIN_SIZE = 3; // The size of the expected maximum error in barycenter computations.
			int ROI0_x_cropped = x * SQUARE_SIZE + (int)globalMovement[_id0].x + MARGIN_SIZE; // top left
			int ROI0_y_cropped = y * SQUARE_SIZE + (int)globalMovement[_id0].y + MARGIN_SIZE; // top left
			int ROI0_w_cropped = SQUARE_SIZE - 2*MARGIN_SIZE;
			int ROI0_h_cropped = SQUARE_SIZE - 2*MARGIN_SIZE;
			Rect ROI0_cropped(ROI0_x_cropped, ROI0_y_cropped, ROI0_w_cropped, ROI0_h_cropped); // A cropped ROI is used (- MARGIN_SIZE pixels at both ends) to avoid tracking pixels that could disappear 
			goodFeaturesToTrack(images[_id0](ROI0_cropped), referencePoints, MAX_POINTS_TRACKED, QUALITY_LEVEL, MIN_DISTANCE, Mat(), 7, false, 0.04);


			// No point stacking because we wont get subpixel accuracy, so we keep things as they are and go on to stack the next square sequence.
			// At these coordinates, the stack will only contain the sharpest fragment that was put in above.
			if (referencePoints.size() == 0)
			{
				continue;
			} 
			

			int nEffectivelyStackedSquares = 1; // To normalize the stack fragment.

			for (int i = 0; i < referencePoints.size(); i++) referencePoints[i] += Point2f((float)MARGIN_SIZE, (float)MARGIN_SIZE);

			int ROI0_x = ROI0_x_cropped - MARGIN_SIZE; // top left
			int ROI0_y = ROI0_y_cropped - MARGIN_SIZE; // top left
			Rect ROI0(ROI0_x, ROI0_y, SQUARE_SIZE, SQUARE_SIZE);

			// match all the other sharpest squares to the original one.
			vector<Point2f> potentialPoints(referencePoints.size());
			vector<Point2f> pointsDetectedInCurrentSquare(referencePoints.size());
			vector<uchar> status;
			vector<float> err;
			TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);
			for (int i = 1; i < nStackedSquares; i++)
			{
				int _id = ids[i];


				

				int ROIi_x = x * SQUARE_SIZE + (int)globalMovement[_id].x; // top left
				int ROIi_y = y * SQUARE_SIZE + (int)globalMovement[_id].y; // top left
				int ROIi_w = SQUARE_SIZE;
				int ROIi_h = SQUARE_SIZE;
				Rect ROIi(ROIi_x , ROIi_y, ROIi_w, ROIi_h);

				// calculate optical flow
				pointsDetectedInCurrentSquare.resize(0);
				status.resize(0);
				err.resize(0);
				//TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);
				calcOpticalFlowPyrLK(images[_id0](ROI0), images[_id](ROIi), referencePoints, potentialPoints, status, err, Size(7, 7), 2, criteria, 0, 1.0E-3);

				// TODO reuse the zero-th pyramid


				Point2f localMovement(.0f,.0f);
				int nPts = 0;
				for (uint j = 0; j < potentialPoints.size(); j++)
				{
					//TODO exclude those with too high a returned error 
					//TODO RANSAC with pointsDetectedInCurrentSquare
					if (status[j] == 1) {
						pointsDetectedInCurrentSquare.push_back(potentialPoints[j]);
						localMovement += (potentialPoints[j] - referencePoints[j]);
						nPts++;
					}
				}

				if (nPts == 0) continue;

				nEffectivelyStackedSquares++;

				localMovement /= (float) nPts;


				Point2f center{ (float) SQUARE_SIZE / 2.f, (float) SQUARE_SIZE / 2.f };
				Point2f intROI_i_tl = Point2f((float)ROIi_x, (float)ROIi_y);
				center += Point2f((float)ROIi_x, (float)ROIi_y) + localMovement; 
				cv::getRectSubPix(images[_id], Size(SQUARE_SIZE, SQUARE_SIZE), center, extractedSquare, CV_32F);

				cv::resize(extractedSquare, upscaledSquare, upscaledSquare.size(), 0, 0, cv::INTER_LINEAR);

				cv::add(upscaledSquare, stack(stackROI), stack(stackROI));
			}

			debugger.at<char>(y, x) = nEffectivelyStackedSquares;
			stack(stackROI) *= 1.f / (float)nEffectivelyStackedSquares;
		}
	}

#ifdef _DEBUG
	//stack *= 255.f; // images were put in [0,1] for visualization
#endif

	debugger.convertTo(debugger, CV_32F);
	debugger *= 1.f / (float) nStackedSquares;

	stack *= 1.f/255.f;
	Rect deconvROI = cv::selectROI(stack);
	Mat deconv = LucyRichardson(stack(deconvROI));

	stack *= 255.f;
	stack.convertTo(stack, CV_8U);

	cv::imwrite("stack5.bmp", stack);

	cv::imshow("Stack", stack);

	int keyboard = waitKey(0);

	return stack;
}


cv::Mat Optimizer::LucyRichardson(cv::Mat img)
{
	float sigmaG = 3.f; //sigma of point spread function (PSF)
	const int N_ITER = 10;
	const float EPSILON = .000001f;


	// Window size of PSF
	int winSize = 10 * sigmaG + 1 ;

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
	for(int j = 0; j < N_ITER; j++) 
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
		GaussianBlur( Y, reBlurred, Size(winSize,winSize), sigmaG, sigmaG );//applying Gaussian filter 
		reBlurred.setTo(EPSILON , reBlurred <= 0); 

		// 2)
		divide(wI, reBlurred, imR);
		imR = imR + EPSILON;

		// 3)
		GaussianBlur( imR, imR, Size(winSize,winSize), sigmaG, sigmaG );//applying Gaussian filter 

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
