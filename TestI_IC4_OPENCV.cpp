#include <iostream>

//#include <ic4/ic4.h>

#include "Optimizer.h"


int main()
{


	Optimizer optimizer;


	auto imVec = optimizer.LoadAndPreProcess();

	auto pointVecs = optimizer.OptFlow(imVec);

	auto stack = optimizer.WarpStack(imVec, pointVecs);


	//float factor = .25f;
	//cv::Mat small;
	//cv::resize(imVec[0], small, cv::Size(), factor, factor);
	//optimizer.WienerFilter(small);



	//auto dst = imVec[0];
	//dst = optimizer.EqualizeHistogram(dst);
	//dst = optimizer.UnsharpMasking(dst);
}
