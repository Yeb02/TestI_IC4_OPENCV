#include <iostream>

//#include <ic4/ic4.h>

#include "Optimizer.h"


int main()
{


	Optimizer optimizer;

	Optimizer::IMAGE_SEQ imSeq = Optimizer::IMAGE_SEQ::TUILES_PROCHES; // TUILES_PROCHES, ARBRES_PROCHES, HORNISGRINDE_HD
	auto imVec = optimizer.LoadAndPreProcess(imSeq);

	auto stack = optimizer.FullFrameSequentialMatcher(imVec);


	//auto pointVecs = optimizer.OptFlow(imVec);

	
	//optimizer.ShowBarycentricStabilization(imVec, pointVecs);
	//optimizer.ShowSharpnessRanking(imVec, pointVecs);



	//auto stack = optimizer.RecursiveMatching(imVec, pointVecs);

	


	//optimizer.DelaunayUnwarping(imVec, pointVecs);

	//float factor = .25f;
	//cv::Mat small;
	//cv::resize(imVec[0], small, cv::Size(), factor, factor);
	//optimizer.WienerFilter(small);



	//auto dst = imVec[0];
	//dst = optimizer.EqualizeHistogram(dst);
	//dst = optimizer.UnsharpMasking(dst);
}
