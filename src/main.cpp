#include <iostream>

//#include <ic4/ic4.h>

#include "Optimizer.h"


int main()
{


	Optimizer optimizer;


	// JARDIN_NO_VIBRATION, JARDIN, HANGAR_1, HANGAR_2, HANGAR_ZOOM, ZI_FLOUE_1, ZI_FLOUE_2, ZI_3, PORT_DU_RHIN_1, CATHEDRALE_1,
	// PORT_DU_RHIN_2, MAISON_1, MAISON_2, MAISON_3, GRAVIERE, ZI_4, USINE_1, USINE_2, MAISON_4
	Optimizer::IMAGE_SEQ imSeq = Optimizer::IMAGE_SEQ::HANGAR_1;

	std::vector<cv::Mat> dstBayerImgs;
	std::vector<cv::Mat> dstRGBImgs;

	optimizer.LoadAndPreProcess(imSeq, dstBayerImgs, dstRGBImgs);

	auto stack = optimizer.FullFrameSequentialMatcher(dstBayerImgs, dstRGBImgs);


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
