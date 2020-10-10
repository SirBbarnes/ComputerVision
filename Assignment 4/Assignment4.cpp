#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	//read image
	Mat image1 = imread("blocks_L-150x150.png");
	Mat image2 = imread("cat.jpg");

	Mat mask1 = Mat::zeros(image1.size(), CV_8U);
	Mat mask2 = Mat::zeros(image2.size(), CV_8U);

	Mat roi1(mask1, cv::Rect(10, 10, 100, 100));
	Mat roi2(mask2, cv::Rect(10, 10, 100, 100));
	roi1 = Scalar(255);
	roi2 = Scalar(255);

	imshow("Image 1", image1);
	imshow("Image 2", image2);
	waitKey();

	//convert image to grayscale
	cvtColor(image1, image1, COLOR_BGR2GRAY);
	cvtColor(image2, image2, COLOR_BGR2GRAY);


	imshow("Image 1", image1);
	imshow("Image 2", image2);
	waitKey();

	//create sift and vector keypoint
	Ptr<xfeatures2d::SiftFeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create();
	std::vector<cv::KeyPoint> keypoint1;
	std::vector<cv::KeyPoint> keypoint2;
	printf("Vector created");
	waitKey();

	//detect keypoints
	Mat descriptor1, descriptor2;
	detector->detectAndCompute(image1, NULL, keypoint1, roi1);
	detector->detectAndCompute(image2, NULL, keypoint2, roi2);
	printf("detected");
	waitKey();

	
	
	//draw keypoints
	//drawKeypoints(image1, keypoint1, descriptor1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//drawKeypoints(image2, keypoint2, descriptor2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("SIFT Image 1 KeyPoints", descriptor1);
	imshow("SIFT Image 2 KeyPoints", descriptor2);
	printf("keypoint");
	waitKey();

	//create brute force matcher
	BFMatcher bf = BFMatcher();
	
	//create vector matches for bf matcher
	vector<vector<DMatch>> matches;
	bf.knnMatch(descriptor1, descriptor2, matches, 2);
	
	vector<DMatch> gMatches;
	for (size_t i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < 0.7 * matches[i][1].distance)
		{
			gMatches.push_back(matches[i][0]);
		}
	}

	//draw matches
	Mat imMatches;
	drawMatches(descriptor1, keypoint1, descriptor2, keypoint2, gMatches, imMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DRAW_OVER_OUTIMG);
	imshow("Matches", imMatches);
	waitKey();
	return 0;
}