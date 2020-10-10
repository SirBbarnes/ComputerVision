#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2//core.hpp"
#include "opencv2/opencv.hpp";

#include <iostream>

using namespace cv;
using namespace std;

Mat MS01, MS02, MS03, MS04, MS05;
Mat OTS201, OTS202, OTS203;
Mat OTSUM01, OTSUM02, OTSUM03, OTSUM04;

Mat dst;


void OtsuMethod(const Mat& Image, const char* input)
{
    Mat src = Image.clone();
    imshow(input, src);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<Vec3b>(i, j) == Vec3b(255, 255, 255))
            {
                src.at<Vec3b>(i, j)[0] = 0;
                src.at<Vec3b>(i, j)[1] = 0;
                src.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }

    imshow(input, src);


    Mat kernel = (Mat_<float>(3, 3) <<
        1, 1, 1,
        1, -8, 1,
        1, 1, 1);

    Mat imgLaplacian;
    filter2D(src, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;

    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

    imshow(input, imgResult);
    waitKey();


    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    imshow(input, bw);
    waitKey();


    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);

    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    imshow(input, dist);
    waitKey(0);


    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);
    imshow(input, dist);
    waitKey();


    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat markers = Mat::zeros(dist.size(), CV_32F);
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
    }
    circle(markers, Point(5, 5), 3, Scalar(255), -1);
    imshow(input, markers * 10000);
    waitKey();

    /*
    watershed(imgResult, markers);
    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);

    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    Mat dst = Mat::zeros(markers.size(), CV_8U);

    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<Vec3b>(i, j) = colors[index - 1];
            }
        }
    }
    imshow(input, dst);
    waitKey();
    */
}


void OtsuMethodMulti(const Mat &Image, const char* input)
{
    Mat src = Image.clone();
    imshow(input, src);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<Vec3b>(i, j) == Vec3b(255, 255, 255))
            {
                src.at<Vec3b>(i, j)[0] = 0;
                src.at<Vec3b>(i, j)[1] = 0;
                src.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }

    imshow(input, src);


    Mat kernel = (Mat_<float>(3, 3) <<
        1, 1, 1,
        1, -8, 1,
        1, 1, 1);

    Mat imgLaplacian;
    filter2D(src, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;

    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

    imshow(input, imgResult);
    waitKey();


    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    imshow(input, bw);
    waitKey();


    threshold(bw, bw, 80, 255, THRESH_BINARY);
    imshow(input, bw);
    waitKey();

    threshold(bw, bw, 160, 255, THRESH_BINARY);
    imshow(input, bw);

    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);

    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    imshow(input, dist);
    waitKey(0);


    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);
    imshow(input, dist);
    waitKey();


    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat markers = Mat::zeros(dist.size(), CV_32F);
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
    }
    circle(markers, Point(5, 5), 3, Scalar(255), -1);
    imshow(input, markers * 10000);
    waitKey();

    /*
    watershed(imgResult, markers);
    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);

    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    Mat dst = Mat::zeros(markers.size(), CV_8U);

    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<Vec3b>(i, j) = colors[index - 1];
            }
        }
    }
    imshow(input, dst);
    waitKey();
    */
}


void MeanShiftMeth(const Mat &Image, const char* input)
{
    Mat src = Image.clone();
    pyrMeanShiftFiltering(src, src, 20, 45, 3);

	imshow(input, src);
	waitKey();
}




void setMeanImages()
{
    destroyAllWindows();
	MS01 = imread("S00-150x150.png");
	MS02 = imread("S02-150x150.png");
	MS03 = imread("S03-150x150.png");
	MS04 = imread("set1Seq2_L-150x150.png");
	MS05 = imread("set2Seq1_L-150x150.png");

    MeanShiftMeth(MS01, "Image 1");
    MeanShiftMeth(MS02, "Image 2");
    MeanShiftMeth(MS03, "Image 3");
    MeanShiftMeth(MS04, "Image 4");
    MeanShiftMeth(MS05, "Image 5");
}

void setOTS2Images()
{
	OTS201 = imread("andreas_L-150x150.png");
	OTS202 = imread("edge_L-150x150.png");
	OTS203 = imread("south_L-150x150.png");

	OtsuMethod(OTS201, "Image 1");
    OtsuMethod(OTS202, "Image 2");
    OtsuMethod(OTS203, "Image 3");
}

void setOTSUMImages()
{
    destroyAllWindows();
	OTSUM01 = imread("blocks_L-150x150.png");
	OTSUM02 = imread("S01-150x150.png");
	OTSUM03 = imread("S04-150x150.png");
	OTSUM04 = imread("S05-150x150.png");

	OtsuMethodMulti(OTSUM01, "Image 1");
	OtsuMethodMulti(OTSUM02, "Image 2");
	OtsuMethodMulti(OTSUM03, "Image 3");
	OtsuMethodMulti(OTSUM04, "Image 4");

}


int main(int argc, char** argv)
{
	setOTS2Images();
    setOTSUMImages();
    setMeanImages();

	return 0;
}