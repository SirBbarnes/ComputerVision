#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2//core.hpp"
#include "opencv2/opencv.hpp"

#include <time.h>
#include <iostream>
#include <stdio.h>

using namespace cv;
using std::cout;


Mat kernel, dst;
Point anchor;
double delta;
int depth;
int kernel_size;
int scale = 1;
Mat newImage;


void boxFilterOrig(const Mat &Image, const char* String)
{
	anchor = Point(-1, -1);
	delta = 0;
	depth = -1;

	kernel_size = 31;
	for(int i = 1; i < kernel_size; i+= 2) 
	{
		kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);

		filter2D(Image, dst, depth, kernel, anchor, delta, BORDER_DEFAULT);
	}
	imshow(String, dst);

}

void boxFilterOpen(const Mat &Image, const char* String)
{
	kernel_size = 31;
	for (int i = 1; i < kernel_size; i += 2)
	{
		blur(Image, dst, Size(i, i), Point(-1, -1));
	}
	imshow(String, dst);

}





void sobFilterXOrig(const Mat& Image, const char* String)
{
	newImage = Image.clone();

	int X = 0, Y = 0;
	int sobelX[3][3] =
	{
		{1, 0, -1},
		{2, 0, -2},
		{1, 0, -1}
	};

	for (int i = 1; i < Image.rows - 1; i++)
	{
		for (int j = 1; j < Image.cols - 1; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				for (int l = 0; l < 3; l++)
				{
					X += sobelX[k][l] * Image.at<uchar>(i + k - 1, j + l - 1);
				}
			}
			int sum = abs(X) + abs(Y);

			newImage.at<uchar>(i, j) = sum;
			X = 0;
			Y = 0;
		}
	}
	imshow(String, newImage);
}

void sobFilterYOrig(const Mat &Image, const char* String)
{
	newImage = Image.clone();

	int X = 0, Y = 0;

	int sobelY[3][3] =
	{
		{1, 2, 1},
		{0, 0, 0},
		{-1, -2, -1}
	};

	for (int i = 1; i < Image.rows - 1; i++)
	{
		for (int j = 1; j < Image.cols - 1; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				for (int l = 0; l < 3; l++)
				{
					Y += sobelY[k][l] * Image.at<uchar>(i + k - 1, j + l - 1);
				}
			}
			int sum = abs(X) + abs(Y);

			newImage.at<uchar>(i, j) = sum;
			X = 0; 
			Y = 0;
		}
	}
	imshow(String, newImage);
}

void sobFilterXYOrig(const Mat &Image, const char* String)
{
	newImage = Image.clone();

	int X = 0, Y = 0;
	int sobelX[3][3] =
	{
		{1, 0, -1},
		{2, 0, -2},
		{1, 0, -1}
	};

	int sobelY[3][3] =
	{
		{1, 2, 1},
		{0, 0, 0},
		{-1, -2, -1}
	};

	for (int i = 1; i < Image.rows - 1; i++)
	{
		for (int j = 1; j < Image.cols - 1; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				for (int l = 0; l < 3; l++)
				{
					X += sobelX[k][l] * Image.at<uchar>(i + k - 1, j + l - 1);
					Y += sobelY[k][l] * Image.at<uchar>(i + k - 1, j + l - 1);
				}
			}
			int sum = abs(X) + abs(Y);

			newImage.at<uchar>(i, j) = sum;
			X = 0; 
			Y = 0;
		}
	}
	imshow(String, newImage);
}

void sobFilterXYOpen(const Mat &Image, const char* String)
{
	Mat grad;
	GaussianBlur(Image, Image, Size(3, 3), 0, 0, BORDER_DEFAULT);

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	Sobel(Image, grad_x, depth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	Sobel(Image, grad_y, depth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	imshow(String, grad);

}





void gaussFilterOpen(const Mat &Image, const char* String)
{
	kernel_size = 31;
	for (int i = 1; i < kernel_size; i += 2)
	{
		GaussianBlur(Image, dst, Size(i, i), 0,0);
		
	}
	imshow(String, dst);
}





int main(int argc, char** argv)
{
	Mat dogImage, grayDogImage;
	Mat biImage, grayBiImage;
	Mat southImage, graySouthImage;
	Mat edgeImage, grayEdgeImage;

	dogImage = imread("dog.bmp");
	cvtColor(dogImage, grayDogImage, COLOR_BGR2GRAY);


	biImage = imread("bicycle.bmp");
	cvtColor(biImage, grayBiImage, COLOR_BGR2GRAY);


	southImage = imread("south_L-150x150.png");
	cvtColor(southImage, graySouthImage, COLOR_BGR2GRAY);

	edgeImage = imread("edge_L-150x150.png");
	cvtColor(edgeImage, grayEdgeImage, COLOR_BGR2GRAY);


	const char* dogBoxOrig = "Box Filter W/O OpenCV";
	const char* dogBoxOP = "Box Filter using OpenCV";
	const char* dogSobXOrig = "Sobel Filter over X-Axis W/O OpenCV";
	const char* dogSobYOrig = "Sobel Filter over Y-Axis W/O OpenCV";
	const char* dogSobXYOrig = "Sobel Filter over X/Y - Axis W/O OpenCV";
	const char* dogSobXYOP = "Sobel Filter over X/Y - Axis using OpenCV";
	const char* dogGaussOP = "Gaussian Filter using OpenCV";


	const char* biBoxOrig = "Box Filter W/O OpenCV";
	const char* biBoxOP = "Box Filter using OpenCV";
	const char* biSobXOrig = "Sobel Filter over X-Axis W/O OpenCV";
	const char* biSobYOrig = "Sobel Filter over Y-Axis W/O OpenCV";
	const char* biSobXYOrig = "Sobel Filter over X/Y Axis W/O OpenCV";
	const char* biSobXYOP = "Sobel Filter over X/Y Axis using OpenCV";
	const char* biGaussOP = "Gaussian Filter using OpenCV";


	const char* southBoxOrig = "Box Filter W/O OpenCV";
	const char* southBoxOP = "Box Filter using OpenCV";
	const char* southSobXOrig = "Sobel Filter over X-Axis W/O OpenCV";
	const char* southSobYOrig = "Sobel Filter over Y-Axis W/O OpenCV";
	const char* southSobXYOrig = "Sobel Filter over X/Y - Axis W/O OpenCV";
	const char* southSobXYOP = "Sobel Filter over X/Y -Axis using OpenCV";
	const char* southGaussOP = "Gaussian Filter using OpenCV";


	const char* edgeBoxOrig = "Box Filter W/O OpenCV";
	const char* edgeBoxOP = "Box Filter using OpenCV";
	const char* edgeSobXOrig = "Sobel Filter over X-Axis W/O OpenCV";
	const char* edgeSobYOrig = "Sobel Filter over Y-Axis W/O OpenCV";
	const char* edgeSobXYOrig = "Sobel Filter over X/Y - Axis W/O OpenCV";
	const char* edgeSobXYOP = "Sobel Filter over X/Y - Axis using OpenCV";
	const char* edgeGaussOP = "Gaussian Filter using OpenCV";



	cout << "Showing Dog Image with each Filter\n";
	boxFilterOrig(dogImage, dogBoxOrig);
	boxFilterOpen(dogImage, dogBoxOP);
	sobFilterXOrig(grayDogImage, dogSobXOrig);
	sobFilterYOrig(grayDogImage, dogSobYOrig);
	sobFilterXYOrig(grayDogImage, dogSobXYOrig);
	sobFilterXYOpen(grayDogImage, dogSobXYOP);
	gaussFilterOpen(dogImage, dogGaussOP);
	waitKey();
	destroyAllWindows();


	
	cout << "\nShowing Bicycle Image with each Filter\n";
	boxFilterOrig(biImage, biBoxOrig);
	boxFilterOpen(biImage, biBoxOP);
	sobFilterXOrig(grayBiImage, biSobXOrig);
	sobFilterYOrig(grayBiImage, biSobYOrig);
	sobFilterXYOrig(grayBiImage, biSobXYOrig);
	sobFilterXYOpen(grayBiImage, biSobXYOP);
	gaussFilterOpen(biImage, biGaussOP);
	waitKey();
	destroyAllWindows();



	cout << "\nShowing South_L Image with each Filter\n";
	boxFilterOrig(southImage, southBoxOrig);
	boxFilterOpen(southImage, southBoxOP);
	sobFilterXOrig(graySouthImage, southSobXOrig);
	sobFilterYOrig(graySouthImage, southSobYOrig);
	sobFilterXYOrig(graySouthImage, southSobXYOrig);
	sobFilterXYOpen(graySouthImage, southSobXYOP);
	gaussFilterOpen(southImage, southGaussOP);
	waitKey();
	destroyAllWindows();



	cout << "\nShowing Edge Image with each Filter\n";
	boxFilterOrig(edgeImage, edgeBoxOrig);
	boxFilterOpen(edgeImage, edgeBoxOP);
	sobFilterXOrig(grayEdgeImage, edgeSobXOrig);
	sobFilterYOrig(grayEdgeImage, edgeSobYOrig);
	sobFilterXYOrig(grayEdgeImage, edgeSobXYOrig);
	sobFilterXYOpen(grayEdgeImage, edgeSobXYOP);
	gaussFilterOpen(edgeImage, edgeGaussOP);
	waitKey();
	destroyAllWindows();


	return 0;
}