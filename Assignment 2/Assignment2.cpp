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
int scale = 1;
Mat newImage;

Mat dogImage, grayDogImage;
Mat biImage, grayBiImage;
Mat southImage, graySouthImage;
Mat edgeImage, grayEdgeImage;


void boxFilterOrig(const Mat &Image, const char* String, int kernel_size)
{
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
	filter2D(Image, dst, depth, kernel, anchor, delta, BORDER_DEFAULT);
	
	imshow(String, dst);
}

void boxFilterOpen(const Mat &Image, const char* String, int kernel_size)
{

	boxFilter(Image, dst, depth, Size(kernel_size, kernel_size), Point(-1, 1), true, BORDER_DEFAULT);

	imshow(String, dst);

}





void sobFilterXOrig(const Mat& Image, const char* String, int kernel_size)
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
			for (int k = 0; k < kernel_size; k++)
			{
				for (int l = 0; l < kernel_size; l++)
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

void sobFilterYOrig(const Mat &Image, const char* String, int kernel_size)
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
			for (int k = 0; k < kernel_size; k++)
			{
				for (int l = 0; l < kernel_size; l++)
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

void sobFilterXYOrig(const Mat &Image, const char* String, int kernel_size)
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
			for (int k = 0; k < kernel_size; k++)
			{
				for (int l = 0; l < kernel_size; l++)
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

void sobFilterXYOpen(const Mat &Image, const char* String, int kernel_size)
{
	Mat grad;
	GaussianBlur(Image, Image, Size(kernel_size, kernel_size), 0, 0, BORDER_DEFAULT);

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	Sobel(Image, grad_x, depth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	Sobel(Image, grad_y, depth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	imshow(String, grad);

}





void gaussFilterOpen(const Mat &Image, const char* String, int kernel_size)
{

	GaussianBlur(Image, dst, Size(kernel_size, kernel_size), 0, 0);
	
	imshow(String, dst);
}




void printDogs(int kernel_size)
{
	dogImage = imread("dog.bmp");
	cvtColor(dogImage, grayDogImage, COLOR_BGR2GRAY);


	const char* dogBoxOrig = "Box Filter W/O OpenCV";
	const char* dogBoxOP = "Box Filter using OpenCV";
	const char* dogSobXOrig = "Sobel Filter over X-Axis W/O OpenCV";
	const char* dogSobYOrig = "Sobel Filter over Y-Axis W/O OpenCV";
	const char* dogSobXYOrig = "Sobel Filter over X/Y - Axis W/O OpenCV";
	const char* dogSobXYOP = "Sobel Filter over X/Y - Axis using OpenCV";
	const char* dogGaussOP = "Gaussian Filter using OpenCV";



	boxFilterOrig(dogImage, dogBoxOrig, kernel_size);
	boxFilterOpen(dogImage, dogBoxOP, kernel_size);
	sobFilterXOrig(grayDogImage, dogSobXOrig, kernel_size);
	sobFilterYOrig(grayDogImage, dogSobYOrig, kernel_size);
	sobFilterXYOrig(grayDogImage, dogSobXYOrig, kernel_size);
	sobFilterXYOpen(grayDogImage, dogSobXYOP, kernel_size);
	gaussFilterOpen(dogImage, dogGaussOP, kernel_size);
	waitKey();
	destroyAllWindows();
}



void printBicycle(int kernel_size)
{
	biImage = imread("bicycle.bmp");
	cvtColor(biImage, grayBiImage, COLOR_BGR2GRAY);


	const char* biBoxOrig = "Box Filter W/O OpenCV";
	const char* biBoxOP = "Box Filter using OpenCV";
	const char* biSobXOrig = "Sobel Filter over X-Axis W/O OpenCV";
	const char* biSobYOrig = "Sobel Filter over Y-Axis W/O OpenCV";
	const char* biSobXYOrig = "Sobel Filter over X/Y Axis W/O OpenCV";
	const char* biSobXYOP = "Sobel Filter over X/Y Axis using OpenCV";
	const char* biGaussOP = "Gaussian Filter using OpenCV";


	boxFilterOrig(biImage, biBoxOrig, kernel_size);
	boxFilterOpen(biImage, biBoxOP, kernel_size);
	sobFilterXOrig(grayBiImage, biSobXOrig, kernel_size);
	sobFilterYOrig(grayBiImage, biSobYOrig, kernel_size);
	sobFilterXYOrig(grayBiImage, biSobXYOrig, kernel_size);
	sobFilterXYOpen(grayBiImage, biSobXYOP, kernel_size);
	gaussFilterOpen(biImage, biGaussOP, kernel_size);
	waitKey();
	destroyAllWindows();
}



void printSouth_L(int kernel_size)
{
	southImage = imread("south_L-150x150.png");
	cvtColor(southImage, graySouthImage, COLOR_BGR2GRAY);


	const char* southBoxOrig = "Box Filter W/O OpenCV";
	const char* southBoxOP = "Box Filter using OpenCV";
	const char* southSobXOrig = "Sobel Filter over X-Axis W/O OpenCV";
	const char* southSobYOrig = "Sobel Filter over Y-Axis W/O OpenCV";
	const char* southSobXYOrig = "Sobel Filter over X/Y - Axis W/O OpenCV";
	const char* southSobXYOP = "Sobel Filter over X/Y -Axis using OpenCV";
	const char* southGaussOP = "Gaussian Filter using OpenCV";


	boxFilterOrig(southImage, southBoxOrig, kernel_size);
	boxFilterOpen(southImage, southBoxOP, kernel_size);
	sobFilterXOrig(graySouthImage, southSobXOrig, kernel_size);
	sobFilterYOrig(graySouthImage, southSobYOrig, kernel_size);
	sobFilterXYOrig(graySouthImage, southSobXYOrig, kernel_size);
	sobFilterXYOpen(graySouthImage, southSobXYOP, kernel_size);
	gaussFilterOpen(southImage, southGaussOP, kernel_size);
	waitKey();
	destroyAllWindows();
}



void printEdge_L(int kernel_size)
{
	edgeImage = imread("edge_L-150x150.png");
	cvtColor(edgeImage, grayEdgeImage, COLOR_BGR2GRAY);


	const char* edgeBoxOrig = "Box Filter W/O OpenCV";
	const char* edgeBoxOP = "Box Filter using OpenCV";
	const char* edgeSobXOrig = "Sobel Filter over X-Axis W/O OpenCV";
	const char* edgeSobYOrig = "Sobel Filter over Y-Axis W/O OpenCV";
	const char* edgeSobXYOrig = "Sobel Filter over X/Y - Axis W/O OpenCV";
	const char* edgeSobXYOP = "Sobel Filter over X/Y - Axis using OpenCV";
	const char* edgeGaussOP = "Gaussian Filter using OpenCV";


	boxFilterOrig(edgeImage, edgeBoxOrig, kernel_size);
	boxFilterOpen(edgeImage, edgeBoxOP, kernel_size);
	sobFilterXOrig(grayEdgeImage, edgeSobXOrig, kernel_size);
	sobFilterYOrig(grayEdgeImage, edgeSobYOrig, kernel_size);
	sobFilterXYOrig(grayEdgeImage, edgeSobXYOrig, kernel_size);
	sobFilterXYOpen(grayEdgeImage, edgeSobXYOP, kernel_size);
	gaussFilterOpen(edgeImage, edgeGaussOP, kernel_size);
	waitKey();
	destroyAllWindows();
}

int main(int argc, char** argv)
{
	cout << "Showing Dog Image with each Filter and a kernel_size of 3x3\n";
	printDogs(3);

	cout << "Showing Dog Image with each Filter and a kernel_size of 7x7\n";
	printDogs(7);




	cout << "\nShowing Bicycle Image with each Filter and a kernel_size of 3x3\n";
	printBicycle(3);

	cout << "Showing Bicycle Image with each Filter and a kernel_size of 7x7\n";
	printBicycle(7);



	cout << "\nShowing South_L Image with each Filter and a kernel_size of 3x3\n";
	printSouth_L(3);

	cout << "Showing South_L Image with each Filter and a kernel_size of 7x7\n";
	printSouth_L(7);


	

	cout << "\nShowing Edge_L Image with each Filter and a kernel_size of 3x3\n";
	printEdge_L(3);

	cout << "Showing Edge_L Image with each Filter and a kernel_size of 7x7";
	printEdge_L(7);

	return 0;
}