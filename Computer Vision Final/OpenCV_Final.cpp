#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

RNG rng(12345);
int main(int argc, char** argv)
{
    Mat frame;
    VideoCapture cap;
    int deviceID = 0;
    int apiID = CAP_ANY;
    cap.open(deviceID, apiID);

    if (!cap.isOpened())
    {
        cerr << "ERROR! Unable to open WebCam\n";
        return -1;
    }

    for (;;)
    {
        int largest_area = 0;
        int largest_contour_index = 0;
        Rect bounding_rect;

        //cap.read(frame);
        cap >> frame;
        //Mat dst(frame.rows, frame.cols, CV_8UC1, Scalar::all(0));

        Mat gray;
        cvtColor(frame, gray, COLOR_RGB2GRAY);

        threshold(gray, gray, 125, 255, THRESH_BINARY);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(gray, contours, hierarchy,RETR_CCOMP, CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); i++)
        {
            Point2f center;
            float radius;

            double a = contourArea(contours[i]);
            double l = arcLength(contours[i], true);
            minEnclosingCircle(contours[i], center, radius);

            char buffer[64] = {0};

            if (a > largest_area)
            {
                largest_area = a;
                largest_contour_index = i;
                bounding_rect = boundingRect(contours[i]);

                printf(buffer, "Area: %.21f", largest_area);
                putText(frame, buffer, center, FONT_HERSHEY_SIMPLEX, .6, Scalar(118,185,0) ,1);
                printf(buffer, "Length: .21f", l);
                putText(frame, buffer, Point(center.x, center.y + 20), FONT_HERSHEY_SIMPLEX, .6, Scalar(118,185,0), 1);
            }

        }
        drawContours(frame, contours, largest_contour_index, Scalar(0,255,0), 2);

        imshow("", frame);
        //imshow("1", gray);

        if (frame.empty())
        {
            cerr << "ERROR! Blank frame grabbed\n";
            break;
        }

        if (waitKey(5) >= 0)
            break;
    }

    return 0;
}

