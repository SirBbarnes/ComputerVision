#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2//core.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <stdio.h>


using namespace cv;
using std::cout;


int alpha_slider = 1, alpha_max = 3;
int beta_slider = 0, beta_max = 100;
Mat src, dst, frame;


void calcHisto(const Mat &Image, Mat &histogImage)
{
    int histSize = 255;
    float range[] = { 0,256 };
    const float* histRanges[] = { range };
    Mat hist;

    calcHist(&Image, 1, 0, Mat(), hist, 1, &histSize, histRanges, true, false);


    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
    {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
    }
    histogImage = histImage;
}


int main( int argc, char** argv )
{
    Mat histImage;
    //String imageName("tesla-cat.jpg"); 
    int frames = 0;
    VideoCapture src("barriers.avi");


    int frame_width = src.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = src.get(CAP_PROP_FRAME_HEIGHT);
    double fps = 25.0;
    String file = "video-output.avi";
    VideoWriter video(file, VideoWriter::fourcc('M','J','P','G'), fps, Size(frame_width, frame_height));
 
    namedWindow("Brightness & Contrast", 1);
    
    
    char TrackbarName1[50] = "Brightness";
    char TrackbarName2[50] = "Contrast";

    createTrackbar(TrackbarName1, "Brightness & Contrast", &beta_slider, beta_max);
    createTrackbar(TrackbarName2, "Brightness & Contrast", &alpha_slider, alpha_max);
    


    cout << "Press 'S' to save video\n";
    cout << "Press Escape to quit";

    while (1)
    {
        src >> frame;
        frames += 1;
        if (frames == src.get(CAP_PROP_FRAME_COUNT)) {
            frames = 0;
            src = VideoCapture("barriers.avi");
        }

        frame.convertTo(frame, -1, 1, getTrackbarPos(TrackbarName1, "Brightness & Contrast"));
        frame.convertTo(frame, -1, getTrackbarPos(TrackbarName2, "Brightness & Contrast"), 0);

        video.write(frame);
            
        imshow("Brightness & Contrast", frame);
        cvtColor(frame, frame, COLOR_BGR2GRAY);

        calcHisto(frame, histImage);
        imshow("Histogram", histImage);
        char c = (char)waitKey(25);

        if (c == 27)
        {
            destroyAllWindows();
            break;
        }
        else if (c == 's')
        {
            

            break;
        }
        
    }
    video.release();
    src.release();
    
    return 0;
}