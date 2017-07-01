#ifndef AUFGABE_3_H_INCLUDED
#define AUFGABE_3_H_INCLUDED

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

double fastComputeIoU(Mat region1, Mat region2);
int extractDetections(Mat img_arg, const char* svm_name, Mat &positions, Mat &det_scores);
void nonMaxSuppression(Mat &positions, Mat &det_scores, int N);
void sortByDetectionScore(Mat &positions, Mat &det_scores);
Mat padImgWithZeros(Mat img, int pad);
Mat drawResults(Mat img, Mat results, Mat groundTruths);
Mat padWithBorderPixels(Mat img, int pad);
double computeMissRate(Mat results, Mat groundTruths);
Mat cloneRowInt(Mat matrix, int row);
Mat cloneRowFloat(Mat matrix, int row);

#endif