#ifndef AUFGABE_2_H_INCLUDED
#define AUFGABE_2_H_INCLUDED

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

void aquireRandomNegatives(Mat img_arg, Mat &labels_arg, Mat &data_arg, int n);
void aquireSimplePositive(Mat img_arg, Mat &labels_arg, Mat &data_arg);
void aquireTestTrainingData(Mat &labels_arg, Mat &data_arg);

#endif