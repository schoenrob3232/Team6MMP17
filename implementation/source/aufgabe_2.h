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
void training_SVM(Mat& data_arg, Mat& labels_arg, const char* name);
Mat showCertainDetections(Mat img_arg, const char* svm_name, double threshold);
void aquireHardestNegative(Mat img_arg, const char* svm_name, Mat &labels_arg, Mat &data_arg, Mat groundTruths_arg);
void aquireMultipleHardNegatives(const char* svm_name, Mat &labels_arg, Mat &data_arg);
void aquireTrainingPositives(Mat &labels_arg, Mat &data_arg);

#endif