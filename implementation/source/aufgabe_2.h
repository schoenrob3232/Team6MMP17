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

void acquireRandomNegatives(Mat img_arg, Mat &labels_arg, Mat &data_arg, int n);
void acquireSimplePositive(Mat img_arg, Mat &labels_arg, Mat &data_arg);
void acquireTestTrainingData(Mat &labels_arg, Mat &data_arg);
void training_SVM(Mat& data_arg, Mat& labels_arg, const char* name);
Mat showCertainDetections(Mat img_arg, const char* svm_name, double threshold);
void acquireHardestNegative(Mat img_arg, const char* svm_name, Mat &labels_arg, Mat &data_arg, Mat groundTruths_arg);
void acquireMultipleHardNegatives(const char* svm_name, Mat &labels_arg, Mat &data_arg);
void acquireTrainingPositives(Mat &labels_arg, Mat &data_arg);
void acquireUltraHardNegatives(const char* svm_name, Mat &labels_arg, Mat &data_arg);

#endif
