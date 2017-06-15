#pragma once
#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat getGroundTruth(string filename);
bool compareToGroundTruth(Mat groundTruth, Mat potentialLocation);
