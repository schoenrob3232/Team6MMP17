#ifndef AUFGABE_1_H_INCLUDED
#define AUFGSBE_1_H_INCLUDED

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
bool compareToAllGroundTruths(Mat allGroundTruths, Mat potentialLocation);
Mat computeHOGBlock(int cell_pos_x, int cell_pos_y, int block_size, double *** hogCells, vector<int> dims);
Mat computeWindowDescriptor(double ***hogCells, vector<int> dims);

#endif
