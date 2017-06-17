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
Mat scaleDownOneStep(Mat img);
void slidingWindow_geruest(Mat img_arg);
void dissolve(double ***hogCells, vector<int> dims);
double ***copyHOGCells(int y, int x, double ***hogCells, vector<int> oldDims, vector<int> &newDims);
void slidingWindowGetData(Mat img_arg, Mat &labels_arg, Mat &data_arg, Mat groundTruths_arg);

#endif
