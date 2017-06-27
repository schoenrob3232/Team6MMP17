#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include "aufgabe_1.h"
#include "aufgabe_2.h"
#include "aufgabe_3.h"
#include "hog.h"


using namespace std;
using namespace cv;


/*
Computes Intersection over Union in O(1) for regions with:
region.at<int>(0, 0) = x1
region.at<int>(0, 1) = y1
region.at<int>(0, 2) = x2
region.at<int>(0, 3) = y2
and x1 < x2 and y1 < y2
Example
			 | x1	   | y1		 | x2	   | y2
____________ | _______ | _______ | _______ | _______
region1		 | 24	   | 1		 | 170	   | 503
region2		 | 39	   | 44		 | 102	   | 432
*/
double fastComputeIoU(Mat region1, Mat region2) {
	int x1_loc1 = region1.at<int>(0, 0);
	int y1_loc1 = region1.at<int>(0, 1);
	int x2_loc1 = region1.at<int>(0, 2);
	int y2_loc1 = region1.at<int>(0, 3);

	int x1_loc2 = region2.at<int>(0, 0);
	int y1_loc2 = region2.at<int>(0, 1);
	int x2_loc2 = region2.at<int>(0, 2);
	int y2_loc2 = region2.at<int>(0, 3);

	int x_overlap, y_overlap, intersection, the_union;
	double intersectionOverUnion;

	//compute x-Axis overlap
	if (x1_loc1 > x2_loc2 || x2_loc1 < x1_loc2) {
		x_overlap = 0;
	} else if (x1_loc1 <= x1_loc2 && x2_loc2 <= x2_loc1) {
		x_overlap = x2_loc2 - x1_loc2;
	} else if (x1_loc2 <= x1_loc1 && x2_loc1 <= x2_loc2) {
		x_overlap = x2_loc1 - x1_loc1;
	} else if (x1_loc1 <= x1_loc2 && x2_loc1 <= x2_loc2) {
		x_overlap = x2_loc1 - x1_loc2;
	} else if (x1_loc2 <= x1_loc1 && x2_loc2 <= x2_loc1) {
		x_overlap = x2_loc2 - x1_loc1;
	}

	//compute Y-Axis overlap
	if (y1_loc1 > y2_loc2 || y2_loc1 < y1_loc2) {
		y_overlap = 0;
	} else if (y1_loc1 <= y1_loc2 && y2_loc2 <= y2_loc1) {
		y_overlap = y2_loc2 - y1_loc2;
	} else if (y1_loc2 <= y1_loc1 && y2_loc1 <= y2_loc2) {
		y_overlap = y2_loc1 - y1_loc1;
	} else if (y1_loc1 <= y1_loc2 && y2_loc1 <= y2_loc2) {
		y_overlap = y2_loc1 - y1_loc2;
	} else if (y1_loc2 <= y1_loc1 && y2_loc2 <= y2_loc1) {
		y_overlap = y2_loc2 - y1_loc1;
	}

	intersection = x_overlap * y_overlap;
	the_union = (x2_loc1 - x1_loc1) * (y2_loc1 - y1_loc1) + (x2_loc2 - x1_loc2) * (y2_loc2 - y1_loc2) - intersection;
	intersectionOverUnion = ((double)intersection) / ((double)the_union);
	return intersectionOverUnion;
}