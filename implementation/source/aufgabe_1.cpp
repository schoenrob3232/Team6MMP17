#include <iostream>
#include <fstream>
#include <string>

#include "aufgabe_1.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

/*
Extracts the ground truth of an annotation (filename) to integer Nx4-Mat (for N persons) with
dimensions / indices:
(person , coordinate)
Example
			| x1	| y1	| x2	| y2	
____________|_______|_______|_______|_______
person 1	| 24	| 1		| 170	| 503
person 2	| 39	| 44	| 102	| 432
*/
Mat getGroundTruth(string filename) {
	string line;
	ifstream file(filename);
	int personCount, x1, k, h;
	Mat persons;
	h = 0;
	while (getline(file, line)) {
		k = 0;
		if (line.find("Objects with ground truth") != -1) {
			for (int i = 0; i < line.size(); i++) {
				if (isdigit(line.at(i))) {
					personCount = line.at(i) - '0';
					for (int j = 1; isdigit(line.at(i + j)); j++) {
						personCount *= 10;
						personCount += line.at(i + j) - '0';
					}
					persons = Mat::zeros(personCount, 4, CV_32S);
					break;
				}
			}
		}
		if (line.find("Bounding box for object") != -1) {
			for (int i = line.find(":"); i < line.size(); i++) {
				if (isdigit(line.at(i))) {
					x1 = line.at(i) - '0';
					int j;
					for (j = 1; isdigit(line.at(i + j)); j++) {
						x1 *= 10;
						x1 += line.at(i + j) - '0';
					}
					i += j;
					persons.at<int>(h, k) = x1;
					k++;
				}
			}
			h++;
		}
	}
	file.close();
	return persons;
}

/*
Gets two 1x4 Mats and checks if their IoU is over 0.500 (50%), 
which is enough to treat the detection as correct.

*/
bool compareToGroundTruth(Mat groundTruth, Mat potentialLocation) {
	int x1_truth, y1_truth, x2_truth, y2_truth, x1_loc, y1_loc, x2_loc, y2_loc, intersection, the_union;
	x1_truth = groundTruth.at<int>(0, 0);
	y1_truth = groundTruth.at<int>(0, 1);
	x2_truth = groundTruth.at<int>(0, 2);
	y2_truth = groundTruth.at<int>(0, 3);
	x1_loc = potentialLocation.at<int>(0, 0);
	y1_loc = potentialLocation.at<int>(0, 1);
	x2_loc = potentialLocation.at<int>(0, 2);
	y2_loc = potentialLocation.at<int>(0, 3);
	double intersectionOverUnion;

	intersection = 0;
	if ((x2_loc - x1_loc) * (y2_loc - y1_loc) < (x2_truth - x1_truth) * (y2_truth - y1_truth)) {
		for (int i = y1_loc; i < y2_loc; i++) {
			for (int j = x1_loc; j < x2_loc; j++) {
				if (i >= y1_truth && j >= y1_truth && i <= y2_truth && j <= x2_truth) {
					intersection++;
				}
			}
		}
	} else {
		for (int i = y1_truth; i < y2_truth; i++) {
			for (int j = x1_truth; j < x2_truth; j++) {
				if (i >= y1_loc && j >= y1_loc && i <= y2_loc && j <= x2_loc) {
					intersection++;
				}
			}
		}
	}

	the_union = (x2_loc - x1_loc) * (y2_loc - y1_loc) + (x2_truth - x1_truth) * (y2_truth - y1_truth) - intersection;
	intersectionOverUnion = ((double)intersection) / ((double)the_union);

	if (intersectionOverUnion >= 0.500) {
		return true;
	} else {
		return false;
	}

}