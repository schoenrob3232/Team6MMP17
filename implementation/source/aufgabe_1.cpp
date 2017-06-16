#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include "aufgabe_1.h"
#include "hog.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#define CELL_SIZE 6
#define BLOCK_SIZE 3

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



/*
Checks if the proposed location fits at least one ground
truth in the image (as returned by the method getGroundTruth(string filename)) 
over 50%. Returns true, if so.

*/
bool compareToAllGroundTruths(Mat allGroundTruths, Mat potentialLocation) {
	Mat groundTruth = Mat::zeros(1, 4, CV_32S);
	int truthCount = allGroundTruths.rows;
	for (int i = 0; i < truthCount; i++) {
		for (int j = 0; j < 4; j++) {
			groundTruth.at<int>(0, j) = allGroundTruths.at<int>(i, j);
			if (compareToGroundTruth(groundTruth, potentialLocation)) {
				return true;
			}
		}
	}
	return false;
}


/*computes a hogblock normalized with L2-Hys*/
Mat computeHOGBlock(int cell_pos_x, int cell_pos_y, int block_size, double *** hogCells, vector<int> dims) {
	int block_len = block_size * block_size * dims[2];
	Mat block = Mat::zeros(1, block_len, CV_32F);
	int dim_z = dims[2];
	int k = 0;
	double sum;

	//compute (raw) block
	for (int i = cell_pos_y; i < cell_pos_y + block_size; i++) {
		for (int j = cell_pos_x; j < cell_pos_x + block_size; j++) {
			for (int l = 0; l < dim_z; l++) {
				block.at<float>(0, k) = hogCells[i][j][l];
				k++;
			}
		}
	}

	//normalize with L2-Hys
	sum = 0;
	for (int i = 0; i < block_len; i++) {
		sum += pow(block.at<float>(0, i), 2.0);
	}
	sum += pow(0.0000001, 2.0);
	sum = sqrt(sum);
	for (int i = 0; i < block_len; i++) {
		block.at<float>(0, i) = block.at<float>(0, i) / sum;
		if (block.at<float>(0, i) > 0.2) {
			block.at<float>(0, i) = 0.2;
		}
	}
	
	return block;
}

/*
Computes HOG descriptor for a whole window
*/
Mat computeWindowDescriptor(double ***hogCells, vector<int> dims) {
	int blocks_y = dims[0] - (BLOCK_SIZE - 1);
	int blocks_x = dims[1] - (BLOCK_SIZE - 1);
	int descriptor_len = BLOCK_SIZE * BLOCK_SIZE * dims[2] * blocks_x * blocks_y;
	Mat descriptor = Mat::zeros(1, descriptor_len, CV_32F);
	Mat block;
	int block_len = BLOCK_SIZE * BLOCK_SIZE * dims[2];
	int k = 0;
	for (int i = 0; i < blocks_y; i++) {
		for (int j = 0; j < blocks_x; j++) {
			block = computeHOGBlock(j, i, BLOCK_SIZE, hogCells, dims);
			for (int l = 0; l < block_len; l++) {
				descriptor.at<float>(0, k) = block.at<float>(0, l);
				k++;
			}
		}
	}
	return descriptor;
}