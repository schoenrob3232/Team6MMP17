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


#define INRIA_PATH "C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\"
#define CELL_SIZE 8
#define BLOCK_SIZE 2
#define CPW_X 8
#define CPW_Y 16

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


/*
Extracts the positions detected by svm_name in img_arg into:

positions: k x 4 Mat with format (example):
			 | x1	   | y1		 | x2	   | y2
____________ | _______ | _______ | _______ | _______
position1	 | 24	   | 1		 | 170	   | 503
position2	 | 39	   | 44		 | 102	   | 432

and
det_scores: k x 1 Mat with Mat.at<float>(i, 0) = "Detection Score of i-th position"
*/
int extractDetections(Mat img_arg, const char* svm_name, Mat &positions, Mat &det_scores) {
	Mat img = img_arg.clone();
	int width = img.cols;
	int height = img.rows;
	int windows_x, windows_y;
	vector<int> dims, cropDims;
	double ***hogCells;
	double ***croppedCells;
	Mat descriptor;
	Mat currentWindowPos;
	int m = 0, window_count = 0;
	CvSVM my_svm;
	my_svm.load(svm_name);
	Mat det_score = Mat::zeros(1, 1, CV_32F);;

	while (height >= 142 && width >= 78) {
		width = img.cols;
		height = img.rows;
		hogCells = computeHoG(img, CELL_SIZE, dims);
		windows_x = dims[1] - (CPW_X - 1);
		windows_y = dims[0] - (CPW_Y - 1);

		//initializing descriptor Mat
		int blocks_y = CPW_Y - (BLOCK_SIZE - 1);
		int blocks_x = CPW_X - (BLOCK_SIZE - 1);
		int descriptor_len = BLOCK_SIZE * BLOCK_SIZE * dims[2] * blocks_x * blocks_y;

		for (int i = 0; i < windows_y; i += 3) {
			for (int j = 0; j < windows_x; j += 3) {
				croppedCells = copyHOGCells(i, j, hogCells, dims, cropDims);
				descriptor = computeWindowDescriptor(croppedCells, cropDims);

				// where is the current window?
				currentWindowPos = Mat::zeros(1, 4, CV_32S);
				currentWindowPos.at<int>(0, 0) = (8 + j * CELL_SIZE) * pow(pow(2.0, 0.2), m);
				currentWindowPos.at<int>(0, 1) = (8 + i * CELL_SIZE) * pow(pow(2.0, 0.2), m);
				currentWindowPos.at<int>(0, 2) = (72 + j * CELL_SIZE) * pow(pow(2.0, 0.2), m);
				currentWindowPos.at<int>(0, 3) = (136 + i * CELL_SIZE) * pow(pow(2.0, 0.2), m);

				window_count++;

				det_score.at<float>(0, 0) = my_svm.predict(descriptor, true);
				if (det_score.at<float>(0, 0) < 0.0) {
					positions.push_back(currentWindowPos);
					det_scores.push_back(det_score);

				}

				dissolve(croppedCells, cropDims);
			}
		}
		dissolve(hogCells, dims);
		img = scaleDownOneStep(img);
		m++;
	}
	return window_count;
}


/*
Performs the non-maximum suppression and only takes the top N remaining detections, 
as requested in task 3.3 and 3.4
*/
void nonMaxSuppression(Mat &positions, Mat &det_scores, int N) {
	int n = positions.rows;
	Mat pos_temp, scores_temp;

	//non-maximum suppression
	sortByDetectionScore(positions, det_scores);
	for (int i = 0; i < n && i < N; i++) {
		pos_temp = Mat::zeros(0, 4, CV_32S);
		scores_temp = Mat::zeros(0, 1, CV_32F);
		for (int k = 0; k <= i; k++) {
			pos_temp.push_back(cloneRowInt(positions, k));
			scores_temp.push_back(cloneRowFloat(det_scores, k));
		}

		for (int j = i + 1; j < n; j++) {
			if (fastComputeIoU(cloneRowInt(positions, i), cloneRowInt(positions, j)) < 0.2) {
				scores_temp.push_back(cloneRowFloat(det_scores, j));
				pos_temp.push_back(cloneRowInt(positions, j));
			}
		}
		
		positions = pos_temp.clone();
		det_scores = scores_temp.clone();
		n = positions.rows;
	}

	//only use top N detections at most
	pos_temp = Mat::zeros(0, 4, CV_32S);
	scores_temp = Mat::zeros(0, 1, CV_32F);
	for (int i = 0; i < n && i < N; i++) {
		pos_temp.push_back(cloneRowInt(positions, i));
		scores_temp.push_back(cloneRowFloat(det_scores, i));
	}
	positions = pos_temp;
	det_scores = scores_temp;
}

/*
clones a matrix row with int format
*/
Mat cloneRowInt(Mat matrix, int row) {
	int cols = matrix.cols;
	Mat cloned_mat = Mat::zeros(1, cols, CV_32S);
	for (int i = 0; i < cols; i++) {
		cloned_mat.at<int>(0, i) = matrix.at<int>(row, i);
	}
	return cloned_mat;
}


/*
clones a matrix row with float format
*/
Mat cloneRowFloat(Mat matrix, int row) {
	int cols = matrix.cols;
	Mat cloned_mat = Mat::zeros(1, cols, CV_32F);
	for (int i = 0; i < cols; i++) {
		cloned_mat.at<float>(0, i) = matrix.at<float>(row, i);
	}
	return cloned_mat;
}


/*
Sorts the rows of positions and det_scores according to det_scores (ascending)
*/
void sortByDetectionScore(Mat &positions, Mat &det_scores) {
	int n = positions.rows;
	Mat pos = Mat::zeros(1, 4, CV_32S); 
	float key;
	int j;
	for (int i = 1; i < n; i++) {
		for (int p = 0; p < 4; p++) {
			pos.at<int>(0, p) = positions.at<int>(i, p);
		}
		key = det_scores.at<float>(i, 0);
		j = i; 
		while (j > 0 && key < det_scores.at<float>(j - 1, 0)) {
			det_scores.at<float>(j, 0) = det_scores.at<float>(j - 1, 0);
			for (int p = 0; p < 4; p++) {
				positions.at<int>(j, p) = positions.at<int>(j - 1, p);
			}
			j--;
		}
		det_scores.at<float>(j, 0) = key;
		for (int p = 0; p < 4; p++) {
			positions.at<int>(j, p) = pos.at<int>(0, p);
		}

	}
}


/*
Pads img with pad zero-pixels on every side
*/
Mat padImgWithZeros(Mat img, int pad) {
	int width_old = img.cols;
	int height_old = img.rows;
	int width_new = width_old + 2 * pad;
	int height_new = height_old + 2 * pad;
	Mat img_work = Mat::zeros(height_new, width_new, CV_8UC3);

	for (int i = pad; i < height_old + pad; i++) {
		for (int j = pad; j < width_old + pad; j++) {
			img_work.at<Vec3b>(i, j) = img.at<Vec3b>(i - pad, j - pad);
		}
	}
	return img_work;
}


/*
Draws the results(green) and groundTruths(red) into the image img.
*/
Mat drawResults(Mat img, Mat results, Mat groundTruths) {
	Mat img_work = img.clone();
	int resCount = results.rows;
	int truCount = groundTruths.rows;

	for (int i = 0; i < resCount; i++) {
		Point p1(results.at<int>(i, 0), results.at<int>(i, 1));
		Point p2(results.at<int>(i, 2), results.at<int>(i, 3));
		Scalar green(0, 255, 0);
		rectangle(img_work, p1, p2, green);
	}

	for (int i = 0; i < truCount; i++) {
		Point p1(groundTruths.at<int>(i, 0), groundTruths.at<int>(i, 1));
		Point p2(groundTruths.at<int>(i, 2), groundTruths.at<int>(i, 3));
		Scalar red(0, 0, 255);
		rectangle(img_work, p1, p2, red);
	}

	return img_work;
}


/*
Pads img by spreading its border pixels to every side by pad pixels
*/
Mat padWithBorderPixels(Mat img, int pad) {
	int width_old = img.cols;
	int height_old = img.rows;
	int width_new = width_old + 2 * pad;
	int height_new = height_old + 2 * pad;
	Mat img_work = Mat::zeros(height_new, width_new, CV_8UC3);

	//padding with zeros
	for (int i = pad; i < height_old + pad; i++) {
		for (int j = pad; j < width_old + pad; j++) {
			img_work.at<Vec3b>(i, j) = img.at<Vec3b>(i - pad, j - pad);
		}
	}

	//padding left border
	for (int i = pad; i < height_old + pad; i++) {
		for (int j = 0; j < pad; j++) {
			img_work.at<Vec3b>(i, j) = img.at<Vec3b>(i - pad, 0);
		}
	}

	//padding right border
	for (int i = pad; i < height_old + pad; i++) {
		for (int j = pad + width_old; j < width_new; j++) {
			img_work.at<Vec3b>(i, j) = img.at<Vec3b>(i - pad, width_old - 1);
		}
	}

	//padding top border
	for (int i = 0; i < pad; i++) {
		for (int j = 0; j < width_new; j++) {
			img_work.at<Vec3b>(i, j) = img_work.at<Vec3b>(pad, j);
		}
	}

	//padding bottom border
	for (int i = pad + height_old; i < height_new; i++) {
		for (int j = 0; j < width_new; j++) {
			img_work.at<Vec3b>(i, j) = img_work.at<Vec3b>(pad + height_old - 1, j);
		}
	}
	return img_work;
}

/*
Computes the miss rate (double in [0, 1]) out of the results and groundTruths

KANN NOCH NICHT VERWENDET WERDEN (VERWENDET ZWEI FEHLERHAFTE FUNKTIONEN)!!!!!!!!!!!!!
*/
double computeMissRate(Mat results, Mat groundTruths) {
	int resCount = results.rows;
	int truCount = groundTruths.rows;
	int missed = 0;

	for (int i = 0; i < truCount; i++) {
		bool found = false;
		for (int j = 0; j < resCount; j++) {
			if (fastComputeIoU(cloneRowInt(results, j), cloneRowInt(groundTruths, i)) >= 0.5) {
				found = true;
				break;
			}
		}
		if (found == false) {
			missed++;
		}
	}
	return ((double)missed / (double)truCount);
}


/*
Eliminates every detection form positions and det_scores with a detection score >= sigma
*/
void suppressThreshold(Mat &positions, Mat &det_scores, float sigma) {
	Mat pos_temp = Mat::zeros(0, 4, CV_32S);
	Mat scores_temp = Mat::zeros(0, 1, CV_32F);
	int n = positions.rows;

	for (int i = 0; i < n; i++) {
		if (det_scores.at<float>(i, 0) < sigma) {
			pos_temp.push_back(cloneRowInt(positions, i));
			scores_temp.push_back(cloneRowFloat(det_scores, i));
		}
	}
	positions = pos_temp;
	det_scores = scores_temp;
}


/*
counts the false positives in results (those who do not fit to anything in groundTruths)
*/
int countFalsePositives(Mat results, Mat groundTruths) {
	int resCount = results.rows;
	int truCount = groundTruths.rows;
	int falsePos = 0;

	for (int i = 0; i < resCount; i++) {
		bool found = false;
		for (int j = 0; j < truCount; j++) {
			if (fastComputeIoU(cloneRowInt(results, i), cloneRowInt(groundTruths, j)) >= 0.5) {
				found = true;
				break;
			}
		}
		if (found == false) {
			falsePos++;
		}
	}
	return falsePos;
}

/*
computes average miss rate and fppw of the whole test set for one sigma, 
and subsequently adds these values to missRates and fppw
*/
void computeDETPoint(Mat &fppw, Mat &missRates, const char* svm, float sigma) {
	ifstream fileNeg("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\Test\\neg.lst");
	ifstream filePos("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\Test\\pos.lst");
	ifstream fileAnnotations("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\Test\\annotations.lst");
	long allWindowsTested = 0;
	long allFalsePositives = 0;
	double collectedMissRates = 0.0;
	int scannedImages = 0;
	String line, annotation;
	Mat sampleImg;
	Mat positions = Mat::zeros(0, 4, CV_32S);
	Mat det_scores = Mat::zeros(0, 1, CV_32F);
	Mat new_fppw = Mat::zeros(1, 1, CV_64F);
	Mat new_missrate = Mat::zeros(1, 1, CV_64F);
	Mat groundTruths;

	while (getline(filePos, line) && getline(fileAnnotations, annotation)) {
		cout << "Scanning: " << line << endl;
		sampleImg = imread(INRIA_PATH + line);
		groundTruths = getGroundTruth(INRIA_PATH + annotation);
		if (!sampleImg.empty()) {
			positions = Mat::zeros(0, 4, CV_32S);
			det_scores = Mat::zeros(0, 1, CV_32F);
			allWindowsTested += extractDetections(sampleImg, svm, positions, det_scores);
			nonMaxSuppression(positions, det_scores, 10);
			suppressThreshold(positions, det_scores, sigma);

			allFalsePositives += countFalsePositives(positions, groundTruths);
			collectedMissRates += computeMissRate(positions, groundTruths);
			scannedImages++;
			
		}
	}

	while (getline(fileNeg, line)) {
		cout << "Scanning: " << line << endl;
		sampleImg = imread(INRIA_PATH + line);
		if (!sampleImg.empty()) {
			positions = Mat::zeros(0, 4, CV_32S);
			det_scores = Mat::zeros(0, 1, CV_32F);
			allWindowsTested += extractDetections(sampleImg, svm, positions, det_scores);
			nonMaxSuppression(positions, det_scores, 10);
			suppressThreshold(positions, det_scores, sigma);

			allFalsePositives += positions.rows;
			scannedImages++;
		}
	}
	new_fppw.at<double>(0, 0) = ((double)allFalsePositives) / ((double)allWindowsTested);
	new_missrate.at<double>(0, 0) = ((double)collectedMissRates) / ((double)scannedImages);
	fppw.push_back(new_fppw);
	missRates.push_back(new_missrate);
}


