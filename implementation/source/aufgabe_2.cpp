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
#include "hog.h"

#define INRIA_PATH "C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\"
#define CELL_SIZE 8
#define BLOCK_SIZE 2
#define CPW_X 8
#define CPW_Y 16

using namespace std;
using namespace cv;

/*
	Adds n negative descriptors from img_arg to data_arg. Also sets additional negative labels to labels_arg.
*/
void aquireRandomNegatives(Mat img_arg, Mat &labels_arg, Mat &data_arg, int n) {
	Mat img = img_arg.clone();
	vector<int> dims, croppedDims;
	Mat sampleDeskriptor;
	double ***hogCells = computeHoG(img, CELL_SIZE, dims);
	double ***croppedCells;
	int windows_x = dims[1] - (CPW_X - 1);
	int windows_y = dims[0] - (CPW_Y - 1);
	srand(time(NULL));
	int x, y;
	
	for (int i = 0; i < n; i++) {
		x = rand() % windows_x;
		y = rand() % windows_y;
		croppedCells = copyHOGCells(y, x, hogCells, dims, croppedDims);
		sampleDeskriptor = computeWindowDescriptor(croppedCells, croppedDims);
		data_arg.push_back(sampleDeskriptor);
		Mat negativeLabels(1, 1, CV_32F);
		negativeLabels.at<float>(0, 0) = -1;
		labels_arg.push_back(negativeLabels);
		dissolve(croppedCells, croppedDims);	
	}
	dissolve(hogCells, dims);
}


/*
	Adds the positive descriptor from img_arg to data_arg; also adds a positive Label to labels_arg
*/
void aquireSimplePositive(Mat img_arg, Mat &labels_arg, Mat &data_arg) {
	Mat img = img_arg.clone();
	if (img.rows == 144 && img.cols == 80) {
		// already fits
	} else if (img.rows == 160 && img.cols == 96) {
		Rect roi(7, 7, 80, 144);
		img = img(roi);
	} else {
		cout << "Image too large. Can't find positive sample without given ground truth." << endl;
		return;
	}

	vector<int> dims;
	Mat sampleDeskriptor;
	Mat positiveLabel = Mat::zeros(1, 1, CV_32F);
	double ***hogCells = computeHoG(img, CELL_SIZE, dims); 
	sampleDeskriptor = computeWindowDescriptor(hogCells, dims);
	data_arg.push_back(sampleDeskriptor);
	positiveLabel.at<float>(0, 0) = 1;
	labels_arg.push_back(positiveLabel);
	dissolve(hogCells, dims);
}


/*Adds training data to the argument Mats
FILENAME WILL HAVE TO BE CHANGED ACCORDINGLY
*/
void aquireTestTrainingData(Mat &labels_arg, Mat &data_arg) {
	string line;
	string subfolder = "train_64x128_H96\\";

	ifstream filePos("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\train_64x128_H96\\pos.lst");
	ifstream fileNeg("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\train_64x128_H96\\neg.lst");
	Mat sampleImg;
	int i = 0;

	while (getline(filePos, line) && i < 5000) {
		sampleImg = imread(INRIA_PATH + subfolder + line);
		if (!sampleImg.empty()) {
			aquireSimplePositive(sampleImg, labels_arg, data_arg);
			i++;
		}
	}
	
	i = 0;
	while (getline(fileNeg, line) && i < 500) {
		sampleImg = imread(INRIA_PATH + subfolder + line);
		if (!sampleImg.empty()) {
			aquireRandomNegatives(sampleImg, labels_arg, data_arg, 10);
			i++;
		}
	}
}

void training_SVM(Mat& data_arg, Mat& labels_arg, const char* name) {

	CvSVMParams parameter;
	CvSVM SVM;
	parameter.svm_type = CvSVM::C_SVC;
	parameter.kernel_type = CvSVM::LINEAR;
	//parameter.degree = 6;
	parameter.C = 0.1;
	parameter.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, (int)50000, 1e-6);

	cout << "Training in process... " << endl;

	SVM.train_auto(data_arg, labels_arg, Mat(), Mat(), parameter);
	cout << "Saving..." << endl;
	SVM.save(name);

	cout << "Training finished successful! " << endl;

}


/*
Draws all detection windows found in img_arg (by svm_name) with a detection score
higher than threshold.
*/
Mat showCertainDetections(Mat img_arg, const char* svm_name, double threshold) {
	Mat img = img_arg.clone();
	Mat img_work = img.clone();
	int width = img.cols;
	int height = img.rows;
	int windows_x, windows_y;
	vector<int> dims, cropDims;
	double ***hogCells;
	double ***croppedCells;
	Mat descriptor, labels;
	Mat currentWindowPos;
	int k = 0, m = 0;
	CvSVM my_svm; 
	my_svm.load(svm_name);

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
		labels = Mat::zeros(1, 1, CV_32F);

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

				if (my_svm.predict(descriptor, true) < -1 * threshold) {
					Point p1(currentWindowPos.at<int>(0, 0), currentWindowPos.at<int>(0, 1));
					Point p2(currentWindowPos.at<int>(0, 2), currentWindowPos.at<int>(0, 3));
					Scalar green(0, 255, 0);
					rectangle(img_work, p1, p2, green);
				}

				k++;
				dissolve(croppedCells, cropDims);
			}
		}
		dissolve(hogCells, dims);
		k = 0;
		img = scaleDownOneStep(img);
		m++;
	}
	return img_work;
}


/*
Auquires the hardest negative on the Mat-Image img_arg which is misdetected by svm_name, and adds it to 
data_arg with a negative label to labels_arg. 
*/
void aquireHardestNegative(Mat img_arg, const char* svm_name, Mat &labels_arg, Mat &data_arg, Mat groundTruths_arg) {
	Mat groundTruths = groundTruths_arg.clone();
	Mat img = img_arg.clone();
	int width = img.cols;
	int height = img.rows;
	int windows_x, windows_y;
	vector<int> dims, cropDims;
	double ***hogCells;
	double ***croppedCells;
	Mat descriptor, labels;
	Mat currentWindowPos;
	int k = 0, m = 0;
	CvSVM my_svm;
	my_svm.load(svm_name);
	Mat hardest_descriptor;
	double temp_predict, minPredict = 0.0;

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
		//data.push_back(Mat::zeros(windows_x * windows_y, descriptor_len, CV_32F));
		labels = Mat::zeros(1, 1, CV_32F);

		for (int i = 0; i < windows_y; i += 3) {
			for (int j = 0; j < windows_x; j += 3) {
				croppedCells = copyHOGCells(i, j, hogCells, dims, cropDims);
				descriptor = computeWindowDescriptor(croppedCells, cropDims);

				
				/* where is the current window?
				currentWindowPos = Mat::zeros(1, 4, CV_32S);
				currentWindowPos.at<int>(0, 0) = 8 + j * CELL_SIZE;
				currentWindowPos.at<int>(0, 1) = 8 + i * CELL_SIZE;
				currentWindowPos.at<int>(0, 2) = 72 + j * CELL_SIZE;
				currentWindowPos.at<int>(0, 3) = 136 + i * CELL_SIZE;*/

				// does the current window fit a ground truth
				temp_predict = my_svm.predict(descriptor, true);
				if (!compareToAllGroundTruths(groundTruths, currentWindowPos) && (minPredict > temp_predict)) {
					minPredict = temp_predict;
					hardest_descriptor = descriptor;
				}


				k++;
				dissolve(croppedCells, cropDims);
			}
		}
		dissolve(hogCells, dims);
		k = 0;

		//verkleinere groundTruths und image
		for (int i = 0; i < groundTruths.rows; i++) {
			groundTruths.at<int>(i, 0) = groundTruths.at<int>(i, 0) / pow(2.0, 0.2);
			groundTruths.at<int>(i, 1) = groundTruths.at<int>(i, 1) / pow(2.0, 0.2);
			groundTruths.at<int>(i, 2) = groundTruths.at<int>(i, 2) / pow(2.0, 0.2);
			groundTruths.at<int>(i, 3) = groundTruths.at<int>(i, 3) / pow(2.0, 0.2);
		}
		img = scaleDownOneStep(img);
		m++;
	}

	if (minPredict < -0.5) {
		labels.at<float>(0, 0) = -1;
		cout << minPredict << "  :minp" << endl;
		data_arg.push_back(hardest_descriptor);
		labels_arg.push_back(labels);
	}
	return;
}


/*
uses aquireHardestNegative to look through the whole negatives-list for hard negatives and
adds them to data_arg // labels_arg.
*/
void aquireMultipleHardNegatives(const char* svm_name, Mat &labels_arg, Mat &data_arg) {
	string line;
	string subfolder = "train_64x128_H96\\";

	ifstream fileNeg("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\train_64x128_H96\\neg.lst");
	Mat sampleImg;
	Mat groundTruth = Mat::zeros(0, 4, CV_32S);

	while (getline(fileNeg, line)) {
		sampleImg = imread(INRIA_PATH + subfolder + line);
		if (!sampleImg.empty()) {
			aquireHardestNegative(sampleImg, svm_name, labels_arg, data_arg, groundTruth);
		}
	}

}

/*
Aquires positive Samples according to annotations
*/
void aquireTrainingPositives(Mat &labels_arg, Mat &data_arg) {
	string line;
	string annotation_line;
	string subfolder = "train_64x128_H96\\";

	ifstream filePos("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\Train\\pos.lst");
	ifstream fileAnnotations("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\Train\\annotations.lst");
	Mat sampleImg;
	Mat annotation;
	int i = 0;

	while (getline(filePos, line) && getline(fileAnnotations, annotation_line) && i < 200) {
		cout << "Fetching annotated positive: " << i << endl;
		sampleImg = imread(INRIA_PATH + line);
		annotation = getGroundTruth(INRIA_PATH + annotation_line);

		if (!sampleImg.empty()) {
			slidingWindowGetPositives(sampleImg, labels_arg, data_arg, annotation);
			i++;
		}
		cout << "Current size of training set: " << data_arg.rows << endl;
	}

	
}