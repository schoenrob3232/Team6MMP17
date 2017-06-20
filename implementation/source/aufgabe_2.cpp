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
	if (img.rows == 142 && img.cols == 78) {
		// already fits
	} else if (img.rows == 160 && img.cols == 96) {
		Rect roi(9, 9, 78, 142);
		img = img(roi);
	} else {
		cout << "Image too large. Can't find positive sample without given ground truth." << endl;
		return;
	}

	vector<int> dims;
	Mat sampleDeskriptor;
	Mat positiveLabel = Mat::zeros(1, 1, CV_32F);
	double ***hogCells = computeHoG(img, CELL_SIZE, dims); 
	cout << dims[0] << "/" << dims[1] << endl;
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

	while (getline(filePos, line) && i < 2000) {
		sampleImg = imread(INRIA_PATH + subfolder + line);
		if (!sampleImg.empty()) {
			aquireSimplePositive(sampleImg, labels_arg, data_arg);
			i++;
		}
	}
	
	i = 0;
	while (getline(fileNeg, line) && i < 200) {
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
	parameter.C = 0.1;
	parameter.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, (int)5e5, 1e-6);

	cout << "Training in process... " << endl;

	SVM.train(data_arg, labels_arg, Mat(), Mat(), parameter);
	SVM.save(name);

	cout << "Training finished successful! " << endl;
}