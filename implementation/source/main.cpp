#include <iostream>

#include "aufgabe_1.h"
#include "aufgabe_2.h"
#include "aufgabe_3.h"
#include "hog.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int testing();
int testing2();

int main() {
	//testing();
	testing2();
	return 0;
}

int testing() {
	// Testing getGroundTruths(string filename).
	// Filename will propably have to be changed!!
	cout << "Ground truths from image INRIAPerson\\Test\\annotations\\crop_000005.txt" << endl;
	cout << getGroundTruth("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\Test\\annotations\\crop_000005.txt") << endl;

	vector<int> dims;
	Mat img1 = imread("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\Test\\neg\\prefecture.jpg");
	double ***hogCells = computeHoG(img1, 6, dims);
	cout << dims[0] << "/" << dims[1] << "/" << dims[2] << endl;
	Mat testdesc = computeWindowDescriptor(hogCells, dims);
	//cout << testdesc << endl;
	imshow("Bild", img1);
	Mat img2 = scaleDownOneStep(img1);
	imshow("Bild_skaliert1", img2);
	Mat img3 = scaleDownOneStep(img2);
	imshow("Bild_skaliert2", img3);
	Mat img4 = scaleDownOneStep(img3);
	imshow("Bild_skaliert3", img4);
	Mat img5 = scaleDownOneStep(img4);
	imshow("Bild_skaliert4", img5);
	Mat img6 = scaleDownOneStep(img5);
	imshow("Bild_skaliert5", img6);
	
	//slidingWindow_geruest(img1);
	Mat groundTruth = getGroundTruth("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\Train\\annotations\\crop_000010.txt");
	Mat labels = Mat::zeros(0, 1, CV_32F);
	Mat image = imread("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\Train\\pos\\crop_000010.png");
	cout << image.cols << endl << image.rows << endl;
	Mat data = Mat::zeros(0, 13440, CV_32F);
	//slidingWindowGetData(image, labels, data, groundTruth);


	aquireTestTrainingData(labels, data);

	cout << "x/y: " << data.cols << "/" << data.rows << endl;
	cout << labels << endl;
	training_SVM(data, labels, "test_svm.xml");
	Mat imagePerson = imread("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\Test\\pos\\crop001501.png");
	Mat detected = showCertainDetections(imagePerson, "test_svm.xml", 0.99);
	cout << "detected --- " << endl;
	imshow("Detection", detected);
	imwrite("C:\\Users\\user\\Documents\\detection.png", detected);
	//aquireMultipleHardNegatives("test_svm.xml", labels, data);
	cout << "x/y : " << data.cols << "/" << data.rows << endl;
	waitKey();
	destroyAllWindows();
	return 0;
}

int testing2() {
	Mat img1 = imread("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\Test\\neg\\prefecture.jpg");
	imshow("Padded", padWithBorderPixels(img1, 40));
	Mat positions = Mat::zeros(0, 4, CV_32S);
	Mat det_scores = Mat::zeros(0, 1, CV_32F);
	Mat imagePerson = imread("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\Test\\pos\\crop001501.png");
	Mat groundTruth = getGroundTruth("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\Test\\annotations\\crop001501.txt");
	cout << groundTruth << endl;
	extractDetections(imagePerson, "test_svm.xml", positions, det_scores);
	nonMaxSuppression(positions, det_scores, 10);
	//cout << positions << endl << det_scores << endl;
	//sortByDetectionScore(positions, det_scores);
	cout << positions << endl << det_scores << endl;
	Mat results = drawResults(imagePerson, positions, groundTruth);
	imshow("Resultate", results);
	waitKey();
	destroyAllWindows();
	return 0;
}