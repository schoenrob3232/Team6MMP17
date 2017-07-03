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

	// Eingabe-Maske für die einzelnen Tasks
	bool running = true;
	while (running) {
		cout << endl;
		cout << "Please enter a number of the task you would like to execute:" << endl;
		cout << " - 01 -- to see the ground truth boxes" << endl;
		cout << " - 02 -- to test the sliding window" << endl;
		cout << " - 03 -- to train the linear SVM" << endl;
		cout << " - 04 -- to train the <> SVM" << endl; // hier <> aussagekräftigen Namen bitte einfügen für die andere
		cout << " - 05 -- to create a classifier" << endl;
		cout << " - 06 -- to enhance a classifier" << endl;
		cout << " - 07 -- to analyze all pictures and save and show the results" << endl;
		cout << " - 11 -- to calculate the data needed for the DET plot + generate x- and y-axis data" << endl;
		cout << " - 12 -- to analyze your own picture" << endl;
		cout << " - 00 -- to end the program" << endl;
		cout << endl;

		cout << ">>> ";
		int task = 0;
		cin >> task;
		cout << endl;

		switch (task) {
		case 00: //end the program
			{
				running = false;
				system("pause");
			}
				break;
	
			case 01: 
			{
	
			}
				break;

			case 02:
			{

			}
				break;

			case 03: 
			{

			}
				break;

			case 04: 
			{

			}
				break;

			case 05: 
			{

			}
				break;

			case 06: 
			{

			}
				break;

			case 07: 
			{

			}
				break;

			case 11:
			{

			}
				break;

			case 12:
			{

			}
				break;

		}
	}
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
	nonMaxSuppression(positions, det_scores, 5);
	//cout << positions << endl << det_scores << endl;
	//sortByDetectionScore(positions, det_scores);
	cout << positions << endl << det_scores << endl;
	Mat results = drawResults(imagePerson, positions, groundTruth);
	imshow("Resultate", results);
	imwrite("C:\\Users\\user\\Documents\\Resultate.png", results);

	waitKey();
	destroyAllWindows();
	return 0;
}