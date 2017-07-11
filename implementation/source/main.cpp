#include <iostream>
#include <fstream>

#include "aufgabe_1.h"
#include "aufgabe_2.h"
#include "aufgabe_3.h"
#include "hog.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <ctime>

using namespace std;
using namespace cv;

int testing();
int testing2();
void computePlotPoints(const char *svm);
void computePlotPoints_hard_negs(const char *svm);
void sortByXVals(Mat &x_Vals, Mat &y_Vals);
void print_plot(Mat fppw_points, Mat missrate_points);

int main() {

	Mat fppw_points = Mat::zeros(6, 1, CV_64F);
	Mat missrate_points = Mat::zeros(6, 1,  CV_64F);

	double wert1x = 0.000850;
	double wert2x = 0.001234;
	double wert3x = 0.001756;
	double wert4x = 0.002517;
	double wert5x = 0.003389;
	double wert6x = 0.004373;

	double wert1y = 0.283046;
	double wert2y = 0.283046;
	double wert3y = 0.283046;
	double wert4y = 0.283046;
	double wert5y = 0.283046;
	double wert6y = 0.283046;

	fppw_points.at<double>(0, 0) = wert1x;
	fppw_points.at<double>(1, 0) = wert2x;
	fppw_points.at<double>(2, 0) = wert3x;
	fppw_points.at<double>(3, 0) = wert4x;
	fppw_points.at<double>(4, 0) = wert5x; 
	fppw_points.at<double>(5, 0) = wert6x;

	missrate_points.at<double>(0, 0) = wert1y;
	missrate_points.at<double>(1, 0) = wert2y;
	missrate_points.at<double>(2, 0) = wert3y;
	missrate_points.at<double>(3, 0) = wert4y;
	missrate_points.at<double>(4, 0) = wert5y;
	missrate_points.at<double>(5, 0) = wert6y;

	print_plot(fppw_points, missrate_points);

	//testing();
	testing2();
	return 0;
}


int testing() {
	// Testing getGroundTruths(string filename).
	// Filename will propably have to be changed!!
	cout << "Ground truths from image INRIAPerson\\Test\\annotations\\crop_000005.txt" << endl;
	cout << getGroundTruth("\\INRIAPerson\\INRIAPerson\\Test\\annotations\\crop_000005.txt") << endl;

	vector<int> dims;
	Mat img1 = imread("\\INRIAPerson\\INRIAPerson\\Test\\neg\\prefecture.jpg");
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
	Mat groundTruth = getGroundTruth("\\INRIAPerson\\INRIAPerson\\Train\\annotations\\crop_000010.txt");
	Mat labels = Mat::zeros(0, 1, CV_32F);
	Mat image = imread("\\INRIAPerson\\INRIAPerson\\Train\\pos\\crop_000010.png");
	cout << image.cols << endl << image.rows << endl;
	Mat data = Mat::zeros(0, 13440, CV_32F);
	//slidingWindowGetData(image, labels, data, groundTruth);


	aquireTestTrainingData(labels, data);
	//aquireTrainingPositives(labels, data);

	cout << "x/y: " << data.cols << "/" << data.rows << endl;
	cout << labels << endl;
	//training_SVM(data, labels, "test_svm.xml");
	Mat imagePerson = imread("\\INRIAPerson\\INRIAPerson\\Test\\pos\\crop001501.png");
	Mat detected = showCertainDetections(imagePerson, "test_svm.xml", 0.49);
	cout << "detected --- " << endl;
	imshow("Detection", detected);
	imwrite("C:\\Users\\user\\Documents\\detection.png", detected);
	aquireMultipleHardNegatives("test_svm.xml", labels, data);
	training_SVM(data, labels, "test_svm_hard_negatives.xml");
	cout << "x/y : " << data.cols << "/" << data.rows << endl;
	waitKey();
	destroyAllWindows();
	return 0;
}

int testing2() {
	Mat img1 = imread("\\INRIAPerson\\INRIAPerson\\Test\\neg\\prefecture.jpg");
	imshow("Padded", padWithBorderPixels(img1, 40));
	Mat positions = Mat::zeros(0, 4, CV_32S);
	Mat det_scores = Mat::zeros(0, 1, CV_32F);
	Mat imagePerson = imread("\\INRIAPerson\\INRIAPerson\\Test\\pos\\person_265.png");
	imagePerson = padImgWithZeros(imagePerson, 40);
	Mat groundTruth = getGroundTruth("\INRIAPerson\\INRIAPerson\\Test\\annotations\\person_265.txt");
	cout << groundTruth << endl;
	cout << "Zeit: 1 : " << time(NULL) << endl;
	extractDetections(imagePerson, "linear_svm_no_hard_negatives_train_auto.xml", positions, det_scores);
	cout << "Zeit: 2 : " << time(NULL) << endl;
	nonMaxSuppression(positions, det_scores, 10);
	//cout << positions << endl << det_scores << endl;
	//sortByDetectionScore(positions, det_scores);
	cout << positions << endl << det_scores << endl;
	Mat results = drawResults(imagePerson, positions, groundTruth);
	///////////////////////////////////////
	computePlotPoints("linear_svm_no_hard_negatives_train_auto.xml");
	computePlotPoints_hard_negs("test_svm_hard_negatives.xml");
	///////////////////////////////////////////
	imshow("Resultate", results);
	imwrite("C:\\Users\\user\\Documents\\Resultate.png", results);

	waitKey();
	destroyAllWindows();
	return 0;
}


/*
computes multiple DET points for SVM trained without hard negatives
*/
void computePlotPoints(const char *svm) {
	Mat fppw_points = Mat::zeros(0, 1, CV_64F);
	Mat missrate_points = Mat::zeros(0, 1, CV_64F);
	float sigma = 0.0; 
	while (sigma >= -1.0) {
		cout << "Computing for sigma = " << sigma << endl;
		computeDETPoint(fppw_points, missrate_points, svm, sigma);
		sigma -= 0.2;
		cout << "Computed!" << endl;
	}
	sortByXVals(fppw_points, missrate_points);

	ofstream file_m, file_f;
	file_m.open("y_value.txt", std::ios_base::app);
	file_f.open("x_value.txt", std::ios_base::app);

	for (int i = 0; i < fppw_points.rows; i++) 
	{
		file_f << to_string(fppw_points.at<double>(i, 0)) << endl;
	}
	
	for (int i = 0; i < missrate_points.rows; i++)
	{
		file_m << to_string(missrate_points.at<double>(i, 0)) << endl;
	}

	file_m.close();
	file_f.close();


	// python script to plot the data
	//system("py plot_script.py"); //findet python36.lib nicht

	//plot mit opencv:
	// kommt noch, weil python leider etwa sspackt -

}



/*
computes multiple DET points for SVM trained with hard negatives
*/
void computePlotPoints_hard_negs(const char *svm) {
	Mat fppw_points = Mat::zeros(0, 1, CV_64F);
	Mat missrate_points = Mat::zeros(0, 1, CV_64F);
	float sigma = 0.0;
	while (sigma >= -1.0) {
		cout << "Computing for sigma = " << sigma << endl;
		computeDETPoint(fppw_points, missrate_points, svm, sigma);
		sigma -= 0.2;
		cout << "Computed!" << endl;
	}
	sortByXVals(fppw_points, missrate_points);
	
	
	///// evtl sinnlos, wenns gleich mit der Matrix funktioniert!
	ofstream file_m, file_f;
	file_m.open("y_value_neg.txt", std::ios_base::app);
	file_f.open("x_value_neg.txt", std::ios_base::app);

	for (int i = 0; i < fppw_points.rows; i++)
	{
		file_f << to_string(fppw_points.at<double>(i, 0)) << endl;
	}

	for (int i = 0; i < missrate_points.rows; i++)
	{
		file_m << to_string(missrate_points.at<double>(i, 0)) << endl;
	}

	file_m.close();
	file_f.close();
	/////////////////////////////////


	// python script to plot the data
	//system("py plot_script.py"); //findet python36.lib nicht

	//plot mit opencv:
	// kommt noch, weil python leider etwas spackt -
	// auch noch ohne Achsenbeschriftung
	print_plot(fppw_points, missrate_points);
}


/*
Sorts the rows of x_Vals and y_Vals according to x_Vals (ascending)
*/
void sortByXVals(Mat &x_Vals, Mat &y_Vals) {
	int n = x_Vals.rows;
	Mat pos = Mat::zeros(1, 1, CV_64F);
	double key;
	int j;
	for (int i = 1; i < n; i++) {
		pos.at<double>(0, 0) = y_Vals.at<double>(i, 0);
		key = x_Vals.at<double>(i, 0);
		j = i;
		while (j > 0 && key < x_Vals.at<double>(j - 1, 0)) {
			x_Vals.at<double>(j, 0) = x_Vals.at<double>(j - 1, 0);
			y_Vals.at<double>(j, 0) = y_Vals.at<double>(j - 1, 0);
			j--;
		}
		x_Vals.at<double>(j, 0) = key;
		y_Vals.at<int>(j, 0) = pos.at<int>(0, 0);
		
	}
}

void print_plot(Mat fppw_points, Mat missrate_points) {
	cv::Mat canvas = cv::Mat::zeros(500, 1020, CV_8UC3);
	int thickness = 1, lineType = 8, shift = 0;

	int len1 = fppw_points.rows; //missrate_points hat selbe Länge

	double fppw_max = fppw_points.at<double>(0, 0);
	double missrate_max = missrate_points.at<double>(0, 0);

	for (int i = 1; i < len1; i++) {
		if (fppw_max < fppw_points.at<double>(i, 0)) {
			fppw_max = fppw_points.at<double>(i, 0);
			cout << fppw_max << endl;
		}

	}

	for (int i = 1; i < len1; i++) {
		if (missrate_max < missrate_points.at<double>(i, 0)) {
			missrate_max = missrate_points.at<double>(i, 0);
		}
	}

	/*double x_pointList[1][6] = { 0.000850, 0.001234, 0.001756, 0.002517, 0.003389, 0.004373 },
		y_pointList[1][6] = { 0.283046,0.283046,0.283046,0.283046,0.283046,0.283046 };*/

	float scalar_in_xRichtung = 0;
	float scalar_in_yRichtung = 0;

	//Skalierung auf ganze Breite - stimmt nicht, ist anscheinend falsch - ich werds anders lösen müssen.
	double abstandx = fppw_max;
	scalar_in_xRichtung = (1000.0  / abstandx);
	cout << abstandx << endl;
	cout << scalar_in_xRichtung << endl;

	double abstandy = missrate_max;
	scalar_in_yRichtung = 240.0 / abstandy;
	cout << fppw_points.at<double>(0, 0) << endl;
	cout << fppw_points.at<double>(1, 0) << endl;
	cout << fppw_points.at<double>(2, 0) << endl;
	cout << fppw_points.at<double>(3, 0) << endl;
	cout << fppw_points.at<double>(4, 0) << endl;
	cout << fppw_points.at<double>(5, 0) << endl;

	Point p1 = Point((scalar_in_xRichtung*(fppw_points.at<double>(0,0)) + 10), 250 - scalar_in_yRichtung*(missrate_points.at<double>(0,0)));
	Point p2 = Point((scalar_in_xRichtung*(fppw_points.at<double>(1,0)) + 10), 250 - scalar_in_yRichtung*(missrate_points.at<double>(1,0)));
	Point p3 = Point((scalar_in_xRichtung*(fppw_points.at<double>(2,0)) + 10), 250 - scalar_in_yRichtung*(missrate_points.at<double>(2,0)));
	Point p4 = Point((scalar_in_xRichtung*(fppw_points.at<double>(3,0)) + 10), 250 - scalar_in_yRichtung*(missrate_points.at<double>(3,0)));
	Point p5 = Point((scalar_in_xRichtung*(fppw_points.at<double>(4,0)) + 10), 250 - scalar_in_yRichtung*(missrate_points.at<double>(4,0)));
	Point p6 = Point((scalar_in_xRichtung*(fppw_points.at<double>(5,0)) + 10), 250 - scalar_in_yRichtung*(missrate_points.at<double>(5,0)));

	rectangle(canvas, Point(10, 0), Point(10, 500), Scalar(255, 255, 255), thickness, lineType, shift); //y Achse
	rectangle(canvas, Point(10, 500), Point(1020, 250), Scalar(255, 255, 255), thickness, lineType, shift); //x Achse
	//line(canvas, Point(10, 0), Point(10, 500), Scalar(255, 255, 255), thickness, lineType, shift);

	//malen des Graphen
	circle(canvas, p1, 3, Scalar(0, 255, 0), -1);
	circle(canvas, p2, 3, Scalar(0, 255, 0), -1);
		line(canvas, p1, p2, Scalar(0, 255, 0), thickness, lineType, shift);
	circle(canvas, p3, 3, Scalar(0, 255, 0), -1);
		line(canvas, p2, p3, Scalar(0, 255, 0), thickness, lineType, shift);
	circle(canvas, p4, 3, Scalar(0, 255, 0), -1);
		line(canvas, p3, p4, Scalar(0, 255, 0), thickness, lineType, shift);
	circle(canvas, p5, 3, Scalar(0, 255, 0), -1);
		line(canvas, p4, p5, Scalar(0, 255, 0), thickness, lineType, shift);
	circle(canvas, p6, 3, Scalar(0, 255, 0), -1);
		line(canvas, p5, p6, Scalar(0, 255, 0), thickness, lineType, shift);

	imshow("DET Curve", canvas);
	waitKey();
}
