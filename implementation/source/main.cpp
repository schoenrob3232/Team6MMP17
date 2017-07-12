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
void print_plot(Mat fppw_points, Mat missrate_points, string name, int blue, int green, int red);
void presentation();

int main() {

	//Test für Plot
	Mat fppw_points = Mat::zeros(6, 1, CV_64F);
	Mat missrate_points = Mat::zeros(6, 1, CV_64F);

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

	int blue = 0, green = 255, red = 0;
	string name = "DET Curve";
	print_plot(fppw_points, missrate_points, name, blue, green, red);

	//testing();
	//testing2();

	presentation();
	return 0;
}


int testing() {
	/*
	// Testing getGroundTruths(string filename).
	// Filename will propably have to be changed!!
	cout << "Ground truths from image INRIAPerson/Test/annotations/crop_000005.txt" << endl;
	cout << getGroundTruth("/INRIAPerson/INRIAPerson/Test/annotations/crop_000005.txt") << endl;

	vector<int> dims;
	Mat img1 = imread("INRIAPerson/INRIAPerson/Test/neg/prefecture.jpg");
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
	*/
	//slidingWindow_geruest(img1);
	Mat groundTruth = getGroundTruth("INRIAPerson/INRIAPerson/Train/annotations/crop_000010.txt");
	Mat labels = Mat::zeros(0, 1, CV_32F);
	Mat image = imread("INRIAPerson/INRIAPerson/Train/pos/crop_000010.png");
	cout << image.cols << endl << image.rows << endl;
	Mat data = Mat::zeros(0, 13440, CV_32F);
	//slidingWindowGetData(image, labels, data, groundTruth);


	aquireTestTrainingData(labels, data);
	//aquireTrainingPositives(labels, data);

	cout << "x/y: " << data.cols << "/" << data.rows << endl;
	cout << labels << endl;
	//training_SVM(data, labels, "test_svm.xml");
	Mat imagePerson = imread("INRIAPerson/INRIAPerson/Test/pos/crop001501.png");
	Mat detected = showCertainDetections(imagePerson, "test_svm.xml", 0.49);
	cout << "detected --- " << endl;
	imshow("Detection", detected);
	imwrite("/home/user/Documents/detection.png", detected);
	aquireMultipleHardNegatives("test_svm.xml", labels, data);
	aquireUltraHardNegatives("test_svm.xml", labels, data);
	training_SVM(data, labels, "svm_all_hard_negatives.xml");
	cout << "x/y : " << data.cols << "/" << data.rows << endl;
	waitKey();
	destroyAllWindows();
	return 0;
}

int testing2() {
	Mat img1 = imread("INRIAPerson/INRIAPerson/Test/neg/prefecture.jpg");
	imshow("Padded", padWithBorderPixels(img1, 40));
	Mat positions = Mat::zeros(0, 4, CV_32S);
	Mat det_scores = Mat::zeros(0, 1, CV_32F);
	Mat imagePerson = imread("INRIAPerson/INRIAPerson/Test/pos/crop001501.png");
	Mat groundTruth = getGroundTruth("INRIAPerson/INRIAPerson/Test/annotations/crop001501.txt");
	cout << groundTruth << endl;
	cout << "Zeit: 1 : " << time(NULL) << endl;
	extractDetections(imagePerson, "svm_all_hard_negatives.xml", positions, det_scores);
	cout << "Zeit: 2 : " << time(NULL) << endl;
	nonMaxSuppression(positions, det_scores, 10);
	//cout << positions << endl << det_scores << endl;
	//sortByDetectionScore(positions, det_scores);
	cout << positions << endl << det_scores << endl;
	Mat results = drawResults(imagePerson, positions, groundTruth);
	///////////////////////////////////////
	//computePlotPoints("linear_svm_no_hard_negatives_train_auto.xml");
	//computePlotPoints_hard_negs("test_svm_hard_negatives.xml");
	///////////////////////////////////////////
	imshow("Resultate", results);
	imwrite("/home/user/Documents/Resultate.png", results);

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

	// save values in .txt to be safe
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
	string name = "DET Curve without hard negatives";
	int blue = 0, green = 255, red = 0;
	print_plot(fppw_points, missrate_points, name, blue, green, red);
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

	// save values in .txt to be safe
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

	// python script to plot the data
	//system("py plot_script.py"); //findet python36.lib nicht

	//plot mit opencv:
	string name = "DET Curve with hard negatives";
	int blue = 0, green = 0, red = 255;
	print_plot(fppw_points, missrate_points, name, blue, green, red);
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
		y_Vals.at<double>(j, 0) = pos.at<double>(0, 0);
	}
}

void print_plot(Mat fppw_points, Mat missrate_points, string name, int blue, int green, int red)
{
	cv::Mat canvas = cv::Mat::zeros(500, 1150, CV_8UC3);
	int thickness = 1, lineType = 8, shift = 0, fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
	double fontScale = 0.5;

	int len1 = fppw_points.rows; //missrate_points hat selbe Länge

	// scaling the graph to the whole image width/height
	double fppw_max = fppw_points.at<double>(0, 0);
	double missrate_max = missrate_points.at<double>(0, 0);

	for (int i = 1; i < len1; i++) {
		if (fppw_max < fppw_points.at<double>(i, 0)) {
			fppw_max = fppw_points.at<double>(i, 0);
		}
	}

	for (int i = 1; i < len1; i++) {
		if (missrate_max < missrate_points.at<double>(i, 0)) {
			missrate_max = missrate_points.at<double>(i, 0);
		}
	}

	float scalar_in_xRichtung = 0, scalar_in_yRichtung = 0;

	/*double wert_x = compare(fppw_max, fppw_max_negs);
	cout << wert_x << endl;
	double wert_y = compare(missrate_max, missrate_max_negs);
	cout << wert_y << endl;*/

	double abstandx = fppw_max;
	scalar_in_xRichtung = (1000.0 / abstandx);

	double abstandy = missrate_max;
	scalar_in_yRichtung = 240.0 / abstandy;

	// generating the points 
	Point p1 = Point((scalar_in_xRichtung*(fppw_points.at<double>(0, 0)) + 100), 250 - scalar_in_yRichtung*(missrate_points.at<double>(0, 0)));
	Point p2 = Point((scalar_in_xRichtung*(fppw_points.at<double>(1, 0)) + 100), 250 - scalar_in_yRichtung*(missrate_points.at<double>(1, 0)));
	Point p3 = Point((scalar_in_xRichtung*(fppw_points.at<double>(2, 0)) + 100), 250 - scalar_in_yRichtung*(missrate_points.at<double>(2, 0)));
	Point p4 = Point((scalar_in_xRichtung*(fppw_points.at<double>(3, 0)) + 100), 250 - scalar_in_yRichtung*(missrate_points.at<double>(3, 0)));
	Point p5 = Point((scalar_in_xRichtung*(fppw_points.at<double>(4, 0)) + 100), 250 - scalar_in_yRichtung*(missrate_points.at<double>(4, 0)));
	Point p6 = Point((scalar_in_xRichtung*(fppw_points.at<double>(5, 0)) + 100), 250 - scalar_in_yRichtung*(missrate_points.at<double>(5, 0)));

	/*Point p1_n = Point((scalar_in_xRichtung*(fppw_points.at<double>(0, 0)) + 10), 250 - scalar_in_yRichtung*(missrate_points.at<double>(0, 0)));
	Point p2_n = Point((scalar_in_xRichtung*(fppw_points.at<double>(1, 0)) + 10), 250 - scalar_in_yRichtung*(missrate_points.at<double>(1, 0)));
	Point p3_n = Point((scalar_in_xRichtung*(fppw_points.at<double>(2, 0)) + 10), 250 - scalar_in_yRichtung*(missrate_points.at<double>(2, 0)));
	Point p4_n = Point((scalar_in_xRichtung*(fppw_points.at<double>(3, 0)) + 10), 250 - scalar_in_yRichtung*(missrate_points.at<double>(3, 0)));
	Point p5_n = Point((scalar_in_xRichtung*(fppw_points.at<double>(4, 0)) + 10), 250 - scalar_in_yRichtung*(missrate_points.at<double>(4, 0)));
	Point p6_n = Point((scalar_in_xRichtung*(fppw_points.at<double>(5, 0)) + 10), 250 - scalar_in_yRichtung*(missrate_points.at<double>(5, 0)));*/

	//Axes
	line(canvas, Point(100, 0), Point(100, 500), Scalar(255, 255, 255), thickness, lineType, shift);//y Achse
	line(canvas, Point(100, 250), Point(1140, 250), Scalar(255, 255, 255), thickness, lineType, shift);//x Achse

	//Numbers on the Axes
	string x1 = to_string(fppw_points.at<double>(0, 0));
	string x2 = to_string(fppw_points.at<double>(1, 0));
	string x3 = to_string(fppw_points.at<double>(2, 0));
	string x4 = to_string(fppw_points.at<double>(3, 0));
	string x5 = to_string(fppw_points.at<double>(4, 0));
	string x6 = to_string(fppw_points.at<double>(5, 0));

	//text on x axis
	putText(canvas, x1, Point((scalar_in_xRichtung*(fppw_points.at<double>(0, 0)) + 65), 265), fontFace, fontScale, Scalar::all(255), thickness, 3);
		line(canvas, Point((scalar_in_xRichtung*(fppw_points.at<double>(0, 0)) + 100), 250), Point((scalar_in_xRichtung*(fppw_points.at<double>(0, 0)) + 100), 245), Scalar::all(255), thickness, lineType, shift);

	putText(canvas, x2, Point((scalar_in_xRichtung*(fppw_points.at<double>(1, 0)) + 65), 265), fontFace, fontScale, Scalar::all(255), thickness, 3);
		line(canvas, Point((scalar_in_xRichtung*(fppw_points.at<double>(1, 0)) + 100), 250), Point((scalar_in_xRichtung*(fppw_points.at<double>(1, 0)) + 100), 245), Scalar::all(255), thickness, lineType, shift);

	putText(canvas, x3, Point((scalar_in_xRichtung*(fppw_points.at<double>(2, 0)) + 65), 265), fontFace, fontScale, Scalar::all(255), thickness, 3);
		line(canvas, Point((scalar_in_xRichtung*(fppw_points.at<double>(2, 0)) + 100), 250), Point((scalar_in_xRichtung*(fppw_points.at<double>(2, 0)) + 100), 245), Scalar::all(255), thickness, lineType, shift);

	putText(canvas, x4, Point((scalar_in_xRichtung*(fppw_points.at<double>(3, 0)) + 65), 265), fontFace, fontScale, Scalar::all(255), thickness, 3);
		line(canvas, Point((scalar_in_xRichtung*(fppw_points.at<double>(3, 0)) + 100), 250), Point((scalar_in_xRichtung*(fppw_points.at<double>(3, 0)) + 100), 245), Scalar::all(255), thickness, lineType, shift);

	putText(canvas, x5, Point((scalar_in_xRichtung*(fppw_points.at<double>(4, 0)) + 65), 265), fontFace, fontScale, Scalar::all(255), thickness, 3);
		line(canvas, Point((scalar_in_xRichtung*(fppw_points.at<double>(4, 0)) + 100), 250), Point((scalar_in_xRichtung*(fppw_points.at<double>(4, 0)) + 100), 245), Scalar::all(255), thickness, lineType, shift);

	putText(canvas, x6, Point((scalar_in_xRichtung*(fppw_points.at<double>(5, 0)) + 65), 265), fontFace, fontScale, Scalar::all(255), thickness, 3);
		line(canvas, Point((scalar_in_xRichtung*(fppw_points.at<double>(5, 0)) + 100), 250), Point((scalar_in_xRichtung*(fppw_points.at<double>(5, 0)) + 100), 245), Scalar::all(255), thickness, lineType, shift);

	//text on y axis
	putText(canvas, x1, Point(15, 255 - (scalar_in_yRichtung*(missrate_points.at<double>(0, 0)))), fontFace, fontScale, Scalar::all(255), thickness, 3);
	line(canvas, Point(98, 250 - scalar_in_yRichtung*(missrate_points.at<double>(0, 0))), Point(105, 250 - scalar_in_yRichtung*(missrate_points.at<double>(0, 0))), Scalar::all(255), thickness, lineType, shift);
	//putText(canvas, x2, Point(2, 250 - (scalar_in_yRichtung*(missrate_points.at<double>(1, 0)))), fontFace, fontScale, Scalar::all(255), thickness, 3);
	//putText(canvas, x3, Point(2, 250 - (scalar_in_yRichtung*(missrate_points.at<double>(2, 0)))), fontFace, fontScale, Scalar::all(255), thickness, 3);
	//putText(canvas, x4, Point(2, 250 - (scalar_in_yRichtung*(missrate_points.at<double>(3, 0)))), fontFace, fontScale, Scalar::all(255), thickness, 3);
	//putText(canvas, x5, Point(2, 250 - (scalar_in_yRichtung*(missrate_points.at<double>(4, 0)))), fontFace, fontScale, Scalar::all(255), thickness, 3);
	//putText(canvas, x6, Point(2, 250 - (scalar_in_yRichtung*(missrate_points.at<double>(5, 0)))), fontFace, fontScale, Scalar::all(255), thickness, 3);

	// draw graph
	circle(canvas, p1, 3, Scalar(blue, green, red), -1);
	circle(canvas, p2, 3, Scalar(blue, green, red), -1);
		line(canvas, p1, p2, Scalar(blue, green, red), thickness, lineType, shift);
	circle(canvas, p3, 3, Scalar(blue, green, red), -1);
		line(canvas, p2, p3, Scalar(blue, green, red), thickness, lineType, shift);
	circle(canvas, p4, 3, Scalar(blue, green, red), -1);
		line(canvas, p3, p4, Scalar(blue, green, red), thickness, lineType, shift);
	circle(canvas, p5, 3, Scalar(blue, green, red), -1);
		line(canvas, p4, p5, Scalar(blue, green, red), thickness, lineType, shift);
	circle(canvas, p6, 3, Scalar(blue, green, red), -1);
		line(canvas, p5, p6, Scalar(blue, green, red), thickness, lineType, shift);

	//circle(canvas, Point(0,0), 3, Scalar(blue, green, red), -1);
	//circle(canvas, Point(1000, 500), 3, Scalar(blue, green, red), -1);

	//// draw the second graph
	//circle(canvas, p1_n, 3, Scalar(255, 0, 0), -1);
	//circle(canvas, p2_n, 3, Scalar(255, 0, 0), -1);
	//	line(canvas, p1_n, p2_n, Scalar(255, 0, 0), thickness, lineType, shift);
	//circle(canvas, p3_n, 3, Scalar(255, 0, 0), -1);
	//	line(canvas, p2_n, p3_n, Scalar(255, 0, 0), thickness, lineType, shift);
	//circle(canvas, p4_n, 3, Scalar(255, 0, 0), -1);
	//	line(canvas, p3_n, p4_n, Scalar(255, 0, 0), thickness, lineType, shift);
	//circle(canvas, p5_n, 3, Scalar(255, 0, 0), -1);
	//	line(canvas, p4_n, p5_n, Scalar(255, 0, 0), thickness, lineType, shift);
	//circle(canvas, p6_n, 3, Scalar(255, 0, 0), -1);
	//	line(canvas, p5_n, p6_n, Scalar(255, 0, 0), thickness, lineType, shift);

	imshow(name, canvas);
	waitKey();
}

//double compare(double wert_x, double wert_y)
//{
//	if (wert_x <= wert_y) {
//
//		return wert_y;
//	}
//	else return wert_x;
//}

void presentation() {

	Mat positions1_init = Mat::zeros(0, 4, CV_32S);
	Mat det_scores1_init = Mat::zeros(0, 1, CV_32F);
	Mat positions2_init = Mat::zeros(0, 4, CV_32S);
	Mat det_scores2_init = Mat::zeros(0, 1, CV_32F);
	Mat positions3_init = Mat::zeros(0, 4, CV_32S);
	Mat det_scores3_init = Mat::zeros(0, 1, CV_32F);
	Mat positions4_init = Mat::zeros(0, 4, CV_32S);
	Mat det_scores4_init = Mat::zeros(0, 1, CV_32F);
	Mat positions5_init = Mat::zeros(0, 4, CV_32S);
	Mat det_scores5_init = Mat::zeros(0, 1, CV_32F);

	Mat positions1_hn = Mat::zeros(0, 4, CV_32S);
	Mat det_scores1_hn = Mat::zeros(0, 1, CV_32F);
	Mat positions2_hn = Mat::zeros(0, 4, CV_32S);
	Mat det_scores2_hn = Mat::zeros(0, 1, CV_32F);
	Mat positions3_hn = Mat::zeros(0, 4, CV_32S);
	Mat det_scores3_hn = Mat::zeros(0, 1, CV_32F);
	Mat positions4_hn = Mat::zeros(0, 4, CV_32S);
	Mat det_scores4_hn = Mat::zeros(0, 1, CV_32F);
	Mat positions5_hn = Mat::zeros(0, 4, CV_32S);
	Mat det_scores5_hn = Mat::zeros(0, 1, CV_32F);

	Mat image1 = imread("INRIAPerson/INRIAPerson/Test/pos/crop001501.png");
	Mat groundTruth1 = getGroundTruth("INRIAPerson/INRIAPerson/Test/annotations/crop001501.txt");
	Mat image2 = imread("INRIAPerson/INRIAPerson/Test/pos/crop001670.png");
	Mat groundTruth2 = getGroundTruth("INRIAPerson/INRIAPerson/Test/annotations/crop001670.txt");
	Mat image3 = imread("INRIAPerson/INRIAPerson/Test/pos/person_039.png");
	Mat groundTruth3 = getGroundTruth("INRIAPerson/INRIAPerson/Test/annotations/person_039.txt");
	Mat image4 = imread("INRIAPerson/INRIAPerson/Test/pos/person_306.png");
	Mat groundTruth4 = getGroundTruth("INRIAPerson/INRIAPerson/Test/annotations/person_306.txt");
	Mat image5 = imread("INRIAPerson/INRIAPerson/Test/pos/person_and_bike_043.png");
	Mat groundTruth5 = getGroundTruth("INRIAPerson/INRIAPerson/Test/annotations/person_and_bike_043.txt");

	if (image1.empty()) {
		cout << 1 << endl;
	}
	if (image2.empty()) {
		cout << 2 << endl;
	}
	if (image3.empty()) {
		cout << 3 << endl;
	}
	if (image4.empty()) {
		cout << 4 << endl;
	}
	if (image5.empty()) {
		cout << 5  << endl;
	}

	extractDetections(image1, "svm_all_hard_negatives.xml", positions1_hn, det_scores1_hn);
	extractDetections(image2, "svm_all_hard_negatives.xml", positions2_hn, det_scores2_hn);
	extractDetections(image3, "svm_all_hard_negatives.xml", positions3_hn, det_scores3_hn);
	extractDetections(image4, "svm_all_hard_negatives.xml", positions4_hn, det_scores4_hn);
	extractDetections(image5, "svm_all_hard_negatives.xml", positions5_hn, det_scores5_hn);


	nonMaxSuppression(positions1_hn, det_scores1_hn, 10);
	nonMaxSuppression(positions2_hn, det_scores2_hn, 10);
	nonMaxSuppression(positions3_hn, det_scores3_hn, 10);
	nonMaxSuppression(positions4_hn, det_scores4_hn, 10);
	nonMaxSuppression(positions5_hn, det_scores5_hn, 10);

	extractDetections(image1, "linear_svm_no_hard_negatives_train_auto.xml", positions1_init, det_scores1_init);
	extractDetections(image2, "linear_svm_no_hard_negatives_train_auto.xml", positions2_init, det_scores2_init);
	extractDetections(image3, "linear_svm_no_hard_negatives_train_auto.xml", positions3_init, det_scores3_init);
	extractDetections(image4, "linear_svm_no_hard_negatives_train_auto.xml", positions4_init, det_scores4_init);
	extractDetections(image5, "linear_svm_no_hard_negatives_train_auto.xml", positions5_init, det_scores5_init);


	nonMaxSuppression(positions1_init, det_scores1_init, 10);
	nonMaxSuppression(positions2_init, det_scores2_init, 10);
	nonMaxSuppression(positions3_init, det_scores3_init, 10);
	nonMaxSuppression(positions4_init, det_scores4_init, 10);
	nonMaxSuppression(positions5_init, det_scores5_init, 10);

	vector<Mat> drawings_init;
	vector<Mat> drawings_hn;

	drawings_init.push_back(drawResults(image1, positions1_init, groundTruth1));
	drawings_init.push_back(drawResults(image2, positions2_init, groundTruth2));
	drawings_init.push_back(drawResults(image3, positions3_init, groundTruth3));
	drawings_init.push_back(drawResults(image4, positions4_init, groundTruth4));
	drawings_init.push_back(drawResults(image5, positions5_init, groundTruth5));

	drawings_hn.push_back(drawResults(image1, positions1_hn, groundTruth1));
	drawings_hn.push_back(drawResults(image2, positions2_hn, groundTruth2));
	drawings_hn.push_back(drawResults(image3, positions3_hn, groundTruth3));
	drawings_hn.push_back(drawResults(image4, positions4_hn, groundTruth4));
	drawings_hn.push_back(drawResults(image5, positions5_hn, groundTruth5));

	for (int i = 0; i < 5; i++) {
		imshow("Resultat - Initiale SVM" + to_string(i+1), drawings_init[i]);
	}

	for (int i = 0; i < 5; i++) {
		imshow("Resultat - SVM mit Hard Negatives" + to_string(i+1), drawings_hn[i]);
	}
	waitKey();
}
