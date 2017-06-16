#include <iostream>

#include "aufgabe_1.h"
#include "hog.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int testing();

int main() {
	testing();
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
	waitKey();
	destroyAllWindows();
	return 0;
}