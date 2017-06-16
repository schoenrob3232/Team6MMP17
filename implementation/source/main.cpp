#include <iostream>

#include "aufgabe_1.h"
#include "hog.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main() {
	// Testing getGroundTruths(string filename).
	// Filename will propably have to be changed!!
	cout << "Ground truths from image INRIAPerson\\Test\\annotations\\crop_000005.txt" << endl;
	cout << getGroundTruth("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\Test\\annotations\\crop_000005.txt") << endl;

	vector<int> dims;
	Mat img1 = imread("C:\\Users\\user\\Documents\\Uni\\MMP\\INRIAPerson\\INRIAPerson\\70X134H96\\Test\\pos\\crop_000001a.png");
	double ***hogCells = computeHoG(img1, 6, dims);
	cout << dims[0] << "/" << dims[1] << "/" << dims[2] << endl;
	Mat testdesc = computeWindowDescriptor(hogCells, dims);
	cout << testdesc << endl;
	getchar();
	return 0;
}