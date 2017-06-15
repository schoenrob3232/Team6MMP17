#include <iostream>
#include <fstream>
#include <string>

#include "aufgabe_1.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

/*
Extracts the ground truth of an annotation (filename) to integer Mat with
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