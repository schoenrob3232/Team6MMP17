#ifndef HOG_H_INCLUDED
#define HOG_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


typedef unsigned char uint8;


// compute HOG features from a color image
//
// param img: CV_8UC3 image
// param cell_size: size of a single squared HOG cell
// param dims: output parameter, dimensions of HOG array (dim_y,dim_x,dim_z)
//
// return: 3D HOG features
double*** computeHoG(const cv::Mat &img, const int sbin, std::vector<int> &dims);


#endif