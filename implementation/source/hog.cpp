
#include <iostream>
#include <cmath>
#include <assert.h>

#include "hog.h"

#define round(x) (floor(x+0.5))

using namespace std;
using namespace cv;

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }



// compute HOG features from a color image
//
// dim_z: 32 = 18 orientations [0,2*pi] + 9 orientations [0,pi] + 4 texture + 1 truncation
//
// param img: CV_8UC3 image
// param cell_size: size of a single squared HOG cell
// param dims: output parameter, dimensions of HOG array (dim_y,dim_x,dim_z)
//
// return: 3D HOG features
double*** computeHoG(const Mat &img, const int cell_size, vector<int> &dims) {
	const double eps = 0.0001; // small value, used to avoid division by zero

	// unit vectors used to compute gradient orientation
	//                    0°      20°     40°     60°     80°      100°     120°     140°     160°
	const double uu[9] = {1.0000, 0.9397, 0.7660, 0.5000, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397};
	const double vv[9] = {0.0000, 0.3420, 0.6428, 0.8660, 0.9848,  0.9848,  0.8660,  0.6428,  0.3420};

	// initialize the function for fast processing
	// pre-compute magnitude and orientation for every possible gradient [-255,255][-255,255]
	static double mag[511][511] = { 0 };
	static unsigned char ori[511][511] = { 0 };
	static bool initOri = false;
	if (!initOri) {
		for (short dy = -255; dy<256; dy++) {
			for (short dx = -255; dx<256; dx++) {
				double best_dot = 0;
				int best_o = 0;
				for (int o = 0; o < 9; o++) {
					double dot = uu[o] * double(dx) + vv[o] * double(dy);
					if (dot > best_dot) {
						best_dot = dot;
						best_o = o;
					} else if (-dot > best_dot) {
						best_dot = -dot;
						best_o = o + 9;
					}
				}
				ori[dy + 255][dx + 255] = best_o; // [0, 17]
				mag[dy + 255][dx + 255] = sqrt((double)(dx*dx) + (double)(dy*dy));
			} // end for dx
		} // end for dy
		initOri = true;
	}

	if (img.channels() != 3 || ((img.type() != CV_8UC3))) {
		cout << "Invalid image input" << endl;
		exit(0);
	}

	// memory for caching orientation histograms & their norms
	int y_cells = (int)round((double)img.rows / (double)cell_size); // num cells in y direction
	int x_cells = (int)round((double)img.cols / (double)cell_size); // num cells is x direction
	double *hist = new double[y_cells * x_cells * 18];
	double *norm = new double[y_cells * x_cells];
	memset(norm, 0, y_cells * x_cells * sizeof(double));
	memset(hist, 0, y_cells * x_cells * 18 * sizeof(double));

	size_t y_visible = y_cells * cell_size; // pixel in y direction
	size_t x_visible = x_cells * cell_size; // pixel in x direction

	// image short cuts
	const size_t img_width = img.cols;
	const size_t img_height = img.rows;
	const size_t num_channels = img.channels();

	const uint8 * im = img.ptr<uint8>();
	for (size_t y = 1; y < y_visible - 1; y++) {
		for (size_t x = 1; x < x_visible - 1; x++) {
			// first (B) color channel
			const uint8 *s = im + min(y, img_height - 2)*img.step + min(x, img_width - 2)*num_channels;
			size_t dx3 = (size_t(255) + *(s + 3)) - *(s - 3);
			size_t dy3 = (size_t(255) + *(s + img.step)) - *(s - img.step);
			double v3 = mag[dy3][dx3]; // dx*dx + dy*dy;

			// second (G) color channel
			s++;
			size_t dx2 = (size_t(255) + *(s + 3)) - *(s - 3);
			size_t dy2 = (size_t(255) + *(s + img.step)) - *(s - img.step);
			double v2 = mag[dy2][dx2]; // dx2*dx2 + dy2*dy2;

			// third (R) color channel
			s++;
			size_t dx = (size_t(255) + *(s + 3)) - *(s - 3);
			size_t dy = (size_t(255) + *(s + img.step)) - *(s - img.step);
			double v = mag[dy][dx]; //dx3*dx3 + dy3*dy3;

			// pick channel with strongest gradient
			if (v2 > v) {
				v = v2;
				dx = dx2;
				dy = dy2;
			}
			if (v3 > v) {
				v = v3;
				dx = dx3;
				dy = dy3;
			}

			int best_o = ori[dy][dx];

			// add to 4 histograms around pixel using linear interpolation
			double xp = ((double)x + 0.5) / (double)cell_size - 0.5;
			double yp = ((double)y + 0.5) / (double)cell_size - 0.5;
			int ixp = (int)floor(xp);
			int iyp = (int)floor(yp);
			double vx0 = xp - ixp;
			double vy0 = yp - iyp;
			double vx1 = 1.0 - vx0;
			double vy1 = 1.0 - vy0;
			//			v = sqrt(v);

			if (ixp >= 0 && iyp >= 0) {
				*(hist + ixp*y_cells + iyp + best_o*y_cells * x_cells) += vx1*vy1*v;
			}

			if (ixp + 1 < x_cells && iyp >= 0) {
				*(hist + (ixp + 1)*y_cells + iyp + best_o*y_cells * x_cells) += vx0*vy1*v;
			}

			if (ixp >= 0 && iyp + 1 < y_cells) {
				*(hist + ixp*y_cells + (iyp + 1) + best_o*y_cells * x_cells) += vx1*vy0*v;
			}

			if (ixp + 1 < x_cells && iyp + 1 < y_cells) {
				*(hist + (ixp + 1)*y_cells + (iyp + 1) + best_o*y_cells * x_cells) += vx0*vy0*v;
			}
		}
	}

	// compute energy in each block by summing over orientations
	for (int o = 0; o < 9; o++) {
		double *src1 = hist + o*y_cells * x_cells;
		double *src2 = hist + (o + 9)*y_cells * x_cells;
		double *dst = norm;
		double *end = norm + x_cells * y_cells;
		while (dst < end) {
			*(dst++) += (*src1 + *src2) * (*src1 + *src2);
			src1++;
			src2++;
		}
	}

	// memory for HOG features
	int dim_y = max(y_cells - 2, 0);
	int dim_x = max(x_cells - 2, 0);
	int dim_z = 27 + 4 + 1;
	double*** featArray = (double***)malloc(dim_y * sizeof(double**));
	for (int i = 0; i < dim_y; ++i) {
		featArray[i] = (double**)malloc(dim_x * sizeof(double*));
		for (int j = 0; j < dim_x; ++j) {
			featArray[i][j] = (double*)malloc(dim_z * sizeof(double));
		}
	}

	dims = vector<int>(3);
	dims[0] = dim_y;
	dims[1] = dim_x;
	dims[2] = dim_z;

	// compute features
	for (int x = 0; x < dim_x; x++) {
		for (int y = 0; y < dim_y; y++) {
			double *dst = &featArray[y][x][0]; //feat + x*out[0] + y;      
			double *src, *p, n1, n2, n3, n4;

			p = norm + (x + 1)*y_cells + y + 1;
			n1 = 1.0 / sqrt(*p + *(p + 1) + *(p + y_cells) + *(p + y_cells + 1) + eps);
			p = norm + (x + 1)*y_cells + y;
			n2 = 1.0 / sqrt(*p + *(p + 1) + *(p + y_cells) + *(p + y_cells + 1) + eps);
			p = norm + x*y_cells + y + 1;
			n3 = 1.0 / sqrt(*p + *(p + 1) + *(p + y_cells) + *(p + y_cells + 1) + eps);
			p = norm + x*y_cells + y;
			n4 = 1.0 / sqrt(*p + *(p + 1) + *(p + y_cells) + *(p + y_cells + 1) + eps);

			double t1 = 0;
			double t2 = 0;
			double t3 = 0;
			double t4 = 0;

			// contrast-sensitive features
			src = hist + (x + 1)*y_cells + (y + 1);
			for (int o = 0; o < 18; o++) {
				double h1 = min(*src * n1, 0.2);
				double h2 = min(*src * n2, 0.2);
				double h3 = min(*src * n3, 0.2);
				double h4 = min(*src * n4, 0.2);
				*dst = 0.5 * (h1 + h2 + h3 + h4);
				t1 += h1;
				t2 += h2;
				t3 += h3;
				t4 += h4;
				dst++;
				src += y_cells * x_cells;
			}

			// contrast-insensitive features
			src = hist + (x + 1)*y_cells + (y + 1);
			for (int o = 0; o < 9; o++) {
				double sum = *src + *(src + 9 * y_cells * x_cells);
				double h1 = min(sum * n1, 0.2);
				double h2 = min(sum * n2, 0.2);
				double h3 = min(sum * n3, 0.2);
				double h4 = min(sum * n4, 0.2);
				*dst = 0.5 * (h1 + h2 + h3 + h4);
				dst++;
				src += y_cells * x_cells;
			}

			// texture features
			*dst = 0.2357 * t1;
			dst++;
			*dst = 0.2357 * t2;
			dst++;
			*dst = 0.2357 * t3;
			dst++;
			*dst = 0.2357 * t4;

			// truncation feature
			dst++;
			*dst = 0;
		}
	}

	delete[] hist; hist = 0;
	delete[] norm; norm = 0;

	return featArray;
}


