/* imdiff.cpp - visual alignment of two images
*
* VS version
* working version as of May 31 2013
*/

// set to 1 if running on cygwin - turns off mouse motion animation, o/w crashes on cygwin
int cygwinbug = 0;

#include <stdio.h>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

Mat im0, im1, im0g, im1g, im0gf, im1gf; // orig images, gray, and gray float versions
Mat gx0, gy0, gx1, gy1, gm0, gm1; // gradients
Mat im1t, im1tg, im1tgf; // transformed image 1
Mat imd; // "difference" image
int mode = 0;
int nccmode = 0;
const int nmodes = 4;
const char *modestr[nmodes] = {
	"diff  ", // color diff 
	"Bleyer", // 0.1 * color diff + 0.9 * gradient diff
	"NCC   ",
	"ICPR  "}; // ICPR 94 gradient diff
//, 	"new gradient diff"};

const char *win = "imdiff";
float dx = 0;
float dy = 0;
float ds = 1;  // motion control multiplier
int xonly = 0; // constrain motion in x dir
float startx;
float starty;
float diffscale = 1;
float step = 0.2f;    // arrow key step size
float ncceps = 1e-2f;

void printhelp()
{
	printf("\
		   drag to change offset, shift-drag for fine control\n\
		   arrows: change offset\n\
		   Space - reset offset\n\
		   A, S - show (blink) orig images\n\
		   D - show diff\n\
		   0, 1, 2, 3, .. - change mode\n\
		   Z, X -  change diff contrast\n\
		   E, R -  change NCC epsilon\n\
		   C, V -  change step size\n\
		   Esc, Q - quit\n");
}



void computeGradientX(Mat img, Mat &gx)
{
	int gdepth = CV_32F; // data type of gradient images
	Sobel(img, gx, gdepth, 1, 0, 3, 1, 0);
}

void computeGradientY(Mat img, Mat &gy)
{
	int gdepth = CV_32F; // data type of gradient images
	Sobel(img, gy, gdepth, 0, 1, 3, 1, 0);
}

void computeGradients(Mat img, Mat &gx, Mat &gy, Mat &gm) 
{
	computeGradientX(img, gx);
	computeGradientY(img, gy);
	magnitude(gx, gy, gm);
}

void info()
{
	//rectangle(imd, Point(0, 0), Point(150, 20), Scalar(100, 100, 100), CV_FILLED); // gray rectangle
	Mat r = imd(Rect(0, imd.rows-18, imd.cols, 18));  // better: darken subregion!
	r *= 0.5;
	char txt[100];
	sprintf_s(txt, 100, "%s dx=%4.1f dy=%4.1f step=%3.1f ncceps=%5g  'h' = help ", 
		modestr[mode], dx, dy, step, ncceps);
	putText(imd, txt, Point(5, imd.rows-4), FONT_HERSHEY_PLAIN, 0.8, Scalar(255, 255, 255));
}

void myImDiff2(Mat a, Mat b, Mat &d)
{
	d = 128 + a - b;
}


void myImDiff(Mat a, Mat b, Mat &d)
{
	if (! d.data || d.rows != a.rows || d.cols != a.cols)
		d = a.clone();
	int w = a.cols, h = a.rows, nb = a.channels();

	for (int y=0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			Vec3b pa = a.at<Vec3b>(y, x);
			Vec3b pb = b.at<Vec3b>(y, x);
			Vec3b pd;// = pa - pb;
			for (int z = 0; z < nb; z++) {
				pd[z] = saturate_cast<uchar>(pa[z] - pb[z] + 128);
			}
			d.at<Vec3b>(y, x) = pd;
		}
	}
}

void myImDiff3(Mat a, Mat b, Mat &d)
{
	if (! d.data || d.rows != a.rows || d.cols != a.cols)
		d = a.clone();
	int w = a.cols, h = a.rows, nb = a.channels();

	for (int y=0; y < h; y++) {
		uchar *pa = a.ptr<uchar>(y);
		uchar *pb = b.ptr<uchar>(y);
		uchar *pd = d.ptr<uchar>(y);
		int wnb = w * nb;
		for (int x = 0; x < wnb; x++) {
			pd[x] = saturate_cast<uchar>(pa[x] - pb[x] + 128);
		}
	}
}

void boxFilter(Mat src, Mat &dst, int n) {
	blur(src, dst, Size(n, n), Point(-1, -1));
}

void ncc(Mat L, Mat R, Mat &imd) {
	int nccsize = 7;
	Mat Lb, Rb;
	boxFilter(L,  Lb, nccsize);
	boxFilter(R,  Rb, nccsize);
	Mat LL = L.mul(L);
	Mat RR = R.mul(R);
	Mat LR = L.mul(R);
	Mat LLb, RRb, LRb;
	boxFilter(LL,  LLb, nccsize);
	boxFilter(RR,  RRb, nccsize);
	boxFilter(LR,  LRb, nccsize);
	Mat LL2 = LLb - Lb.mul(Lb);
	Mat RR2 = RRb - Rb.mul(Rb);
	Mat LR2 = LRb - Lb.mul(Rb);
	Mat den = LL2.mul(RR2) + ncceps;
	sqrt(den, den);
	Mat ncc = LR2 / den;
	ncc.convertTo(imd, CV_8U, 128, 128);
}

// 3x3 NCC, taken from Sudipta's code
void ncc2(Mat L, Mat R, Mat &imd) {
	int w = L.cols, h = L.rows;
	if (! imd.data)
		imd = Mat_<uchar>(h,w);

	for (int y=1; y < h-1; y++) {

		//get starting src block
		float L10_ = L.at<float>(-1 + y, -1 + 1); float L20_ = L.at<float>(-1 + y,  0 + 1);
		float L11_ = L.at<float>( 0 + y, -1 + 1); float L21_ = L.at<float>( 0 + y,  0 + 1);
		float L12_ = L.at<float>(+1 + y, -1 + 1); float L22_ = L.at<float>(+1 + y,  0 + 1);

		float R10_ = R.at<float>(-1 + y, -1 + 1); float R20_ = R.at<float>(-1 + y,  0 + 1);
		float R11_ = R.at<float>( 0 + y, -1 + 1); float R21_ = R.at<float>( 0 + y,  0 + 1);
		float R12_ = R.at<float>(+1 + y, -1 + 1); float R22_ = R.at<float>(+1 + y,  0 + 1);

		for (int x = 1; x < w-1; x++) {
			//shift over src block
			float L00_ = L10_; L10_ = L20_; L20_ = L.at<float>(-1 + y, +1 + x);
			float L01_ = L11_; L11_ = L21_; L21_ = L.at<float>( 0 + y, +1 + x);
			float L02_ = L12_; L12_ = L22_; L22_ = L.at<float>(+1 + y, +1 + x);

			float R00_ = R10_; R10_ = R20_; R20_ = R.at<float>(-1 + y, +1 + x);
			float R01_ = R11_; R11_ = R21_; R21_ = R.at<float>( 0 + y, +1 + x);
			float R02_ = R12_; R12_ = R22_; R22_ = R.at<float>(+1 + y, +1 + x);

			float Lavg = 0.111111111111111f * (L00_ + L10_ + L20_ + L01_ + L11_ + L21_ + L02_ + L12_ + L22_);
			float Ravg = 0.111111111111111f * (R00_ + R10_ + R20_ + R01_ + R11_ + R21_ + R02_ + R12_ + R22_);

			float L00 = L00_ - Lavg; float L10 = L10_ - Lavg; float L20 = L20_ - Lavg;
			float L01 = L01_ - Lavg; float L11 = L11_ - Lavg; float L21 = L21_ - Lavg;
			float L02 = L02_ - Lavg; float L12 = L12_ - Lavg; float L22 = L22_ - Lavg;

			float R00 = R00_ - Ravg; float R10 = R10_ - Ravg; float R20 = R20_ - Ravg;
			float R01 = R01_ - Ravg; float R11 = R11_ - Ravg; float R21 = R21_ - Ravg;
			float R02 = R02_ - Ravg; float R12 = R12_ - Ravg; float R22 = R22_ - Ravg;

			float LL =
				L00 * L00 + L10 * L10 + L20 * L20 +
				L01 * L01 + L11 * L11 + L21 * L21 +
				L02 * L02 + L12 * L12 + L22 * L22;

			float RR =
				R00 * R00 + R10 * R10 + R20 * R20 +
				R01 * R01 + R11 * R11 + R21 * R21 +
				R02 * R02 + R12 * R12 + R22 * R22;

			float LR =
				L00 * R00 + L10 * R10 + L20 * R20 +
				L01 * R01 + L11 * R11 + L21 * R21 +
				L02 * R02 + L12 * R12 + L22 * R22;

			// This value is good for images with very little noise
			float ncc = LR / sqrt(LL * RR + ncceps); // add small value to avoid divide by zero
			int score = (int)((1.0-ncc)*128.0f);

			// Check that variance of intensities in the left image is greater than the noise threshold, 
			// otherwise set all scores to zero.
			//if (LL > m_delta) {

			//if (m_delta < 0 && LL < -m_delta) {
			// Soft threshold on LL variance (Rick, 04/10/13)
			//score = 1 + int(score * (LL / -m_delta) * (LL / -m_delta));
			//}
			//}

			imd.at<uchar>(y, x) = saturate_cast<uchar>(255-score);
		}
	}
}

void imdiff()
{
	float s = 1;
	//Mat T0 = (Mat_<float>(2,3) << s, 0,  0, 0, s,  0); 
	Mat T1 = (Mat_<float>(2,3) << s, 0, dx, 0, s, dy); 
	//Mat im0t;
	//warpAffine(im0, im0t, T0, im0.size());
	if (mode == 0) { // difference of images
		warpAffine(im1, im1t, T1, im1.size());
		addWeighted(im0, diffscale, im1t, -diffscale, 128, imd);
	} else if (mode == 1) { // Bleyer weighted sum of color and gradient diff
		warpAffine(im1, im1t, T1, im1.size());
		cvtColor(im1t, im1tg, CV_BGR2GRAY );  
		Mat cdiff, gdiff;
		absdiff(im0, im1t, cdiff);
		cvtColor(cdiff, cdiff, CV_BGR2GRAY);
		computeGradientX(im1tg, gx1);
		absdiff(gx0, gx1, gdiff);
		gdiff.convertTo(imd, CV_8U, 10, 0);
		float sc = diffscale;
		addWeighted(cdiff, 0.1*sc, gdiff, 0.9*sc, 0, imd, CV_8U);
		imd = 255 - imd;
		//still need to truncate diffs

	} else if (mode == 2) { // NCC
		warpAffine(im1gf, im1tgf, T1, im1.size());
		if (nccmode == 0)
			ncc(im0gf, im1tgf, imd);
		else
			ncc2(im0gf, im1tgf, imd);
	} else if (mode == 3) { // ICPR gradient measure
		warpAffine(im1g, im1tg, T1, im1.size());
		computeGradients(im1tg, gx1, gy1, gm1);
		gm1 += gm0; // sum of the gradient magnitudes s
		gx1 -= gx0; // compute magnitude of difference
		gy1 -= gy0;
		Mat gmag;
		magnitude(gx1, gy1, gmag); // magnitude of difference d
		addWeighted(gm1, 0.5, gmag, -1, 128, imd, CV_8U); // result is s/2 - d
	} else { // new gradient measure
		warpAffine(im1g, im1tg, T1, im1.size());
		computeGradients(im1tg, gx1, gy1, gm1);
		// gradient dot prod
		gx1 = gx1.mul(gx0);
		gy1 = gy1.mul(gy0);
		Mat dot = gx1 + gy1;

		Mat minlen, maxlen;
		maxlen = max(gm0, gm1);
		minlen = min(gm0, gm1);

		gm1 = gm1.mul(gm0);
		gm1 = max(gm1, 1e-10);
		dot /= gm1;
		dot = max(dot, 0);

		int k = 10; // exponent
		pow(dot, k, dot);
		// now multiply by minlen / maxlen
		// if minlen too small, just set to zero
		//double thresh = 1e-10;
		//threshold(minlen, minlen, thresh, 1, THRESH_TOZERO);
		//divide(minlen, maxlen, minlen);


		dot = dot.mul(minlen);
		dot.convertTo(imd, CV_8U, 20, 0);
	}

	//imd = diffscale * imd + (1 - diffscale) * 128;
	info();

	imshow(win, imd);
}


static void onMouse( int event, int x, int y, int flags, void* )
{
	x = (short)x; // seem to be short values passed in, cast needed for negative values during dragging
	y = (short)y;
	//printf("x=%d y=%d    ", x, y);
	//printf("dx=%g dy=%g\n", dx, dy);
	if (event == CV_EVENT_LBUTTONDOWN) {
		ds = (float)((flags & CV_EVENT_FLAG_SHIFTKEY) ? 0.1 : 1.0); // fine motion control if Shift is down
		xonly = flags & CV_EVENT_FLAG_CTRLKEY;  // xonly motion if Control is down
		startx = ds*x - dx;
		starty = ds*y - dy;
		//} else if (event == CV_EVENT_LBUTTONUP) {
		//	imdiff();
	} else if (event == CV_EVENT_MOUSEMOVE && flags & CV_EVENT_FLAG_LBUTTON) {
		//startx < 9999) {
		dx = ds*x - startx;
		if (!xonly)
			dy = ds*y - starty;
		if (!cygwinbug)
			imdiff();
	}
}

Mat pyrImg(vector<Mat> pyr) 
{
	Mat im = pyr[0];
	int w = im.cols, h = im.rows;
	Mat pim(Size(3*w/2+4, h+20), CV_8UC3);
	im.copyTo(pim(Rect(0, 0, w, h)));

	int x = w+2;
	int y = 0;
	for (int i = 1; i < (int)pyr.size(); i++) {
		int w1 = pyr[i].cols, h1 = pyr[i].rows;
		pyr[i].copyTo(pim(Rect(x, y, w1, h1)));
		y += h1 + 2;
	}
	return pim;
}


int main(int argc, char ** argv)
{
	setvbuf(stdout, (char*)NULL, _IONBF, 0); // fix to flush stdout when called from cygwin

	if (argc < 3) {
		fprintf(stderr, "usage: %s im1 im2\n", argv[0]);
		exit(1);
	}

	im0 = imread(argv[1], 1);
	if (!im0.data) { 
		fprintf(stderr, "cannot read image %s\n", argv[1]); 
		exit(1); 
	}
	im1 = imread(argv[2], 1);
	if (!im1.data) { 
		fprintf(stderr, "cannot read image %s\n", argv[2]); 
		exit(1); 
	}

	int maxlevels = 4;  // if > 0, create pyramid

	if (maxlevels > 1) {
		vector<Mat> pyr0, pyr1;
		buildPyramid(im0, pyr0, maxlevels);
		im0 = pyrImg(pyr0).clone();
		buildPyramid(im1, pyr1, maxlevels);
		im1 = pyrImg(pyr1).clone();
	}

	// compute graylevel and float versions
	cvtColor(im0, im0g, CV_BGR2GRAY );  
	cvtColor(im1, im1g, CV_BGR2GRAY );  
	im0g.convertTo(im0gf, CV_32F);
	im1g.convertTo(im1gf, CV_32F);

	// smooth
	//int k = 3;
	//double sigma = 0.5;
	//GaussianBlur(im0g, im0g, Size(k, k), sigma, sigma);
	//GaussianBlur(im1g, im1g, Size(k, k), sigma, sigma);

	// compute gradients for im0
	computeGradients(im0g, gx0, gy0, gm0);

	namedWindow(win, CV_WINDOW_AUTOSIZE);
	setMouseCallback(win, onMouse);
	imdiff();

	while(1) {
		int c = waitKey(0);
		switch(c) {
		case 27: // ESC
		case 'q':
			return 0;
		case 7602176: // F5
			{
				Mat m1 = imd;
				break;	// can set a breakpoint here, and then use F5 to stop and restart
			}
		case 'h':
		case '?':
			printhelp(); break;
		case 2424832: case 65361: // left arrow
			dx -= step; imdiff(); break;
		case 2555904: case 65363: // right arrow
			dx += step; imdiff(); break;
		case 2490368: case 65362: // up arrow
			dy -= step; imdiff(); break; 
		case 2621440: case 65364: // down arrow
			dy += step; imdiff(); break;
		case ' ': // reset
			dx = 0; dy = 0; imdiff(); break;
		case 'a': // show original left image
			imshow(win, im0); break;
		case 's': // show original right image
			imshow(win, im1t); break;
		case 'd': // back to diff
			imdiff(); break;
		case 'z': // decrease contrast
			diffscale /= 1.5; imdiff(); break;
		case 'x': // increase contrast
			diffscale *= 1.5; imdiff(); break;
		case 'e': // decrease eps
			ncceps /= 2; imdiff(); break;
		case 'r': // increase eps
			ncceps *= 2; imdiff(); break;
		case 'c': // decrease step
			step /= 2; imdiff(); break;
		case 'v': // increase step
			step *= 2; imdiff(); break;
		case '1': case '2': case '3': case '4':  // change mode
		case '5': case '6': case '7': case '8': case '9':
			mode = min(c - '1', nmodes-1);
			printf("using mode %s\n", modestr[mode]);
			imdiff(); break;
		case 'n': // change nccmode
			nccmode = ! nccmode;
			printf("using %s\n", nccmode? "ncc2 - sudipta" : "ncc - opencv");
			imdiff(); break;
		default:
			printf("key %d (%c %d) pressed\n", c, (char)c, (char)c);
		}
	}

	return 0;
}
