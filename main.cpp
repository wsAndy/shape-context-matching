#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <math.h>

#include "lap.h"

using namespace std;
using namespace cv;

#define OUTLIER_THRESHOLD 1.2

const Scalar RED = Scalar(0,0,255);
const Scalar PINK = Scalar(230,130,255);
const Scalar BLUE = Scalar(255,0,0);
const Scalar LIGHTBLUE = Scalar(255,255,160);
const Scalar GREEN = Scalar(0,255,0);
const Scalar WHITE = Scalar(255, 255, 255);

double dist(Point& p1, Point& p2)
{
	return sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y));
}


/*
void makeImage(int **hist, int csize)
{
	for (int i=0; i<csize; i++)
	{
		int maxval = 0;

		for (int j=0; j<5; j++)
		{
			for (int k=0; k<12; k++)
			{
				if (maxval < hist[i][j*12+k]) maxval = hist[i][j*12+k];
			}
		}
		maxval = maxval + 1;

		int **img = (int **) malloc(sizeof(int *) * 120);
		for (int j=0; j<120; j++) img[j] = (int *) malloc(sizeof(int) * 288);

		bool nonzero = false;

		for (int j=0; j<5; j++)
		{
			for (int k=0; k<12; k++)
			{
				int val = hist[i][j*12+k];
				val = (int) ((val * 256.0) / (maxval * 1.0));

				if (!nonzero)
					if (val > 0)
						nonzero = true;

				for (int m=0; m<24; m++)
				{
					for (int n=0; n<24; n++)
					{
						img[j*24+m][k*24+n] = val;
					}
				}
			}
		}

		if (!nonzero)
		{
			for (int j=0; j<120; j++) free(img[j]);
			free(img);

			continue;
		}

		ostringstream convert;
		convert << i;
		string result = convert.str();
		if (result.size() == 1) result = "000" + result;
		else if (result.size() == 2) result = "00" + result;
		else if (result.size() == 3) result = "0" + result;

		string fname = "results/hist_" + result + ".pgm";
		ofstream outfile(fname.c_str());
		outfile << "P2" << endl;
		outfile << "288 120" << endl;
		for (int j=0; j<120; j++)
		{
			for (int k=0; k<288; k++)
			{
				outfile << img[j][k] << endl;
			}
		}

		cout << "done with " << fname << endl;

		for (int j=0; j<120; j++) free(img[j]);
		free(img);
	}

	return;
}

*/

double angle(Point& p1, Point& p2)
{
	int ydif = p2.y - p1.y;
	int xdif = p2.x - p1.x;

	if (xdif == 0)
	{
		if (ydif < 0) return (-1.0 * M_PI/2.0);
		else return (M_PI / 2.0);
	}

	float slope = ydif / xdif;
	double theta = atan(slope);

	if (theta > 0 && xdif > 0) theta = theta;
	if (theta > 0 && xdif < 0) theta = theta + M_PI;
	else if (theta < 0 && xdif > 0) theta = theta + M_PI * 3.0 / 4.0;
	else theta = theta + M_PI / 2.0;

	return theta;
}

vector<Point> getContourPtsFromFile(char* fname)
{
	int totalSize, innerSize, x, y;
	ifstream infile(fname);
	infile >> totalSize;

	vector<Point> cpts;

	for (int i=0; i<totalSize; i++)
	{
		infile >> innerSize;
		for (int j=0; j<innerSize; j++)
		{
			infile >> x >> y;
			cpts.push_back(Point(x, y));
		}
	}

	return cpts;
}


//int** getHistogramFromContourPts(vector<Point>& contourPts)
vector< vector<int> > getHistogramFromContourPts(vector<Point>& contourPts)
{
	// get the average distance
	double avgDistance = 0;
	int numPairs = 0;

	for (int i=0; i<contourPts.size(); i++)
	{
		Point p1 = contourPts[i];

		for (int j=i+1; j<contourPts.size(); j++)
		{
			Point p2 = contourPts[j];

			double distance = dist(p1, p2);
			avgDistance += distance;
			numPairs++;
		}
	}

	avgDistance = avgDistance / numPairs;

	double maxLogDistance = -9999999.9;
	double minLogDistance = 999999.9;
	for (int i=0; i<contourPts.size(); i++)
	{
		Point p1 = contourPts[i];

		for (int j=i+1; j<contourPts.size(); j++)
		{
			Point p2 = contourPts[j];

			double distance = dist(p1, p2);
			distance /= avgDistance;

			distance = log(distance);
			if (distance > maxLogDistance) maxLogDistance = distance;
			if (distance < minLogDistance) minLogDistance = max(0.0, distance);
		}
	}

	double radialBound = maxLogDistance + (maxLogDistance - minLogDistance) * 0.01;
	double intervalSize = radialBound / 5.0;
	double angleSize = M_PI / 6.0;

	cout << radialBound << endl;

//	int **histogram;
//	histogram = (int **) malloc(sizeof(int *) * contourPts.size());
//	for (int i=0; i<contourPts.size(); i++)
//	{
//		histogram[i] = (int *) malloc(sizeof(int) * 60);
//		for (int j=0; j<60; j++)
//		{
//			histogram[i][j] = 0;
//		}
//	}
    vector< vector<int> > histogram;
    for(int i = 0; i < contourPts.size(); ++i)
    {
        vector<int> tmp;
        for(int j = 0; j < 60; j ++)
        {
            tmp.push_back(0);
        }
        histogram.push_back(tmp);
    }

	for (int i=0; i<contourPts.size(); i++)
	{
		Point p1 = contourPts[i];

		for (int j=0; j<contourPts.size(); j++)
		{
			if (i == j) continue;

			Point p2 = contourPts[j];

			double ang = angle(p1, p2);
			int angleBin = (int) floor(ang / angleSize);

			double distance = dist(p1, p2);
			distance /= avgDistance;

			distance = max(0.0, log(distance));
			int distanceBin = (int) floor(distance / intervalSize);

			//if (distanceBin * 12 + angleBin >= 60) cout << distanceBin * 12 + angleBin << endl;

			histogram[i][distanceBin * 12 + angleBin]++;
		}
	}

    return histogram;
}

void cleanup(int **hist1, int size1, int **hist2, int size2)
{
    int s = max(size1, size2);

    for (int i=0; i<s; i++)
    {
        if (i < size1) free(hist1[i]);
        if (i < size2) free(hist2[i]);
    }

    free(hist1);
    free(hist2);

    return;
}

//void getChiStatistic(double **stats, int **histogram1, int size1, int **histogram2, int size2)
void getChiStatistic(vector< vector<double> >& stats,
                     vector< vector<int> >& histogram1, int size1,
                     vector< vector<int> >& histogram2, int size2)
{
	int size = max(size1, size2);

	for (int i=0; i<size; i++)
	{
		for (int j=0; j<size; j++)
		{
			if (i >= size1 || j >= size2)
			{
				stats[i][j] = OUTLIER_THRESHOLD;
				continue;
			}

			double summation = 0;
			for (int k=0; k<60; k++)
			{
				double diff = histogram1[i][k] - histogram2[j][k];
				double sum = histogram1[i][k] + histogram2[j][k];
				summation = summation + diff*diff / sum;
			}
			summation = summation / 2;

			stats[i][j] = summation;
		}
	}

	return;
}

pair<Point,Point> getMinMax(vector<Point>& cpts1, vector<Point>& cpts2)
{
	int mx = 9999999, my = 99999999;
	int Mx = -999999999, My = -999999999;

	int s = max(cpts1.size(), cpts2.size());
	for (int i=0; i<s; i++)
	{
		if (i < cpts1.size())
		{
			if (mx > cpts1[i].x) mx = cpts1[i].x;
			if (my > cpts1[i].y) my = cpts1[i].y;

			if (Mx < cpts1[i].x) Mx = cpts1[i].x;
			if (My < cpts1[i].y) My = cpts1[i].y;
		}

		if (i < cpts2.size())
		{
			if (mx > cpts2[i].x) mx = cpts2[i].x;
			if (my > cpts2[i].y) my = cpts2[i].y;

			if (Mx < cpts2[i].x) Mx = cpts2[i].x;
			if (My < cpts2[i].y) My = cpts2[i].y;
		}
	}

	return make_pair(Point(mx,my), Point(Mx,My));
}

vector <Point> getSampledPoints(vector<Point>& v, int sr)
{
	vector<Point> result;
	for (int i=0; i<v.size(); i+=sr)
		result.push_back(v[i]);

	return result;
}

int main(int argc, char **argv)
{/*
	if (argc != 3)
	{
		cout << "usage :: ./match <img1_contours> <img2_contours>" << endl;
		return -1;
    }*/

//	double **chiStatistics;

    vector< vector<double> > chiStatistics;

    Mat img1 = imread("../img/color-cam0-f000.jpg");
    Mat img2 = imread("../img/color-cam2-f000.jpg");

    resize(img1,img1,cv::Size(int(img1.cols/2),int(img1.rows/2)));
    resize(img2,img2,cv::Size(int(img2.cols/2),int(img2.rows/2)));

//    Mat sh = img1.clone();

    cvtColor(img1,img1,CV_RGB2GRAY);
    cvtColor(img2,img2,CV_RGB2GRAY);
    Mat edge1,edge2;
    Canny(img1,edge1,40,120,3);
    Canny(img2,edge2,40,120,3);
    int rows = img1.rows;
    int cols = img1.cols;

    threshold(edge1,edge1,40,255,CV_THRESH_BINARY);
    threshold(edge2,edge2,40,255,CV_THRESH_BINARY);

//    imshow("ed1",edge1);
//    imshow("ed2",edge2);
//    waitKey(0);


    vector <Point> contourPts1; // = getContourPtsFromFile(argv[1]);
    vector <Point> contourPts2; // = getContourPtsFromFile(argv[2]);


    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            if(edge1.at<uchar>(i,j) > 200)
            {
                contourPts1.push_back(cv::Point(j,i));
//                circle(sh,Point(j,i),2,Scalar(0,255,0),1,8);
            }

            if(edge2.at<uchar>(i,j) > 200)
            {
                contourPts2.push_back(cv::Point(j,i));
            }
        }
    }

    contourPts1 = getSampledPoints(contourPts1,10);
    contourPts2 = getSampledPoints(contourPts2,10);


//    imshow("sh1",sh);
//    waitKey(0);
    cout << contourPts1.size() << "  " << contourPts2.size() <<endl;

//    return 0;


	int minsize = min(contourPts1.size(), contourPts2.size());
	int maxsize = max(contourPts1.size(), contourPts2.size());
	int samplingRate = maxsize / minsize;
	if (contourPts1.size() == maxsize)
		contourPts1 = getSampledPoints(contourPts1, samplingRate);
	else
		contourPts2 = getSampledPoints(contourPts2, samplingRate);

	cout << contourPts1.size() << " " << contourPts2.size() << endl;

	pair<Point,Point> mm = getMinMax(contourPts1, contourPts2);
	int sizex = mm.second.x + (int) ((mm.second.x - mm.first.x) * 0.1);
	int sizey = mm.second.y + (int) ((mm.second.y - mm.first.y) * 0.1);

//	int **histogram1 = getHistogramFromContourPts(contourPts1);
    vector< vector<int> > histogram1 = getHistogramFromContourPts(contourPts1);
	int size1 = contourPts1.size();
    cout <<" size1 = " << size1 << endl;

//	int **histogram2 = getHistogramFromContourPts(contourPts2);
    vector< vector<int> > histogram2 = getHistogramFromContourPts(contourPts2);
	int size2 = contourPts2.size();
    cout << "size2 = " << size2 << endl;

	int size = max(size1, size2);

    for(int i = 0; i < size; ++i)
    {
        vector<double> tmp;
        for(int j = 0; j < size; ++j)
        {
            tmp.push_back(-1.0);
        }
        chiStatistics.push_back(tmp);
    }


	getChiStatistic(chiStatistics, histogram1, size1, histogram2, size2);

//    double *u, *v;
//    int *colsol, *rowsol;

//    rowsol = (int *) malloc(sizeof(int) * size);
//    colsol = (int *) malloc(sizeof(int) * size);
//    u = (double *) malloc(sizeof(double) * size);
//    v = (double *) malloc(sizeof(double) * size);

    vector<double> u,v;
    vector<int> colsol,rowsol;
    for(int i = 0; i < size; ++i)
    {
        u.push_back(0);
        v.push_back(0);
        rowsol.push_back(0);
        colsol.push_back(0);
    }



	struct timeval t1, t2;

	gettimeofday(&t1, NULL);
	lap(size, chiStatistics, rowsol, colsol, u, v);
	gettimeofday(&t2, NULL);

	cout << "time taken :: " << (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0 << " s" << endl;


    double max_dist = -1, min_dist = 1000;

//	Mat resimg = Mat::zeros(sizey, sizex, CV_8UC3);
//    Mat resimg2 = Mat::zeros(sizey, sizex, CV_8UC3);
//	int count = 0;
	for (int i=0; i<size; i++)
	{
		int j = rowsol[i];
		if (i>=contourPts1.size() || j>=contourPts2.size())
			continue;

		Point p1 = contourPts1[i];
		Point p2 = contourPts2[j];

        double tmp_dist = dist(p1,p2);
        if(tmp_dist > max_dist)
        {
            max_dist = tmp_dist;
        }
        if(tmp_dist < min_dist)
        {
            min_dist = tmp_dist;
        }

//		if (count % 4 == 0)
//		{
//			circle(resimg, p1, 1, RED, 4, 8);
//			circle(resimg, p2, 1, BLUE, 4, 8);
//            line(resimg, p1, p2, WHITE, 0.2, 8);
//		}
//		count = count + 1;
	}



    vector<int> link1,link2;
    for(int i = 0; i < size; ++i)
    {
        int j = rowsol[i];
        if (i>=contourPts1.size() || j>=contourPts2.size())
            continue;

        Point p1 = contourPts1[i];
        Point p2 = contourPts2[j];

        double tmp_dist = dist(p1,p2);

        if(tmp_dist < 1.7 * min_dist || tmp_dist < max_dist/10)
        {
            link1.push_back(i);
            link2.push_back(j);
        }
    }




//    for(int i = 0; i < link1.size() ; ++i )
//    {

//        Point p1 = contourPts1[ link1[i] ];
//        Point p2 = contourPts2[ link2[i] ];
//        circle(resimg2, p1, 1, RED, 4, 8);
//        circle(resimg2, p2, 1, BLUE, 4, 8);
//        line(resimg2, p1, p2, WHITE, 0.2, 8);
//    }



//    free(colsol); free(rowsol);
//    free(u); free(v);


//    imshow("mapping", resimg);
//    imshow("map_clear",resimg2);

//    imwrite("./matching.jpg", resimg);
//    imwrite("./match_clear.jpg", resimg2);
//    waitKey(0);

	return 0;
}
