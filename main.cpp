#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/viz/vizcore.hpp"
#include "opencv2/viz/viz3d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::viz;
using namespace std;

struct Colours { int red; int green; int blue; };
struct dataType { Point3d point; int red; int green; int blue; };
typedef dataType SpacePoint;
typedef struct PCT_STRUCT { Point2d *P2; Point3d *P3; }Entry;
void ratioTest(std::vector<std::vector<cv::DMatch>> &matches, std::vector<cv::DMatch> &goodMatches);
void symmetryTest(const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2, std::vector<cv::DMatch>& symMatches);
Mat_<double> IterativeTriangulation(Point3d u, Matx34d P, Point3d u1, Matx34d P1);
Mat_<double> LinearLSTriangulation(Point3d u, Matx34d P, Point3d u1, Matx34d P1);
vector<SpacePoint> triangulation(std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2, Mat &K, Matx34d &P, Matx34d & P1, vector<SpacePoint> pointCloud);
Matx34d tableProcess(Matx34d P1, vector<cv::Point2f> newKeyPoints, vector<cv::Point2f> oldKeyPoints, Mat K);
void downsample(Mat *image);

vector<Entry> table;
int entry_num = 0;

int main(int argc, char** argv)
{
	string prefix = "00";
	string extension = ".png";
	string imageName1, imageName2;
	int pictureNumber1 = 0;
	int pictureNumber2 = 1;
	string stringpicturenumber1 = static_cast<ostringstream*>(&(ostringstream() << pictureNumber1))->str();
	string stringpicturenumber2 = static_cast<ostringstream*>(&(ostringstream() << pictureNumber2))->str();
	imageName1 = prefix + stringpicturenumber1 + extension;
	imageName2 = prefix + stringpicturenumber2 + extension;

	bool firstTwoImages = true;
	Mat K;
	Matx34d P;
	Matx34d P1;
	vector<SpacePoint> pointCloud;
	vector<Colours> colours;

	while (pictureNumber1 < 9)
	{
		cout << "Load Base Image Files" << endl;
		Mat BaseImageLeft = imread(imageName1, -1);
		Mat BaseImageRight = imread(imageName2, -1);
		downsample(&BaseImageLeft);
		downsample(&BaseImageRight);

		if (!BaseImageLeft.data || !BaseImageRight.data)
			printf(" --(!) Error reading images \n");

		//-- Step 1: Detect the keypoints using SURF Detector --//
		cout << "Detect Feature Points" << endl;
		cv::Ptr<SURF> detector = SURF::create(10);
		std::vector<KeyPoint> keypoints_1, keypoints_2;
		detector->detect(BaseImageLeft, keypoints_1);
		detector->detect(BaseImageRight, keypoints_2);


		//-- Step 2: Calculate descriptors (feature vectors) --//
		cout << "Extract Feature Points" << endl;
		cv::Ptr<SURF> extractor = SURF::create();
		cv::Mat descriptors_1, descriptors_2;
		extractor->compute(BaseImageLeft, keypoints_1, descriptors_1);
		extractor->compute(BaseImageRight, keypoints_2, descriptors_2);


		//-- Step 3: Matching descriptor vectors using FLANN matcher --//
		cout << "Match Feature Points" << endl;
		FlannBasedMatcher matcher;
		std::vector<std::vector<cv::DMatch>> matches1, matches2;
		std::vector<cv::DMatch> goodMatches1, goodMatches2, goodMatches, outMatches;
		matcher.knnMatch(descriptors_1, descriptors_2, matches1, 2); // find 2 nearest neighbours, match.size() = query.rowsize()
		matcher.knnMatch(descriptors_2, descriptors_1, matches2, 2);
		ratioTest(matches1, goodMatches1);
		ratioTest(matches2, goodMatches2);
		symmetryTest(goodMatches1, goodMatches2, goodMatches); // double check

		if (firstTwoImages)
		{
			cv::Mat fundamentalMatrix;
			std::vector<cv::Point2f> points1, points2;

			if (goodMatches.size() < 30)
				cerr << "Error: Not Enough Matches" << endl;
			else
			{
				for (std::vector<cv::DMatch>::const_iterator it = goodMatches.begin(); it != goodMatches.end(); ++it)
				{
					// Get the position of left keypoints
					float x = keypoints_1[it->queryIdx].pt.x;
					float y = keypoints_1[it->queryIdx].pt.y;
					points1.push_back(cv::Point2f(x, y));

					// Get the position of right keypoints
					x = keypoints_2[it->trainIdx].pt.x;
					y = keypoints_2[it->trainIdx].pt.y;
					points2.push_back(cv::Point2f(x, y));
				}

				cout << "Compute Fundamental Matrix" << endl;
				std::vector<uchar> inliers(points1.size(), 0);
				fundamentalMatrix = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2), inliers, CV_FM_RANSAC, 3.0, 0.8); // Compute fundamental matrix using RANSAC
				// extract the surviving (inliers) matches
				std::vector<cv::DMatch>::const_iterator	itM = goodMatches.begin();
				for (std::vector<uchar>::const_iterator itIn = inliers.begin(); itIn != inliers.end(); ++itIn, ++itM)
					if (*itIn) // it is a valid match
						outMatches.push_back(*itM);
				if (outMatches.size() < 25)
					cerr << "Error: Not Enough Matches" << endl;

				points1.clear();
				points2.clear();
				for (std::vector<cv::DMatch>::const_iterator it = outMatches.begin(); it != outMatches.end(); ++it)
				{
					// Get the position of left keypoints
					float x = keypoints_1[it->queryIdx].pt.x;
					float y = keypoints_1[it->queryIdx].pt.y;
					points1.push_back(cv::Point2f(x, y));

					// Get the position of right keypoints
					x = keypoints_2[it->trainIdx].pt.x;
					y = keypoints_2[it->trainIdx].pt.y;
					points2.push_back(cv::Point2f(x, y));
				}
				fundamentalMatrix = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2), CV_FM_8POINT);
			}


			//-- Step 4: Get the color of the matched points --//
			cout << "Get Point Colors" << endl;

			for (int i = 0; i < points1.size(); i++)
			{
				int x = int(points1.at(i).x + 0.5);
				int y = int(points1.at(i).y + 0.5);
				Point3_<uchar> *p = BaseImageLeft.ptr<Point3_<uchar>>(y, x);
				Colours pointColour;
				pointColour.blue = int(p->x);
				pointColour.green = int(p->y);
				pointColour.red = int(p->z);
				colours.push_back(pointColour);
			}

			// -- Step 5 Triangulation --//
			cout << "Triangulation" << endl;
			double pX = BaseImageLeft.cols / 2.0;
			double pY = BaseImageRight.rows / 2.0;
			K = (Mat_<double>(3, 3) << 1000, 0, pX, 0, 1000, pY, 0, 0, 1);

			Mat_<double> E = K.t() * fundamentalMatrix * K; // E = (K').transpose() * F * K

			SVD svd(E, SVD::MODIFY_A);
			Matx33d W(0, -1, 0, 1, 0, 0, 0, 0, 1); // equation $9.13 page 258
			Matx33d Wt(0, 1, 0, -1, 0, 0, 0, 0, 1);

			Mat_<double> R1 = svd.u * Mat(W) * svd.vt; // equation $9.14 page 258
			Mat_<double> R2 = svd.u * Mat(Wt) * svd.vt;
			Mat_<double> t1 = svd.u.col(2); // t = U(0, 0, 1).transpose() = u3 page 259
			Mat_<double> t2 = -svd.u.col(2);

			Mat Ptemp = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
			P = Matx34d(Ptemp);
			// TODO: need to make sure det(P1si) > 0
			Mat P1s1 = (Mat_<double>(3, 4) << R1(0, 0), R1(0, 1), R1(0, 2), t2(0), R1(1, 0), R1(1, 1), R1(1, 2), t2(1), R1(2, 0), R1(2, 1), R1(2, 2), t2(2));
			P1 = Matx34d(P1s1);

			pointCloud = triangulation(points1, points2, K, P, P1, pointCloud);


			// -- Step 6 Registration point cloud --//
			int threeD_Start = pointCloud.size() - points2.size();

			for (int i = 0; i < points2.size(); i++)
			{
				Point2d *twoD = (Point2d *)malloc(sizeof(Point2d));
				Point3d *threeD = (Point3d *)malloc(sizeof(Point3d));

				twoD->x = points2.at(i).x;
				twoD->y = points2.at(i).y;

				threeD->x = pointCloud.at(threeD_Start + i).point.x;
				threeD->y = pointCloud.at(threeD_Start + i).point.y;
				threeD->z = pointCloud.at(threeD_Start + i).point.z;

				if (entry_num >= table.size())
					table.resize(500 + table.size());

				Entry e;
				e.P2 = twoD;
				e.P3 = threeD;
				table[entry_num] = e;
				entry_num++;
			}

			firstTwoImages = false;

			cout << "Complete processing " << imageName1 << "and " << imageName2 << "\n" << endl;
			/*
			cout << "Write Point Cloud" << endl;
			ofstream outfile("test.ply");
			outfile << "ply\n" << "format ascii 1.0\n" << "element face 0\n";
			outfile << "property list uchar int vertex_indices\n" << "element vertex " << pointCloud.size() << "\n";
			outfile << "property float x\n" << "property float y\n" << "property float z\n";
			outfile << "property uchar diffuse_red\n" << "property uchar diffuse_green\n" << "property uchar diffuse_blue\n";
			outfile << "end_header\n" << "0 0 0 255 0 0\n";;
			for (int i = 0; i < pointCloud.size(); i++)
			{
				outfile << pointCloud.at(i).point.x << " ";
				outfile << pointCloud.at(i).point.y << " ";
				outfile << pointCloud.at(i).point.z << " ";
				outfile << colours.at(i).blue << " ";
				outfile << colours.at(i).green << " ";
				outfile << colours.at(i).red << " ";
				outfile << "\n";
			}

			outfile.close();
			*/
		}
		else
		{
			cv::Mat fundamentalMatrix;
			std::vector<cv::Point2f> points1, points2;
			cout << "Compute Fundamental Matrix" << endl;
			if (goodMatches.size() < 30)
				cerr << "Error: Not Enough Matches" << endl;
			else
			{
				for (std::vector<cv::DMatch>::const_iterator it = goodMatches.begin(); it != goodMatches.end(); ++it)
				{
					float x = keypoints_1[it->queryIdx].pt.x;
					float y = keypoints_1[it->queryIdx].pt.y;
					points1.push_back(cv::Point2f(x, y));

					x = keypoints_2[it->trainIdx].pt.x;
					y = keypoints_2[it->trainIdx].pt.y;
					points2.push_back(cv::Point2f(x, y));
				}

				std::vector<uchar> inliers(points1.size(), 0);
				fundamentalMatrix = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2), inliers, CV_FM_RANSAC, 3.0, 0.8);

				std::vector<cv::DMatch>::const_iterator	itM = goodMatches.begin();
				for (std::vector<uchar>::const_iterator itIn = inliers.begin(); itIn != inliers.end(); ++itIn, ++itM)
					if (*itIn)
						outMatches.push_back(*itM);
				if (outMatches.size() < 25)
					cerr << "Error: Not Enough Matches" << endl;

				points1.clear();
				points2.clear();
				for (std::vector<cv::DMatch>::const_iterator it = outMatches.begin(); it != outMatches.end(); ++it)
				{
					float x = keypoints_1[it->queryIdx].pt.x;
					float y = keypoints_1[it->queryIdx].pt.y;
					points1.push_back(cv::Point2f(x, y));

					x = keypoints_2[it->trainIdx].pt.x;
					y = keypoints_2[it->trainIdx].pt.y;
					points2.push_back(cv::Point2f(x, y));
				}
			}

			// -- Step 4 Triangulation --//
			cout << "Triangulation" << endl;
			P = P1;
			P1 = tableProcess(P1, points2, points1, K);

			pointCloud = triangulation(points1, points2, K, P, P1, pointCloud);

			//-- Step 5: Get the color of the matched points --//
			cout << "Get Point Colors" << endl;
			for (int i = 0; i < points1.size(); i++)
			{
				int x = int(points1.at(i).x + 0.5);
				int y = int(points1.at(i).y + 0.5);
				Point3_<uchar> *p = BaseImageLeft.ptr<Point3_<uchar>>(y, x);
				Colours pointColour;
				pointColour.blue = int(p->x);
				pointColour.green = int(p->y);
				pointColour.red = int(p->z);
				colours.push_back(pointColour);
			}

			int threeD_Start = pointCloud.size() - points2.size();

			for (int i = 0; i < points2.size(); i++)
			{
				bool found = false;

				for (int j = 0; j < entry_num; j++)
					if (table[j].P2->x == points2.at(i).x && table[j].P2->y == points2.at(i).y)
						found = true;
				
				if (!found)
				{
					Point2d *twoD = (Point2d *)malloc(sizeof(Point2d));
					Point3d *threeD = (Point3d *)malloc(sizeof(Point3d));

					twoD->x = points2.at(i).x;
					twoD->y = points2.at(i).y;

					threeD->x = pointCloud.at(threeD_Start + i).point.x;
					threeD->y = pointCloud.at(threeD_Start + i).point.y;
					threeD->z = pointCloud.at(threeD_Start + i).point.z;

					if (entry_num >= table.size())
						table.resize(500 + table.size());

					Entry e;
					e.P2 = twoD;
					e.P3 = threeD;
					table[entry_num] = e;
					entry_num++;
				}

			}
		}

		cout << "Complete processing " << imageName1 << "and " << imageName2 << "\n" << endl;

		pictureNumber1++;
		pictureNumber2++;
		stringpicturenumber1 = static_cast<ostringstream*>(&(ostringstream() << pictureNumber1))->str();
		stringpicturenumber2 = static_cast<ostringstream*>(&(ostringstream() << pictureNumber2))->str();
		imageName1 = prefix + stringpicturenumber1 + extension;
		imageName2 = prefix + stringpicturenumber2 + extension;
	}

	//-- Step 7 Write point cloud into ply file --//
	cout << "Write Point Cloud" << endl;
	ofstream outfile("pointcloud.ply");
	outfile << "ply\n" << "format ascii 1.0\n" << "element face 0\n";
	outfile << "property list uchar int vertex_indices\n" << "element vertex " << pointCloud.size() << "\n";
	outfile << "property float x\n" << "property float y\n" << "property float z\n";
	outfile << "property uchar diffuse_red\n" << "property uchar diffuse_green\n" << "property uchar diffuse_blue\n";
	outfile << "end_header\n" << "0 0 0 255 0 0\n";;
	for (int i = 0; i < pointCloud.size(); i++)
	{
		outfile << pointCloud.at(i).point.x << " ";
		outfile << pointCloud.at(i).point.y << " ";
		outfile << pointCloud.at(i).point.z << " ";
		outfile << colours.at(i).blue << " ";
		outfile << colours.at(i).green << " ";
		outfile << colours.at(i).red << " ";
		outfile << "\n";
	}
	outfile.close();
	cout << "Done" << endl;

	return 0;
}

void ratioTest(std::vector<std::vector<cv::DMatch>> &matches, std::vector<cv::DMatch> &goodMatches)
{
	for (std::vector<std::vector<cv::DMatch>>::iterator	matchIterator = matches.begin(); matchIterator != matches.end(); ++matchIterator)
		if (matchIterator->size() > 1)
			if ((*matchIterator)[0].distance < (*matchIterator)[1].distance * 0.8) // check distance ratio
				goodMatches.push_back((*matchIterator)[0]);
}

void symmetryTest(const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2, std::vector<cv::DMatch>& symMatches)
{
	symMatches.clear();
	for (vector<DMatch>::const_iterator matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1)
		for (vector<DMatch>::const_iterator matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2)
			if ((*matchIterator1).queryIdx == (*matchIterator2).trainIdx && (*matchIterator2).queryIdx == (*matchIterator1).trainIdx)
				symMatches.push_back(DMatch((*matchIterator1).queryIdx, (*matchIterator1).trainIdx, (*matchIterator1).distance));
}

vector<SpacePoint> triangulation(std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2, Mat &K, Matx34d &P, Matx34d & P1, vector<SpacePoint> pointCloud)
{
	// http://www.ics.uci.edu/~dramanan/teaching/cs217_spring09/lec/stereo.pdf
	Mat kInverse = K.inv();

	vector<SpacePoint> tempCloud = pointCloud;
	for (int i = 0; i < points1.size(); i++)
	{
		Point3d point3D1(points1.at(i).x, points1.at(i).y, 1);
		Mat_<double> mapping3D1 = kInverse * Mat_<double>(point3D1); // K.inverse() * (x, y, 1).transpose() = (X, Y, Z).transpose()
		point3D1.x = mapping3D1(0);
		point3D1.y = mapping3D1(1);
		point3D1.z = mapping3D1(2);

		Point3d point3D2(points2.at(i).x, points2.at(i).y, 1);
		Mat_<double> mapping3D2 = kInverse * Mat_<double>(point3D2);
		point3D2.x = mapping3D2(0);
		point3D2.y = mapping3D2(1);
		point3D2.z = mapping3D2(2);

		Mat_<double> X = IterativeTriangulation(point3D1, P, point3D2, P1);

		SpacePoint Location3D;
		Location3D.point.x = X(0);
		Location3D.point.y = X(1);
		Location3D.point.z = X(2);

		tempCloud.push_back(Location3D);
	}

	pointCloud = tempCloud;

	return tempCloud;
}


Mat_<double> IterativeTriangulation(Point3d u, Matx34d P, Point3d u1, Matx34d P1)
{
	double wi = 1, wi1 = 1;
	Mat_<double> X(4, 1);

	Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);
	X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;

	for (int i = 0; i<10; i++) 
	{ //Hartley suggests 10 iterations at most		

		//recalculate weights
		double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
		double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);

		//breaking point
		if (fabsf(wi - p2x) <= 0.0001 && fabsf(wi1 - p2x1) <= 0.0001) 
			break;

		wi = p2x;
		wi1 = p2x1;

		//reweight equations and solve
		Matx43d A(	(u.x*P(2, 0) - P(0, 0)) / wi, (u.x*P(2, 1) - P(0, 1)) / wi, (u.x*P(2, 2) - P(0, 2)) / wi,
					(u.y*P(2, 0) - P(1, 0)) / wi, (u.y*P(2, 1) - P(1, 1)) / wi, (u.y*P(2, 2) - P(1, 2)) / wi,
					(u1.x*P1(2, 0) - P1(0, 0)) / wi1, (u1.x*P1(2, 1) - P1(0, 1)) / wi1, (u1.x*P1(2, 2) - P1(0, 2)) / wi1,
					(u1.y*P1(2, 0) - P1(1, 0)) / wi1, (u1.y*P1(2, 1) - P1(1, 1)) / wi1, (u1.y*P1(2, 2) - P1(1, 2)) / wi1
				 );
		Mat_<double> B =(Mat_<double>(4, 1) <<  -(u.x*P(2, 3) - P(0, 3)) / wi,
												-(u.y*P(2, 3) - P(1, 3)) / wi,
												-(u1.x*P1(2, 3) - P1(0, 3)) / wi1,
												-(u1.y*P1(2, 3) - P1(1, 3)) / wi1
						);

		solve(A, B, X_, DECOMP_SVD);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
	}
	return X;

}

Mat_<double> LinearLSTriangulation(Point3d u, Matx34d P, Point3d u1, Matx34d P1)
{
	//	http://cmp.felk.cvut.cz/cmp/courses/TDV/2012W/lectures/tdv-2012-07-anot.pdf
	//	solve || D*X || = 0

	Matx43d A;
	A <<u.x*P(2, 0) - P(0, 0), u.x*P(2, 1) - P(0, 1), u.x*P(2, 2) - P(0, 2),
		u.y*P(2, 0) - P(1, 0), u.y*P(2, 1) - P(1, 1), u.y*P(2, 2) - P(1, 2),
		u1.x*P1(2, 0) - P1(0, 0), u1.x*P1(2, 1) - P1(0, 1), u1.x*P1(2, 2) - P1(0, 2),
		u1.y*P1(2, 0) - P1(1, 0), u1.y*P1(2, 1) - P1(1, 1), u1.y*P1(2, 2) - P1(1, 2);
	Matx41d B;
	B <<-(u.x*P(2, 3) - P(0, 3)),
		-(u.y*P(2, 3) - P(1, 3)),
		-(u1.x*P1(2, 3) - P1(0, 3)),
		-(u1.y*P1(2, 3) - P1(1, 3));

	Mat_<double> X;
	solve(A, B, X, DECOMP_SVD);

	return X;
}

Matx34d tableProcess(Matx34d P1, vector<cv::Point2f> newKeyPoints, vector<cv::Point2f> oldKeyPoints, Mat K)
{
	vector<Point2d> foundPoints2d;
	vector<Point3d> foundPoints3d;
	vector<KeyPoint> newKeyPoints_notIn;
	vector<KeyPoint> oldKeyPoints_notIn;

	for (int i = 0; i < oldKeyPoints.size(); i++)
	{
		bool found = false;
		int index = 0;
		for (int j = 0; j < entry_num; j++)
			if (table[j].P2->x == oldKeyPoints.at(i).x && table[j].P2->y == oldKeyPoints.at(i).y)
			{
				found = true;
				index = j;
			}

		if (found)
		{
			Point3d newPoint;
			newPoint.x = table[index].P3->x;
			newPoint.y = table[index].P3->y;
			newPoint.z = table[index].P3->z;
			Point2d newPoint2;
			newPoint2.x = newKeyPoints.at(i).x;
			newPoint2.y = newKeyPoints.at(i).y;
			foundPoints3d.push_back(newPoint);
			foundPoints2d.push_back(newPoint2);
			if (entry_num >= table.size())
				table.resize(500 + table.size());

			Entry e;
			e.P2 = &newPoint2;
			e.P3 = &newPoint;
			table[entry_num] = e;
			entry_num++;
		}
	}

	int size = foundPoints3d.size();

	Mat_<double> found3dPoints(size, 3);
	Mat_<double> found2dPoints(size, 2);

	for (int i = 0; i < size; i++)
	{

		found3dPoints(i, 0) = foundPoints3d.at(i).x;
		found3dPoints(i, 1) = foundPoints3d.at(i).y;
		found3dPoints(i, 2) = foundPoints3d.at(i).z;

		found2dPoints(i, 0) = foundPoints2d.at(i).x;
		found2dPoints(i, 1) = foundPoints2d.at(i).y;

	}

	Mat_<double> temp1(found3dPoints);
	Mat_<double> temp2(found2dPoints);

	Mat P0(P1);

	Mat r(P0, Rect(0, 0, 3, 3));
	Mat t(P0, Rect(3, 0, 1, 3));

	Mat r_rog;
	cv::Rodrigues(r, r_rog);


	Mat dist = Mat::zeros(1, 4, CV_32F);
	double _dc[] = { 0, 0, 0, 0 };

	cv::solvePnP(found3dPoints, found2dPoints, K, Mat(1, 4, CV_64FC1, _dc), r_rog, t, false);

	cout << "Got new Camera matrix" << endl;

	Mat_<double> R1(3, 3);
	Mat_<double> t1(t);

	cv::Rodrigues(r_rog, R1);

	Mat camera = (Mat_<double>(3, 4) << R1(0, 0), R1(0, 1), R1(0, 2), t1(0),
										R1(1, 0), R1(1, 1), R1(1, 2), t1(1),
										R1(2, 0), R1(2, 1), R1(2, 2), t1(2));

	return Matx34d(camera);
}

void downsample(Mat *image)
{
	int maxRows = 1800;
	int maxCols = 1600;
	Mat modifyImage = *image;
	int height = modifyImage.rows;
	int width = modifyImage.cols;
	//account for odds
	if (height % 2 != 0)
		height--;
	if (width % 2 != 0)
		width--;
	//form new images:
	Mat evenSize(modifyImage, Rect(0, 0, width - 1, height - 1));
	Mat downSize;
	while (height * width > maxRows * maxCols)
	{
		pyrDown(evenSize, downSize, Size(width / 2, height / 2));
		//set new image to the downsized one
		*image = downSize;
		//do again and account for odds
		height = downSize.rows;
		width = downSize.cols;
		if (height % 2 != 0)
			height--;
		if (width % 2 != 0)
			width--;
		Mat next(downSize, Rect(0, 0, width - 1, height - 1));
		evenSize = next;
	}
}