#define _DEBUG
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;

# define pi 3.14159265358979323846

void genTransform(DMatch match, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, double &e, double &theta, double &tx, double &ty)
{
    e = (double) (1.0*keypoints2[match.queryIdx].size)/keypoints1[match.trainIdx].size; 
    theta = keypoints2[match.queryIdx].angle - keypoints1[match.trainIdx].angle;
    tx = keypoints2[match.queryIdx].pt.x - e*(keypoints1[match.trainIdx].pt.x * cos(theta*pi/180) - keypoints1[match.trainIdx].pt.y * sin(theta*pi/180)); 
    ty = keypoints2[match.queryIdx].pt.y - e*(keypoints1[match.trainIdx].pt.x * sin(theta*pi/180) + keypoints1[match.trainIdx].pt.y * cos(theta*pi/180));
}

int computeConsensus(vector<DMatch> &matches, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<int> &selected, double e, double theta, double tx, double ty)
{
    int cons = 0;
    selected.clear();
    for (int i=0; i<(int)matches.size(); i++)
    {
    	double x1 = keypoints1[matches[i].trainIdx].pt.x;
    	double y1 = keypoints1[matches[i].trainIdx].pt.y;
    	double x2 = keypoints2[matches[i].queryIdx].pt.x;
    	double y2 = keypoints2[matches[i].queryIdx].pt.y;
    	double x2t = e*(cos(theta*pi/180)*x1-sin(theta*pi/180)*y1)+tx; 
    	double y2t = e*(sin(theta*pi/180)*x1+cos(theta*pi/180)*y1)+ty; 
    	double ex = sqrt(pow(x2-x2t,2)+pow(y2-y2t,2)); 
    	if (ex < 40)
    	{
    		selected.push_back(i);
    		cons++;
    	}
    }
    return cons;
}

bool ransac(vector<DMatch> &matches, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &accepted)
{
	vector<int> selected;
	double e, theta, tx, ty;
	for (int j=1; j<=200; j++){
		int ind = rand() % matches.size();
		genTransform(matches[ind], keypoints1, keypoints2, e, theta, tx, ty);
		int consensus = computeConsensus(matches, keypoints1, keypoints2, selected, e, theta, tx, ty);
		if (consensus > 100)
		{
			for (int i=0; i<(int)selected.size(); i++)
				accepted.push_back(matches[selected[i]]);
			return true;
		}
	}
	return false;
}

int main(void)
{
	Mat input1, input2; // Crear matriz de OpenCV
	input1 = imread("uch049a.jpg"); //Leer imagen
	input2 = imread("uch049b.jpg"); //Leer imagen

	if(input1.empty() || input2.empty()) // No encontro la imagen
	{
		cout<<"Imagen no encontrada"<<endl;
		return 1; // Sale del programa anormalmente
	}

	SurfFeatureDetector detector;
	vector<KeyPoint> keypoints1;
	detector.detect(input1, keypoints1);

	vector<KeyPoint> keypoints2;
	detector.detect(input2, keypoints2);

	SurfDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(input1, keypoints1, descriptors1);
	extractor.compute(input2, keypoints2, descriptors2);

	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(descriptors2, descriptors1, matches);

	vector<DMatch> accepted;
	ransac(matches, keypoints1, keypoints2, accepted);

	// drawing the results
	Mat output;
	drawKeypoints(input1, keypoints1, output);
	imshow("keypoints1", output);
	imwrite("pi00a.jpg",output);
	drawKeypoints(input2, keypoints2, output);
	imshow("keypoints2", output);
	imwrite("pi00b.jpg", output);

	//imwrite("sift_result.jpg", output);
	namedWindow("matches", 1);
	Mat img_matches;
	drawMatches(input2, keypoints2, input1, keypoints1, matches, img_matches);
	imshow("matches", img_matches);
	imwrite("matches5.jpg", img_matches);

	Mat img_accepted;
	drawMatches(input2, keypoints2, input1, keypoints1, accepted, img_matches);
	imshow("accepted", img_matches);
	imwrite("accepted8.jpg", img_matches);

	cout << "Presione ENTER en una ventana o CTRL-C para salir" << endl;
	waitKey(0);

	return 0;
}
