#define _DEBUG
#include <iostream>
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

void ecualizar(Mat input, Mat output)
{
	// POR HACER: programar la ecualizacion de histograma aca
	float hist[256];
	float lookup[256];
	float d = 1.0/(input.rows*input.cols); //Para normalizar el histograma
	for (int i=0; i<256; i++)
		hist[i] = 0;
	for (int r=0; r<input.rows; r++)
	{
		for (int c=0; c<input.cols; c++)
		{
			int ind = input.at<unsigned char>(r,c);
			hist[ind] = hist[ind]+d;
		}
	}
	float sum = 0.0;
	for (int i=0; i<256; i++)
	{
		sum += hist[i];
		lookup[i] = sum*255+0.5;
	}
	for (int i=0; i<input.rows; i++)
	{
		for (int j=0; j<input.cols; j++)
			output.at<unsigned char>(i,j) = lookup[input.at<unsigned char>(i,j)];
	}
	return;
}

int main(void)
{
	Mat originalRGB = imread("huason.jpg",1); //Leer imagen

	if(originalRGB.empty()) // No encontro la imagen
	{
		cout << "Imagen no encontrada" << endl;
		return 1;
	}

	
	Mat original;
	cvtColor(originalRGB, original, CV_BGR2GRAY);

	//imshow("huasoG", original);   // Mostrar imagen
	//imwrite("huasoG.jpg", original);
	
	Mat output = Mat::zeros(original.rows, original.cols, CV_8UC1);
	ecualizar(original, output);

	imshow("huasonE", output);   // Mostrar imagen
	imwrite("huasonE.jpg", output); // Grabar imageEn
	cvWaitKey(0); // Pausa, permite procesamiento intEerno de OpenCV

	return 0; // Sale del programa normalmente
}
