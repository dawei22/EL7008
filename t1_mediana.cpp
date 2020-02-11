#define _DEBUG
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <algorithm>

using namespace std;
using namespace cv;

void mediana(Mat input, Mat output)
{
	// POR HACER: programar el filtro de mediana 3x3 aca

	for (int i=1 ; i<input.rows-1 ; i++) //Acá me muevo en la imagen
	{
		for (int j=1 ; j<input.cols-1 ; j++)
		{
			vector<float> vals;
			for (int k=0 ; k<3 ; k++)   // 3x3 dimension de la ventana, me muevo por la misma
			{
				for (int l=0 ; l<3 ; l++)
				{
					vals.push_back(input.at<float>(i+(k-1),j+(l-1))); // Agrego cada valor de la ventana a un vector
				}
			}
			sort(&vals[0],&vals[9]); //Ordeno el vector
			//cout << "La mediana es " << vals[4] << endl;
			output.at<float>(i,j) = vals[4]; //Recupero la mediana
		}
	}



	//vector<float> vals;
	//vals.push_back(1);
	//vals.push_back(2);
	//vals.push_back(2);
	//vals.push_back(4);
	//vals.push_back(4);
	//sort(&vals[0], &vals[5]);
	//cout << "La mediana es " << vals[2] << endl;
}

int main(void)
{
	Mat originalRGB = imread("loro.png",1); //Leer imagen

	if(originalRGB.empty()) // No encontro la imagen
	{
		cout << "Imagen no encontrada" << endl;
		return 1;
	}
	
	Mat original;
	cvtColor(originalRGB, original, CV_BGR2GRAY);

	//imshow("loroG", original);   // Mostrar imagen
	//imwrite("loroG.jpg", original); // Grabar imagen
	
	Mat input;
	original.convertTo(input, CV_32FC1);
	
	Mat output = Mat::zeros(input.rows, input.cols, CV_32FC1);	
	mediana(input, output);

	Mat last;
	output.convertTo(last, CV_8UC1);

	imshow("loroF", last);   // Mostrar imagen
	imwrite("loroF.jpg", last); // Grabar imagen
	cvWaitKey(0); // Pausa, permite procesamiento interno de OpenCV

	return 0; // Sale del programa normalmente
}
