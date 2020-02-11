#define _DEBUG
#include <iostream>
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

void convolucion(Mat input, Mat mask, Mat output)
{
	// POR HACER: Programar la convolucion aca

	int centroX = mask.cols/2;
	int centroY = mask.rows/2;
	for (int m=0; m<input.rows ; m++)
	{
		for (int n=0 ; n<input.cols; n++)  // Para llenar cada pixel del output
		{
			for (int i=0 ; i<mask.rows ; i++)
			{
				int a = mask.rows-1-i; // Voy a recorrer el kernel alrevés, para simular que fue volteado

				for (int j=0 ; j<mask.cols ; j++)
				{
					int b = mask.cols-1-j; // Así, para (i,j) = (0,0), (a,b) = (2,2)

					int Fil = m + (i-centroY);
					int Col = n + (j-centroX);

					// El centro del kernel se ajusta el pixel de salida (m,n). Recorro el kernel (con i-centroY, j-centroX) y 
					//voy verificando si para cada pixel del kernel, existe algún pixel en el input (que no se salga de los bordes) 
					//tal que se pueda realizar la multiplicación.  

					if (Fil>=0 && Fil<input.rows && Col>=0 && Col<input.cols)
						output.at<float>(m,n) += input.at<float>(Fil,Col) * mask.at<float>(a,b);
 
				}
			}
		}
	}
}

int main(void)
{
	Mat originalRGB = imread("gato.jpg"); //Leer imagen

	if(originalRGB.empty()) // No encontro la imagen
	{
		cout << "Imagen no encontrada" << endl;
		return 1;
	}
	
	Mat original;
	cvtColor(originalRGB, original, CV_BGR2GRAY);
	
	Mat input;
	original.convertTo(input, CV_32FC1);

	//imshow("gatoG", original);   // Mostrar imagen
	//imwrite("gatoG.jpg", original); // Grabar imagen
	
	float maskval[9] = {1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,};  // pasa bajo recto
	//float maskval[9] = {-1.0, 0, 1.0,-1.0, 0, 1.0,-1.0, 0, 1.0}; //Prewitt vertical
	//float maskval[3] = {1.0/3, 1.0/3, 1.0/3}; // filtro pasa bajo unidimensional
	//float maskval[9] = {-1.0, -1.0, -1.0, 0, 0, 0, 1.0, 1.0, 1.0}; //Prewitt horizontal
	//float maskval[25] = {1.0/100,4.02/100,7.0/100,4.0/100,1.0/100,4.0/100,20.0/100,33.0/100,20.0/100,4.0/100,7.0/100,33.0/100,55.0/100,33.0/100,7.0/100,4.0/100,20.0/100,33.0/100,20.0/100,4.0/100,1.0/100,4.0/100,7.0/100,4.0/100,1.0/100}; //gaussiano 5x5 s=1 r=2
	Mat mask = Mat(3, 3, CV_32FC1, maskval);

	Mat output = Mat::zeros(input.rows, input.cols, CV_32FC1);	
	convolucion(input, mask, output);
	output = abs(output);

	Mat last;
	output.convertTo(last, CV_8UC1);

	imshow("gatogau", last);   // Mostrar imagen
	imwrite("gatogau.jpg", last); // Grabar imagen
	cvWaitKey(0); // Pausa, permite procesamiento interno de OpenCV

	return 0; // Sale del programa normalmente
}
