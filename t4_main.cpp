#define _DEBUG
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

#define Rows 3
#define Cols 16
#define Factor 0.0634936

float skinMean[Rows][Cols] = {
	{73.53, 249.71, 161.68, 186.07, 189.26, 247,150.1, 206.85, 212.78, 234.87, 151.19, 120.52, 192.20, 214.29, 99.57, 238.88},
	{29.94, 233.94, 116.25, 136.62, 98.37, 152.2, 72.66, 171.09, 152.82, 175.43, 97.74, 77.55, 119.62, 136.08, 54.33, 203.18},
	{17.76,217.49, 96.95, 114.4, 51.18, 90.84, 37.76, 156.34, 120.04, 138.94, 74.59, 59.82, 82.32, 87.24, 38.06, 176.91}};

float NoskinMean[Rows][Cols] ={
	{254.37,9.39,96.57,160.44,74.89,121.83,202.18,193.06,51.88,30.88,44.97,236.02,207.86,99.83,135.06,135.96},
	{254.41,8.09,96.95,162.49,63.23,60.88,154.88,201.93,57.14,26.84,85.96,236.27,191.2,148.11,131.92,103.89},
	{253.82,8.52,91.53,159.06,46.33,18.31,91.04,206.55,61.55,25.32,131.95,230.7,164.12,188.17,123.1,66.88}};

float skinWeight[Cols] = {0.0294,0.0331,0.0654,0.0756,0.0554,0.0314,0.0454,0.0469,0.0956,0.0763,0.11,0.0676,0.0755,0.05,0.0667,0.0749};

float NoskinWeight[Cols] = {0.0637,0.0516,0.0864,0.0636,0.0747,0.0365,0.0349,0.0649,0.0656,0.1189,0.0362,0.0849,0.0368,0.0389,0.0943,0.0477};

float skinCovariance[Rows][Cols] = {
	{765.4,39.94,291.03,274.95,633.18,65.23,408.63,530.08,160.57,163.8,425.4,330.45,152.76,204.9,448.13,178.38},
	{121.44,154.44,60.48,64.6,222.4,691.53,200.77,155.08,84.52,121.57,73.56,70.34,92.14,140.17,90.18,156.27},
	{112.8,396.05,162.85,198.27,250.69,609.92,257.57,572.79,243.9,279.22,175.11,151.82,259.15,270.19,151.29,404.99}};

float NoskinCovariance[Rows][Cols] = {
	{2.77,46.84,280.69,335.98,414.84,2502.24,957.42,562.88,344.11,222.07,651.32,225.03,494.04,955.88,350.35,806.44},
	{2.81,33.59,156.79,115.89,245.95,1383.53,1766.94,190.23,191.77,118.65,840.52,117.29,237.69,654.95,130.3,642.2},
	{5.46,32.48,436.58,591.24,361.27,237.18,1582.52,447.28,433.4,182.41,963.67,331.95,533.52,916.7,388.43,350.36}};

int VP = 0;
int FP = 0;	
int VN = 0;
int FN = 0;

float e1;
float e2;

float Determinante1[Cols];
float Determinante2[Cols];


void iniciar(){
	for (int i=0; i<Cols; i++){
		Determinante1[i] = 1.0/sqrt(skinCovariance[0][i]*skinCovariance[1][i]*skinCovariance[2][i]);
		Determinante2[i] = 1.0/sqrt(NoskinCovariance[0][i]*NoskinCovariance[1][i]*NoskinCovariance[2][i]);
	}
}


bool Piel(uchar b, uchar g, uchar r, float umbral){
	float Pskin = 0;
	float Pnoskin = 0;
 	for (int i=0; i<Cols; i++){
 		e1 = exp(-0.5*((pow(r-skinMean[0][i],2)/skinCovariance[0][i])+(pow(g-skinMean[1][i],2)/skinCovariance[1][i])+(pow(b-skinMean[2][i],2)/skinCovariance[2][i])));	
 		e2 = exp(-0.5*((pow(r-NoskinMean[0][i],2)/NoskinCovariance[0][i])+(pow(g-NoskinMean[1][i],2)/NoskinCovariance[1][i])+(pow(b-NoskinMean[2][i],2)/NoskinCovariance[2][i])));
 		Pskin = Pskin + skinWeight[i]*Factor*Determinante1[i]*e1;
 		Pnoskin = Pnoskin + NoskinWeight[i]*Factor*Determinante2[i]*e2;
 	}
 	if (Pskin/Pnoskin > umbral)
 		return true;
 	else{
 		//cout << "Pskin ="<< Pskin << endl;
 		//cout << "Pnoskin = "<< Pnoskin << endl;
 		return false;
 	}
 		
}

void Detector(float umbral, Mat teorico, Mat output){
	for (int j=0; j<output.rows; j++){
		for (int i=0; i<output.cols; i++){
			Vec3b pixel = output.at<Vec3b>(j,i);
			uchar blue = pixel.val[0];
			uchar green = pixel.val[1];
			uchar red = pixel.val[2];
			bool piel = Piel(blue,green,red,umbral);
			if (!piel){
				output.at<Vec3b>(j,i)[0] = 255;
				output.at<Vec3b>(j,i)[1] = 255;
				output.at<Vec3b>(j,i)[2] = 255;
			}
			if (piel==true){
				if (teorico.at<Vec3b>(j,i)[0] == 255 & teorico.at<Vec3b>(j,i)[1] == 255  & teorico.at<Vec3b>(j,i)[2] == 255){
					FP++;
				}
				else{
					VP++;
				}
			}
			else{
				if (teorico.at<Vec3b>(j,i)[0] == 255 & teorico.at<Vec3b>(j,i)[1] == 255  & teorico.at<Vec3b>(j,i)[2] == 255){
					VN++;
				}
				else{
					FN++;
				}
			}
		}
	}
}


int main(void)
{
	iniciar();
	string line;
	string img;
	string groundtruth;

	ifstream file;
	ofstream roc;
	roc.open("ROC.txt");

	Mat imagen; 
	Mat teorico;
	Mat output;

	float inicio = 4; 
	float paso = 1;
	float final = 4;

	int tamano = ((final-inicio)/paso) +1 ;
	float umbral [tamano];

	float aux=inicio;
	int l = 0;
	while (aux<=final){
		umbral[l] = aux;
		aux = aux+paso;
		l++;
	}

	for (int k=0; k<tamano; k++){
		file.open("img.txt");
		while (getline(file,line)){
			img = line.substr(0,8);
			groundtruth = line.substr(8,8);			
			imagen = imread(img);
			teorico = imread(groundtruth); 
			output = imagen; 
			if(imagen.empty() | teorico.empty()) // No encontro la imagen
			{
				cout<<"Imagen no encontrada"<<endl;
				return 1; // Sale del programa anormalmente
			}
			Detector(umbral[k],teorico, output);
			string name = img+"_"+".jpg";
			imwrite(name, output);
		}

		stringstream aux1;
		stringstream aux2;
		stringstream aux3;
		stringstream aux4;
		stringstream aux5;

		aux1 << VP;
		aux2 << FP;
		aux3 << VN;
		aux4 << FN;
		aux5 << umbral[k];

		string vp = aux1.str();
		string fp = aux2.str();
		string vn = aux3.str();
		string fn = aux4.str();
		string Theta = aux5.str();

		roc << vp <<" "<< fp << " "<< vn << " "<< fn << " "<<Theta << "\n";
		file.close();
		VP = 0;
		FP = 0;
		VN = 0;
		FN = 0;

	}
	roc.close();

	//cout << "Verdaderos  = " << 1.0*VP/(pixeles) << endl;
	//cout << "Falsos positivos = " << 1.0*FP/(pixeles) << endl;

	//imshow("Piel",output);
	//imshow("Teorico",input2);
	

	


	//Mat img_matches;

	//imshow("matches", img_matches);
	//imwrite("matches5.jpg", img_matches);

	//cout << "Presione ENTER en una ventana o CTRL-C para salir" << endl;
	//waitKey(0);

	//return 0;
}
