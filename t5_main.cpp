#define _DEBUG
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <fstream>
#include <string>
#include <cv.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

#define alto 40 
#define ancho 80 
#define bajo  80
#define celdas_x 2
#define celdas_y 2
#define number_male_train 140
#define number_female_train 140
#define number_male_test 60
#define number_female_test 60

const char lookup[256] = {
0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10,
11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15,
16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21,
22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28,
29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33,
58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34,
58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35,
36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40,
58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41,
42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46,
47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57 };

int eye_prom[2];
int Left_eye[2];
int Right_eye[2];

void CropFace(int Left_eye[2], int Right_eye[2], Mat original, Mat new_image2, string line_img){

	eye_prom[0] = (Left_eye[0]+Right_eye[0])/2;
	eye_prom[1] = (Left_eye[1]+Right_eye[1])/2;
  int x = eye_prom[0]-(ancho/2);
  int y = eye_prom[1]-alto;
  int h = (eye_prom[1]-y)+bajo;

	Rect Roi(x,y,ancho,h);

	Mat new_image = original(Roi);
	Size size(original.rows,original.cols);
	resize(new_image,new_image2,size);
  imwrite("Gender/Female/Female_test/"+line_img, new_image2);

}


void histo(const Mat& in, Mat& resultado){
  for (int i=0; i< in.rows;i++){
    for (int j=0; j< in.cols; j++){
      int valor = in.at<unsigned char>(i,j);
      resultado.at<float>(0,lookup[valor]) += 1;
    }
  }
}

void LBP(const Mat& input_lbp, Mat& output_lbp) {
  
    for(int i=1;i<input_lbp.rows-1;i++) {
        for(int j=1;j<input_lbp.cols-1;j++) {
            unsigned char center = input_lbp.at<unsigned char>(i,j);
            unsigned char code = 0;
            code |= (input_lbp.at<unsigned char>(i-1,j-1) >= center) << 7;
            code |= (input_lbp.at<unsigned char>(i-1,j) >= center) << 6;
            code |= (input_lbp.at<unsigned char>(i-1,j+1) >= center) << 5;
            code |= (input_lbp.at<unsigned char>(i,j+1) >= center) << 4;
            code |= (input_lbp.at<unsigned char>(i+1,j+1) >= center) << 3;
            code |= (input_lbp.at<unsigned char>(i+1,j) >= center) << 2;
            code |= (input_lbp.at<unsigned char>(i+1,j-1) >= center) << 1;
            code |= (input_lbp.at<unsigned char>(i,j-1) >= center) << 0;
            output_lbp.at<unsigned char>(i-1,j-1) = code;
        }
    }
}

Mat HLBP(const Mat& imagen_lbp, int grid_x, int grid_y) {

    Mat histogram;
    int paso_x = imagen_lbp.cols/grid_x;
  	int paso_y = imagen_lbp.rows/grid_y;

  	for (int i=0; i<grid_y; i++){
  		for (int j=0; j<grid_x; j++){
  			Mat region_imagen_lbp(imagen_lbp,Range(i*paso_y,(i+1)*paso_y),Range(j*paso_x,(j+1)*paso_x)); //Cuadrante
        Mat resultado = Mat::zeros(1,59,CV_32FC1);
        histo(region_imagen_lbp,resultado);

        resultado /= (int)region_imagen_lbp.total();

        histogram.push_back(resultado);
  		}
  	}
    Mat r1(histogram,Range(0,1),Range(0,59));
    Mat r2(histogram,Range(1,2),Range(0,59));
    Mat r3(histogram,Range(2,3),Range(0,59));
    Mat r4(histogram,Range(3,4),Range(0,59));
    Mat array[] ={r1,r2,r3,r4};
    Mat out;
    hconcat(array,4,out);
    return out;
    
  }


int main(void)
{
  //Para procesar datos, descomentar
  // ifstream Img;
  // ifstream Eye;
  // ifstream Eye_position;

  // string line_img;
  // string line_eye;
  // string line_position;

  // vector<string> eye_position;

  // Img.open("Gender/Female/female_test.txt");
  // Eye.open("Gender/Female/female_eye_test.txt");
  // while (getline(Img,line_img)){
  //     getline(Eye,line_eye);
  //     Eye_position.open(line_eye.c_str());
  //     Eye_position >> Left_eye[0];
  //     Eye_position >> Left_eye[1];
  //     Eye_position >> Right_eye[0];
  //     Eye_position >> Right_eye[1];

  //     Mat imagen_gray = imread("Gender/Female/"+line_img,CV_LOAD_IMAGE_GRAYSCALE);
  //     if(imagen_gray.empty()) // No encontro la imagen
  //     {
  //       cout<<"Imagen no encontrada"<<endl;
  //       return 1; // Sale del programa anormalmente
  //     }

  //     Mat new_image2;
  //     CropFace(Left_eye,Right_eye,imagen_gray,new_image2,line_img);
  //     Eye_position.close();
  // }

  //Para graficar LBP

  //Acá se asume que ya se tienen los datos procesados

  string img_male;
  string img_female;
  string male_test;
  string female_test;
  ifstream img_female_train;
  ifstream img_male_train;
  ifstream img_male_test;
  ifstream img_female_test;

  img_male_train.open("Gender/Male/Male_train/male_process_train.txt");

  Mat Histo;

  while (getline(img_male_train,img_male)){
    Mat img_gray = imread("Gender/Male/Male_train/"+img_male);
    if(img_gray.empty()) // No encontro la imagen
     {
        cout<<"Imagen no encontrada"<<endl;
        return 1; // Sale del programa anormalmente
      }

    Mat img_lbp = Mat::zeros(img_gray.rows-2,img_gray.cols-2,CV_8UC1);
    LBP(img_gray,img_lbp);

    Mat hist = HLBP(img_lbp,celdas_x,celdas_y);
    Histo.push_back(hist);
  }
  img_male_train.close();


  Mat labels_male(number_male_train,1,CV_32FC1,Scalar(1.0));

  img_female_train.open("Gender/Female/Female_train/female_process_train.txt");

  while (getline(img_female_train,img_female)){
    Mat img_gray_female = imread("Gender/Female/Female_train/"+img_female);
    if(img_gray_female.empty()) // No encontro la imagen
     {
        cout<<"Imagen no encontrada"<<endl;
        return 1; // Sale del programa anormalmente
      }
    
    Mat img_lbp_female = Mat::zeros(img_gray_female.rows-2,img_gray_female.cols-2,CV_8UC1);
    LBP(img_gray_female,img_lbp_female);

    Mat hist_female = HLBP(img_lbp_female,celdas_x,celdas_y);
    Histo.push_back(hist_female);
  }

  img_female_train.close();

  Mat labels_female(number_female_train,1,CV_32FC1,Scalar(0.0));

  Mat labelsMat;
  vconcat(labels_male,labels_female,labelsMat);

  //Preparar test

  img_male_test.open("Gender/Male/Male_test/male_process_test.txt");
  

  Mat Histo_male_test;
  Mat Histo_female_test;

  while (getline(img_male_test,male_test)){
    Mat img_gray_male_test = imread("Gender/Male/Male_test/"+male_test);
    if(img_gray_male_test.empty()) // No encontro la imagen
     {
        cout<<"Imagen no encontrada"<<endl;
        return 1; // Sale del programa anormalmente
      }
   
    Mat img_lbp_male_test = Mat::zeros(img_gray_male_test.rows-2,img_gray_male_test.cols-2,CV_8UC1);
    LBP(img_gray_male_test,img_lbp_male_test);

    Mat hist_male_test = HLBP(img_lbp_male_test,celdas_x,celdas_y);
    Histo_male_test.push_back(hist_male_test);
  }
  img_male_test.close();

  img_female_test.open("Gender/Female/Female_test/female_process_test.txt");

   while (getline(img_female_test,female_test)){
    Mat img_gray_female_test = imread("Gender/Female/Female_test/"+female_test);
    if(img_gray_female_test.empty()) // No encontro la imagen
     {
        cout<<"Imagen no encontrada"<<endl;
        return 1; // Sale del programa anormalmente
      }
   
    Mat img_lbp_female_test = Mat::zeros(img_gray_female_test.rows-2,img_gray_female_test.cols-2,CV_8UC1);
    LBP(img_gray_female_test,img_lbp_female_test);

    Mat hist_female_test = HLBP(img_lbp_female_test,celdas_x,celdas_y);
    Histo_female_test.push_back(hist_female_test);
  }

  img_female_test.close();


 //Entreno clasificador


  CvSVMParams params;
  params.svm_type = CvSVM::C_SVC;
  params.kernel_type = CvSVM::RBF;
  //params.kernel_type = CvSVM::LINEAR;
  params.gamma =3.5;
  params.C = 15;

  params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 1e-8);


  CvSVM SVM;
  SVM.train(Histo, labelsMat, Mat(), Mat(), params);

//Prediccion
  Mat male_result;
  Mat female_result;

  SVM.predict(Histo_male_test,male_result);
  SVM.predict(Histo_female_test,female_result);

//Estadistica

  int vp_male = sum(male_result)[0];
  int fn_male = number_male_test-vp_male;

  cout << "vp hombre= " << vp_male << " " << "fn hombres = " << fn_male << " "<< "tasa acierto= " << 1.0*vp_male/number_male_test<< endl;


  int vp_female = number_female_test-sum(female_result)[0];
  int fn_female = number_female_test-vp_female ;

   cout << "vp mujer= " << vp_female << " " << "fn mujer= " << fn_female << " " << "tasa acierto = " << 1.0*vp_female/number_female_test << endl;

   int n = SVM.get_support_vector_count();

   cout << "vector soporte= " << n << endl;


}
