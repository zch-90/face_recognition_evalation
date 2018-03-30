// This file is exploited for testing the face detection algorithm
//
// Code exploited by 2016 Feng Wang <feng.wff@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD lisence.
#include <chrono>
#include <cstdlib>
#include <memory>
#include <Windows.h>

#include "boost/make_shared.hpp"
#include "TestFaceDetection.inc.h"

caffe::CaffeBinding* kCaffeBinding = new caffe::CaffeBinding();

using namespace FaceInception;

int CaptureDemo(CascadeCNN cascade) {
  VideoCapture cap(0);
  if (!cap.isOpened()) {
    return -1;
  }
  Mat frame;
  Mat edges;

  bool stop = false;
  Rect_<double> bakFaceRect;
  int bak_time = 0;
  while (!stop) {
    cap >> frame;
    if (frame.empty()) {
      cout << "cannot read from camera!" << endl;
      Sleep(100);
      continue;
    }
    vector<vector<Point2d>> points;
    std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
    double min_face_size = 40;
    auto result = cascade.GetDetection(frame, 12 / min_face_size, { 0.6, 0.7, 0.7 }, true, 0.7, true, points);
    std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
    cout << "detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << endl;
    for (int i = 0; i < result.size(); i++) {
      rectangle(frame, result[i].first, Scalar(255, 0, 0), 4);
      for (int p = 0; p < 5; p++) {
        circle(frame, points[i][p], 2, Scalar(0, 255, 255), -1);
      }
    }
    //resize(image, image, Size(0, 0), 0.5, 0.5);
    imshow("capture", frame);
    waitKey(1);
  }
}
void ScanList(string root_folder, CascadeCNN cascade) {
  std::ifstream infile(root_folder.c_str());
  string filename;
  char this_line[65536];
  int label;

  while (!infile.eof()) {
    memset(this_line, 0, sizeof(this_line));
    infile.getline(this_line, 65536);
    if (strlen(this_line) == 0) continue;
    this_line[strlen(this_line) - 1] = '\0';
    std::stringstream stream;
    stream << this_line;
    stream >> filename >> label;
    cout << filename << endl;
    try {
      Mat image = imread(filename);
      vector<vector<Point2d>> points;
      auto result = cascade.GetDetection(image, 1, { 0.6, 0.7, 0.995 }, true, 0.3, true, points);
      for (int i = 0; i < result.size(); i++) {
        rectangle(image, result[i].first, Scalar(255, 0, 0), 4);
        for (int p = 0; p < 5; p++) {
          circle(image, points[i][p], 2, Scalar(0, 255, 255), -1);
        }
      }
      //resize(image, image, Size(0, 0), 0.5, 0.5);
      imshow("boxes", image);
      waitKey(0);
    }
    catch (std::exception e) {}
  }
}

void TestFDDBPrecision(CascadeCNN& cascade, string fddb_folder = "G:\\FDDB\\", bool save = true, bool show = false) {
  string list_folder = fddb_folder + "FDDB-folds\\";
  int total = 0, match = 0;
  FILE *fpout = fopen((list_folder + "FDDB-RectangleList.txt").c_str(), "w");
  FILE *fp = fopen((list_folder + "FDDB-EllipseList.txt").c_str(), "r");
  char image_filename[255], last_filename[255];
  int count;
  while (!feof(fp)) {
    vector<RotatedRect> rects;
    int this_total = 0, this_match = 0;
    fscanf(fp, "%s\n", image_filename);
    if (strcmp(image_filename, last_filename) == 0) break;
    strcpy(last_filename, image_filename);
    Mat image = imread(fddb_folder + image_filename + ".jpg");
    vector <vector<Point2d>> points;
    auto result = cascade.GetDetection(image, 12.0 / 12.0, { 0.6, 0.7, 0.7 }, true, 0.7, show, points);
    if (save) {
      fprintf(fpout, "%s\n", image_filename);
      fprintf(fpout, "%d\n", result.size());
      for (auto& r : result) {
        //r.first.y -= r.first.height * 0.1;
        //r.first.height *= 1.2;
        //r.first.width *= 1.2;
        //fixRect(r.first, image.size());
        fprintf(fpout, "%f %f %f %f %f\n", r.first.x, r.first.y, r.first.width, r.first.height, r.second);
      }
    }
    fscanf(fp, "%d", &count);
    for (int c = 0; c < count; c++) {
      RotatedRect r;
      fscanf(fp, "%f %f %f %f %f 1", &r.size.width, &r.size.height, &r.angle, &r.center.x, &r.center.y);
      //r.center.y += r.size.height * 0.2;
      r.size.height *= 2;
      r.size.width *= 2;
      r.angle *= 180 / CV_PI;
      bool matched = false;
      for (auto& rect : result) {
        if (IoU(rect.first, r, image.size()) > 0.5) {
          matched = true;
        }
      }
      if (matched) match++, this_match++;
      total++;
      this_total++;
      rects.push_back(r);
      //ellipse(image, r, Scalar(255, 0, 0));
      //rectangle(image, r.boundingRect(), Scalar(255, 0, 0), 1);
    }
    cout << image_filename << " " << this_match << this_total << endl;
    if (show) {
      //if (this_match != this_total) {
      for (auto& r : rects) {
        ellipse(image, r, Scalar(255, 0, 0), 1);
        //rectangle(image,r.boundingRect(), Scalar(255, 0, 0), 1);
      }
      for (auto& p : points) {
        for (int j = 0; j < 5; j++) {
          circle(image, p[j], 2, Scalar(0, 255, 255), -1);
        }
      }
      for (auto& rect : result) {
        rectangle(image, rect.first, Scalar(0, 0, 255), 1);
      }
      imshow("image", image);
      waitKey(0);
      // }
    }
  }
  fclose(fp);
  fclose(fpout);
  cout << "total recall:" << (double)match / (double)total * 100 << "%" << endl;
}

int main(int argc, char* argv[])
{
  //CascadeCNN cascade("G:\\WIDER\\face_detection\\bak3\\cascade_12_memory_nobn1.prototxt", "G:\\WIDER\\face_detection\\bak3\\cascade12-_iter_490000.caffemodel",
  //                   "G:\\WIDER\\face_detection\\bak3\\cascade_24_memory_full.prototxt", "G:\\WIDER\\face_detection\\bak3\\cascade24-_iter_145000.caffemodel",
  //                   "G:\\WIDER\\face_detection\\bak3\\cascade_48_memory_full.prototxt", "G:\\WIDER\\face_detection\\bak3\\cascade48-_iter_225000.caffemodel");
  string model_folder = "D:\\face project\\MTCNN_face_detection_alignment\\code\\codes\\MTCNNv2\\model\\";
  CascadeCNN cascade(model_folder+"det1-memory.prototxt", model_folder + "det1.caffemodel",
                     model_folder + "det1-memory-stitch.prototxt", model_folder + "det1.caffemodel",
                     model_folder + "det2-memory.prototxt", model_folder + "det2.caffemodel",
                     model_folder + "det3-memory.prototxt", model_folder + "det3.caffemodel",
                     model_folder + "det4-memory.prototxt", model_folder + "det4.caffemodel",
                     0);
  vector<Point2d> target_points = { {30.2946,51.6963},{65.5318,51.5014},{48.0252,71.7366},{33.5493,92.3655},{62.7299,92.2041} };
  //CaptureDemo(cascade);

  double min_face_size = 40;

  //ScanList("H:\\lfw\\list.txt", cascade);
  Mat image = imread("D:\\face project\\images\\test.jpg");
  //Mat image = imread("C:\\lena.png");
  std::vector<std::pair<Rect, double>> location_and_scale;
  Mat stitch_image = getPyramidStitchingImage2(image, location_and_scale);
  resize(stitch_image, stitch_image, Size(0, 0), 0.25, 0.25);
  imwrite("stitch_image.png", stitch_image);
  //resize(image, image, Size(640, 480));
  
  //Mat image = imread("G:\\WIDER\\face_detection\\pack\\1[00_00_26][20160819-181452-0].BMP");
  //Mat image = imread("D:\\face project\\FDDB\\2002/07/25/big/img_1047.jpg");
  
  //Mat image = imread("D:\\face project\\FDDB\\2003/01/13/big/img_1087.bmp");
  cout << image.cols<<","<<image.rows << endl;
  vector<vector<Point2d>> points;
  std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
  auto result = cascade.GetDetection(image, 12.0 / min_face_size, { 0.6, 0.7, 0.7 }, true, 0.7, true, points);
  points.clear();//The first run is slow because it need to allocate memory.
  result = cascade.GetDetection(image, 12.0 / min_face_size, { 0.6, 0.7, 0.7 }, true, 0.7, true, points);
  std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
  cout << "detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << endl;

  cout << "===========================================================" << endl;
  points.clear();//The first run is slow because it need to allocate memory.
  p0 = std::chrono::system_clock::now();
  result = cascade.GetDetection(image, 12.0 / min_face_size, { 0.6, 0.7, 0.7 }, true, 0.7, true, points);
  p1 = std::chrono::system_clock::now();
  cout << "detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << endl;
  cout << "detected " << result.size() << " faces" << endl;

  Mat show_image = image.clone();
  vector<Mat> croppedImages;
  for (int i = 0; i < result.size(); i++) {
    //cout << "face box:" << result[i].first << " confidence:" << result[i].second << endl;
    //rectangle(show_image, result[i].first, Scalar(255, 0, 0), 2);
    if (points.size() >= i + 1) {
      for (int p = 0; p < 5; p++) {
        circle(show_image, points[i][p], 2, Scalar(0, 255, 255), -1);
      }
      Mat trans_inv;
      Mat trans = findSimilarityTransform(points[i], target_points, trans_inv);
      Mat cropImage;
      warpAffine(image, cropImage, trans, Size(96, 112));
      //imshow("cropImage", cropImage);
      croppedImages.push_back(cropImage);
      vector<Point2d> rotatedVertex;
      transform(getVertexFromBox(Rect(0,0,96,112)), rotatedVertex, trans_inv);
      for (int i = 0; i < 4; i++)
        line(show_image, rotatedVertex[i], rotatedVertex[(i + 1) % 4], Scalar(0, 255, 0), 2);
    }
  }
  while (show_image.cols > 1000) {
    resize(show_image, show_image, Size(0, 0), 0.75, 0.75);
  }
  imshow("final", show_image);
  waitKey(0);
  //imwrite("output.jpg", image);
  //TestFDDBPrecision(cascade, "D:\\face project\\FDDB\\", true, true);
  delete kCaffeBinding;
  system("pause");
	return 0;
}

