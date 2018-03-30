#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono> // std::chrono::system_clock

#include "util\FileManager.inc.h"
#include "util\BoundingBox.inc.h"
#include "CaffeBinding.h"
#include "SaveHDF5.inc.h"
#include "TestFaceDetection.inc.h"
#include <boost/shared_ptr.hpp>
extern boost::shared_ptr<caffe::CaffeBinding> kCaffeBinding;

using namespace cv;
using namespace std;
using namespace FaceInception;

typedef std::mt19937 RANDOM_ENGINE;

class WiderFace {
public:

  WiderFace() : foreground_thresh(0.65),
    part_thresh(0.4),
    background_thresh(0.3),
    positive_ratio(0.35),
    negative_ratio(0.35),
    patch_per_face(9),
    prnd(time(NULL)) { }
  
  void ExtractPatch(int size, string list_file = "G:\\WIDER\\wider_face_val_list.txt",
                    string root_folder = "G:\\WIDER\\WIDER_val\\images\\",
                    string output_dir = "G:\\WIDER\\val") {
    output_dir += to_string(size) + "\\";
    _mkdir(output_dir.c_str());
    _mkdir((output_dir + "positive").c_str());
    _mkdir((output_dir + "negative").c_str());
    _mkdir((output_dir + "part").c_str());
    HDF5Manager hdf5(output_dir, "/data", "/label", "wider-", 3, size, size, 10);
    FILE *fp = fopen((output_dir + "list.txt").c_str(), "w");
    std::ifstream infile(list_file.c_str());
    string filename;
    char this_line[65536];
    int image_file_id = 0;
    char image_file_name[255];
    const int exceed_try = 20;

    while (!infile.eof()) {
      memset(this_line, 0, sizeof(this_line));
      infile.getline(this_line, 65536);
      if (strlen(this_line) == 0) continue;
      this_line[strlen(this_line) - 1] = '\0';
      std::stringstream stream;
      stream << this_line;
      stream >> filename;
      double xx, yy, ww, hh;
      auto labels_ptr = make_shared<std::vector<Rect2d> >();
      while (!stream.eof()) {
        stream >> xx >> yy >> ww >> hh;
        if (ww>12 && hh>12)
          labels_ptr->push_back(Rect2d(xx, yy, ww, hh));
      }
      if (labels_ptr->size() == 0) continue;
      cout << filename << " " << labels_ptr->size() << endl;
      Mat image = imread(root_folder + filename);
      if (image.empty()) continue;
      Mat sub_image;
      Mat sub_small_image;
      for (int f = 0; f <= 1; f++) {
        if (f == 1) {
          flip(image, image, 1);
          for (int j = 0; j < labels_ptr->size(); j++) {
            labels_ptr->at(j).x = image.cols - 1 - (labels_ptr->at(j).x + labels_ptr->at(j).width);
          }
        }
        for (int j = 0; j < labels_ptr->size(); j++) {
          int per_face = 0;
          if (list_file.find("train") != list_file.npos && labels_ptr->at(j).x >= 0 && labels_ptr->at(j).y >= 0
              && labels_ptr->at(j).x + labels_ptr->at(j).width <= image.cols - 1 && labels_ptr->at(j).y + labels_ptr->at(j).height <= image.rows-1)
          {
            try {
              sub_image = image(labels_ptr->at(j));
              resize(sub_image, sub_small_image, Size(size, size), 0, 0, CV_INTER_LINEAR);
              hdf5.saveMat(sub_small_image, { 1, 2, 0, 0, 0, 0, 1, 1, 1, 1 });
              /*sprintf(image_file_name, "positive\\%07d.bmp", image_file_id);
              imwrite(output_dir + image_file_name, sub_small_image);
              fprintf(fp, "%s %.3lf %.3lf %.3lf %.3lf %.3lf\r\n", image_file_name, 1.0, 0.0, 0.0, 0.0, 0.0);
              image_file_id++;*/
            }
            catch (cv::Exception e) {
              cout << e.what() << endl;
            }
          }
          patch_per_face = 3;
          if (labels_ptr->at(j).width > 16)patch_per_face = 6;
          if (labels_ptr->at(j).width > 24)patch_per_face = 9;
          for (int k = 0; k < patch_per_face; k++) {
            try {
              int x, y, w, h, t = 0;
              double iou;
              if (std::bernoulli_distribution(positive_ratio)(prnd)) {
                do {
                  if (++t > exceed_try) break;
                  x = std::uniform_int_distribution<int>(-labels_ptr->at(j).width * (sqrt(1 / foreground_thresh) - 1), labels_ptr->at(j).width * (sqrt(1 / foreground_thresh) - 1))(prnd);
                  x += labels_ptr->at(j).x;
                  y = std::uniform_int_distribution<int>(-labels_ptr->at(j).height * (sqrt(1 / foreground_thresh) - 1), labels_ptr->at(j).height * (sqrt(1 / foreground_thresh) - 1))(prnd);
                  y += labels_ptr->at(j).y;
                  w = std::uniform_int_distribution<int>(labels_ptr->at(j).width * foreground_thresh, labels_ptr->at(j).width / foreground_thresh)(prnd);
                  //h = std::uniform_int_distribution<int>(labels_ptr->at(j).height * foreground_thresh, labels_ptr->at(j).height / foreground_thresh)(prnd);
                  h = std::uniform_int_distribution<int>(w / 1.3, w *1.3)(prnd);
                  iou = IoU(Rect2d(x, y, w, h), labels_ptr->at(j));
                } while (iou < foreground_thresh || x < 0 || y<0 || x + w>image.cols - 1 || y + h>image.rows - 1 || w < size / 2);
                if (t <= exceed_try) {
                  sub_image = image(Rect(x, y, w, h));
                  resize(sub_image, sub_small_image, Size(size, size), 0, 0, CV_INTER_LINEAR);
                  //sprintf(image_file_name, "positive\\%07d.bmp", image_file_id);
                  //imwrite(output_dir + image_file_name, sub_small_image);
                  //image_file_id++;
                  //fprintf(fp, "%s %.3lf %.3lf %.3lf %.3lf %.3lf\r\n", image_file_name, iou, bb_target.x, bb_target.y, bb_target.width, bb_target.height);
                  auto bb_target = BoundingBoxRegressionTarget(Rect(x, y, w, h), labels_ptr->at(j));
                  hdf5.saveMat(sub_small_image, { 1, 1, 
                               (float)bb_target.x, (float)bb_target.y, (float)bb_target.width, (float)bb_target.height,
                               1, 1, 1, 1 });
                  per_face++;
                  //printf("%s %.3lf %.3lf %.3lf %.3lf %.3lf\r\n", image_file_name, iou, bb_target.x, bb_target.y, bb_target.width, bb_target.height);
                  //imshow("positive", sub_image);
                  //waitKey(0);
                }
                else {
                  //cout << "failed to get a positive!"<<endl;
                  //cout << "f";
                }
              }
              else if (std::bernoulli_distribution(negative_ratio / (1 - positive_ratio))(prnd)) {
                do {
                  if (++t > exceed_try) break;
                  x = std::uniform_int_distribution<int>(-labels_ptr->at(j).width, labels_ptr->at(j).width)(prnd);
                  x += labels_ptr->at(j).x;
                  y = std::uniform_int_distribution<int>(-labels_ptr->at(j).height, labels_ptr->at(j).height)(prnd);
                  y += labels_ptr->at(j).y;
                  w = std::uniform_int_distribution<int>(labels_ptr->at(j).width * foreground_thresh, labels_ptr->at(j).width / foreground_thresh)(prnd);
                  //h = std::uniform_int_distribution<int>(labels_ptr->at(j).height * foreground_thresh, labels_ptr->at(j).height / foreground_thresh)(prnd);
                  h = std::uniform_int_distribution<int>(w / 1.5, w *1.5)(prnd);
                  iou = IoU(Rect2d(x, y, w, h), labels_ptr->at(j));
                  for (auto other_rect : *labels_ptr) {
                    double other_iou = IoU(Rect2d(x, y, w, h), other_rect);
                    if (other_iou > iou) iou = other_iou;
                  }
                } while (iou < 0 || iou > background_thresh || x < 0 || y<0 || x + w>image.cols - 1 || y + h>image.rows - 1 || w < size / 2);
                if (t <= exceed_try) {
                  sub_image = image(Rect(x, y, w, h));
                  resize(sub_image, sub_small_image, Size(size, size), 0, 0, CV_INTER_LINEAR);
                  //sprintf(image_file_name, "negative\\%07d.bmp", image_file_id);
                  //imwrite(output_dir + image_file_name, sub_small_image);
                  //fprintf(fp, "%s %.3lf %.3lf %.3lf %.3lf %.3lf\r\n", image_file_name, iou, bb_target.x, bb_target.y, bb_target.width, bb_target.height);
                  //image_file_id++
                  auto bb_target = BoundingBoxRegressionTarget(Rect(x, y, w, h), labels_ptr->at(j));
                  hdf5.saveMat(sub_small_image, { 0, 1,
                               (float)bb_target.x, (float)bb_target.y, (float)bb_target.width, (float)bb_target.height,
                               0, 0, 0, 0 });
                  per_face++;
                  //printf("%s %.3lf %.3lf %.3lf %.3lf %.3lf\r\n", image_file_name, iou, bb_target.x, bb_target.y, bb_target.width, bb_target.height);
                  //imshow("negative", sub_image);
                  //waitKey(0);
                }
                else {
                  //cout << "failed to get a negative!"<<endl;
                  //cout << "f";
                }
              }
              else {
                do {
                  if (++t > exceed_try) break;
                  x = std::uniform_int_distribution<int>(-labels_ptr->at(j).width, labels_ptr->at(j).width)(prnd);
                  x += labels_ptr->at(j).x;
                  y = std::uniform_int_distribution<int>(-labels_ptr->at(j).height, labels_ptr->at(j).height)(prnd);
                  y += labels_ptr->at(j).y;
                  w = std::uniform_int_distribution<int>(labels_ptr->at(j).width * foreground_thresh, labels_ptr->at(j).width / foreground_thresh)(prnd);
                  //h = std::uniform_int_distribution<int>(labels_ptr->at(j).height * foreground_thresh, labels_ptr->at(j).height / foreground_thresh)(prnd);
                  h = std::uniform_int_distribution<int>(w / 1.3, w *1.3)(prnd);
                  iou = IoU(Rect2d(x, y, w, h), labels_ptr->at(j));
                  for (auto other_rect : *labels_ptr) {
                    double other_iou = IoU(Rect2d(x, y, w, h), other_rect);
                    if (other_iou > iou) iou = other_iou;
                  }
                } while (iou < part_thresh || iou > foreground_thresh || x < 0 || y<0 || x + w>image.cols - 1 || y + h>image.rows - 1 || w < size / 2);
                if (t <= exceed_try) {
                  sub_image = image(Rect(x, y, w, h));
                  resize(sub_image, sub_small_image, Size(size, size), 0, 0, CV_INTER_LINEAR);
                  //sprintf(image_file_name, "part\\%07d.bmp", image_file_id);
                  //imwrite(output_dir + image_file_name, sub_small_image);
                  //fprintf(fp, "%s %.3lf %.3lf %.3lf %.3lf %.3lf\r\n", image_file_name, iou, bb_target.x, bb_target.y, bb_target.width, bb_target.height);
                  //image_file_id++;
                  auto bb_target = BoundingBoxRegressionTarget(Rect(x, y, w, h), labels_ptr->at(j));
                  hdf5.saveMat(sub_small_image, { 1, 0,
                               (float)bb_target.x, (float)bb_target.y, (float)bb_target.width, (float)bb_target.height,
                               1, 1, 1, 1 });
                  per_face++;
                  //printf("%s %.3lf %.3lf %.3lf %.3lf %.3lf\r\n", image_file_name, iou, bb_target.x, bb_target.y, bb_target.width, bb_target.height);
                  //imshow("part", sub_image);
                  //waitKey(0);
                }
                else {
                  //cout << "failed to get a part!"<<endl;
                  //cout << "f";
                }
              }
            }
            catch (cv::Exception e) {
              cout << e.what() << endl;
            }
          }
          cout << per_face;
        }
        cout << endl;
      }
    }

    fclose(fp);
  }

  void LoadModel(string net_definition, string weights, int input_height, int input_width) {
    net_id = kCaffeBinding->AddNet(net_definition, weights);
    net_input_height = input_height;
    net_input_width = input_width;
  }

  void ExtractNegative(int size, string list_file = "G:\\WIDER\\wider_face_train_list.txt",
                       string root_folder = "G:\\WIDER\\WIDER_train\\images\\",
                       string output_dir = "G:\\WIDER\\train_negative") {
    negative_per_image = 200;
    output_dir += to_string(size) + "\\";
    _mkdir(output_dir.c_str());
    HDF5Manager hdf5(output_dir,"/n_data","/n_label","negative"+to_string(size)+"-",3,size,size);
    FILE *fp = fopen((output_dir + "list.txt").c_str(), "w");
    std::ifstream infile(list_file.c_str());
    string filename;
    char this_line[65536];
    int image_file_id = 0;
    char image_file_name[255];
    const int exceed_try = 20;

    while (!infile.eof()) {
      memset(this_line, 0, sizeof(this_line));
      infile.getline(this_line, 65536);
      if (strlen(this_line) == 0) continue;
      this_line[strlen(this_line) - 1] = '\0';
      std::stringstream stream;
      stream << this_line;
      stream >> filename;
      double xx, yy, ww, hh;
      auto labels_ptr = make_shared<std::vector<Rect2d> >();
      while (!stream.eof()) {
        stream >> xx >> yy >> ww >> hh;
        labels_ptr->push_back(Rect2d(xx, yy, ww, hh));
      }
      cout << filename << " ";
      Mat image = imread(root_folder + filename);
      if (image.empty()) continue;

      int x, y, w, h;
      std::vector<Mat> subimages;
      std::vector<Rect> subRects;
      int valid_count = 0;
      Mat small_image;
      for (int i = 0; i < negative_per_image; i++) {
        double max_iou=0;
        int t = 0;
        do {
          if (++t > exceed_try) break;
          x = std::uniform_int_distribution<int>(0, image.cols - size * 2)(prnd);
          y = std::uniform_int_distribution<int>(0, image.rows - size * 2)(prnd);
          w = std::uniform_int_distribution<int>(12, size * 8)(prnd);
          h = std::uniform_int_distribution<int>(max(12, w / 2), w * 2)(prnd);
          for (auto other_rect : *labels_ptr) {
            double other_iou = IoU(Rect2d(x, y, w, h), other_rect);
            if (other_iou > max_iou) max_iou = other_iou;
          }
        } while (max_iou > 0.1 || x < 0 || y<0 || x + w>image.cols - 1 || y + h>image.rows - 1 || w < size / 2);
        if (t <= exceed_try) {
          Mat sub_image = image(Rect(x, y, w, h));
          Mat sub_small_image;
          resize(sub_image, sub_small_image, Size(size, size), 0, 0, CV_INTER_LINEAR);
          hdf5.saveMat(sub_small_image, { 0, 1});
          valid_count++;
        }
      }
      cout << valid_count << endl;
    }
  }

  void MineNegative(int size, int stride, string list_file = "G:\\WIDER\\wider_face_train_list.txt",
                    string root_folder = "G:\\WIDER\\WIDER_train\\images\\",
                    string output_dir = "G:\\WIDER\\train_negative",
                    bool mine_hard = true) {
    negative_per_image = 20;
    output_dir += to_string(size) + "\\";
    _mkdir(output_dir.c_str());
    HDF5Manager hdf5(output_dir);
    FILE *fp = fopen((output_dir + "list.txt").c_str(), "w");
    std::ifstream infile(list_file.c_str());
    string filename;
    char this_line[65536];
    int image_file_id = 0;
    char image_file_name[255];
    const int exceed_try = 20;

    while (!infile.eof()) {
      memset(this_line, 0, sizeof(this_line));
      infile.getline(this_line, 65536);
      if (strlen(this_line) == 0) continue;
      this_line[strlen(this_line) - 1] = '\0';
      std::stringstream stream;
      stream << this_line;
      stream >> filename;
      double xx, yy, ww, hh;
      auto labels_ptr = make_shared<std::vector<Rect2d> >();
      while (!stream.eof()) {
        stream >> xx >> yy >> ww >> hh;
        labels_ptr->push_back(Rect2d(xx, yy, ww, hh));
      }
      cout << filename << " ";
      Mat image = imread(root_folder + filename);
      if (image.empty()) continue;
      
      int x, y, w, h;
      std::vector<Mat> subimages;
      std::vector<Rect> subRects;
      int valid_count = 0;
      Mat small_image;
      resize(image, small_image, Size(net_input_width, net_input_height));
      subimages.push_back(small_image);
      //imshow("subimage", small_image);
      //waitKey(1);
      subRects.push_back(Rect(0, 0, image.cols, image.rows));
      for (int slice = 0; slice < 9; slice++) {
        Mat sub_image;
        do {
          x = std::uniform_int_distribution<int>(0, image.cols - 1 - net_input_width)(prnd);
          y = std::uniform_int_distribution<int>(0, image.cols - 1 - net_input_width)(prnd);
          w = std::uniform_int_distribution<int>(net_input_width / 4, net_input_width * 4)(prnd);
          h = std::uniform_int_distribution<int>(net_input_width / 4, net_input_width * 4)(prnd);
          //h = std::uniform_int_distribution<int>(w / 1.5, w * 1.5)(prnd);
        } while (x < 0 || y<0 || x + w>image.cols - 1 || y + h>image.rows - 1);
        resize(image(Rect(x, y, w, h)), sub_image, Size(net_input_width, net_input_height));
        subimages.push_back(sub_image);
        subRects.push_back(Rect(x, y, w, h));
      }
      auto result = kCaffeBinding->Forward(subimages, net_id);
      for (int slice = 0; slice < 10; slice++) {
        const float *slice_ptr = result[1].data + slice * result[1].size[1] * result[1].size[2] * result[1].size[3];
        //if (slice == 0) {
          //Mat conf_slice = Mat(result[1].size[2], result[1].size[3], CV_32F, (float *)slice_ptr);
          //Mat slice_show;
          //conf_slice.convertTo(slice_show, CV_8UC1, 255, -128);
          //imshow("confidence", slice_show);
          //waitKey(0);
        //}
        double negative_confidence_thresh = 0.3;
        if (!mine_hard) negative_confidence_thresh = 0.5;
        vector<float> confidences;
        for (int ox = 0; ox < result[1].size[3]; ox += size / 4) {
          for (int oy = 0; oy < result[1].size[2]; oy += size / 4) {
            if (slice_ptr[oy * result[1].size[3] + ox] < negative_confidence_thresh) {
              confidences.push_back(slice_ptr[oy * result[1].size[3] + ox]);
            }
          }
        }
        if (mine_hard) {
          if (confidences.size() > negative_per_image) {
            std::sort(confidences.begin(), confidences.end());
            negative_confidence_thresh = confidences[negative_per_image];
          }
        }
        int negative_count = confidences.size();
        for (int ox = 0; ox < result[1].size[3]; ox += size / 4) {
          for (int oy = 0; oy < result[1].size[2]; oy += size / 4) {
            if (slice_ptr[oy * result[1].size[3] + ox] < negative_confidence_thresh && (mine_hard ||
                (!mine_hard && std::bernoulli_distribution((double)negative_per_image / (double)negative_count)(prnd)))) {
              try {
                x = ox * stride;
                y = oy * stride;
                w = size;
                h = size;
                Rect2d rect = Rect2d(x, y, w, h);
                bool valid = true;
                for (int j = 0; j < labels_ptr->size(); j++) {
                  Rect2d gt_rect = labels_ptr->at(j);
                  gt_rect.x -= subRects[slice].x;
                  gt_rect.y -= subRects[slice].y;
                  gt_rect.width *= (double)subRects[slice].width / (double)net_input_width;
                  gt_rect.height *= (double)subRects[slice].height / (double)net_input_height;
                  if (IoU(rect, gt_rect) > 0.1) {
                    valid = false;
                  }
                }
                if (!valid) continue;
                Mat sub_image = subimages[slice](rect).clone();
                sprintf(image_file_name, "negative\\%07d.bmp", image_file_id);
                hdf5.saveMat(sub_image, { 0, 1 });
                //imwrite(output_dir + image_file_name, sub_image);
                fprintf(fp, "%s\r\n", image_file_name);
                image_file_id++;
                valid_count++;
              }
              catch (cv::Exception e) {

              }
            }
          }
        }
      }
      cout << valid_count << endl;
    }
  }

  void MineFromProposal(int size, CascadeCNN cascade, string list_file = "G:\\WIDER\\wider_face_train_list.txt",
                        string root_folder = "G:\\WIDER\\WIDER_train\\images\\",
                        string output_dir = "G:\\WIDER\\train_negative",
                        string data_name = "n_data", string label_name = "n_label", string prefix = "negative12-",
                        double min_confidence=0.7, bool negative_only = true) {
    negative_per_image = 200;
    output_dir += to_string(size) + "\\";
    _mkdir(output_dir.c_str());
    HDF5Manager hdf5(output_dir, data_name, label_name, prefix, 3, size, size, negative_only ? 2 : 10);
    FILE *fp = fopen((output_dir + "list.txt").c_str(), "w");
    std::ifstream infile(list_file.c_str());
    string filename;
    char this_line[65536];
    int image_file_id = 0;
    char image_file_name[255];
    const int exceed_try = 20;

    while (!infile.eof()) {
      memset(this_line, 0, sizeof(this_line));
      infile.getline(this_line, 65536);
      if (strlen(this_line) == 0) continue;
      this_line[strlen(this_line) - 1] = '\0';
      std::stringstream stream;
      stream << this_line;
      stream >> filename;
      double xx, yy, ww, hh;
      auto labels_ptr = make_shared<std::vector<Rect2d> >();
      while (!stream.eof()) {
        stream >> xx >> yy >> ww >> hh;
        labels_ptr->push_back(Rect2d(xx, yy, ww, hh));
      }
      cout << filename << " ";
      Mat image = imread(root_folder + filename);
      if (image.empty()) continue;
      int mined_positive = 0, mined_negative = 0, mined_part = 0;
      auto result = cascade.getNet12Proposal(image, min_confidence, true, 0.3);
      int result_size = result.size();
      for (auto& rect_and_score : result) {
        double max_iou = 0;
        Rect2d max_iou_rect;
        for (auto& positive_rect : *labels_ptr) {
          double iou = IoU(rect_and_score.first, positive_rect);
          if (iou > max_iou) {
            max_iou = iou;
            max_iou_rect = positive_rect;
          }
        }
        Mat sub_image = image(rect_and_score.first);
        resize(sub_image, sub_image, Size(size, size));
        Mat image_for_save = sub_image;
        if (max_iou <background_thresh) {
          if (negative_only) {
            hdf5.saveMat(image_for_save, { 0, 1 });
          }
          else if (std::bernoulli_distribution(0.1)(prnd)) {
            hdf5.saveMat(image_for_save, { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 });
            mined_negative++;
          }
        }
        else if (!negative_only) {
          auto bb_target = BoundingBoxRegressionTarget(rect_and_score.first, max_iou_rect);
          if (max_iou > foreground_thresh) {
            hdf5.saveMat(image_for_save, { 1, 1,
                         (float)bb_target.x, (float)bb_target.y, (float)bb_target.width, (float)bb_target.height,
                         1, 1, 1, 1 });
            mined_positive++;
          }
          else if (max_iou > part_thresh && max_iou < foreground_thresh) {
            hdf5.saveMat(image_for_save, { 1, 0,
                         (float)bb_target.x, (float)bb_target.y, (float)bb_target.width, (float)bb_target.height,
                         1, 1, 1, 1 });
            mined_part++;
          }
        }
      }
      cout << labels_ptr->size() <<" " << mined_positive << " " << mined_part << " " << mined_negative << endl;
      //for (auto& rect : result) {
      //  rectangle(image, rect.first, Scalar(255, 0, 0), 1);
      //}
      //for (auto& rect : *labels_ptr) {
      //  rectangle(image, rect, Scalar(0, 0, 255), 1);
      //}
      //imshow("boxes", image);
      //waitKey(0);
    }
  }

  void MineHardFromProposal(int size, CascadeCNN cascade, string list_file = "G:\\WIDER\\wider_face_train_list.txt",
                        string root_folder = "G:\\WIDER\\WIDER_train\\images\\",
                        string output_dir = "G:\\WIDER\\train_negative",
                        string data_name = "n_data", string label_name = "n_label", string prefix = "negative12-",
                        double min_positive_confidence = 0.7, double min_negative_confidence = 0.7, bool negative_only = true,
                        int net_id = 0) {
    negative_per_image = 20;
    output_dir += to_string(size) + "\\";
    _mkdir(output_dir.c_str());
    HDF5Manager hdf5(output_dir, data_name, label_name, prefix, 3, size, size, negative_only ? 2 : 10);
    FILE *fp = fopen((output_dir + "list.txt").c_str(), "w");
    std::ifstream infile(list_file.c_str());
    string filename;
    char this_line[65536];
    int image_file_id = 0;
    char image_file_name[255];
    const int exceed_try = 20;

    while (!infile.eof()) {
      memset(this_line, 0, sizeof(this_line));
      infile.getline(this_line, 65536);
      if (strlen(this_line) == 0) continue;
      this_line[strlen(this_line) - 1] = '\0';
      std::stringstream stream;
      stream << this_line;
      stream >> filename;
      double xx, yy, ww, hh;
      auto labels_ptr = make_shared<std::vector<Rect2d> >();
      while (!stream.eof()) {
        stream >> xx >> yy >> ww >> hh;
        labels_ptr->push_back(Rect2d(xx, yy, ww, hh));
      }
      cout << filename << " ";
      Mat input_image = imread(root_folder + filename);
      if (input_image.empty()) continue;
      int mined_count = 0;
      int mined_positive = 0, mined_negative = 0, mined_part = 0;
      double scale_decay_ = 0.717;

      int short_side = min(input_image.cols, input_image.rows);
      //resize(input_image, input_image, Size(640, 480));

      vector<Mat> pyramid;
      Mat small_image;
      if (input_image.cols < 1500 && input_image.rows < 1500) {
        pyramid.push_back(input_image);
      }
      resize(input_image, small_image, Size(input_image.cols * scale_decay_, input_image.rows *scale_decay_));
      if (small_image.cols < 1500 && small_image.rows < 1500) {
        pyramid.push_back(small_image);
      }
      do {
        resize(small_image, small_image, Size(small_image.cols * scale_decay_, small_image.rows *scale_decay_));
        if (small_image.cols < 1500 && small_image.rows < 1500) {
          pyramid.push_back(small_image);
        }
      } while (floor(small_image.rows * scale_decay_) > size && floor(small_image.cols * scale_decay_) > size);
      assert(pyramid[pyramid.size() - 1].cols > size);
      assert(pyramid[pyramid.size() - 1].rows > size);

      for (int p = 0; p < pyramid.size(); p++) {
        double scale = (double)pyramid[p].rows / (double)input_image.rows;
        std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
        auto net12output = kCaffeBinding->Forward({ pyramid[p] }, net_id);
        for (int xx = 0; xx < net12output[0].size[3]; xx++) {
          for (int yy = 0; yy < net12output[0].size[2]; yy++) {
            double x = net12output[0].data[0 * net12output[0].size[2] * net12output[0].size[3] + yy * net12output[0].size[3] + xx];
            double y = net12output[0].data[1 * net12output[0].size[2] * net12output[0].size[3] + yy * net12output[0].size[3] + xx];
            double w = net12output[0].data[2 * net12output[0].size[2] * net12output[0].size[3] + yy * net12output[0].size[3] + xx];
            double h = net12output[0].data[3 * net12output[0].size[2] * net12output[0].size[3] + yy * net12output[0].size[3] + xx];
            double score = net12output[0].data[4 * net12output[0].size[2] * net12output[0].size[3] + yy * net12output[0].size[3] + xx];
            assert(w == size && h == size);
            double max_iou = 0;
            Rect2d max_iou_rect;
            for (auto& positive_rect : *labels_ptr) {
              double iou = IoU(Rect2d(x / scale, y / scale, w / scale, h / scale), positive_rect);
              if (iou > max_iou) {
                max_iou = iou;
                max_iou_rect = positive_rect;
              }
            }
            if (max_iou < background_thresh && score > min_negative_confidence &&
                std::bernoulli_distribution(0.002)(prnd)) {
              try {
                if (checkRect(Rect(x, y, w, h), pyramid[p].size())) {
                  Mat image_for_save = pyramid[p](Rect(x, y, w, h)).clone();
                  if (negative_only)
                    hdf5.saveMat(image_for_save, { 0, 1 });
                  else
                    hdf5.saveMat(image_for_save, { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 });
                  mined_count++;
                  mined_negative++;
                }
              }
              catch (cv::Exception e) {}
            }
            if (!negative_only) {
              if (max_iou > foreground_thresh && score < 1 - min_positive_confidence) {
                try {
                  if (checkRect(Rect(x, y, w, h), pyramid[p].size())) {
                    Mat image_for_save = pyramid[p](Rect(x, y, w, h)).clone();
                    auto bb_target = BoundingBoxRegressionTarget(Rect2d(x / scale, y / scale, w / scale, h / scale), max_iou_rect);
                    hdf5.saveMat(image_for_save, { 1, 1,
                                 (float)bb_target.x, (float)bb_target.y, (float)bb_target.width, (float)bb_target.height,
                                 1, 1, 1, 1 });
                    mined_count++;
                    mined_positive++;
                  }
                }
                catch (cv::Exception e) {}
              }
              //if (max_iou > part_thresh && max_iou < foreground_thresh && score < 1 - min_confidence / 2) {
              //  try {
              //    if (checkRect(Rect(x, y, w, h), pyramid[p].size())) {
              //      Mat image_for_save = pyramid[p](Rect(x, y, w, h)).clone();
              //      auto bb_target = BoundingBoxRegressionTarget(Rect2d(x / scale, y / scale, w / scale, h / scale), max_iou_rect);
              //      hdf5.saveMat(image_for_save, { 1, 0,
              //                   (float)bb_target.x, (float)bb_target.y, (float)bb_target.width, (float)bb_target.height,
              //                   1, 1, 1, 1 });
              //      mined_count++;
              //      mined_part++;
              //    }
              //  }
              //  catch (cv::Exception e) {}
              //}
            }
          }
        }
      }
      if (negative_only) {
        cout << mined_count << endl;
      }
      else {
        cout << mined_positive << " " << mined_part << " " << mined_negative << endl;
      }
    }
  }

  double foreground_thresh;
  double part_thresh;
  double background_thresh;
  double positive_ratio;
  double negative_ratio;
  int patch_per_face;
  std::vector<std::pair<std::string, shared_ptr<std::vector<Rect2d> > > > lines;
  RANDOM_ENGINE prnd;

  int net_id;
  int net_input_width, net_input_height;
  int negative_per_image;
};