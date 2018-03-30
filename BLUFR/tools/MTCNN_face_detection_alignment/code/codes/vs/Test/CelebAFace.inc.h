#pragma once
#include <opencv2/opencv.hpp>
#include <memory>

#include "util\FileManager.inc.h"
#include "util\BoundingBox.inc.h"
#include "CaffeBinding.h"
#include "SaveHDF5.inc.h"
#include <boost/shared_ptr.hpp>
extern boost::shared_ptr<caffe::CaffeBinding> kCaffeBinding;

using namespace cv;
using namespace std;
using namespace FaceInception;

typedef std::mt19937 RANDOM_ENGINE;

class CelebAFace {
public:

  CelebAFace() : foreground_thresh(0.65),
    part_thresh(0.4),
    background_thresh(0.3),
    positive_ratio(0.35),
    negative_ratio(0.35),
    patch_per_face(5),
    prnd(time(NULL)){}
  
  FaceAndPoints extract_face(cv::Mat& input_image, vector<Point2d> points, int point_count = 5,
                    int new_width = 12, int new_height = 12, int max_random_shift = 10,
                    float max_shear_ratio = 0, float max_aspect_ratio = 0, float max_rotate_angle = 15,
                    float min_random_scale = 0.8, float max_random_scale = 1.2,
                    bool face_mirror = true) {
    Rect2d bounding_box = FaceInception::calcRect(input_image, points);
    cv::Point2d face_center;
    face_center.x = bounding_box.x + bounding_box.width / 2;
    face_center.y = bounding_box.y + bounding_box.height / 2;
    double face_scale = bounding_box.width;
    face_center.x += std::uniform_int_distribution<int>(-max_random_shift, max_random_shift)(prnd);
    face_center.y += std::uniform_int_distribution<int>(-max_random_shift, max_random_shift)(prnd);
    std::uniform_real_distribution<float> rand_uniform(0, 1);
    // shear
    float s = rand_uniform(prnd) * max_shear_ratio * 2 - max_shear_ratio;
    // rotate
    int angle = std::uniform_int_distribution<int>(
      -max_rotate_angle, max_rotate_angle)(prnd);
    float a = cos(angle / 180.0 * CV_PI);
    float b = sin(angle / 180.0 * CV_PI);
    // scale
    float scale = rand_uniform(prnd) *
      (max_random_scale - min_random_scale) + min_random_scale;
    scale = scale * new_width / face_scale;
    // aspect ratio
    float ratio = rand_uniform(prnd) *
      max_aspect_ratio * 2 - max_aspect_ratio + 1;
    float hs = 2 * scale / (1 + ratio);
    float ws = ratio * hs;
    int flip = 1;
    if (face_mirror) {
      flip = std::uniform_int_distribution<int>(0, 1)(prnd)* 2 - 1;
    }
    hs *= flip;

    cv::Mat M(2, 3, CV_32F);
    M.at<float>(0, 0) = hs * a - s * b * ws;
    M.at<float>(1, 0) = -b * ws;
    M.at<float>(0, 1) = hs * b + s * a * ws;
    M.at<float>(1, 1) = a * ws;
    M.at<float>(0, 2) = new_width / 2 - M.at<float>(0, 0) * face_center.x - M.at<float>(0, 1) * face_center.y;
    M.at<float>(1, 2) = new_height / 2 - M.at<float>(1, 0) * face_center.x - M.at<float>(1, 1) * face_center.y;
    //LOG(INFO) << M.at<float>(0, 0) << " " << M.at<float>(1, 0) << " " << M.at<float>(0, 1) << " " << M.at<float>(1, 1) << " " << new_width << " " << new_height << " " << flip;
    cv::Mat temp_;
    cv::warpAffine(input_image, temp_, M, cv::Size(new_width, new_height),
                   cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(128));
    vector<Point2d> wrappedPoint;
    for (int j = 0; j < point_count; j++) {
      float x = M.at<float>(0, 0)*points[j].x + M.at<float>(0, 1) * points[j].y + M.at<float>(0, 2);
      float y = M.at<float>(1, 0)*points[j].x + M.at<float>(1, 1) * points[j].y + M.at<float>(1, 2);
      wrappedPoint.push_back(Point2d(x - new_width / 2, y - new_height / 2));
    }
    if (flip == -1) {
      std::swap(wrappedPoint[0], wrappedPoint[1]);
      std::swap(wrappedPoint[3], wrappedPoint[4]);
    }
    FaceAndPoints result = { temp_.clone(), wrappedPoint };
    return result;
  }

  void GetCelebAData(int size = 12, string list_file = "G:\\celebrity\\list_landmark_attr_val.txt",
                    string root_folder = "G:\\celebrity\\img_celeba\\",
                    string output_folder = "G:\\celebrity\\detection_data\\") {
    _mkdir(output_folder.c_str());
    std::ifstream infile(list_file.c_str());
    string filename;
    char this_line[1024];
    int label_count = 0;
    std::vector<std::pair<std::string, boost::shared_ptr<std::vector<int> > > > lines;
    HDF5Manager hdf5(output_folder, "/a_data", "/a_label", "celeb-", 3, size, size, 62);
    patch_per_face = 3;

    while (!infile.eof()) {
      infile.getline(this_line, 1024);
      std::stringstream stream;
      stream << this_line;
      stream >> filename;
      int label;
      boost::shared_ptr<std::vector<int> > labels_ptr(new std::vector<int>);
      while (!stream.eof()) {
        stream >> label;
        labels_ptr->push_back(label);
      }
      if (label_count == 0) {
        label_count = labels_ptr->size();
      }
      else {
        if (label_count != labels_ptr->size()) {
          cout << "get list error!" << endl;
          return;
        }
      }
      lines.push_back(std::make_pair(filename, labels_ptr));
    }
    cout << "Totally " << lines.size() << " images." << endl;

    for (int i = 0; i < lines.size(); i++) {
      string facefile = root_folder + lines[i].first;
      cout << facefile << " " << lines[i].second->size()<<endl;
      cv::Mat image = cv::imread(facefile);
      auto ground_truth = *lines[i].second;
      std::vector<Point2d> points;
      for (int j = 0; j < 5; j++) {
        points.push_back(Point2d(ground_truth[j * 2], ground_truth[j * 2 + 1]));
      }
      vector<float> label;
      label.reserve(62);
      label.push_back(1);
      label.push_back(1);
      for (int p = 0; p < 5; ++p) {
        label.push_back(points[p].x);
        label.push_back(points[p].y);
      }
      for (int p = 0; p < 10; ++p) {
        label.push_back(1);
      }
      for (int p = 10; p < ground_truth.size(); p++) {
        label.push_back(ground_truth[p]);
      }

      Mat sub_image;
      Rect2d bounding_box = FaceInception::calcRect(image, points);
      if (!checkRect(bounding_box, image.size())) {
        continue;
      }
      resize(image(bounding_box), sub_image, Size(size, size));
      for (int p = 0; p < 5; ++p) {
        label[p*2 + 2] = (points[p].x - bounding_box.x) * (double)size / bounding_box.width - size / 2;
        label[p*2 + 3] = (points[p].y - bounding_box.y) * (double)size / bounding_box.height - size / 2;
      }
      label[1] = 2;
      hdf5.saveMat(sub_image, label);
      //for (int p = 0; p < 5; p++) {
      //  circle(sub_image, Point2d(label[p * 2 + 2], label[p * 2 + 3]) + Point2d(sub_image.cols / 2, sub_image.rows / 2),
      //         2, Scalar(0, 0, 255), -1);
      //}
      //circle(sub_image, Point2d(label[1 * 2 + 2], label[1 * 2 + 3]) + Point2d(sub_image.cols / 2, sub_image.rows / 2),
      //       2, Scalar(255, 0, 0), -1);//ÓÒÑÛÍ¿³ÉÀ¶É«
      //imshow("check", sub_image);
      //waitKey(0);

      for (int f = 0; f < patch_per_face; f++) {
        FaceAndPoints image_points;
        bool valid_sample = true;
        int t = 0;
        do {
          t++;
          image_points = extract_face(image, points, 5, size, size);
          for (int p = 0; p < 5; ++p) {
            if (image_points.points[p].x < -0.45 * (double)image_points.image.cols ||
                image_points.points[p].x > 0.45 * (double)image_points.image.cols ||
                image_points.points[p].y < -0.45 * (double)image_points.image.rows ||
                image_points.points[p].y > 0.45 * (double)image_points.image.rows) {
              valid_sample = false;
            }
          }
        } while (!valid_sample && t < 20);
        if (!valid_sample) continue;
        label[1] = 1;
        for (int p = 0; p < 5; ++p) {
          label[p*2 + 2] = image_points.points[p].x;
          label[p*2 + 3] = image_points.points[p].y;
        }
        //for (int p = 0; p < label.size(); p++) {
        //  cout << label[p] << " ";
        //}
        //cout << endl;
        hdf5.saveMat(image_points.image, label);
        //for (int p = 0; p < 5; p++) {
        //  circle(image_points.image, image_points.points[p] + Point2d(image_points.image.cols / 2, image_points.image.rows / 2),
        //         1, Scalar(0, 0, 255), -1);
        //}
        //circle(image_points.image, image_points.points[1] + Point2d(image_points.image.cols / 2, image_points.image.rows / 2),
        //       1, Scalar(255, 0, 0), -1);//ÓÒÑÛÍ¿³ÉÀ¶É«
        //imshow("check",image_points.image);
        //waitKey(0);
      }
    }
    return;
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