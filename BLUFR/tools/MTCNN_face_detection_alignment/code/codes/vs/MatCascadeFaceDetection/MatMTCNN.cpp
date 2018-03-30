// MatMTCNN.cpp : 
//

#include <io.h>
#include <direct.h>
#include <string>
#include <iomanip>
#include <Windows.h>
#include <AtlBase.h>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include "mex.h"

#include "../Test/util/BoundingBox.inc.h"
#include "../Test/TestFaceDetection.inc.h"

using namespace std;
using namespace FaceInception;
using namespace cv;
#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

CascadeCNN* cascade;
caffe::CaffeBinding* kCaffeBinding = new caffe::CaffeBinding();
cv::Mat image;
vector<double> confidence_threshold = { 0.6, 0.7, 0.7 };

// Do CHECK and throw a Mex error if check fails
inline void mxCHECK(bool expr, const char* msg) {
  if (!expr) {
    mexErrMsgTxt(msg);
  }
}
inline void mxERROR(const char* msg) { mexErrMsgTxt(msg); }

// Check if a file exists and can be opened
bool mxCHECK_FILE_EXIST(const char* file) {
  std::ifstream f(file);
  if (!f.good()) {
    f.close();
    std::string msg("Could not open file ");
    msg += file;
    mxERROR(msg.c_str());
    return false;
  }
  f.close();
  return true;
}

// Convert vector<int> to matlab row vector
static mxArray* int_vec_to_mx_vec(const vector<int>& int_vec) {
  mxArray* mx_vec = mxCreateDoubleMatrix(int_vec.size(), 1, mxREAL);
  double* vec_mem_ptr = mxGetPr(mx_vec);
  for (int i = 0; i < int_vec.size(); i++) {
    vec_mem_ptr[i] = static_cast<double>(int_vec[i]);
  }
  return mx_vec;
}

// Convert vector<double> to matlab row vector
static mxArray* double_vec_to_mx_vec(const vector<double>& double_vec) {
  mxArray* mx_vec = mxCreateDoubleMatrix(double_vec.size(), 1, mxREAL);
  double* vec_mem_ptr = mxGetPr(mx_vec);
  for (int i = 0; i < double_vec.size(); i++) {
    vec_mem_ptr[i] = static_cast<double>(double_vec[i]);
  }
  return mx_vec;
}

// Convert vector<string> to matlab cell vector of strings
static mxArray* str_vec_to_mx_strcell(const vector<std::string>& str_vec) {
  mxArray* mx_strcell = mxCreateCellMatrix(str_vec.size(), 1);
  for (int i = 0; i < str_vec.size(); i++) {
    mxSetCell(mx_strcell, i, mxCreateString(str_vec[i].c_str()));
  }
  return mx_strcell;
}

// Copy matlab array to Blob data or diff
static void ReadMat(const mxArray* mx_mat) {
  const size_t* mat_size = mxGetDimensions(mx_mat);
  int rows = mat_size[1];
  int cols = mat_size[0];
  image = cv::Mat(rows, cols, CV_8UC3);
  const unsigned char* mat_mem_ptr = reinterpret_cast<const unsigned char*>(mxGetData(mx_mat));
  
  std::vector<cv::Mat> channels; // B, G, R channels
  channels.push_back(Mat(rows, cols, CV_8UC1));
  channels.push_back(Mat(rows, cols, CV_8UC1));
  channels.push_back(Mat(rows, cols, CV_8UC1));

  memcpy(channels[2].ptr(), (void *)mat_mem_ptr, rows*cols * sizeof(UINT8));
  memcpy(channels[1].ptr(), (void *)(mat_mem_ptr + rows*cols), rows*cols * sizeof(UINT8));
  memcpy(channels[0].ptr(), (void *)(mat_mem_ptr + 2 * rows*cols), rows*cols * sizeof(UINT8));
  
  cv::transpose(channels[0], channels[0]);
  cv::transpose(channels[1], channels[1]);
  cv::transpose(channels[2], channels[2]);

  cv::merge(channels, image);
}

// Copy matlab array to Blob data or diff
static mxArray* WriteMat(const cv::Mat& opencv_mat) {
  std::vector<mwSize> dims(3);

  dims[0] = static_cast<mwSize>(opencv_mat.rows);
  dims[1] = static_cast<mwSize>(opencv_mat.cols);
  dims[2] = 3;

  mxArray* mx_mat =
    mxCreateNumericArray(3, dims.data(), mxUINT8_CLASS, mxREAL);
  float* mat_mem_ptr = reinterpret_cast<float*>(mxGetData(mx_mat));
  std::vector<cv::Mat> channels; // B, G, R channels
  cv::split(opencv_mat, channels);

  cv::transpose(channels[0], channels[0]);
  cv::transpose(channels[1], channels[1]);
  cv::transpose(channels[2], channels[2]);

  memcpy((void *)mat_mem_ptr, channels[2].ptr(), dims[0] * dims[1] * sizeof(UINT8));
  memcpy((void *)(mat_mem_ptr + dims[0] * dims[1]), channels[1].ptr(), dims[0] * dims[1] * sizeof(UINT8));
  memcpy((void *)(mat_mem_ptr + 2 * dims[0] * dims[1]), channels[0].ptr(), dims[0] * dims[1] * sizeof(UINT8));
  return mx_mat;
}

static void set_device(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsDouble(prhs[0]),
          "Usage: MatMTCNN('set_device', device_id)");
  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  kCaffeBinding->SetDevice(device_id);
}

void InitModel(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsChar(prhs[0]) && mxIsDouble(prhs[1]),
          "Usage: MatMTCNN('init_model', model_path)");
  if (cascade != NULL) {
    mexPrintf("MTCNN already inited, skip...\n");
  }
  char* model_folder_char = mxArrayToString(prhs[0]);
  string model_folder = model_folder_char;
  int gpu_id = static_cast<int>(mxGetScalar(prhs[1]));
  mxCHECK_FILE_EXIST((model_folder + "det1-memory.prototxt").c_str());
  mxCHECK_FILE_EXIST((model_folder + "det1.caffemodel").c_str());
  mxCHECK_FILE_EXIST((model_folder + "det1-memory-stitch.prototxt").c_str());
  mxCHECK_FILE_EXIST((model_folder + "det1.caffemodel").c_str());
  mxCHECK_FILE_EXIST((model_folder + "det2-memory.prototxt").c_str());
  mxCHECK_FILE_EXIST((model_folder + "det2.caffemodel").c_str());
  mxCHECK_FILE_EXIST((model_folder + "det3-memory.prototxt").c_str());
  mxCHECK_FILE_EXIST((model_folder + "det3.caffemodel").c_str());
  mxCHECK_FILE_EXIST((model_folder + "det4-memory.prototxt").c_str());
  mxCHECK_FILE_EXIST((model_folder + "det4.caffemodel").c_str());
  cascade = new CascadeCNN(model_folder + "det1-memory.prototxt", model_folder + "det1.caffemodel",
                           model_folder + "det1-memory-stitch.prototxt", model_folder + "det1.caffemodel",
                           model_folder + "det2-memory.prototxt", model_folder + "det2.caffemodel",
                           model_folder + "det3-memory.prototxt", model_folder + "det3.caffemodel",
                           model_folder + "det4-memory.prototxt", model_folder + "det4.caffemodel",
                           gpu_id);
  mxFree(model_folder_char);
}

void ReleaseModel(MEX_ARGS) {
  delete cascade;
  delete kCaffeBinding;
  cascade = NULL;
  kCaffeBinding = NULL;
}

void SetThreshold(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsDouble(prhs[0]) && mxGetNumberOfElements(prhs[0]) == 3,
          "Usage: MatMTCNN('set_threshold', [0.6, 0.7, 0.7])");
  double* threshold_ptr = (double *)mxGetData(prhs[0]);
  confidence_threshold[0] = threshold_ptr[0];
  confidence_threshold[1] = threshold_ptr[1];
  confidence_threshold[2] = threshold_ptr[2];
  mexPrintf(("Threshold set to (" + to_string(confidence_threshold[0]) + ", "
             + to_string(confidence_threshold[1]) + ", "
             + to_string(confidence_threshold[2]) + ")\n").c_str());
}

static void Detect(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsUint8(prhs[0]) && mxIsDouble(prhs[1]),
          "Usage: MatMTCNN('detect', image, min_face)");
  ReadMat(prhs[0]);
  //mexPrintf("Read Mat to OpenCV done.\n");
  double min_face = mxGetScalar(prhs[1]);
  if (cascade != NULL) {
    vector<vector<Point2d>> points;
    //std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
    auto rect_and_score = cascade->GetDetection(image, 12.0 / min_face, confidence_threshold, true, 0.7, true, points);
    //std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
    //mexPrintf("get %d faces in %f ms\n", result_size, (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000);
    int result_size = rect_and_score.size();
    
    if (result_size > 0) {
      mxArray* mx_bounding_box = mxCreateDoubleMatrix(result_size, 4, mxREAL);
      double* bounding_box_data = (double *)mxGetData(mx_bounding_box);
      for (int i = 0; i < result_size; i++) {
        bounding_box_data[i] = rect_and_score[i].first.x;
        bounding_box_data[1 * result_size + i] = rect_and_score[i].first.y;
        bounding_box_data[2 * result_size + i] = rect_and_score[i].first.width;
        bounding_box_data[3 * result_size + i] = rect_and_score[i].first.height;
      }

      mxArray* mx_score = mxCreateDoubleMatrix(result_size, 1, mxREAL);
      double* score_data = (double *)mxGetData(mx_score);
      for (int i = 0; i < result_size; i++) {
        score_data[i] = rect_and_score[i].second;
      }

      int points_data_num = points[0].size() * 2;
      mxArray* mx_points = mxCreateDoubleMatrix(result_size, points_data_num, mxREAL);
      double* points_data = (double *)mxGetData(mx_points);
      for (int i = 0; i < result_size; i++) {
        for (int j = 0; j < points[i].size(); j++) {
          points_data[j * 2 * result_size + i] = points[i][j].x;
          points_data[(j * 2 + 1) * result_size + i] = points[i][j].y;
        }

      }

      const char* result_fields[3] = { "bounding_box", "score", "points" };
      mxArray* mx_result = mxCreateStructMatrix(1, 1, 3,
                                                result_fields);
      mxSetField(mx_result, 0, "bounding_box", mx_bounding_box);
      mxSetField(mx_result, 0, "score", mx_score);
      mxSetField(mx_result, 0, "points", mx_points);
      plhs[0] = mx_result;
    }
    else {
      const char* result_fields[3] = { "bounding_box", "score", "points" };
      mxArray* mx_result = mxCreateStructMatrix(1, 1, 3,
                                                result_fields);
      mxArray* mx_bounding_box = mxCreateDoubleMatrix(0, 0, mxREAL);
      mxArray* mx_score = mxCreateDoubleMatrix(0, 0, mxREAL);
      mxArray* mx_points = mxCreateDoubleMatrix(0, 0, mxREAL);
      mxSetField(mx_result, 0, "bounding_box", mx_bounding_box);
      mxSetField(mx_result, 0, "score", mx_score);
      mxSetField(mx_result, 0, "points", mx_points);
      plhs[0] = mx_result;
    }
  }
  else {
    mxERROR("Please call MatMTCNN(\"init_model\", model_path) first!");
  }
}

static void ForceDetect(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsUint8(prhs[0]) && mxIsDouble(prhs[1]),
          "Usage: MatMTCNN('force_detect', image, coarse_rect)");
  ReadMat(prhs[0]);
  //mexPrintf("Read Mat to OpenCV done.\n");
  double* rect_data = static_cast<double*>(mxGetData(prhs[1]));
  cv::Rect2d coarse_rect(rect_data[0], rect_data[1], rect_data[2], rect_data[3]);
  if (cascade != NULL) {
    vector<vector<Point2d>> points;
    auto rect_and_score = cascade->ForceGetLandmark(image, coarse_rect, 0.7, points);
    int result_size = rect_and_score.size();
    if (result_size > 0) {
      mxArray* mx_bounding_box = mxCreateDoubleMatrix(result_size, 4, mxREAL);
      double* bounding_box_data = (double *)mxGetData(mx_bounding_box);
      for (int i = 0; i < result_size; i++) {
        bounding_box_data[i] = rect_and_score[i].first.x;
        bounding_box_data[1 * result_size + i] = rect_and_score[i].first.y;
        bounding_box_data[2 * result_size + i] = rect_and_score[i].first.width;
        bounding_box_data[3 * result_size + i] = rect_and_score[i].first.height;
      }

      mxArray* mx_score = mxCreateDoubleMatrix(result_size, 1, mxREAL);
      double* score_data = (double *)mxGetData(mx_score);
      for (int i = 0; i < result_size; i++) {
        score_data[i] = rect_and_score[i].second;
      }

      int points_data_num = points[0].size() * 2;
      mxArray* mx_points = mxCreateDoubleMatrix(result_size, points_data_num, mxREAL);
      double* points_data = (double *)mxGetData(mx_points);
      for (int i = 0; i < result_size; i++) {
        for (int j = 0; j < points[i].size(); j++) {
          points_data[j * 2 * result_size + i] = points[i][j].x;
          points_data[(j * 2 + 1) * result_size + i] = points[i][j].y;
        }

      }

      const char* result_fields[3] = { "bounding_box", "score", "points" };
      mxArray* mx_result = mxCreateStructMatrix(1, 1, 3,
                                                result_fields);
      mxSetField(mx_result, 0, "bounding_box", mx_bounding_box);
      mxSetField(mx_result, 0, "score", mx_score);
      mxSetField(mx_result, 0, "points", mx_points);
      plhs[0] = mx_result;
    }
    else {
      const char* result_fields[3] = { "bounding_box", "score", "points" };
      mxArray* mx_result = mxCreateStructMatrix(1, 1, 3,
                                                result_fields);
      mxArray* mx_bounding_box = mxCreateDoubleMatrix(0, 0, mxREAL);
      mxArray* mx_score = mxCreateDoubleMatrix(0, 0, mxREAL);
      mxArray* mx_points = mxCreateDoubleMatrix(0, 0, mxREAL);
      mxSetField(mx_result, 0, "bounding_box", mx_bounding_box);
      mxSetField(mx_result, 0, "score", mx_score);
      mxSetField(mx_result, 0, "points", mx_points);
      plhs[0] = mx_result;
    }
  }
  else {
    mxERROR("Please call MatMTCNN(\"init_model\", model_path) first!");
  }
}

/** -----------------------------------------------------------------
** Available commands.
**/
struct handler_registry {
  string cmd;
  void(*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "init_model",         InitModel },
  { "release_model",      ReleaseModel },
  { "set_threshold",      SetThreshold },
  { "detect",             Detect },
  { "set_device",         set_device },
  { "force_detect",      ForceDetect },
  // The end.
  { "END",                NULL },
};

void mexFunction(MEX_ARGS) {
  mexLock();  // Avoid clearing the mex file.
  mxCHECK(nrhs > 0, "Usage: MatMTCNN(api_command, arg1, arg2, ...)");
  {// Handle input command
    char* cmd = mxArrayToString(prhs[0]);
    bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
        handlers[i].func(nlhs, plhs, nrhs - 1, prhs + 1);
        dispatched = true;
        break;
      }
    }
    if (!dispatched) {
      ostringstream error_msg;
      error_msg << "Unknown command '" << cmd << "'";
      mxERROR(error_msg.str().c_str());
    }
    mxFree(cmd);
  }
}