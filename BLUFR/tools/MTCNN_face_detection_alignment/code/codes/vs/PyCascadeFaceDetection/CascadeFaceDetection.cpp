#include "CascadeFaceDetection.h"
#include "CaffeBinding.h"
#include "../Test/util/BoundingBox.inc.h"
#include "../Test/TestFaceDetection.inc.h"
#include "pyboostcvconverter.hpp"

using namespace std;
caffe::CaffeBinding* kCaffeBinding = new caffe::CaffeBinding();

namespace FaceInception {
  CascadeCNN* cascade;
  CascadeFaceDetection::CascadeFaceDetection() {
    cout << "Please specify the net models." << endl;
  }

  PyObject* pyopencv_from(const FaceInformation &info) {
    PyObject* seq = PyList_New(3);
    //ERRWRAP2(
    PyObject* rect = Py_BuildValue("(dddd)", info.boundingbox.x, info.boundingbox.y, info.boundingbox.width, info.boundingbox.height);
    PyList_SET_ITEM(seq, 0, rect);
    PyObject* score = PyFloat_FromDouble(info.confidence);
    PyList_SET_ITEM(seq, 1, score);
    PyObject* points = PyList_New(5);
    for (int i = 0; i < 5; i++) {
      PyObject* item = Py_BuildValue("(dd)", info.points[i].x, info.points[i].y);
      PyList_SET_ITEM(points, i, item);
    }
    PyList_SET_ITEM(seq, 2, points);
    //)
    return seq;
  }

  static inline PyObject* pyopencv_from_face_info_vec(const std::vector<FaceInformation>& value) {
    int i, n = (int)value.size();
    PyObject* seq = PyList_New(n);
    //ERRWRAP2(
    for (i = 0; i < n; i++) {
      PyObject* item = pyopencv_from(value[i]);
      if (!item)
        break;
      PyList_SET_ITEM(seq, i, item);
    }
    //if (i < n) {
    //  Py_DECREF(seq);
    //  return 0;
    //}
    //)
    return seq;
  }

  CascadeFaceDetection::CascadeFaceDetection(std::string net12_definition, std::string net12_weights,
                                             std::string net12_stitch_definition, std::string net12_stitch_weights,
                                             std::string net24_definition, std::string net24_weights,
                                             std::string net48_definition, std::string net48_weights,
                                             std::string netLoc_definition, std::string netLoc_weights,
                                             int gpu_id) {
    cascade = new CascadeCNN(net12_definition, net12_weights,
                             net12_stitch_definition, net12_stitch_weights,
                             net24_definition, net24_weights,
                             net48_definition, net48_weights,
                             netLoc_definition, netLoc_weights,
                             gpu_id);
  }

  std::vector<FaceInformation> CascadeFaceDetection::Predict(cv::Mat& input_image, vector<double> thresholds, double min_face) {
    std::vector<FaceInformation> result;
    vector<vector<Point2d>> points;
    if (cascade != NULL) {
      auto rect_and_score = cascade->GetDetection(input_image, 12.0 / min_face, thresholds, true, 0.7, true, points);
      for (int i = 0; i < rect_and_score.size();i++) {
        result.push_back(FaceInformation{ rect_and_score[i].first, rect_and_score[i].second, points[i] });
      }
    }
    return result;
  }

  PyObject * CascadeFaceDetection::Predict(PyObject * input) {
    Mat input_image = FaceInception::fromNDArrayToMat(input);
    if (!input_image.data) return nullptr;
    auto faces = Predict(input_image, { 0.6, 0.6, 0.7 });
    return pyopencv_from_face_info_vec(faces);
  }

  PyObject * CascadeFaceDetection::Predict(PyObject * input, PyObject * confidence_threshold, PyObject * min_face) {
    Mat input_image;
    ERRWRAP2(input_image = FaceInception::fromNDArrayToMat(input));
    if (!input_image.data || !PyFloat_Check(min_face)) return nullptr;
    vector<double> thesholds(3);
    PyArg_ParseTuple(confidence_threshold, "dddd", &thesholds[0], &thesholds[1], &thesholds[2]);
    auto faces = Predict(input_image, thesholds, PyFloat_AsDouble(min_face));
    return pyopencv_from_face_info_vec(faces);
  }

  PyObject* CascadeFaceDetection::ForceGetLandmark(PyObject* input, PyObject * CoarseRect) {
    Mat input_image;
    ERRWRAP2(input_image = FaceInception::fromNDArrayToMat(input));
    cv::Rect2d r;
    PyArg_ParseTuple(CoarseRect, "dddd", &r.x, &r.y, &r.width, &r.height);
    cout << r << endl;
    vector<vector<Point2d>> points;
    auto rect_and_score = cascade->ForceGetLandmark(input_image, r, 0.7, points);
    vector<FaceInformation> result;
    for (int i = 0; i < rect_and_score.size(); i++) {
      result.push_back(FaceInformation{ rect_and_score[i].first, rect_and_score[i].second, points[i] });
    }
    return pyopencv_from_face_info_vec(result);
  }

  CascadeFaceDetection::~CascadeFaceDetection() {
    delete cascade;
    delete kCaffeBinding;
  }
}