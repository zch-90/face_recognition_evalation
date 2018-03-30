#pragma once

#ifdef CASCADEDETECTION_EXPORTS
#define CASCADE_DLL __declspec(dllexport)
#else
#define CASCADE_DLL __declspec(dllimport)
#endif

#include <opencv2\opencv.hpp>
#include <Python.h>

namespace FaceInception {
  struct FaceInformation {
    cv::Rect2d boundingbox;
    float confidence;
    std::vector<cv::Point2d> points;
  };
  class CASCADE_DLL CascadeFaceDetection {
  public:
    CascadeFaceDetection();
    CascadeFaceDetection(std::string net12_definition, std::string net12_weights,
                         std::string net12_stitch_definition, std::string net12_stitch_weights,
                         std::string net24_definition, std::string net24_weights,
                         std::string net48_definition, std::string net48_weights,
                         std::string netLoc_definition, std::string netLoc_weights,
                         int gpu_id = -1);
    // confidence_threshold default: (0.6, 0.7, 0.7), min_face default: 24
    std::vector<FaceInformation> Predict(cv::Mat& input_image, vector<double> thresholds = {0.6, 0.6, 0.7}, double min_face = 24.0);
    PyObject* Predict(PyObject* input);
    PyObject* Predict(PyObject* input, PyObject * confidence_threshold, PyObject * min_face);
    PyObject* ForceGetLandmark(PyObject* input, PyObject * CoarseRect);
    ~CascadeFaceDetection();
  };
}