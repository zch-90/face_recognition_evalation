#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "pyboostcvconverter.hpp"
#include "CascadeFaceDetection.h"

namespace FaceInception {

    using namespace boost::python;


#if (PY_VERSION_HEX >= 0x03000000)

    static void *init_ar() {
#else
        static void init_ar(){
#endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (CascadeFaceDetection) {
        //using namespace XM;
        init_ar();

        //initialize converters
        to_python_converter<cv::Mat,
                FaceInception::matToNDArrayBoostConverter>();
        FaceInception::matFromNDArrayBoostConverter();

        //class_<std::vector<FaceInformation>>("pyvector_faceinfo")
        //  .def(vector_indexing_suite<std::vector<FaceInformation>>());

        //class_<std::vector<Point2d>>("pyvector_point2d")
        //  .def(vector_indexing_suite<std::vector<Point2d>>());

        std::vector<FaceInformation>(CascadeFaceDetection::*Predict_func0)(cv::Mat&, double, double) = &CascadeFaceDetection::Predict;
        PyObject*(CascadeFaceDetection::*Predict_func1)(PyObject*, PyObject*, PyObject*) = &CascadeFaceDetection::Predict;
        

        class_<CascadeFaceDetection>("CascadeCNN", init<std::string, std::string,
                                     std::string, std::string,
                                     std::string, std::string,
                                     std::string, std::string,
                                     std::string, std::string,
                                     int>())
          .def("Predict", Predict_func1)
          .def("ForceGetLandmark", &CascadeFaceDetection::ForceGetLandmark);
    }

} //end namespace FaceInception
