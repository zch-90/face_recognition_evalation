#pragma once
#include "hdf5.h"
#include "hdf5_hl.h"
#include <string>
#include "opencv2\opencv.hpp"

#define HDF5_DATA_DATASET_NAME "/n_data"
#define HDF5_DATA_LABEL_NAME "/n_label"

using namespace cv;
using namespace std;

class HDF5Manager {
public:
  //chunk暂时必须等于total_num，hdf5这bug。。。
  HDF5Manager(string root_folder, string data_name = HDF5_DATA_DATASET_NAME, string label_name = HDF5_DATA_LABEL_NAME,
              string file_prefix = "negative12-", int channel = 3, int height = 12, int width = 12, int label_size = 2) :
              file_prefix_(file_prefix), file_id_(0), file_num_(0), chunk_(10000), total_num_(10000), this_num_(0), this_chunk_(0),
  mean_value(128), scale_value(0.0078125), num_axes_data(4), num_axes_label(2), pointer_chunk(0),
  channel_(channel), height_(height), width_(width), label_size_(label_size),
  root_folder_(root_folder),data_name_(data_name), label_name_(label_name) {
    fplist = fopen((root_folder + "hdf5_list.txt").c_str(), "w");
    dims_data = new hsize_t[num_axes_data];
    dims_data[0] = chunk_;
    dims_data[1] = channel_;
    dims_data[2] = height_;
    dims_data[3] = width_;
    dims_label = new hsize_t[num_axes_label];
    dims_label[0] = chunk_;
    dims_label[1] = label_size_;
    data_buffer = new float[chunk_ * channel_ * height * width_];
    label_buffer = new float[chunk_ * label_size_];
    createNew();
  }

  ~HDF5Manager() {
    close();
  }

  void createNew() {
    if (file_num_ > 0) herr_t status = H5Fclose(file_id_);
    string file_name = root_folder_ + file_prefix_ + to_string(file_num_++) + ".h5";
    cout << "Opening new hdf5 file:" << file_name << endl;
    fprintf(fplist, "%s\n", file_name.c_str());
    file_id_ = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                         H5P_DEFAULT);
  }

  void close() {
    if (pointer_chunk != 0) {
      dims_data[0] = pointer_chunk;
      dims_label[0] = pointer_chunk;
      herr_t status = H5LTmake_dataset_float(
        file_id_, data_name_.c_str(), num_axes_data, dims_data, data_buffer);
      if (status < 0) cout << "save data failed!";
      status = H5LTmake_dataset_float(
        file_id_, label_name_.c_str(), num_axes_label, dims_label, label_buffer);
      if (status < 0) cout << "save label failed!";
      H5Fflush(file_id_, H5F_SCOPE_GLOBAL);
    }
    if (file_num_ > 0) herr_t status = H5Fclose(file_id_);
    fclose(fplist);
    delete[] dims_data;
    delete[] dims_label;
    delete[] data_buffer;
    delete[] label_buffer;
  }

  void saveMat(Mat image, vector<float> label) {
    assert(label.size() == label_size_);
    float* transformed_data = data_buffer + pointer_chunk * channel_ * width_ * height_;
    int top_index;
    for (int h = 0; h < height_; ++h) {
      const uchar* ptr = image.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < width_; ++w) {
        int w_idx = w;
        for (int c = 0; c < channel_; ++c) {
          top_index = (c * height_ + h) * width_ + w;
          float pixel = static_cast<float>(ptr[img_index++]);
          transformed_data[top_index] = (pixel - mean_value) * scale_value;
        }
      }
    }
    memcpy(label_buffer + pointer_chunk * label_size_, label.data(), label_size_ * sizeof(float));
    pointer_chunk++;
    if (pointer_chunk == chunk_) {
      herr_t status = H5LTmake_dataset_float(
        file_id_, data_name_.c_str(), num_axes_data, dims_data, data_buffer);
      if (status < 0) cout << "save data failed!";
      status = H5LTmake_dataset_float(
        file_id_, label_name_.c_str(), num_axes_label, dims_label, label_buffer);
      if (status < 0) cout << "save label failed!";
      H5Fflush(file_id_, H5F_SCOPE_GLOBAL);
      this_chunk_++;
      pointer_chunk = 0;
      if (this_chunk_ * chunk_ == total_num_) {
        createNew();
        this_chunk_ = 0;
      }
    }
  }

  static void shuffle_data_list(string data_list,int shuffle_group = 5,
                                string data_name = HDF5_DATA_DATASET_NAME, string label_name = HDF5_DATA_LABEL_NAME) {
    std::vector<std::string> hdf_filenames_;
    unsigned int num_files_;
    unsigned int current_file_;
    hsize_t current_row_;
    std::vector<unsigned int> data_permutation_;
    std::vector<unsigned int> file_permutation_;
    std::ifstream source_file(data_list.c_str());
    if (source_file.is_open()) {
      std::string line;
      while (source_file >> line) {
        hdf_filenames_.push_back(line);
      }
    }
    else {
      cout << "Failed to open source file: " << data_list;
    }
    source_file.close();
    num_files_ = hdf_filenames_.size();
    current_file_ = 0;
    cout << "Number of HDF5 files: " << num_files_;

    file_permutation_.clear();
    file_permutation_.resize(num_files_);
    // Default to identity permutation.
    for (int i = 0; i < num_files_; i++) {
      file_permutation_[i] = i;
    }

    // Shuffle if needed.
    std::random_shuffle(file_permutation_.begin(), file_permutation_.end());

    // get data info
    for (unsigned int i = 0; i < num_files_ / shuffle_group; i++) {
      vector<hid_t> file_ids;
      vector<string> file_names;
      int data_ndims, label_ndims;
      std::vector<hsize_t> data_dims, label_dims;
      vector<float*> data_buffers, label_buffers;
      int data_length, label_length;

      int files_in_group = shuffle_group;
      if (i == num_files_ / shuffle_group - 1) files_in_group = num_files_ - i * shuffle_group;

      for (int j = 0; j < files_in_group; j++) {
        string current_filename = hdf_filenames_[file_permutation_[current_file_]].c_str();
        file_names.push_back(current_filename);
        hid_t file_id = H5Fopen(current_filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        file_ids.push_back(file_id);
        if (j == 0) {
          herr_t status = H5LTget_dataset_ndims(file_id, data_name.c_str(), &data_ndims);
          if (status < 0) {
            cout << "failed to load " << current_filename << " dataset_name:" << data_name<<endl;
            return;
          }
          data_dims.reserve(data_ndims);
          status = H5LTget_dataset_ndims(file_id, label_name.c_str(), &label_ndims);
          if (status < 0) {
            cout << "failed to load " << current_filename << " dataset_name:" << data_name << endl;
            return;
          }
          label_dims.reserve(label_ndims);
          H5T_class_t class_;
          status = H5LTget_dataset_info(file_id, data_name.c_str(), data_dims.data(), &class_, NULL);
          data_length = 1;
          for (auto counti : data_dims) {
            data_length *= counti;
          }
          label_length = 1;
          for (auto counti : label_dims) {
            label_length *= counti;
          }
        }
        float* data_buffer = new float[data_length];
        float* label_buffer = new float[label_length];
        herr_t status = H5LTread_dataset_float(file_id, data_name.c_str(), data_buffer);
        status = H5LTread_dataset_float(file_id, label_name.c_str(), label_buffer);
        data_buffers.push_back(data_buffer);
        label_buffers.push_back(label_buffer);
      }
      float* current_data = new float[data_length];
      float* label_data = new float[label_length];

    }
  }

  std::string root_folder_;
  std::string file_prefix_;
  hid_t file_id_;
  int file_num_;
  int chunk_, this_chunk_;
  int total_num_;
  int this_num_;
  float mean_value;
  float scale_value;
  FILE* fplist;
  int channel_, height_, width_, label_size_;
  int num_axes_data, num_axes_label;
  hsize_t *dims_data, *dims_label;
  float *data_buffer, *label_buffer;
  int pointer_chunk;
  string data_name_;
  string label_name_;
};