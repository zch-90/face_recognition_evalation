#pragma once
#include <random> // std::default_random_engine
#include <chrono> // std::chrono::system_clock
#include <iostream>
#include <io.h>
#include <direct.h>
#include <string>
#include <iomanip>
#include <Windows.h>
#include <AtlBase.h>
#include <fstream>

void getFiles(std::string path, std::vector<std::pair<std::string, int> >& files, bool create_list = false) {
  //文件句柄  
  long   hFile = 0;
  //文件信息  
  struct _finddata_t fileinfo;

  std::string p;

  static int auto_label = -1;
  int label = 0;
  p.clear();
  if (create_list) {
    FILE *fp = fopen(p.assign(path).append("/label.txt").c_str(), "r");
    if (fp == NULL) {
      label = auto_label++;
      std::cout << path << " label:" << label << std::endl;
    }
    else {
      fscanf(fp, "%d", &label);
      if (auto_label<label) auto_label = label + 1;
      fclose(fp);
    }
    _mkdir(p.assign(path).insert(3, "align-").c_str());
  }
  
  bool getImage = false;
  if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1) {

    do {
      //如果是目录,迭代之
      //如果不是,加入列表
      if ((fileinfo.attrib   &   _A_SUBDIR)) {
        if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
          getFiles(p.assign(path).append("/").append(fileinfo.name), files);
      }
      else {
        if (strstr(fileinfo.name, "png") != NULL || strstr(fileinfo.name, "bmp") != NULL || strstr(fileinfo.name, "jpg") != NULL || strstr(fileinfo.name, "tif") != NULL) {
          files.push_back(make_pair(p.assign(path).append("\\").append(fileinfo.name), label));
          getImage = true;
        }
      }
    } while (_findnext(hFile, &fileinfo) == 0);
    //if(!getImage) auto_label--;
    _findclose(hFile);
  }
}