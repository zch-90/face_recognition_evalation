% --------------------------------------------------------
% Copyright (c) Weiyang Liu, Yandong Wen
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to evaluate the performance of the trained model on LFW dataset.
% We perform 10-fold cross validation, using cosine similarity as metric.
% More details about the testing protocol can be found at http://vis-www.cs.umass.edu/lfw/#views.
% 
% Usage:
% cd $SPHEREFACE_ROOT/test
% run code/evaluation.m
% --------------------------------------------------------

function extract_features()

clear;clc;close all;
cd('../')

%% caffe setttings
if exist('/home/ml/caffe/matlab/+caffe', 'dir')
  addpath('/home/ml/caffe/matlab');
else
  error('Please run this demo from caffe/matlab/demo');
end
gpu = 1;
if gpu
   gpu_id = 0;
   caffe.set_mode_gpu();
   caffe.set_device(gpu_id);
else
   caffe.set_mode_cpu();
end
caffe.reset_all();

%% deploy.prototxt
model   = '../model/sphereface_deploy.prototxt';
%% caffemodel
weights = '../model/sphereface_model.caffemodel';
net     = caffe.Net(model, weights, 'test');
net.save('../model/sphereface_model1.caffemodel');

%% compute features
pairs = parseList('result/lfw-112X96_5749_13233.txt', fullfile(pwd, 'result/lfw-112X96'));

for i = 1:length(pairs)
    fprintf('extracting deep features from the %dth face pair...\n', i);
    pairs(i).features = extractDeepFeature(pairs(i).files, net);
    Descriptors(i,:) = (pairs(i).features)';
end
save ../data/sphereface.mat Descriptors

function pairs = parseList(list, folder)
    i    = 0;
    fid  = fopen(list);
    line = fgets(fid);
    while ischar(line)
          strings = strsplit(line, ' ');
          if length(strings) == 2
              i = i + 1;
              pairs(i).files = fullfile(folder, strings{1});
          end
          line = fgets(fid);
    end
    fclose(fid);
end

function feature = extractDeepFeature(file, net)
    img     = single(imread(file));
    img     = (img - 127.5)/128;
    img     = permute(img, [2,1,3]);
    img     = img(:,:,[3,2,1]);
    res     = net.forward({img});
    res_    = net.forward({flip(img, 1)});
    feature = double([res{1}; res_{1}]);
end

end