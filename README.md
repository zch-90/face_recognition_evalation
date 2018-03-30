# face_recognition_evalation
人脸识别BLUFR评价方法：

1. 图片预处理
cd preprocess/code
run face_detect_demo.m //人脸、关键点检测
run face_align_demo.m //人脸对齐，裁剪大小为112x96，保存到result/lfw-112x96
python lfw_txt_maker.py //获取图片列表
run extract_features.m  //提取特征，并保存到../data/名字.mat

2. 获取ROC曲线、CMC曲线
cd code 
run demo_pca.m //画出训练模型的ROC曲线、CMC曲线，并保存到../result/
