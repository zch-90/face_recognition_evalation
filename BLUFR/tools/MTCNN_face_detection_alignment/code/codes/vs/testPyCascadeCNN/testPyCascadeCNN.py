print "loading library..."
import cv2
import sys;
import os;
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../x64/Release"))
import CascadeFaceDetection

model_folder = "D:/face project/MTCNN_face_detection_alignment/code/codes/MTCNNv2/model/"

print "loading model from " + model_folder + "..."
CascadeCNN = CascadeFaceDetection.CascadeCNN(model_folder + "det1-memory.prototxt", model_folder + "det1.caffemodel",
                     model_folder + "det1-memory-stitch.prototxt", model_folder + "det1.caffemodel",
                     model_folder + "det2-memory.prototxt", model_folder + "det2.caffemodel",
                     model_folder + "det3-memory.prototxt", model_folder + "det3.caffemodel",
                     model_folder + "det4-memory.prototxt", model_folder + "det4.caffemodel",
                     0) # 0 means to use the first gpu, -1 means to use cpu.
print "done."
I = cv2.imread("E:/datasets/lfw/Muhammad_Saeed_al-Sahhaf/Muhammad_Saeed_al-Sahhaf_0003.jpg")
print I.shape
# CascadeCNN.Predict(image, min_threshold, min_face)
#result = CascadeCNN.Predict(I, 0.9, 10.0) # speed is slow in the first detection
#result = CascadeCNN.Predict(I, 0.9, 10.0)
result = CascadeCNN.ForceGetLandmark(I, (65.0, 85.0, 100.0, 120.0));
for face in result:
    cv2.rectangle(I, (int(face[0][0]), int(face[0][1])), (int(face[0][0]+face[0][2]), int(face[0][1]+face[0][3])), (255, 100, 0), 2)
    for i in range(5):
        cv2.circle(I, (int(face[2][i][0]), int(face[2][i][1])), 1, (0, 0, 255), 2)

#J = cv2.resize(I, (0,0), fx=0.5, fy=0.5)
cv2.imshow("detection result", I)
cv2.waitKey(0)