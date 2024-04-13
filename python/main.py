#!/bin/env python3

import cv2
import matplotlib.pyplot as plt  ###如无则pip安装
from yolov8face import YOLOface_8n
from face_68landmarks import face_68_landmarks
from face_recognizer import face_recognize
from face_swap import swap_face
from face_enhancer import enhance_face
import os
import argparse;


def main(weights_dir, output_dir, source, target):
    source_img = cv2.imread(source)
    target_img = cv2.imread(target)
    
    detect_face_net = YOLOface_8n(os.path.join(weights_dir , "yoloface_8n.onnx"))
    detect_68landmarks_net = face_68_landmarks(os.path.join(weights_dir , "2dfan4.onnx"))
    face_embedding_net = face_recognize(os.path.join(weights_dir , "arcface_w600k_r50.onnx"))
    swap_face_net = swap_face(os.path.join(weights_dir , "inswapper_128.onnx"), "./python/model_matrix.npy")
    enhance_face_net = enhance_face(os.path.join(weights_dir , "gfpgan_1.4.onnx"))

    boxes, _, _ = detect_face_net.detect(source_img)
    position = 0  ###一张图片里可能有多个人脸，这里只考虑1个人脸的情况
    bounding_box = boxes[position]
    _, face_landmark_5of68 = detect_68landmarks_net.detect(source_img, bounding_box)
    source_face_embedding, _ = face_embedding_net.detect(source_img, face_landmark_5of68)

    boxes, _, _ = detect_face_net.detect(target_img)
    position = 0  ###一张图片里可能有多个人脸，这里只考虑1个人脸的情况
    bounding_box = boxes[position]
    _, target_landmark_5 = detect_68landmarks_net.detect(target_img, bounding_box)

    swapimg = swap_face_net.process(target_img, source_face_embedding, target_landmark_5)
    resultimg = enhance_face_net.process(swapimg, target_landmark_5)
    
    plt.subplot(1, 2, 1)
    plt.imshow(source_img[:,:,::-1])  ###plt库显示图像是RGB顺序
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(target_img[:,:,::-1])
    plt.axis('off')
    # plt.show()
    plt.savefig(os.path.join(output_dir , "source_target.jpg"), dpi=600, bbox_inches='tight') ###保存高清图

    cv2.imwrite(os.path.join(output_dir , "result.jpg"), resultimg)
    
    # cv2.namedWindow('resultimg', 0)
    # cv2.imshow('resultimg', resultimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='facefusion-onnxrun')
    parser.add_argument('--weights_dir', type=str, default='./weights', help='模型文件目录 default ./weights')
    parser.add_argument('--output_dir', type=str, default='./sample_out', help='输出结果目录 default ./sample_out')
    parser.add_argument('source', type=str, help='source image')
    parser.add_argument('target', type=str, help='target image')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args.weights_dir, args.output_dir, args.source, args.target)
    



    
