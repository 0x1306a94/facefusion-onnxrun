#include "face68landmarks.h"
#include "faceenhancer.h"
#include "facerecognizer.h"
#include "faceswap.h"
#include "yolov8face.h"

#include <argparse/argparse.hpp>
#include <filesystem>

namespace fs = std::filesystem;

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {

    argparse::ArgumentParser program("facefusion-onnxrun");
    program.add_argument("--weights")
        .required()
        .help("weights files dir.");

    program.add_argument("--out")
        .required()
        .help("out dir.");

    program.add_argument("--source")
        .required()
        .help("source image.");

    program.add_argument("--target")
        .required()
        .help("target image.");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return EXIT_FAILURE;
    }

    string weights_dir = program.get<std::string>("--weights");
    string out_dir = program.get<std::string>("--out");
    string source_path = program.get<std::string>("--source");
    string target_path = program.get<std::string>("--target");

    if (fs::path(out_dir).is_relative()) {
        out_dir = fs::canonical(out_dir).string();
    }

    if (!fs::exists(out_dir)) {
        fs::create_directories(out_dir);
    }

    if (fs::path(weights_dir).is_relative()) {
        weights_dir = fs::canonical(weights_dir).string();
    }

    std::cerr << "weights_dir: " << weights_dir << std::endl;
    std::cerr << "out_dir: " << out_dir << std::endl;
    ////图片路径和onnx文件的路径，要确保写正确，才能使程序正常运行的
    Yolov8Face detect_face_net(fs::path(weights_dir).append("yoloface_8n.onnx").string());
    Face68Landmarks detect_68landmarks_net(fs::path(weights_dir).append("2dfan4.onnx").string());
    FaceEmbdding face_embedding_net(fs::path(weights_dir).append("arcface_w600k_r50.onnx").string());
    SwapFace swap_face_net(fs::path(weights_dir).append("inswapper_128.onnx").string(), fs::path(weights_dir).append("model_matrix.bin").string());
    FaceEnhance enhance_face_net(fs::path(weights_dir).append("gfpgan_1.4.onnx").string());

    Mat source_img = imread(fs::canonical(source_path).string());
    Mat target_img = imread(fs::canonical(target_path).string());

    vector<Bbox> boxes;
    detect_face_net.detect(source_img, boxes);
    int position = 0;  ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
    vector<Point2f> face_landmark_5of68;
    vector<Point2f> face68landmarks = detect_68landmarks_net.detect(source_img, boxes[position], face_landmark_5of68);
    vector<float> source_face_embedding = face_embedding_net.detect(source_img, face_landmark_5of68);

    detect_face_net.detect(target_img, boxes);
    position = 0;  ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
    vector<Point2f> target_landmark_5;
    detect_68landmarks_net.detect(target_img, boxes[position], target_landmark_5);

    Mat swapimg = swap_face_net.process(target_img, source_face_embedding, target_landmark_5);
    Mat resultimg = enhance_face_net.process(swapimg, target_landmark_5);

    imwrite(fs::path(out_dir).append("resultimg.jpg").string(), resultimg);

    /*static const string kWinName = "Deep learning face swap use onnxruntime";
    namedWindow(kWinName, WINDOW_NORMAL);
    imshow(kWinName, resultimg);
    waitKey(0);
    destroyAllWindows();*/
}
