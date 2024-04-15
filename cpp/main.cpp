#include "face68landmarks.h"
#include "faceenhancer.h"
#include "facerecognizer.h"
#include "faceswap.h"
#include "yolov8face.h"

#ifdef USE_OPENCV_HIGHGUI
#include <opencv2/highgui.hpp>
#endif

#include <argparse/argparse.hpp>
#include <chrono>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

using namespace cv;
using namespace std;

class Timer {
  public:
    Timer(const std::string &name = "Timer", std::ostream &os = std::cerr)
        : m_name(name)
        , m_os(os)
        , m_start_time_point(std::chrono::high_resolution_clock::now()) {}

    void reset(bool print_on_reset = true, const std::string &new_name = "Timer") {
        if (!m_printed && print_on_reset) {
            print();
        }
        m_name = new_name;
        m_start_time_point = std::chrono::high_resolution_clock::now();
        m_printed = false;
    }

    double elapsed() const {
        auto end_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time_point - m_start_time_point);
        return duration.count();
    }

    void print() const {
        auto end_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time_point - m_start_time_point);
        m_os << m_name << " elapsed time: " << duration.count() << " ms" << std::endl;
        m_printed = true;
    }

    void abort(bool print_on_abort = false) {
        m_abort = true;
        if (!m_printed && print_on_abort) {
            print();
        }
    }

    ~Timer() {
        if (!m_printed && !m_abort) {
            print();
        }
    }

  private:
    std::string m_name;
    std::ostream &m_os;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start_time_point;
    mutable bool m_printed;
    bool m_abort;
};

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
  
    if (!fs::exists(out_dir)) {
        fs::create_directories(out_dir);
    }

    if (fs::path(out_dir).is_relative()) {
        out_dir = fs::canonical(out_dir).string();
    }

    if (fs::path(weights_dir).is_relative()) {
        weights_dir = fs::canonical(weights_dir).string();
    }

    std::cerr << "weights_dir: " << weights_dir << std::endl;
    std::cerr << "out_dir: " << out_dir << std::endl;

    std::stringstream elapsed_log;
    Timer timer("Load Model", elapsed_log);
    ////图片路径和onnx文件的路径，要确保写正确，才能使程序正常运行的
    Yolov8Face detect_face_net(fs::path(weights_dir).append("yoloface_8n.onnx").string());
    Face68Landmarks detect_68landmarks_net(fs::path(weights_dir).append("2dfan4.onnx").string());
    FaceEmbdding face_embedding_net(fs::path(weights_dir).append("arcface_w600k_r50.onnx").string());
    SwapFace swap_face_net(fs::path(weights_dir).append("inswapper_128.onnx").string(), fs::path(weights_dir).append("model_matrix.bin").string());
    FaceEnhance enhance_face_net(fs::path(weights_dir).append("gfpgan_1.4.onnx").string());
    timer.reset(true, "Load Image");

    Mat source_img = imread(fs::canonical(source_path).string());
    Mat target_img = imread(fs::canonical(target_path).string());
    timer.reset(true, "Detect Face Source Image");
    vector<Bbox> boxes;
    detect_face_net.detect(source_img, boxes);
    timer.reset(true, "Detect Landmarks Source Image");
    int position = 0;  ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
    vector<Point2f> face_landmark_5of68;
    vector<Point2f> face68landmarks = detect_68landmarks_net.detect(source_img, boxes[position], face_landmark_5of68);
    timer.reset(true, "Detect Embedding Source Image");
    vector<float> source_face_embedding = face_embedding_net.detect(source_img, face_landmark_5of68);
    timer.reset(true, "Detect Face Target Image");
    detect_face_net.detect(target_img, boxes);
    timer.reset(true, "Detect Landmarks Target Image");
    position = 0;  ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
    vector<Point2f> target_landmark_5;
    detect_68landmarks_net.detect(target_img, boxes[position], target_landmark_5);
    timer.reset(true, "Swap Face");
    Mat swapimg = swap_face_net.process(target_img, source_face_embedding, target_landmark_5);
    timer.reset(true, "Enhance Face");
    Mat resultimg = enhance_face_net.process(swapimg, target_landmark_5);
    timer.abort(true);
    std::cerr << "###################### elapsed summary ######################\n"
              << elapsed_log.str() << std::endl;

    int max_height = std::max(source_img.rows, target_img.rows);
    cv::Mat merged_image(max_height, source_img.cols + target_img.cols, source_img.type(), cv::Scalar(128, 128, 128));

    cv::Rect roi1(cv::Rect(0, (max_height - source_img.rows) * 0.5, source_img.cols, source_img.rows));
    source_img.copyTo(merged_image(roi1));

    cv::Rect roi2(cv::Rect(source_img.cols, (max_height - target_img.rows) * 0.5, target_img.cols, target_img.rows));
    target_img.copyTo(merged_image(roi2));

    imwrite(fs::path(out_dir).append("source_target.jpg").string(), merged_image);

    imwrite(fs::path(out_dir).append("result.jpg").string(), resultimg);

    //    static const string kWinName = "Deep learning face swap use onnxruntime";
#ifdef USE_OPENCV_HIGHGUI
    namedWindow("source_target.jpg", WINDOW_NORMAL);
    namedWindow("result.jpg", WINDOW_NORMAL);
    imshow("source_target.jpg", merged_image);
    imshow("result.jpg", resultimg);
    waitKey(0);
    destroyAllWindows();
#endif

    return EXIT_SUCCESS;
}
