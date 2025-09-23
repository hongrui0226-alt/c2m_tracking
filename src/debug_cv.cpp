#include <gflags/gflags.h>
#include "tracker.hpp"

DEFINE_string(process_folder, "/home/sunrise/qimeng/dataset/videos", "Process folder which is used to track");
DEFINE_string(model_path, "yolo11n-modified.bin", "Model path");

int main(int argc, char** argv) {
    Tracker tracker(FLAGS_model_path);
    google::ParseCommandLineFlags(&argc, &argv, true);
    tracker.cv_debug_FromImgsFolder(FLAGS_process_folder);
    // tracker.track_FromVideosFolder(FLAGS_process_folder);

    return 0;
}