#include <gflags/gflags.h>
#include "tracker.hpp"

DEFINE_string(process_folder, "/home/sunrise/qimeng/dataset/videos", "Process folder which is used to track");
DEFINE_string(model_path, "You should input model path", "Model path");

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    Tracker tracker(FLAGS_model_path);
    tracker.track_FromImgsFolder(FLAGS_process_folder);
    // tracker.track_FromVideosFolder(FLAGS_process_folder);

    return 0;
}