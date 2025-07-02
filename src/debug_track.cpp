#include <gflags/gflags.h>
#include "tracker.hpp"

DEFINE_string(process_folder, "/home/sunrise/qimeng/dataset/videos", "Process folder which is used to track");

int main(int argc, char** argv) {
    Tracker tracker;
    google::ParseCommandLineFlags(&argc, &argv, true);
    tracker.track_FromImgsFolder(FLAGS_process_folder);
    // tracker.track_FromVideosFolder(FLAGS_process_folder);

    return 0;
}