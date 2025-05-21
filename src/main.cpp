#include "config.h"
#include <iostream>
#include <stdexcept>

void run_scheduler(const Config& cfg);

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        print_config(cfg);
        run_scheduler(cfg);
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
