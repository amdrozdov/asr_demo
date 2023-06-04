#include "asr_app.h"

void usage(int argc, char ** argv, const app_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "  -m, --model FNAME, model path(required)\n");
    fprintf(stderr, "  -f, --file FNAME, input audio file name(optional)\n");
    fprintf(stderr, "\n");
}


bool arg_parse(int argc, char ** argv, app_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            usage(argc, argv, params);
            return false;
        }
        else if (arg == "-m"   || arg == "--model") {
            params.model = argv[++i];
        }
        else if (arg == "-f"   || arg == "--file") {
            params.input_wav = argv[++i];
        }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            usage(argc, argv, params);
            return false;
        }
    }
    if(params.model == ""){
        usage(argc, argv, params);
        return false;
    }
    return true;
}


int main(int argc, char ** argv) {
    // Parse arguments
    app_params params;
    if(!arg_parse(argc, argv, params)){
        return 1;
    }
    params.keep_ms = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    // Create app and run it
    AsrApp app(params);
    if(params.input_wav != ""){
        app.offline_inference();
	return 0;
    }
    app.realtime_inference();

    // Cleanup
    delete &app;
    return 0;
}
