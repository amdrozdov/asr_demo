#include "third_party/common.h"
#include "third_party/common-sdl.h"
#include "third_party/whisper.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>


struct app_params {
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;

    std::string model     = "";
    std::string input_wav = "";
};


struct realtime_audio {
    std::vector<float> pcmf32;
    std::vector<float> pcmf32_new;
    std::vector<float> pcmf32_old;
};


class AsrApp
{
    private:
    struct app_params params;
    realtime_audio audio_buf;

    struct whisper_context *ctx;

    int n_samples_step;
    int n_samples_len;
    int n_samples_keep;
    int n_samples_30s;
    int n_new_line;

    void read_audio(audio_async *audio);
    whisper_full_params default_asr_params();
    void inference_result();

    public:
    void offline_inference();
    void realtime_inference();

    AsrApp(struct app_params asr_params){
        params = asr_params;
        ctx = whisper_init_from_file(params.model.c_str());
    }
    ~AsrApp(){
        whisper_print_timings(ctx);
        whisper_free(ctx);
    }
};
