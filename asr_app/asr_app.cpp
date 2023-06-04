#include "asr_app.h"
#include <iostream>
#include <string>
#include <curl/curl.h>


static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

void send_data(const char *data){
  /* In real life we will send data to the extarnal service or queue,
   *  but for simplified demo we are sending data directly over HTTP protocol*/
  CURL *curl;
  CURLcode res;
  std::string readBuffer;

  curl = curl_easy_init();
  if(curl) {
    curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:5000/speech");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);

    struct curl_slist *hs=NULL;
    hs = curl_slist_append(hs, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hs);
    res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
  }
}


whisper_full_params AsrApp::default_asr_params(){
    // In real life we will take inference settings from the config file
    // for simplicity I will keep them hardcoded
    whisper_full_params wparams = whisper_full_default_params(
        WHISPER_SAMPLING_GREEDY
    );

    wparams.strategy = WHISPER_SAMPLING_GREEDY;
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = true;
    wparams.print_special    = false;
    wparams.translate        = false;
    wparams.language         = "en";
    wparams.detect_language  = false;
    wparams.n_threads        = 4;
    wparams.offset_ms        = 0;
    wparams.duration_ms      = 0;
    wparams.token_timestamps = false;
    wparams.thold_pt         = 0.01f;
    wparams.max_len          = 0;
    wparams.split_on_word    = false;
    wparams.speed_up         = false;
    wparams.initial_prompt   = "";
    wparams.greedy.best_of   = 2;
    wparams.beam_search.beam_size = -1;
    wparams.temperature_inc  = wparams.temperature_inc;
    wparams.entropy_thold    = 2.40f;
    wparams.logprob_thold    = -1.00f;

    return wparams;
}


void AsrApp::inference_result(){
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; i++) {
	// print inference result
        const char * text = whisper_full_get_segment_text(ctx, i);
        printf("%s", text);

	// send to the service
	std::string json = "{\"text\" : \""+std::string(text)+"\"}";
	send_data(json.c_str());

        fflush(stdout);
    }
}


void AsrApp::offline_inference(){
    std::vector<float> pcmf32_in;
    std::vector<std::vector<float>> pcmf32_stereo_in;

    if (!::read_wav(params.input_wav, pcmf32_in, pcmf32_stereo_in, false)) {
        fprintf(
            stderr, "error: failed to read WAV file '%s'\n",
	    params.input_wav.c_str()
	);
        exit(1);
    }
    whisper_full_params wparams = default_asr_params();

    printf("Inference in progress...\n");
    int result_err = whisper_full_parallel(
        ctx, wparams, pcmf32_in.data(), pcmf32_in.size(), 1
    );

    if (result_err) {
        fprintf(
            stderr, "%s: failed to process audio\n",
	    params.input_wav.c_str()
	);
        exit(1);
    }
    printf("Output utterances:\n");
    inference_result();
    printf("\nDone\n");
}


void AsrApp::read_audio(audio_async *audio){
    while (true) {
	audio->get(params.step_ms, audio_buf.pcmf32_new);
	if ((int) audio_buf.pcmf32_new.size() > 2*n_samples_step) {
	    fprintf(
                stderr,
		"\n\n%s: Cannot process audio fast enough\n\n",
		__func__
	    );
	    audio->clear();
	    continue;
	}
	if ((int) audio_buf.pcmf32_new.size() >= n_samples_step) {
	    audio->clear();
	    break;
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const int n_samples_new = audio_buf.pcmf32_new.size();
    // take up to params.length_ms audio from previous iteration
    const int n_samples_take = std::min(
        (int) audio_buf.pcmf32_old.size(),
	std::max(0, n_samples_keep + n_samples_len - n_samples_new)
    );

    audio_buf.pcmf32.resize(n_samples_new + n_samples_take);
    int idx = 0;
    for (int i = 0; i < n_samples_take; i++) {
	idx = audio_buf.pcmf32_old.size() - n_samples_take + i;
	audio_buf.pcmf32[i] = audio_buf.pcmf32_old[idx];
    }
    memcpy(
        audio_buf.pcmf32.data() + n_samples_take,
	audio_buf.pcmf32_new.data(),
	n_samples_new*sizeof(float)
    );
    audio_buf.pcmf32_old = audio_buf.pcmf32;
}


void AsrApp::realtime_inference(){
    // Initialize audio interface
    audio_async audio(params.length_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        exit(1);
    }
    audio.resume();

    n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;
    n_new_line = std::max(1, params.length_ms / params.step_ms - 1);

    /* realtime_audio audio_buf; */
    audio_buf.pcmf32 = std::vector<float>(n_samples_30s, 0.0f);
    audio_buf.pcmf32_new = std::vector<float>(n_samples_30s, 0.0f);
    audio_buf.pcmf32_old = std::vector<float>();

    std::vector<whisper_token> prompt_tokens;

    int n_iter = 0;
    bool is_running = true;
    printf("Stream is online. Ready to transcribe\n");
    fflush(stdout);

    auto t_last  = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;

    // main audio loop
    while (is_running) {
        // Catch ctrl+c
        is_running = sdl_poll_events();
        if (!is_running) {
            break;
        }

	// Get new audio chunk
	read_audio(&audio);

        // Run the inference
        whisper_full_params wparams = default_asr_params();
        wparams.single_segment   = true;
        wparams.prompt_tokens    = prompt_tokens.data();
        wparams.prompt_n_tokens  = prompt_tokens.size();

	int result_err = whisper_full(
            ctx, wparams,
	    audio_buf.pcmf32.data(), audio_buf.pcmf32.size()
	);
	if(result_err) {
            fprintf(stderr, "failed to process audio\n");
            exit(1);
        }

        // Print long empty line to clear the previous line
        printf("\33[2K\r");
        printf("%s", std::string(100, ' ').c_str());
        printf("\33[2K\r");
        // Print current output state
        inference_result();

        n_iter++;
        if (!(n_iter % n_new_line)) {
            // Print next line (we are done with current utterance)
            printf("\n");
            // Append current audio to next iteration
            audio_buf.pcmf32_old = std::vector<float>(
                audio_buf.pcmf32.end() - n_samples_keep, audio_buf.pcmf32.end()
            );
            // Append tokens from current iteration as prompt
            prompt_tokens.clear();
            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; i++) {
                const int token_count = whisper_full_n_tokens(ctx, i);
                for (int j = 0; j < token_count; j++) {
                    prompt_tokens.push_back(
		        whisper_full_get_token_id(ctx, i, j)
	            );
                }
            }
        }
        fflush(stdout);
    }
    audio.pause();
}
