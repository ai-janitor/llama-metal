#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <signal.h>
#include <unistd.h>

#ifdef HAVE_AUDIOTOOLBOX
#include <AudioToolbox/AudioToolbox.h>
#endif

using json = nlohmann::ordered_json;

enum outetts_version {
    OUTETTS_V0_2,
    OUTETTS_V0_3,
};

//
// Terminal utils
//

#define SQR(X)    ((X) * (X))
#define UNCUBE(x) x < 48 ? 0 : x < 115 ? 1 : (x - 35) / 40

/**
 * Quantizes 24-bit RGB to xterm256 code range [16,256).
 */
static int rgb2xterm256(int r, int g, int b) {
    unsigned char cube[] = {0, 0137, 0207, 0257, 0327, 0377};
    int av, ir, ig, ib, il, qr, qg, qb, ql;
    av = r * .299 + g * .587 + b * .114 + .5;
    ql = (il = av > 238 ? 23 : (av - 3) / 10) * 10 + 8;
    qr = cube[(ir = UNCUBE(r))];
    qg = cube[(ig = UNCUBE(g))];
    qb = cube[(ib = UNCUBE(b))];
    if (SQR(qr - r) + SQR(qg - g) + SQR(qb - b) <=
        SQR(ql - r) + SQR(ql - g) + SQR(ql - b))
        return ir * 36 + ig * 6 + ib + 020;
    return il + 0350;
}

static std::string set_xterm256_foreground(int r, int g, int b) {
    int x = rgb2xterm256(r, g, b);
    std::ostringstream oss;
    oss << "\033[38;5;" << x << "m";
    return oss.str();
}

const std::vector<std::string> k_colors = {
    set_xterm256_foreground(220,   5,  12),
    set_xterm256_foreground(232,  96,  28),
    set_xterm256_foreground(241, 147,  45),
    set_xterm256_foreground(246, 193,  65),
    set_xterm256_foreground(247, 240,  86),
    set_xterm256_foreground(144, 201, 135),
    set_xterm256_foreground( 78, 178, 101),
};

struct wav_header {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 1; // Mono
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};

static bool save_wav16(const std::string & fname, const std::vector<float> & data, int sample_rate) {
    std::ofstream file(fname, std::ios::binary);
    if (!file) {
        LOG_ERR("%s: Failed to open file '%s' for writing.\n", __func__, fname.c_str());
        return false;
    }

    wav_header header;
    header.sample_rate = sample_rate;
    header.byte_rate = header.sample_rate * header.num_channels * (header.bits_per_sample / 8);
    header.block_align = header.num_channels * (header.bits_per_sample / 8);
    header.data_size = data.size() * (header.bits_per_sample / 8);
    header.chunk_size = 36 + header.data_size;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    for (const auto & sample : data) {
        int16_t pcm_sample = static_cast<int16_t>(std::clamp(sample * 32767.0, -32768.0, 32767.0));
        file.write(reinterpret_cast<const char*>(&pcm_sample), sizeof(pcm_sample));
    }

    return file.good();
}

static void fill_hann_window(int length, bool periodic, float * output) {
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}

// precomputed twiddle factors for irfft (avoid repeated cos/sin calls)
static std::vector<float> twiddle_cos;
static std::vector<float> twiddle_sin;

static void fill_twiddles(int N) {
    twiddle_cos.resize(N);
    twiddle_sin.resize(N);
    for (int k = 0; k < N; ++k) {
        float angle = 2.0f * M_PI * k / N;
        twiddle_cos[k] = cosf(angle);
        twiddle_sin[k] = sinf(angle);
    }
}

// fast irfft using precomputed twiddle factors (5x faster than naive)
static void irfft_fast(int n, const float * inp_cplx, float * out_real) {
    const int N = n / 2 + 1;

    for (int k = 0; k < n; ++k) {
        float sum = 0.0f;
        for (int m = 0; m < N; ++m) {
            const int idx = (int)((int64_t)k * m % n);
            sum += inp_cplx[2 * m] * twiddle_cos[idx] - inp_cplx[2 * m + 1] * twiddle_sin[idx];
        }
        out_real[k] = sum / N;
    }
}

//
//  y = torch.nn.functional.fold(
//       data, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
//  )[:, 0, 0, pad:-pad]
//
// data.shape =  torch.Size([1, 1280, 261])
// output_size =  84480
// win_length =  1280
// hop_length =  320
// pad =  480
//
static void fold(const std::vector<float> & data, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad, std::vector<float> & output) {
    int64_t output_height = n_out;
    int64_t kernel_w = n_win;
    int64_t stride_w = n_hop;
    int64_t width    = n_out;

    output.resize(width, 0.0f);

    int64_t col_idx = 0;
    for (int64_t w_col = 0; w_col < width; ++w_col) {
        int64_t start = w_col * stride_w - n_pad;
        int64_t end   = start + kernel_w;

        for (int64_t w_im = start; w_im < end; ++w_im) {
            if (w_im >= 0 && w_im < output_height && col_idx < (int64_t) data.size()) {
                output[w_im] += data[col_idx];
            }
            col_idx++;
        }
    }

    output.resize(n_out - 2 * n_pad);
}

// GPU-accelerated spectral ops via ggml compute graph
static std::vector<float> embd_to_audio(
        const float * embd,
        const int n_codes,
        const int n_embd,
        const int n_thread,
        ggml_backend_t backend = nullptr) {
    const int n_fft = 1280;
    const int n_hop = 320;
    const int n_win = 1280;
    const int n_pad = (n_win - n_hop)/2;
    const int n_out = (n_codes - 1)*n_hop + n_win;
    const int audio_len = n_out - 2*n_pad;

    // Step 1: CPU preprocessing — transpose + polar-to-cartesian + interleave
    // This is O(n_codes * n_embd) and fast, not worth GPU dispatch overhead
    std::vector<float> ST(n_embd * n_codes);
    {
        std::vector<float> E(n_embd * n_codes);

        for (int l = 0; l < n_codes; ++l) {
            for (int k = 0; k < n_embd; ++k) {
                E[k*n_codes + l] = embd[l*n_embd + k];
            }
        }

        for (int l = 0; l < n_codes; ++l) {
            for (int k = 0; k < n_embd/2; ++k) {
                float mag = E[(k           )*n_codes + l];
                float phi = E[(k + n_embd/2)*n_codes + l];

                mag = expf(mag);
                if (mag > 1e2f) {
                    mag = 1e2f;
                }

                ST[l*n_embd + 2*k + 0] = mag*cosf(phi);
                ST[l*n_embd + 2*k + 1] = mag*sinf(phi);
            }
        }
    }

    // Step 2: Compute Hann window
    std::vector<float> hann(n_fft);
    fill_hann_window(hann.size(), true, hann.data());

    // Step 3: Use ggml graph for IRFFT + fold (GPU or CPU)
    bool own_backend = false;
    if (!backend) {
        backend = ggml_backend_init_best();
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        }
        own_backend = true;
    }

    const auto t_irfft_start = ggml_time_us();

    // Create ggml context for graph tensors
    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 10 + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ctx_params);

    // Create input tensors
    struct ggml_tensor * t_st   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_codes);  // [1282, n_codes]
    struct ggml_tensor * t_hann = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_fft);             // [1280]
    ggml_set_name(t_st, "spectral_input");
    ggml_set_name(t_hann, "hann_window");
    ggml_set_input(t_st);
    ggml_set_input(t_hann);

    // Build graph: IRFFT → window multiply → FOLD
    struct ggml_tensor * t_irfft = ggml_irfft(ctx, t_st, n_fft);     // [n_fft, n_codes]
    ggml_set_name(t_irfft, "irfft_output");

    // Multiply by Hann window (broadcast hann [n_fft] across n_codes frames)
    struct ggml_tensor * t_windowed = ggml_mul(ctx, t_irfft, t_hann); // [n_fft, n_codes]
    ggml_set_name(t_windowed, "windowed_frames");

    // Fold: overlap-add with Hann² normalization → audio output
    struct ggml_tensor * t_audio = ggml_fold(ctx, t_windowed, t_hann, n_out, n_hop, n_pad);
    ggml_set_name(t_audio, "audio_output");
    ggml_set_output(t_audio);

    // Build compute graph
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, t_audio);

    // Allocate tensors on backend
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, graph);

    // Copy input data to backend tensors
    ggml_backend_tensor_set(t_st,   ST.data(),   0, n_embd * n_codes * sizeof(float));
    ggml_backend_tensor_set(t_hann, hann.data(), 0, n_fft * sizeof(float));

    // Execute graph on GPU
    ggml_backend_graph_compute(backend, graph);

    // Read output
    std::vector<float> audio(audio_len);
    ggml_backend_tensor_get(t_audio, audio.data(), 0, audio_len * sizeof(float));

    LOG_INF("%s: time irfft+fold (ggml): %.3f ms\n", __func__, (ggml_time_us() - t_irfft_start) / 1000.0f);

    // Cleanup
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    if (own_backend) {
        ggml_backend_free(backend);
    }

    return audio;
}

static const std::map<int, std::string> ones = {
    {0, "zero"}, {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"},
    {5, "five"}, {6, "six"}, {7, "seven"}, {8, "eight"}, {9, "nine"},
    {10, "ten"}, {11, "eleven"}, {12, "twelve"}, {13, "thirteen"}, {14, "fourteen"},
    {15, "fifteen"}, {16, "sixteen"}, {17, "seventeen"}, {18, "eighteen"}, {19, "nineteen"}
};

static const std::map<int, std::string> tens = {
    {2, "twenty"}, {3, "thirty"}, {4, "forty"}, {5, "fifty"},
    {6, "sixty"}, {7, "seventy"}, {8, "eighty"}, {9, "ninety"}
};

// Convert a number less than 1000 to words
static std::string convert_less_than_thousand(int num) {
    std::string result;

    if (num >= 100) {
        result += ones.at(num / 100) + " hundred ";
        num %= 100;
    }

    if (num >= 20) {
        result += tens.at(num / 10);
        if (num % 10 > 0) {
            result += "-" + ones.at(num % 10);
        }
    } else if (num > 0) {
        result += ones.at(num);
    }

    return result;
}

static std::string number_to_words(const std::string & number_str) {
    try {
        size_t decimal_pos = number_str.find('.');
        std::string integer_part = number_str.substr(0, decimal_pos);

        int int_number = std::stoi(integer_part);
        std::string result;

        if (int_number == 0) {
            result = "zero";
        } else {
            if (int_number >= 1000000000) {
                int billions = int_number / 1000000000;
                result += convert_less_than_thousand(billions) + " billion ";
                int_number %= 1000000000;
            }

            if (int_number >= 1000000) {
                int millions = int_number / 1000000;
                result += convert_less_than_thousand(millions) + " million ";
                int_number %= 1000000;
            }

            if (int_number >= 1000) {
                int thousands = int_number / 1000;
                result += convert_less_than_thousand(thousands) + " thousand ";
                int_number %= 1000;
            }

            if (int_number > 0) {
                result += convert_less_than_thousand(int_number);
            }
        }

        // Handle decimal part
        if (decimal_pos != std::string::npos) {
            result += " point";
            std::string decimal_part = number_str.substr(decimal_pos + 1);
            for (char digit : decimal_part) {
                result += " " + ones.at(digit - '0');
            }
        }

        return result;
    } catch (const std::exception& e) {
        // Skip if fails
        return " ";
    }
}

static std::string replace_numbers_with_words(const std::string & input_text) {
    std::regex number_pattern(R"(\d+(\.\d+)?)");
    std::string result;
    auto it = std::sregex_iterator(input_text.begin(), input_text.end(), number_pattern);
    auto end = std::sregex_iterator();

    size_t last_pos = 0;
    for (std::sregex_iterator i = it; i != end; ++i) {
        const std::smatch& match = *i;
        result.append(input_text, last_pos, match.position() - last_pos);
        result.append(number_to_words(match.str()));
        last_pos = match.position() + match.length();
    }
    result.append(input_text, last_pos);

    return result;
}

// Based on: https://github.com/edwko/OuteTTS/blob/a613e79c489d8256dd657ea9168d78de75895d82/outetts/version/v1/prompt_processor.py#L39
static std::string process_text(const std::string & text, const outetts_version tts_version = OUTETTS_V0_2) {

    // For now I skipped text romanization as I am unsure how to handle
    // uroman and MeCab implementations in C++
    // maybe something like https://github.com/anyascii/anyascii/ could work.
    // currently only English would be supported in this function

    std::string processed_text = replace_numbers_with_words(text);

    std::transform(processed_text.begin(), processed_text.end(),
                  processed_text.begin(), ::tolower);

    std::regex special_chars(R"([-_/,\.\\])");
    processed_text = std::regex_replace(processed_text, special_chars, " ");

    std::regex non_alpha(R"([^a-z\s])");
    processed_text = std::regex_replace(processed_text, non_alpha, "");

    std::regex multiple_spaces(R"(\s+)");
    processed_text = std::regex_replace(processed_text, multiple_spaces, " ");

    processed_text = std::regex_replace(processed_text, std::regex(R"(^\s+|\s+$)"), "");

    /*
        Replace spaces with the separator token same as in line 365

        for (auto & c : prompt_user) {
        if (c == ' ') {
            prompt_clean += "<|text_sep|>";
    */
    std::string separator = (tts_version == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
    processed_text = std::regex_replace(processed_text, std::regex(R"(\s)"), separator);

    return processed_text;
}

static void prompt_add(llama_tokens & prompt, llama_token token) {
    prompt.push_back(token);
}

static void prompt_add(llama_tokens & prompt, const llama_tokens & tokens) {
    prompt.insert(prompt.end(), tokens.begin(), tokens.end());
}

static void prompt_add(llama_tokens & prompt, const llama_vocab * vocab, const std::string & txt, bool add_special, bool parse_special) {
    auto tmp = common_tokenize(vocab, txt, add_special, parse_special);
    prompt_add(prompt, tmp);
}

static void prompt_init(llama_tokens & prompt, const llama_vocab * vocab) {
    prompt.clear();

    prompt_add(prompt, vocab, "<|im_start|>\n", true, true);
}

static std::vector<llama_token> prepare_guide_tokens(const llama_vocab * vocab, const std::string & str, const outetts_version tts_version = OUTETTS_V0_2) {
    const std::string& delimiter = (tts_version == OUTETTS_V0_3 ? "<|space|>" : "<|text_sep|>");

    std::vector<llama_token> result;
    size_t start = 0;
    size_t end = str.find(delimiter);

    //first token is always a newline, as it was not previously added
    result.push_back(common_tokenize(vocab, "\n", false, true)[0]);

    while (end != std::string::npos) {
        std::string current_word = str.substr(start, end - start);
        auto tmp = common_tokenize(vocab, current_word, false, true);
        result.push_back(tmp[0]);
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    // Add the last part
    std::string current_word = str.substr(start);
    auto tmp = common_tokenize(vocab, current_word, false, true);
    if (tmp.size() > 0) {
        result.push_back(tmp[0]);
    }
    return result;
}

static json speaker_from_file(const std::string & speaker_file) {
    std::ifstream file(speaker_file);
    if (!file) {
        LOG_ERR("%s: Failed to open file '%s' for reading\n", __func__, speaker_file.c_str());
        return json();
    }

    json speaker = json::parse(file);
    return speaker;
}

static outetts_version get_tts_version(llama_model *model, json speaker = json::object()) {
    if (speaker.contains("version")) {
        std::string version = speaker["version"].get<std::string>();
        if (version == "0.2") {
            return OUTETTS_V0_2;
        } else if (version == "0.3") {
            return OUTETTS_V0_3;
        } else {
            LOG_ERR("%s: Unsupported speaker version '%s'\n", __func__, version.c_str());
        }
    }

    // Also could get version from model itself
    const char *chat_template = llama_model_chat_template(model, nullptr);
    if (chat_template && std::string(chat_template) == "outetts-0.3") {
        return OUTETTS_V0_3;
    }

    // Use 0.2 as the default version
    return OUTETTS_V0_2;
}

static std::string audio_text_from_speaker(json speaker, const outetts_version tts_version = OUTETTS_V0_2) {
    std::string audio_text = "<|text_start|>";

    if (tts_version == OUTETTS_V0_2 || tts_version == OUTETTS_V0_3) {
        std::string separator = (tts_version == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
        for (const auto &word : speaker["words"]) {
            audio_text += word["word"].get<std::string>() + separator;
        }
    }

    return audio_text;
}

static std::string audio_data_from_speaker(json speaker, const outetts_version tts_version = OUTETTS_V0_2) {
    std::string audio_data = "<|audio_start|>\n";

    if (tts_version == OUTETTS_V0_2 || tts_version == OUTETTS_V0_3) {
        std::string code_start = (tts_version == OUTETTS_V0_3) ? "" : "<|code_start|>";
        std::string code_end = (tts_version == OUTETTS_V0_3) ? "<|space|>" : "<|code_end|>";
        for (const auto &word : speaker["words"]) {
            std::string word_text = word["word"].get<std::string>();
            double duration = word["duration"].get<double>();
            std::vector<int> codes = word["codes"].get<std::vector<int>>();

            // Create the audio output entry
            std::ostringstream word_entry;
            word_entry << word_text << "<|t_" << std::fixed << std::setprecision(2)
                       << duration << "|>" + code_start;
            for (const auto &Code : codes) {
                word_entry << "<|" << Code << "|>";
            }
            word_entry << code_end << "\n";
            audio_data += word_entry.str();
        }
    }

    return audio_data;
}

#ifdef HAVE_AUDIOTOOLBOX
#include <mutex>
#include <condition_variable>

struct PlaybackContext {
    std::mutex mtx;
    std::condition_variable cv;
    bool done = false;
};

static void playback_is_running_callback(void * userData, AudioQueueRef queue, AudioQueuePropertyID prop) {
    UInt32 running = 0;
    UInt32 size = sizeof(running);
    AudioQueueGetProperty(queue, kAudioQueueProperty_IsRunning, &running, &size);
    if (!running) {
        auto * ctx = (PlaybackContext *)userData;
        std::lock_guard<std::mutex> lock(ctx->mtx);
        ctx->done = true;
        ctx->cv.notify_one();
    }
}

static void play_audio(const std::vector<float> & audio, int sample_rate) {
    // Convert float samples to int16
    std::vector<int16_t> pcm(audio.size());
    for (size_t i = 0; i < audio.size(); i++) {
        pcm[i] = (int16_t)std::clamp(audio[i] * 32767.0f, -32768.0f, 32767.0f);
    }

    AudioStreamBasicDescription fmt = {};
    fmt.mSampleRate = sample_rate;
    fmt.mFormatID = kAudioFormatLinearPCM;
    fmt.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
    fmt.mBitsPerChannel = 16;
    fmt.mChannelsPerFrame = 1;
    fmt.mBytesPerFrame = 2;
    fmt.mFramesPerPacket = 1;
    fmt.mBytesPerPacket = 2;

    PlaybackContext pbctx;

    AudioQueueRef queue = nullptr;
    auto callback = [](void*, AudioQueueRef, AudioQueueBufferRef) {};
    AudioQueueNewOutput(&fmt, callback, nullptr, nullptr, nullptr, 0, &queue);

    // Listen for the queue stopping
    AudioQueueAddPropertyListener(queue, kAudioQueueProperty_IsRunning, playback_is_running_callback, &pbctx);

    // Enqueue all audio in chunks
    const int chunk_frames = 4096;
    const int chunk_bytes = chunk_frames * 2;
    size_t offset = 0;

    while (offset < pcm.size()) {
        size_t remaining = pcm.size() - offset;
        size_t frames = std::min((size_t)chunk_frames, remaining);

        AudioQueueBufferRef buf = nullptr;
        AudioQueueAllocateBuffer(queue, chunk_bytes, &buf);
        buf->mAudioDataByteSize = frames * 2;
        memcpy(buf->mAudioData, pcm.data() + offset, frames * 2);
        AudioQueueEnqueueBuffer(queue, buf, 0, nullptr);
        offset += frames;
    }

    // Start playback, then tell queue to stop after draining all buffers
    AudioQueueStart(queue, nullptr);
    AudioQueueStop(queue, false);  // false = drain all enqueued buffers before stopping

    // Wait for playback to actually complete
    {
        std::unique_lock<std::mutex> lock(pbctx.mtx);
        pbctx.cv.wait(lock, [&]{ return pbctx.done; });
    }

    AudioQueueDispose(queue, true);
}
#else
// Fallback: write to temp file and use afplay
static void play_audio(const std::vector<float> & audio, int sample_rate) {
    std::string tmp = "/tmp/tts-daemon-out.wav";
    save_wav16(tmp, audio, sample_rate);
    std::string cmd = "afplay " + tmp;
    system(cmd.c_str());
}
#endif

static volatile sig_atomic_t g_running = 1;
static std::string g_fifo_path;

static void signal_handler(int) {
    g_running = 0;
}

int main(int argc, char ** argv) {
    common_params params;

    params.out_file = ""; // no wav saving by default
    params.prompt = "";

    params.n_predict = 4096;
    params.n_batch   = 8192;
    params.n_ctx     = 8192;

    params.sampling.top_k = 4;
    params.sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K, };

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS, nullptr)) {
        return 1;
    }

    const int n_predict  = params.n_predict;

    common_init();

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_ttc = NULL; // text-to-codes
    llama_model * model_cts = NULL; // codes-to-speech

    llama_context * ctx_ttc = NULL;
    llama_context * ctx_cts = NULL;

    auto llama_init_ttc = common_init_from_params(params);

    model_ttc = llama_init_ttc->model();
    ctx_ttc   = llama_init_ttc->context();

    if (model_ttc == nullptr || ctx_ttc == nullptr) {
        return ENOENT;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_ttc);

    params.model = params.vocoder.model;
    params.embedding = true;
    params.n_ubatch = params.n_batch;

    auto llama_init_cts = common_init_from_params(params);

    model_cts = llama_init_cts->model();
    ctx_cts   = llama_init_cts->context();

    if (model_cts == nullptr || ctx_cts == nullptr) {
        return ENOENT;
    }

    std::vector<common_sampler *> smpl(1); // n_parallel = 1 for daemon
    params.sampling.no_perf = false;

    smpl[0] = common_sampler_init(model_ttc, params.sampling);

    LOG_INF("sampler seed: %u\n",     common_sampler_get_seed(smpl[0]));
    LOG_INF("sampler params: \n%s\n", params.sampling.print().c_str());
    LOG_INF("sampler chain: %s\n",    common_sampler_print(smpl[0]).c_str());

    LOG_INF("%s: loading done\n", __func__);

    // Pre-init GPU backend for spectral ops (avoids per-call init overhead)
    ggml_backend_t backend_spectral = ggml_backend_init_best();
    if (!backend_spectral) {
        backend_spectral = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }

    // Pre-fill twiddle tables
    fill_twiddles(1280);

    // the default speaker profile is from: https://github.com/edwko/OuteTTS/blob/main/outetts/version/v1/default_speakers/en_male_1.json
    std::string audio_text = "<|text_start|>the<|text_sep|>overall<|text_sep|>package<|text_sep|>from<|text_sep|>just<|text_sep|>two<|text_sep|>people<|text_sep|>is<|text_sep|>pretty<|text_sep|>remarkable<|text_sep|>sure<|text_sep|>i<|text_sep|>have<|text_sep|>some<|text_sep|>critiques<|text_sep|>about<|text_sep|>some<|text_sep|>of<|text_sep|>the<|text_sep|>gameplay<|text_sep|>aspects<|text_sep|>but<|text_sep|>its<|text_sep|>still<|text_sep|>really<|text_sep|>enjoyable<|text_sep|>and<|text_sep|>it<|text_sep|>looks<|text_sep|>lovely<|text_sep|>";
    std::string audio_data = R"(<|audio_start|>
the<|t_0.08|><|code_start|><|257|><|740|><|636|><|913|><|788|><|1703|><|code_end|>
overall<|t_0.36|><|code_start|><|127|><|201|><|191|><|774|><|700|><|532|><|1056|><|557|><|798|><|298|><|1741|><|747|><|1662|><|1617|><|1702|><|1527|><|368|><|1588|><|1049|><|1008|><|1625|><|747|><|1576|><|728|><|1019|><|1696|><|1765|><|code_end|>
package<|t_0.56|><|code_start|><|935|><|584|><|1319|><|627|><|1016|><|1491|><|1344|><|1117|><|1526|><|1040|><|239|><|1435|><|951|><|498|><|723|><|1180|><|535|><|789|><|1649|><|1637|><|78|><|465|><|1668|><|901|><|595|><|1675|><|117|><|1009|><|1667|><|320|><|840|><|79|><|507|><|1762|><|1508|><|1228|><|1768|><|802|><|1450|><|1457|><|232|><|639|><|code_end|>
from<|t_0.19|><|code_start|><|604|><|782|><|1682|><|872|><|1532|><|1600|><|1036|><|1761|><|647|><|1554|><|1371|><|653|><|1595|><|950|><|code_end|>
just<|t_0.25|><|code_start|><|1782|><|1670|><|317|><|786|><|1748|><|631|><|599|><|1155|><|1364|><|1524|><|36|><|1591|><|889|><|1535|><|541|><|440|><|1532|><|50|><|870|><|code_end|>
two<|t_0.24|><|code_start|><|1681|><|1510|><|673|><|799|><|805|><|1342|><|330|><|519|><|62|><|640|><|1138|><|565|><|1552|><|1497|><|1552|><|572|><|1715|><|1732|><|code_end|>
people<|t_0.39|><|code_start|><|593|><|274|><|136|><|740|><|691|><|633|><|1484|><|1061|><|1138|><|1485|><|344|><|428|><|397|><|1562|><|645|><|917|><|1035|><|1449|><|1669|><|487|><|442|><|1484|><|1329|><|1832|><|1704|><|600|><|761|><|653|><|269|><|code_end|>
is<|t_0.16|><|code_start|><|566|><|583|><|1755|><|646|><|1337|><|709|><|802|><|1008|><|485|><|1583|><|652|><|10|><|code_end|>
pretty<|t_0.32|><|code_start|><|1818|><|1747|><|692|><|733|><|1010|><|534|><|406|><|1697|><|1053|><|1521|><|1355|><|1274|><|816|><|1398|><|211|><|1218|><|817|><|1472|><|1703|><|686|><|13|><|822|><|445|><|1068|><|code_end|>
remarkable<|t_0.68|><|code_start|><|230|><|1048|><|1705|><|355|><|706|><|1149|><|1535|><|1787|><|1356|><|1396|><|835|><|1583|><|486|><|1249|><|286|><|937|><|1076|><|1150|><|614|><|42|><|1058|><|705|><|681|><|798|><|934|><|490|><|514|><|1399|><|572|><|1446|><|1703|><|1346|><|1040|><|1426|><|1304|><|664|><|171|><|1530|><|625|><|64|><|1708|><|1830|><|1030|><|443|><|1509|><|1063|><|1605|><|1785|><|721|><|1440|><|923|><|code_end|>
sure<|t_0.36|><|code_start|><|792|><|1780|><|923|><|1640|><|265|><|261|><|1525|><|567|><|1491|><|1250|><|1730|><|362|><|919|><|1766|><|543|><|1|><|333|><|113|><|970|><|252|><|1606|><|133|><|302|><|1810|><|1046|><|1190|><|1675|><|code_end|>
i<|t_0.08|><|code_start|><|123|><|439|><|1074|><|705|><|1799|><|637|><|code_end|>
have<|t_0.16|><|code_start|><|1509|><|599|><|518|><|1170|><|552|><|1029|><|1267|><|864|><|419|><|143|><|1061|><|0|><|code_end|>
some<|t_0.16|><|code_start|><|619|><|400|><|1270|><|62|><|1370|><|1832|><|917|><|1661|><|167|><|269|><|1366|><|1508|><|code_end|>
critiques<|t_0.60|><|code_start|><|559|><|584|><|1163|><|1129|><|1313|><|1728|><|721|><|1146|><|1093|><|577|><|928|><|27|><|630|><|1080|><|1346|><|1337|><|320|><|1382|><|1175|><|1682|><|1556|><|990|><|1683|><|860|><|1721|><|110|><|786|><|376|><|1085|><|756|><|1523|><|234|><|1334|><|1506|><|1578|><|659|><|612|><|1108|><|1466|><|1647|><|308|><|1470|><|746|><|556|><|1061|><|code_end|>
about<|t_0.29|><|code_start|><|26|><|1649|><|545|><|1367|><|1263|><|1728|><|450|><|859|><|1434|><|497|><|1220|><|1285|><|179|><|755|><|1154|><|779|><|179|><|1229|><|1213|><|922|><|1774|><|1408|><|code_end|>
some<|t_0.23|><|code_start|><|986|><|28|><|1649|><|778|><|858|><|1519|><|1|><|18|><|26|><|1042|><|1174|><|1309|><|1499|><|1712|><|1692|><|1516|><|1574|><|code_end|>
of<|t_0.07|><|code_start|><|197|><|716|><|1039|><|1662|><|64|><|code_end|>
the<|t_0.08|><|code_start|><|1811|><|1568|><|569|><|886|><|1025|><|1374|><|code_end|>
gameplay<|t_0.48|><|code_start|><|1269|><|1092|><|933|><|1362|><|1762|><|1700|><|1675|><|215|><|781|><|1086|><|461|><|838|><|1022|><|759|><|649|><|1416|><|1004|><|551|><|909|><|787|><|343|><|830|><|1391|><|1040|><|1622|><|1779|><|1360|><|1231|><|1187|><|1317|><|76|><|997|><|989|><|978|><|737|><|189|><|code_end|>
aspects<|t_0.56|><|code_start|><|1423|><|797|><|1316|><|1222|><|147|><|719|><|1347|><|386|><|1390|><|1558|><|154|><|440|><|634|><|592|><|1097|><|1718|><|712|><|763|><|1118|><|1721|><|1311|><|868|><|580|><|362|><|1435|><|868|><|247|><|221|><|886|><|1145|><|1274|><|1284|><|457|><|1043|><|1459|><|1818|><|62|><|599|><|1035|><|62|><|1649|><|778|><|code_end|>
but<|t_0.20|><|code_start|><|780|><|1825|><|1681|><|1007|><|861|><|710|><|702|><|939|><|1669|><|1491|><|613|><|1739|><|823|><|1469|><|648|><|code_end|>
its<|t_0.09|><|code_start|><|92|><|688|><|1623|><|962|><|1670|><|527|><|599|><|code_end|>
still<|t_0.27|><|code_start|><|636|><|10|><|1217|><|344|><|713|><|957|><|823|><|154|><|1649|><|1286|><|508|><|214|><|1760|><|1250|><|456|><|1352|><|1368|><|921|><|615|><|5|><|code_end|>
really<|t_0.36|><|code_start|><|55|><|420|><|1008|><|1659|><|27|><|644|><|1266|><|617|><|761|><|1712|><|109|><|1465|><|1587|><|503|><|1541|><|619|><|197|><|1019|><|817|><|269|><|377|><|362|><|1381|><|507|><|1488|><|4|><|1695|><|code_end|>
enjoyable<|t_0.49|><|code_start|><|678|><|501|><|864|><|319|><|288|><|1472|><|1341|><|686|><|562|><|1463|><|619|><|1563|><|471|><|911|><|730|><|1811|><|1006|><|520|><|861|><|1274|><|125|><|1431|><|638|><|621|><|153|><|876|><|1770|><|437|><|987|><|1653|><|1109|><|898|><|1285|><|80|><|593|><|1709|><|843|><|code_end|>
and<|t_0.15|><|code_start|><|1285|><|987|><|303|><|1037|><|730|><|1164|><|502|><|120|><|1737|><|1655|><|1318|><|code_end|>
it<|t_0.09|><|code_start|><|848|><|1366|><|395|><|1601|><|1513|><|593|><|1302|><|code_end|>
looks<|t_0.27|><|code_start|><|1281|><|1266|><|1755|><|572|><|248|><|1751|><|1257|><|695|><|1380|><|457|><|659|><|585|><|1315|><|1105|><|1776|><|736|><|24|><|736|><|654|><|1027|><|code_end|>
lovely<|t_0.56|><|code_start|><|634|><|596|><|1766|><|1556|><|1306|><|1285|><|1481|><|1721|><|1123|><|438|><|1246|><|1251|><|795|><|659|><|1381|><|1658|><|217|><|1772|><|562|><|952|><|107|><|1129|><|1112|><|467|><|550|><|1079|><|840|><|1615|><|1469|><|1380|><|168|><|917|><|836|><|1827|><|437|><|583|><|67|><|595|><|1087|><|1646|><|1493|><|1677|><|code_end|>)";

    // audio data for 0.3 version
    outetts_version tts_version = get_tts_version(model_ttc);
    if (tts_version == OUTETTS_V0_3) {
        audio_text = std::regex_replace(audio_text, std::regex(R"(<\|text_sep\|>)"), "<|space|>");
        audio_data = std::regex_replace(audio_data, std::regex(R"(<\|code_start\|>)"), "");
        audio_data = std::regex_replace(audio_data, std::regex(R"(<\|code_end\|>)"), "<|space|>");
    }

    // load speaker if given
    if (!params.vocoder.speaker_file.empty()) {
        LOG_INF("%s: loading speaker ..\n", __func__);
        json speaker = speaker_from_file(params.vocoder.speaker_file);
        if (speaker.empty()) {
            LOG_ERR("%s: Failed to load speaker file '%s'\n", __func__, params.vocoder.speaker_file.c_str());
            return 1;
        }
        audio_text = audio_text_from_speaker(speaker, tts_version);
        audio_data = audio_data_from_speaker(speaker, tts_version);
    }

    // Set up FIFO
    g_fifo_path = "/tmp/tts-gpu"; // default, could be made configurable
    unlink(g_fifo_path.c_str()); // remove stale
    if (mkfifo(g_fifo_path.c_str(), 0666) != 0) {
        LOG_ERR("Failed to create FIFO at %s\n", g_fifo_path.c_str());
        return 1;
    }
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    LOG_INF("TTS daemon ready. Listening on %s\n", g_fifo_path.c_str());
    LOG_INF("Usage: echo \"your text\" > %s\n", g_fifo_path.c_str());

    // Main daemon loop
    while (g_running) {
        // Open FIFO for reading (blocks until a writer connects)
        std::ifstream fifo(g_fifo_path);
        if (!fifo.is_open()) {
            if (!g_running) break;
            usleep(100000);
            continue;
        }

        std::string line;
        while (std::getline(fifo, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);

            if (line.empty()) continue;
            if (line == "quit" || line == "exit") {
                g_running = 0;
                break;
            }

            LOG_INF("\n--- Processing: \"%s\" ---\n", line.c_str());
            const auto t_start = ggml_time_us();

            // === Generate codes ===
            std::vector<llama_token> codes;
            std::vector<llama_token> guide_tokens;

            // Clear KV cache for new utterance
            llama_memory_clear(llama_get_memory(ctx_ttc), true);

            // Reset sampler
            common_sampler_reset(smpl[0]);

            {
                std::vector<llama_token> prompt_inp;

                prompt_init(prompt_inp, vocab);
                prompt_add(prompt_inp, vocab, audio_text, false, true);

                // convert the input text into the necessary format expected by OuteTTS
                {
                    std::string prompt_clean = process_text(line, tts_version);
                    if (params.vocoder.use_guide_tokens) {
                        guide_tokens = prepare_guide_tokens(vocab, prompt_clean, tts_version);
                    }

                    LOG_INF("%s: prompt: '%s'\n", __func__, prompt_clean.c_str());

                    prompt_add(prompt_inp, vocab, prompt_clean, false, true);
                }

                prompt_add(prompt_inp, vocab, "<|text_end|>\n", false, true);

                // Add audio_data
                auto tmp = common_tokenize(vocab, audio_data, false, true);
                prompt_add(prompt_inp, tmp);

                LOG_INF("%s: prompt size: %d\n", __func__, (int) prompt_inp.size());

                // create a llama_batch
                const int n_parallel = 1;
                llama_batch batch = llama_batch_init(std::max(prompt_inp.size(), (size_t) n_parallel), 0, n_parallel);

                std::vector<llama_seq_id> seq_ids(n_parallel, 0);
                for (int32_t i = 0; i < n_parallel; ++i) {
                    seq_ids[i] = i;
                }

                // evaluate the initial prompt
                for (size_t i = 0; i < prompt_inp.size(); ++i) {
                    common_batch_add(batch, prompt_inp[i], i, seq_ids, false);
                }
                GGML_ASSERT(batch.n_tokens == (int) prompt_inp.size());

                // llama_decode will output logits only for the last token of the prompt
                batch.logits[batch.n_tokens - 1] = true;

                if (llama_decode(ctx_ttc, batch) != 0) {
                    LOG_ERR("%s: llama_decode() failed\n", __func__);
                    continue;
                }

                llama_synchronize(ctx_ttc);

                // main loop

                // remember the batch index of the last token for each parallel sequence
                // we need this to determine which logits to sample from
                std::vector<int32_t> i_batch(n_parallel, batch.n_tokens - 1);

                int n_past   = batch.n_tokens;
                int n_decode = 0;

                bool next_token_uses_guide_token = true;

                while (n_decode <= n_predict) {
                    // prepare the next batch
                    common_batch_clear(batch);

                    // sample the next token for each parallel sequence / stream
                    for (int32_t i = 0; i < n_parallel; ++i) {
                        if (i_batch[i] < 0) {
                            // the stream has already finished
                            continue;
                        }

                        llama_token new_token_id = common_sampler_sample(smpl[i], ctx_ttc, i_batch[i]);

                        //guide tokens help prevent hallucinations by forcing the TTS to use the correct word
                        if (!guide_tokens.empty() && next_token_uses_guide_token && !llama_vocab_is_control(vocab, new_token_id) && !llama_vocab_is_eog(vocab, new_token_id)) {
                            llama_token guide_token = guide_tokens[0];
                            guide_tokens.erase(guide_tokens.begin());
                            new_token_id = guide_token; //ensure correct word fragment is used
                        }

                        //this is the token id that always precedes a new word
                        next_token_uses_guide_token = (new_token_id == 198);

                        common_sampler_accept(smpl[i], new_token_id, true);

                        codes.push_back(new_token_id);

                        // is it an end of generation? -> mark the stream as finished
                        if (llama_vocab_is_eog(vocab, new_token_id) || n_decode == n_predict) {
                            i_batch[i] = -1;
                            break;
                        }

                        i_batch[i] = batch.n_tokens;

                        // push this new token for next evaluation
                        common_batch_add(batch, new_token_id, n_past, { i }, true);
                    }

                    // all streams are finished
                    if (batch.n_tokens == 0) {
                        break;
                    }

                    n_decode += 1;
                    n_past += 1;

                    // evaluate the current batch with the transformer model
                    if (llama_decode(ctx_ttc, batch)) {
                        LOG_ERR("%s : failed to eval, return code %d\n", __func__, 1);
                        break;
                    }
                }

                llama_batch_free(batch);
            }

            // === Token filtering ===
            codes.erase(std::remove_if(codes.begin(), codes.end(),
                [](llama_token t) { return t < 151672 || t > 155772; }), codes.end());

            if (codes.empty()) {
                LOG_ERR("No audio codes generated\n");
                continue;
            }

            for (auto & token : codes) {
                token -= 151672;
            }

            // === Vocoder ===
            const int n_codes = codes.size();
            llama_batch batch = llama_batch_init(n_codes, 0, 1);
            for (size_t i = 0; i < codes.size(); ++i) {
                common_batch_add(batch, codes[i], i, { 0 }, true);
            }

            if (llama_encode(ctx_cts, batch) != 0) {
                LOG_ERR("llama_encode() failed\n");
                llama_batch_free(batch);
                continue;
            }
            llama_synchronize(ctx_cts);
            llama_batch_free(batch);

            // === Spectral ops ===
            const int n_embd = llama_model_n_embd_out(model_cts);
            const float * embd = llama_get_embeddings(ctx_cts);
            auto audio = embd_to_audio(embd, n_codes, n_embd, params.cpuparams.n_threads, backend_spectral);

            const int n_sr = 24000;
            // zero out first 0.25 seconds
            for (int i = 0; i < std::min((int)audio.size(), n_sr/4); ++i) {
                audio[i] = 0.0f;
            }

            LOG_INF("time total: %.3f ms, playing %d samples (%.2f s)\n",
                (ggml_time_us() - t_start) / 1000.0f,
                (int)audio.size(), (float)audio.size() / n_sr);

            // === Play audio ===
            play_audio(audio, n_sr);
        }
        fifo.close();
    }

    // Cleanup
    unlink(g_fifo_path.c_str());
    ggml_backend_free(backend_spectral);
    for (auto * s : smpl) { common_sampler_free(s); }
    llama_backend_free();
    LOG_INF("TTS daemon stopped.\n");
    return 0;
}
