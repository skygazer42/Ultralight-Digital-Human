#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>
#include <MNN/Interpreter.hpp>

class LandmarkSmoother;
class BBoxStabilizer;
class MelExtractor;

class DigitalHumanEngine {
public:
    struct Options {
        std::string dataset_dir;
        std::string model_path;
        std::string audio_feat_path;
        std::string audio_model_path;
        std::string asr{"wenet"};
        std::string variant{"base"};
        double fps{20.0};
        bool use_smooth{false};
        bool use_soft_mask{false};
        bool use_seamless{false};
    };

    explicit DigitalHumanEngine(const Options& opt);
    ~DigitalHumanEngine();

    DigitalHumanEngine(const DigitalHumanEngine&) = delete;
    DigitalHumanEngine& operator=(const DigitalHumanEngine&) = delete;

    void reset();

    struct Result {
        bool has_frame{false};
        cv::Mat frame;               // BGR
        std::vector<int16_t> audio;  // PCM 16kHz
    };

    Result process(const int16_t* pcm, size_t len);

private:
    void load_dataset();
    void prepare_session();
    void prepare_audio_encoder();
    void fill_audio_window(std::vector<float>& dst);
    void update_audio_features();
    std::vector<float> compute_audio_feature(const std::vector<float>& mel);
    void advance_index();

    Options opt_;

    std::shared_ptr<MNN::Interpreter> unet_;
    MNN::Session* session_{nullptr};
    MNN::Tensor* input_img_{nullptr};
    MNN::Tensor* input_audio_{nullptr};

    std::shared_ptr<MNN::Interpreter> audio_net_;
    MNN::Session* audio_session_{nullptr};
    MNN::Tensor* audio_input_{nullptr};
    MNN::Tensor* audio_output_{nullptr};
    bool realtime_audio_{false};

    std::vector<cv::Mat> frames_;
    std::vector<std::vector<cv::Point2f>> landmarks_;
    std::vector<std::vector<float>> audio_feats_;

    std::unique_ptr<LandmarkSmoother> smoother_;
    std::unique_ptr<BBoxStabilizer> stabilizer_;
    std::unique_ptr<MelExtractor> mel_;

    std::vector<float> sample_buffer_;
    std::vector<float> feature_template_;

    size_t audio_index_{0};
    int img_index_{0};
    int step_stride_{0};
    bool silence_{true};
    std::vector<int16_t> audio_play_list_;
};
