#include "digital_human_engine.hpp"

#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <iterator>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <limits>

#include <MNN/Tensor.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include "npy_reader.hpp"

#define LOG_TAG "DHNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

constexpr double PI = 3.14159265358979323846;

class OneEuroFilter {
public:
    OneEuroFilter() = default;
    explicit OneEuroFilter(double te, double mincutoff = 1.0, double beta = 0.007, double dcutoff = 1.0)
        : te_(te), mincutoff_(mincutoff), beta_(beta), dcutoff_(dcutoff) {}

    double filter(double x) {
        if (!initialized_) {
            prev_x_ = x;
            prev_dx_ = 0.0;
            initialized_ = true;
            return x;
        }
        double dx = (x - prev_x_) / te_;
        double dx_hat = exponential_smoothing(dx, prev_dx_, alpha(dcutoff_));
        double cutoff = mincutoff_ + beta_ * std::abs(dx_hat);
        double x_hat = exponential_smoothing(x, prev_x_, alpha(cutoff));
        prev_x_ = x_hat;
        prev_dx_ = dx_hat;
        return x_hat;
    }

private:
    double te_{1.0 / 25.0};
    double mincutoff_{1.0};
    double beta_{0.007};
    double dcutoff_{1.0};
    double prev_x_{0.0};
    double prev_dx_{0.0};
    bool initialized_{false};

    static double exponential_smoothing(double x_cur, double x_prev, double a) {
        return a * x_cur + (1.0 - a) * x_prev;
    }

    double alpha(double cutoff) const {
        double tau = 1.0 / (2.0 * PI * cutoff);
        return 1.0 / (1.0 + tau / te_);
    }
};

class LandmarkSmoother {
public:
    LandmarkSmoother(int num_points, double fps) {
        double te = 1.0 / fps;
        filters_.resize(static_cast<size_t>(num_points));
        for (auto& pair : filters_) {
            pair.first = OneEuroFilter(te);
            pair.second = OneEuroFilter(te);
        }
    }

    void smooth(std::vector<cv::Point2f>& pts) {
        if (filters_.empty()) return;
        if (pts.size() != filters_.size()) return;
        for (size_t i = 0; i < pts.size(); ++i) {
            pts[i].x = static_cast<float>(filters_[i].first.filter(pts[i].x));
            pts[i].y = static_cast<float>(filters_[i].second.filter(pts[i].y));
        }
    }

private:
    std::vector<std::pair<OneEuroFilter, OneEuroFilter>> filters_;
};

class BBoxStabilizer {
public:
    BBoxStabilizer(double ema, double max_move_ratio, double max_scale_ratio)
        : ema_(ema), max_move_ratio_(max_move_ratio), max_scale_ratio_(max_scale_ratio) {}

    void reset() { has_prev_ = false; }

    void stabilize(double& cx, double& cy, double& side, double ref_scale) {
        if (!has_prev_) {
            prev_cx_ = cx;
            prev_cy_ = cy;
            prev_side_ = side;
            has_prev_ = true;
            return;
        }

        double sx = ema_ * cx + (1.0 - ema_) * prev_cx_;
        double sy = ema_ * cy + (1.0 - ema_) * prev_cy_;
        double ss = ema_ * side + (1.0 - ema_) * prev_side_;

        double max_move = std::max(1.0, max_move_ratio_ * ref_scale);
        sx = prev_cx_ + std::clamp(sx - prev_cx_, -max_move, max_move);
        sy = prev_cy_ + std::clamp(sy - prev_cy_, -max_move, max_move);

        double scale = std::clamp(ss / prev_side_, 1.0 - max_scale_ratio_, 1.0 + max_scale_ratio_);
        ss = prev_side_ * scale;

        prev_cx_ = sx;
        prev_cy_ = sy;
        prev_side_ = ss;

        cx = sx;
        cy = sy;
        side = ss;
    }

private:
    double ema_;
    double max_move_ratio_;
    double max_scale_ratio_;
    bool has_prev_{false};
    double prev_cx_{0.0};
    double prev_cy_{0.0};
    double prev_side_{0.0};
};

class MelExtractor {
public:
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int N_FFT = 800;
    static constexpr int HOP_SIZE = 200;
    static constexpr int WIN_SIZE = 800;
    static constexpr int MEL_BINS = 80;
    static constexpr int MEL_FRAMES = 16;
    static constexpr int WINDOW_SAMPLES = HOP_SIZE * MEL_FRAMES;

    MelExtractor() {
        build_window();
        build_mel_filter();
        min_level_ = std::exp(MIN_LEVEL_DB / 20.0 * std::log(10.0));
    }

    std::vector<float> compute(const std::vector<float>& segment) const {
        std::vector<float> output(MEL_BINS * MEL_FRAMES, 0.f);
        std::vector<double> frame(N_FFT);
        std::vector<double> spectrum(N_FFT / 2 + 1);

        for (int frame_idx = 0; frame_idx < MEL_FRAMES; ++frame_idx) {
            int offset = frame_idx * HOP_SIZE;
            for (int n = 0; n < N_FFT; ++n) {
                int idx = offset + n;
                double x = (idx < static_cast<int>(segment.size())) ? segment[idx] : 0.0;
                double prev = (idx <= 0) ? 0.0 : segment[idx - 1];
                double pre = x - PREEMPHASIS * prev;
                frame[n] = pre * window_[n];
            }

            for (int k = 0; k <= N_FFT / 2; ++k) {
                double real = 0.0;
                double imag = 0.0;
                double angle_base = -2.0 * PI * k / N_FFT;
                for (int n = 0; n < N_FFT; ++n) {
                    double angle = angle_base * n;
                    double value = frame[n];
                    real += value * std::cos(angle);
                    imag += value * std::sin(angle);
                }
                spectrum[k] = std::sqrt(real * real + imag * imag);
            }

            for (int m = 0; m < MEL_BINS; ++m) {
                double energy = 0.0;
                for (int k = 0; k <= N_FFT / 2; ++k) {
                    energy += mel_filter_[m][k] * spectrum[k];
                }
                energy = std::max(energy, min_level_);
                double db = 20.0 * std::log10(energy) - REF_LEVEL_DB;
                double norm = (2.0 * MAX_ABS_VALUE) * ((db - MIN_LEVEL_DB) / (-MIN_LEVEL_DB)) - MAX_ABS_VALUE;
                norm = std::clamp(norm, -MAX_ABS_VALUE, MAX_ABS_VALUE);
                output[m * MEL_FRAMES + frame_idx] = static_cast<float>(norm);
            }
        }

        return output;
    }

private:
    static constexpr double PREEMPHASIS = 0.97;
    static constexpr double FMIN = 55.0;
    static constexpr double FMAX = 7600.0;
    static constexpr double MIN_LEVEL_DB = -100.0;
    static constexpr double REF_LEVEL_DB = 20.0;
    static constexpr double MAX_ABS_VALUE = 4.0;

    std::vector<double> window_;
    std::vector<std::vector<double>> mel_filter_;
    double min_level_{0.0};

    static double hz_to_mel(double hz) {
        return 2595.0 * std::log10(1.0 + hz / 700.0);
    }

    static double mel_to_hz(double mel) {
        return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
    }

    void build_window() {
        window_.resize(N_FFT);
        for (int i = 0; i < N_FFT; ++i) {
            window_[i] = 0.5 - 0.5 * std::cos(2.0 * PI * i / (WIN_SIZE - 1));
        }
    }

    void build_mel_filter() {
        int n_freq = N_FFT / 2 + 1;
        mel_filter_.assign(MEL_BINS, std::vector<double>(n_freq, 0.0));

        std::vector<double> mel_points(MEL_BINS + 2);
        double mel_min = hz_to_mel(FMIN);
        double mel_max = hz_to_mel(FMAX);
        for (int i = 0; i < MEL_BINS + 2; ++i) {
            mel_points[i] = mel_min + (mel_max - mel_min) * i / (MEL_BINS + 1);
        }

        std::vector<double> hz_points(MEL_BINS + 2);
        for (int i = 0; i < MEL_BINS + 2; ++i) {
            hz_points[i] = mel_to_hz(mel_points[i]);
        }

        std::vector<int> bins(MEL_BINS + 2);
        for (int i = 0; i < MEL_BINS + 2; ++i) {
            bins[i] = static_cast<int>(std::floor((N_FFT + 1) * hz_points[i] / SAMPLE_RATE));
            bins[i] = std::clamp(bins[i], 0, n_freq - 1);
        }

        for (int m = 1; m <= MEL_BINS; ++m) {
            int left = bins[m - 1];
            int center = bins[m];
            int right = bins[m + 1];
            for (int k = left; k < center; ++k) {
                if (center - left > 0)
                    mel_filter_[m - 1][k] = (static_cast<double>(k) - left) / (center - left);
            }
            for (int k = center; k < right; ++k) {
                if (right - center > 0)
                    mel_filter_[m - 1][k] = (right - static_cast<double>(k)) / (right - center);
            }
        }
    }
};

static std::tuple<int, int, int, int> clip_square(double cx, double cy, double side, int W, int H) {
    int xmin = std::max(0, static_cast<int>(std::round(cx - side / 2)));
    int ymin = std::max(0, static_cast<int>(std::round(cy - side / 2)));
    int xmax = std::min(W, xmin + static_cast<int>(std::round(side)));
    int ymax = std::min(H, ymin + static_cast<int>(std::round(side)));
    int new_side = std::min(xmax - xmin, ymax - ymin);
    xmax = xmin + new_side;
    ymax = ymin + new_side;
    return {xmin, ymin, xmax, ymax};
}

static std::tuple<int, int, int, int> compute_mouth_roi(
    const std::vector<cv::Point2f>& lms,
    const cv::Size& img_size,
    BBoxStabilizer& stabilizer,
    bool use_stabilizer) {

    const int H = img_size.height;
    const int W = img_size.width;
    const size_t n = lms.size();

    if (n >= 68) {
        double x0 = lms[0].x;
        double x16 = lms[16].x;
        double face_w = std::abs(x16 - x0);
        double brow_y = std::numeric_limits<double>::max();
        for (int i = 17; i < 27; ++i) {
            brow_y = std::min(brow_y, static_cast<double>(lms[i].y));
        }
        double chin_y = lms[8].y;
        double face_h = std::max(1.0, chin_y - brow_y);

        double ear_outset_ratio = 0.06;
        double eye_margin_ratio = 0.02;
        double chin_extra_ratio = 0.06;
        double safe_enlarge = 1.0;

        double x_left = x0 - ear_outset_ratio * face_w;
        double x_right = x16 + ear_outset_ratio * face_w;

        std::vector<double> eye_y;
        eye_y.reserve(12);
        for (int i = 36; i < 48; ++i) eye_y.push_back(lms[i].y);
        std::nth_element(eye_y.begin(), eye_y.begin() + eye_y.size() * 3 / 4, eye_y.end());
        double eye_lower = eye_y[eye_y.size() * 3 / 4];
        double y_top = eye_lower + eye_margin_ratio * face_h;
        double y_bot = chin_y + chin_extra_ratio * face_h;

        double width_need = x_right - x_left;
        double height_need = y_bot - y_top;
        double side = std::max(width_need, height_need) * safe_enlarge;

        double cx = (x_left + x_right) * 0.5;
        double cy = y_top + side * 0.5;

        if (use_stabilizer) {
            double ref = std::max(face_w, face_h);
            stabilizer.stabilize(cx, cy, side, ref);
        }
        return clip_square(cx, cy, side, W, H);
    }

    int xmin = static_cast<int>(lms[1].x);
    int ymin = static_cast<int>(lms[52].y);
    int xmax = static_cast<int>(lms[31].x);
    int width = xmax - xmin;
    int expand = static_cast<int>(width * 0.10);
    xmin = std::max(0, xmin - expand);
    xmax = std::min(W, xmax + expand);
    ymin = std::max(0, ymin - expand);
    width = xmax - xmin;
    int ymax = std::min(H, ymin + width);

    double cx = (xmin + xmax) * 0.5;
    double cy = (ymin + ymax) * 0.5;
    double side = std::max(1, xmax - xmin);

    if (use_stabilizer) {
        stabilizer.stabilize(cx, cy, side, side);
        std::tie(xmin, ymin, xmax, ymax) = clip_square(cx, cy, side, W, H);
    }
    return {xmin, ymin, xmax, ymax};
}

static std::vector<cv::Point2f> load_landmarks(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open landmark file: " + path);
    }
    std::vector<cv::Point2f> pts;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        float x, y;
        ss >> x >> y;
        pts.emplace_back(x, y);
    }
    return pts;
}

static cv::Mat create_soft_mask(const cv::Size& size, double inner_ratio = 0.7) {
    cv::Mat mask(size, CV_32F, cv::Scalar(0));
    cv::Point center(size.width / 2, size.height / 2);
    cv::Size axes(static_cast<int>(size.width * inner_ratio * 0.5),
                  static_cast<int>(size.height * inner_ratio * 0.5));
    cv::ellipse(mask, center, axes, 0.0, 0.0, 360.0, cv::Scalar(1.0), cv::FILLED);
    cv::GaussianBlur(mask, mask, cv::Size(21, 21), 0);
    return mask;
}

DigitalHumanEngine::DigitalHumanEngine(const Options& opt) : opt_(opt) {
    realtime_audio_ = !opt_.audio_model_path.empty();

    load_dataset();
    prepare_session();

    if (opt_.use_smooth) {
        smoother_ = std::make_unique<LandmarkSmoother>(static_cast<int>(landmarks_.front().size()), opt_.fps);
    }
    stabilizer_ = std::make_unique<BBoxStabilizer>(0.55, 0.03, 0.06);

    if (realtime_audio_) {
        prepare_audio_encoder();
        sample_buffer_.clear();
        sample_buffer_.reserve(8000);
        feature_template_.assign(512, 0.0f);
    } else {
        if (opt_.audio_feat_path.empty()) {
            throw std::runtime_error("audio_feat_path is required when audio_model_path is empty");
        }
        NpyArray npy = load_npy_float32(opt_.audio_feat_path);
        if (npy.shape.size() < 2) throw std::runtime_error("Unexpected audio npy shape.");
        size_t feat_dim = npy.shape.back();
        size_t total = npy.data.size() / feat_dim;
        audio_feats_.assign(total, std::vector<float>(feat_dim));
        for (size_t i = 0; i < total; ++i) {
            std::copy(npy.data.begin() + i * feat_dim,
                      npy.data.begin() + (i + 1) * feat_dim,
                      audio_feats_[i].begin());
        }
        feature_template_.assign(feat_dim, 0.0f);
    }

    audio_play_list_.assign(13440, 0);
    LOGI("DigitalHumanEngine initialized (realtime_audio=%d)", realtime_audio_ ? 1 : 0);
}

DigitalHumanEngine::~DigitalHumanEngine() {
    if (session_) {
        unet_->releaseSession(session_);
    }
    if (audio_session_ && audio_net_) {
        audio_net_->releaseSession(audio_session_);
    }
}

void DigitalHumanEngine::reset() {
    audio_index_ = 0;
    img_index_ = 0;
    step_stride_ = 0;
    silence_ = true;
    audio_play_list_.assign(13440, 0);
    if (realtime_audio_) {
        sample_buffer_.clear();
        audio_feats_.clear();
    }
    if (opt_.use_smooth) {
        smoother_ = std::make_unique<LandmarkSmoother>(static_cast<int>(landmarks_.front().size()), opt_.fps);
    }
    stabilizer_->reset();
}

DigitalHumanEngine::Result DigitalHumanEngine::process(const int16_t* pcm, size_t len) {
    Result result;
    const size_t chunk = 160;
    if (len != chunk) {
        LOGE("Expected 160 samples, got %zu", len);
    }

    bool chunk_has_signal = false;
    for (size_t i = 0; i < len; ++i) {
        if (pcm[i] != 0) {
            chunk_has_signal = true;
            break;
        }
    }
    silence_ = !chunk_has_signal ? true : false;

    audio_play_list_.insert(audio_play_list_.end(), pcm, pcm + len);

    if (realtime_audio_) {
        for (size_t i = 0; i < len; ++i) {
            sample_buffer_.push_back(static_cast<float>(pcm[i]) / 32768.0f);
        }
        update_audio_features();
    }

    if (realtime_audio_ && audio_feats_.size() < 16) {
        goto AUDIO_RETURN;
    }

    advance_index();

    cv::Mat img = frames_[img_index_].clone();
    auto pts = landmarks_[img_index_];
    if (opt_.use_smooth && smoother_) {
        smoother_->smooth(pts);
    }

    auto [xmin, ymin, xmax, ymax] = compute_mouth_roi(
        pts, img.size(), *stabilizer_, true);

    xmin = std::clamp(xmin, 0, img.cols - 1);
    xmax = std::clamp(xmax, 1, img.cols);
    ymin = std::clamp(ymin, 0, img.rows - 1);
    ymax = std::clamp(ymax, 1, img.rows);

    cv::Rect roi(xmin, ymin, xmax - xmin, ymax - ymin);
    cv::Mat crop = img(roi).clone();
    const bool variant_328 = (opt_.variant == "328");
    const int inner_size = variant_328 ? 320 : 160;
    const int border = 4;
    const int full_size = inner_size + border * 2;
    cv::resize(crop, crop, cv::Size(full_size, full_size), 0, 0, cv::INTER_CUBIC);

    cv::Mat inner = crop(cv::Rect(border, border, inner_size, inner_size)).clone();
    cv::Mat mask_img = inner.clone();
    cv::rectangle(mask_img, cv::Rect(5, 5, inner_size - 10, inner_size - 10), cv::Scalar(0, 0, 0), cv::FILLED);

    std::vector<float> img_tensor(inner_size * inner_size * 6);
    for (int y = 0; y < inner_size; ++y) {
        const cv::Vec3b* ptr_inner = inner.ptr<cv::Vec3b>(y);
        const cv::Vec3b* ptr_mask = mask_img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < inner_size; ++x) {
            const cv::Vec3b& p = ptr_inner[x];
            const cv::Vec3b& pm = ptr_mask[x];
            size_t offset = static_cast<size_t>(y * inner_size + x);
            img_tensor[offset] = p[2] / 255.0f;
            img_tensor[offset + inner_size * inner_size] = p[1] / 255.0f;
            img_tensor[offset + inner_size * inner_size * 2] = p[0] / 255.0f;
            img_tensor[offset + inner_size * inner_size * 3] = pm[2] / 255.0f;
            img_tensor[offset + inner_size * inner_size * 4] = pm[1] / 255.0f;
            img_tensor[offset + inner_size * inner_size * 5] = pm[0] / 255.0f;
        }
    }

    std::vector<float> audio_window;
    if (realtime_audio_) {
        const size_t feat_dim = feature_template_.size();
        audio_window.assign(16 * feat_dim, 0.0f);
        for (int i = 0; i < 16; ++i) {
            const auto& feat = audio_feats_[audio_feats_.size() - 16 + i];
            std::copy(feat.begin(), feat.end(), audio_window.begin() + i * feat_dim);
        }
    } else {
        audio_window.resize(audio_feats_.front().size() * 8);
        fill_audio_window(audio_window);
    }

    std::vector<float> audio_tensor;
    std::vector<int> audio_shape;

    if (variant_328) {
        if (opt_.asr == "ave") {
            audio_shape = {1, 32, 16, 16};
            audio_tensor.assign(32 * 16 * 16, 0.0f);
            for (int frame = 0; frame < 16; ++frame) {
                const float* src = audio_window.data() + frame * 512;
                for (int c = 0; c < 32; ++c) {
                    std::copy(src + c * 16, src + (c + 1) * 16,
                              audio_tensor.begin() + c * 16 * 16 + frame * 16);
                }
            }
        } else if (opt_.asr == "hubert") {
            audio_shape = {1, 32, 32, 32};
            audio_tensor = audio_window;
        } else {
            audio_shape = {1, 256, 16, 32};
            audio_tensor = audio_window;
        }
    } else {
        if (opt_.asr == "hubert") audio_shape = {1, 16, 32, 32};
        else audio_shape = {1, 128, 16, 32};
        audio_tensor = audio_window;
    }

    auto* host_img = new MNN::Tensor(input_img_, MNN::Tensor::CAFFE);
    std::copy(img_tensor.begin(), img_tensor.end(), host_img->host<float>());
    input_img_->copyFromHostTensor(host_img);
    delete host_img;

    auto* host_audio = new MNN::Tensor(input_audio_, MNN::Tensor::CAFFE);
    std::copy(audio_tensor.begin(), audio_tensor.end(), host_audio->host<float>());
    input_audio_->copyFromHostTensor(host_audio);
    delete host_audio;

    unet_->runSession(session_);

    MNN::Tensor* output = unet_->getSessionOutput(session_, "output");
    auto* host_output = new MNN::Tensor(output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(host_output);

    const float* data = host_output->host<float>();
    cv::Mat pred(inner_size, inner_size, CV_32FC3);
    for (int y = 0; y < inner_size; ++y) {
        for (int x = 0; x < inner_size; ++x) {
            int idx = y * inner_size + x;
            float r = data[idx] * 255.0f;
            float g = data[inner_size * inner_size + idx] * 255.0f;
            float b = data[inner_size * inner_size * 2 + idx] * 255.0f;
            pred.at<cv::Vec3f>(y, x) = cv::Vec3f(b, g, r);
        }
    }
    delete host_output;

    pred.convertTo(pred, CV_8U);

    if (opt_.use_soft_mask) {
        cv::Mat mask = create_soft_mask(pred.size());
        cv::Mat mask3;
        cv::merge(std::vector<cv::Mat>{mask, mask, mask}, mask3);
        pred.convertTo(pred, CV_32F);
        inner.convertTo(inner, CV_32F);
        cv::multiply(pred, mask3, pred);
        cv::multiply(inner, cv::Scalar(1.0f, 1.0f, 1.0f) - mask3, inner);
        cv::add(pred, inner, pred);
        pred.convertTo(pred, CV_8U);
    }

    pred.copyTo(crop(cv::Rect(border, border, inner_size, inner_size)));
    cv::Mat resized;
    cv::resize(crop, resized, roi.size(), 0, 0, cv::INTER_CUBIC);

    if (opt_.use_seamless) {
        cv::Mat mask(roi.size(), CV_8U, cv::Scalar(255));
        cv::Point center(roi.x + roi.width / 2, roi.y + roi.height / 2);
        try {
            cv::seamlessClone(resized, img, mask, center, img, cv::NORMAL_CLONE);
        } catch (const cv::Exception&) {
            resized.copyTo(img(roi));
        }
    } else {
        resized.copyTo(img(roi));
    }

    result.has_frame = true;
    result.frame = std::move(img);

AUDIO_RETURN:
    if (!audio_play_list_.empty()) {
        size_t take = std::min<size_t>(len, audio_play_list_.size());
        result.audio.insert(result.audio.end(), audio_play_list_.begin(), audio_play_list_.begin() + take);
        audio_play_list_.erase(audio_play_list_.begin(), audio_play_list_.begin() + take);
    }

    return result;
}

void DigitalHumanEngine::load_dataset() {
    frames_.clear();
    landmarks_.clear();

    cv::String pattern = opt_.dataset_dir + "/full_body_img/*.jpg";
    std::vector<cv::String> img_files;
    cv::glob(pattern, img_files, false);
    std::sort(img_files.begin(), img_files.end());

    if (img_files.empty()) throw std::runtime_error("No frames found under " + opt_.dataset_dir);

    for (const auto& file : img_files) {
        cv::Mat img = cv::imread(file);
        if (img.empty()) continue;
        frames_.push_back(img);
        std::string idx = file.substr(file.find_last_of("/\\") + 1);
        idx = idx.substr(0, idx.find('.'));
        auto lms = load_landmarks(opt_.dataset_dir + "/landmarks/" + idx + ".lms");
        landmarks_.push_back(std::move(lms));
    }
}

void DigitalHumanEngine::prepare_session() {
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_AUTO;
    config.numThread = 4;
    unet_.reset(MNN::Interpreter::createFromFile(opt_.model_path.c_str()));
    if (!unet_) {
        throw std::runtime_error("Failed to load MNN model: " + opt_.model_path);
    }
    session_ = unet_->createSession(config);
    input_img_ = unet_->getSessionInput(session_, "input");
    input_audio_ = unet_->getSessionInput(session_, "audio");
}

void DigitalHumanEngine::prepare_audio_encoder() {
    if (opt_.audio_model_path.empty()) return;
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_AUTO;
    config.numThread = 2;
    audio_net_.reset(MNN::Interpreter::createFromFile(opt_.audio_model_path.c_str()));
    if (!audio_net_) {
        throw std::runtime_error("Failed to load audio encoder: " + opt_.audio_model_path);
    }
    audio_session_ = audio_net_->createSession(config);
    audio_input_ = audio_net_->getSessionInput(audio_session_, "input");
    audio_output_ = audio_net_->getSessionOutput(audio_session_, "output");
    mel_ = std::make_unique<MelExtractor>();
}

void DigitalHumanEngine::fill_audio_window(std::vector<float>& dst) {
    if (realtime_audio_) {
        return;
    }

    const int window = 8;
    const size_t feat_dim = audio_feats_.front().size();
    dst.assign(feat_dim * window * 2, 0.0f);
    for (int offset = 0, t = static_cast<int>(audio_index_) - window; t < static_cast<int>(audio_index_) + window; ++t, ++offset) {
        float* ptr = dst.data() + offset * feat_dim;
        if (t >= 0 && t < static_cast<int>(audio_feats_.size())) {
            const auto& src = audio_feats_[static_cast<size_t>(t)];
            std::copy(src.begin(), src.end(), ptr);
        } else {
            std::fill(ptr, ptr + feat_dim, 0.0f);
        }
    }
}

void DigitalHumanEngine::update_audio_features() {
    if (!realtime_audio_ || !mel_) return;
    const int hop = MelExtractor::HOP_SIZE;
    const int window = MelExtractor::WINDOW_SAMPLES;
    while (sample_buffer_.size() >= static_cast<size_t>(window)) {
        std::vector<float> segment(sample_buffer_.begin(), sample_buffer_.begin() + window);
        std::vector<float> mel = mel_->compute(segment);
        std::vector<float> feature = compute_audio_feature(mel);
        audio_feats_.push_back(std::move(feature));
        if (audio_feats_.size() > 128) {
            audio_feats_.erase(audio_feats_.begin(), audio_feats_.begin() + (audio_feats_.size() - 128));
        }
        sample_buffer_.erase(sample_buffer_.begin(), sample_buffer_.begin() + hop);
    }
}

std::vector<float> DigitalHumanEngine::compute_audio_feature(const std::vector<float>& mel) {
    if (!audio_net_) return feature_template_;
    auto* host = new MNN::Tensor(audio_input_, MNN::Tensor::CAFFE);
    std::copy(mel.begin(), mel.end(), host->host<float>());
    audio_input_->copyFromHostTensor(host);
    delete host;

    audio_net_->runSession(audio_session_);
    auto* out = audio_output_;
    auto* host_out = new MNN::Tensor(out, MNN::Tensor::CAFFE);
    out->copyToHostTensor(host_out);
    size_t size = host_out->elementSize();
    std::vector<float> feat(size);
    std::copy(host_out->host<float>(), host_out->host<float>() + size, feat.begin());
    delete host_out;
    return feat;
}

void DigitalHumanEngine::advance_index() {
    if (img_index_ > static_cast<int>(frames_.size()) - 2) step_stride_ = -1;
    if (img_index_ < 1) step_stride_ = 1;
    img_index_ += step_stride_;
    audio_index_ = std::min(audio_index_ + 1, audio_feats_.empty() ? static_cast<size_t>(0) : audio_feats_.size() - 1);
}
