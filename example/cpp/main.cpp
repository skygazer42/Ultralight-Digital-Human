#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/videoio.hpp>

namespace fs = std::filesystem;

// ---------------------------- Utility: simple npy reader ----------------------------

struct NpyArray {
    std::vector<size_t> shape;
    std::vector<float> data;
};

NpyArray load_npy_float32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open npy file: " + path);
    }

    char magic[6];
    f.read(magic, 6);
    if (f.gcount() != 6 || std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("Invalid npy magic in " + path);
    }

    char ver[2];
    f.read(ver, 2);
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);  // little endian

    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    if (header.find("'descr': '<f4'") == std::string::npos) {
        throw std::runtime_error("Only float32 little-endian arrays are supported: " + path);
    }
    if (header.find("'fortran_order': False") == std::string::npos) {
        throw std::runtime_error("Only C-order arrays are supported: " + path);
    }

    auto shape_pos = header.find("'shape':");
    if (shape_pos == std::string::npos) {
        throw std::runtime_error("Failed to parse shape from npy header: " + path);
    }
    auto start = header.find('(', shape_pos);
    auto end = header.find(')', start);
    if (start == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("Invalid shape tuple in npy header: " + path);
    }
    auto shape_str = header.substr(start + 1, end - start - 1);

    std::vector<size_t> shape;
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',')) {
            ++pos;
        }
        if (pos >= shape_str.size()) break;
        size_t next = shape_str.find_first_of(",)", pos);
        auto token = shape_str.substr(pos, next - pos);
        shape.push_back(static_cast<size_t>(std::stoul(token)));
        pos = next;
    }
    if (shape.empty()) {
        shape.push_back(1);
    }

    size_t total = 1;
    for (auto v : shape) total *= v;

    std::vector<float> data(total);
    f.read(reinterpret_cast<char*>(data.data()), total * sizeof(float));
    if (!f) {
        throw std::runtime_error("Failed to read npy payload: " + path);
    }
    return {shape, std::move(data)};
}

// ---------------------------- Landmark smoother ----------------------------

class OneEuroFilter {
public:
    OneEuroFilter() = default;
    OneEuroFilter(double te, double mincutoff = 1.0, double beta = 0.007, double dcutoff = 1.0)
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
        double tau = 1.0 / (2.0 * M_PI * cutoff);
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

    void reset() {
        filters_.clear();
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

// ---------------------------- ROI computation ----------------------------

struct StabilizerState {
    bool has_prev{false};
    double cx{0.0};
    double cy{0.0};
    double side{0.0};
};

class BBoxStabilizer {
public:
    BBoxStabilizer(double ema, double max_move_ratio, double max_scale_ratio)
        : ema_(ema), max_move_ratio_(max_move_ratio), max_scale_ratio_(max_scale_ratio) {}

    void reset() { state_ = {}; }

    void stabilize(double& cx, double& cy, double& side, double ref_scale) {
        if (!state_.has_prev) {
            state_ = {true, cx, cy, side};
            return;
        }
        double px = state_.cx;
        double py = state_.cy;
        double ps = state_.side;

        double sx = ema_ * cx + (1.0 - ema_) * px;
        double sy = ema_ * cy + (1.0 - ema_) * py;
        double ss = ema_ * side + (1.0 - ema_) * ps;

        double max_move = std::max(1.0, max_move_ratio_ * ref_scale);
        sx = px + std::clamp(sx - px, -max_move, max_move);
        sy = py + std::clamp(sy - py, -max_move, max_move);

        double scale = std::clamp(ss / ps, 1.0 - max_scale_ratio_, 1.0 + max_scale_ratio_);
        ss = ps * scale;

        cx = sx;
        cy = sy;
        side = ss;
        state_ = {true, sx, sy, ss};
    }

private:
    double ema_;
    double max_move_ratio_;
    double max_scale_ratio_;
    StabilizerState state_;
};

std::tuple<int, int, int, int> clip_square(double cx, double cy, double side, int W, int H) {
    int xmin = std::max(0, static_cast<int>(std::round(cx - side / 2)));
    int ymin = std::max(0, static_cast<int>(std::round(cy - side / 2)));
    int xmax = std::min(W, xmin + static_cast<int>(std::round(side)));
    int ymax = std::min(H, ymin + static_cast<int>(std::round(side)));
    int new_side = std::min(xmax - xmin, ymax - ymin);
    xmax = xmin + new_side;
    ymax = ymin + new_side;
    return {xmin, ymin, xmax, ymax};
}

std::tuple<int, int, int, int> compute_mouth_roi(
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
        auto [xmin, ymin, xmax, ymax] = clip_square(cx, cy, side, W, H);
        return {xmin, ymin, xmax, ymax};
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

// ---------------------------- Soft mask ----------------------------

cv::Mat create_soft_mask(const cv::Size& size, double inner_ratio = 0.7) {
    cv::Mat mask(size, CV_32F, cv::Scalar(0));
    cv::Point center(size.width / 2, size.height / 2);
    cv::Size axes(static_cast<int>(size.width * inner_ratio * 0.5),
                  static_cast<int>(size.height * inner_ratio * 0.5));
    cv::ellipse(mask, center, axes, 0.0, 0.0, 360.0, cv::Scalar(1.0), cv::FILLED);
    cv::GaussianBlur(mask, mask, cv::Size(21, 21), 0);
    return mask;
}

// ---------------------------- Landmark loader ----------------------------

std::vector<cv::Point2f> load_landmarks(const fs::path& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open landmark file: " + path.string());
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

// ---------------------------- Audio helpers ----------------------------

std::vector<float> gather_audio_window(
    const std::vector<std::vector<float>>& features,
    int index,
    int window_radius) {
    const int total = static_cast<int>(features.size());
    const size_t feat_dim = features.front().size();
    const int frame_count = window_radius * 2;
    std::vector<float> result(feat_dim * frame_count, 0.0f);
    int offset = 0;
    for (int t = index - window_radius; t < index + window_radius; ++t, ++offset) {
        float* dst = result.data() + offset * static_cast<int>(feat_dim);
        if (t >= 0 && t < total) {
            const auto& src = features[static_cast<size_t>(t)];
            std::copy(src.begin(), src.end(), dst);
        } else {
            std::fill(dst, dst + feat_dim, 0.0f);
        }
    }
    return result;
}

// ---------------------------- Tensor helpers ----------------------------

std::vector<float> build_image_input(
    const cv::Mat& inner,
    const cv::Mat& masked,
    int channels = 3) {
    const int H = inner.rows;
    const int W = inner.cols;
    std::vector<float> tensor(static_cast<size_t>(channels * 2 * H * W));
    // BGR order
    for (int y = 0; y < H; ++y) {
        const cv::Vec3b* row_inner = inner.ptr<cv::Vec3b>(y);
        const cv::Vec3b* row_mask = masked.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < channels; ++c) {
                size_t offset = static_cast<size_t>(c * H * W + y * W + x);
                tensor[offset] = row_inner[x][c] / 255.0f;
                tensor[offset + channels * H * W] = row_mask[x][c] / 255.0f;
            }
        }
    }
    return tensor;
}

cv::Mat tensor_to_bgr(const float* data, int C, int H, int W) {
    std::vector<cv::Mat> channels;
    channels.reserve(C);
    for (int c = 0; c < C; ++c) {
        cv::Mat channel(H, W, CV_32F);
        for (int y = 0; y < H; ++y) {
            float* dst = channel.ptr<float>(y);
            const float* src = data + c * H * W + y * W;
            std::copy(src, src + W, dst);
        }
        channels.push_back(channel);
    }
    cv::Mat merged;
    cv::merge(channels, merged);
    merged = merged * 255.0f;
    cv::Mat out;
    merged.convertTo(out, CV_8U);
    return out;
}

// ---------------------------- Argument parsing ----------------------------

struct Options {
    fs::path data_path;
    fs::path audio_npy;
    fs::path unet_path;
    fs::path output_path;
    std::string asr{"wenet"};
    std::string variant{"base"};
    double fps{20.0};
    bool use_smooth{false};
    bool use_soft_mask{false};
    bool use_seamless{false};
};

Options parse_args(int argc, char** argv) {
    if (argc < 9) {
        throw std::runtime_error(
            "Usage: ultralight_dh_onnx "
            "--data <dataset_dir> --audio <audio.npy> --unet <unet.onnx> "
            "--output <mp4> [--variant base|328] [--asr wenet|hubert|ave] "
            "[--fps 20|25] [--use-smooth] [--use-soft-mask] [--use-seamless]");
    }

    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> char* {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
            return argv[++i];
        };

        if (arg == "--data") opt.data_path = fs::path(next());
        else if (arg == "--audio") opt.audio_npy = fs::path(next());
        else if (arg == "--unet") opt.unet_path = fs::path(next());
        else if (arg == "--output") opt.output_path = fs::path(next());
        else if (arg == "--asr") opt.asr = next();
        else if (arg == "--variant") opt.variant = next();
        else if (arg == "--fps") opt.fps = std::stod(next());
        else if (arg == "--use-smooth") opt.use_smooth = true;
        else if (arg == "--use-soft-mask") opt.use_soft_mask = true;
        else if (arg == "--use-seamless") opt.use_seamless = true;
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (!fs::exists(opt.data_path)) throw std::runtime_error("Dataset not found: " + opt.data_path.string());
    if (!fs::exists(opt.audio_npy)) throw std::runtime_error("Audio npy not found: " + opt.audio_npy.string());
    if (!fs::exists(opt.unet_path)) throw std::runtime_error("UNet ONNX not found: " + opt.unet_path.string());
    return opt;
}

// ---------------------------- Main pipeline ----------------------------

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        fs::path img_dir = opt.data_path / "full_body_img";
        fs::path lms_dir = opt.data_path / "landmarks";
        if (!fs::exists(img_dir) || !fs::exists(lms_dir)) {
            throw std::runtime_error("Dataset directory must contain full_body_img/ and landmarks/");
        }

        std::vector<cv::Mat> frames;
        std::vector<std::vector<cv::Point2f>> landmarks;

        for (int idx = 0;; ++idx) {
            fs::path img_path = img_dir / (std::to_string(idx) + ".jpg");
            fs::path lms_path = lms_dir / (std::to_string(idx) + ".lms");
            if (!fs::exists(img_path) || !fs::exists(lms_path)) break;
            cv::Mat img = cv::imread(img_path.string());
            if (img.empty()) break;
            frames.push_back(img);
            landmarks.push_back(load_landmarks(lms_path));
        }
        if (frames.empty()) throw std::runtime_error("No frames found under " + img_dir.string());

        NpyArray audio_npy = load_npy_float32(opt.audio_npy.string());
        if (audio_npy.shape.size() < 2) throw std::runtime_error("Expected 2D audio array.");
        size_t audio_dim = audio_npy.shape.back();
        size_t audio_frames = audio_npy.data.size() / audio_dim;

        std::vector<std::vector<float>> audio_feats(audio_frames, std::vector<float>(audio_dim));
        for (size_t i = 0; i < audio_frames; ++i) {
            std::copy(audio_npy.data.begin() + i * audio_dim,
                      audio_npy.data.begin() + (i + 1) * audio_dim,
                      audio_feats[i].begin());
        }

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ultralight_dh");
        Ort::SessionOptions session_opts;
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session session(env, opt.unet_path.c_str(), session_opts);
        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name_img = session.GetInputNameAllocated(0, allocator);
        auto input_name_audio = session.GetInputNameAllocated(1, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);

        const std::string mode = opt.asr;
        const std::string variant = opt.variant;

        const bool variant_328 = (variant == "328");
        const int inner_size = variant_328 ? 320 : 160;
        const int border = 4;
        const int full_size = inner_size + border * 2;  // 168 or 328

        BBoxStabilizer stabilizer(0.55, 0.03, 0.06);
        LandmarkSmoother smoother(static_cast<int>(landmarks.front().size()), opt.fps);

        cv::VideoWriter writer;
        cv::Size frame_size(frames.front().cols, frames.front().rows);
        if (!writer.open(opt.output_path.string(),
                         cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                         opt.fps, frame_size)) {
            throw std::runtime_error("Failed to open video writer: " + opt.output_path.string());
        }

        int len_img = static_cast<int>(frames.size()) - 1;
        int step_stride = 0;
        int img_idx = 0;

        std::cout << "Processing " << audio_feats.size() << " audio frames..." << std::endl;

        for (size_t i = 0; i < audio_feats.size(); ++i) {
            if (img_idx > len_img - 1) step_stride = -1;
            if (img_idx < 1) step_stride = 1;
            img_idx += step_stride;
            int frame_id = std::clamp(img_idx, 0, len_img);

            cv::Mat img = frames[frame_id].clone();
            auto pts = landmarks[frame_id];
            if (opt.use_smooth) smoother.smooth(pts);

            auto [xmin, ymin, xmax, ymax] = compute_mouth_roi(
                pts, img.size(), stabilizer, true);

            xmin = std::clamp(xmin, 0, img.cols - 1);
            xmax = std::clamp(xmax, 1, img.cols);
            ymin = std::clamp(ymin, 0, img.rows - 1);
            ymax = std::clamp(ymax, 1, img.rows);
            if (xmax - xmin < full_size || ymax - ymin < full_size) {
                writer.write(img);
                continue;
            }

            cv::Rect roi(xmin, ymin, xmax - xmin, ymax - ymin);
            cv::Mat crop = img(roi).clone();
            cv::resize(crop, crop, cv::Size(full_size, full_size), 0, 0, cv::INTER_CUBIC);

            cv::Mat inner = crop(cv::Rect(border, border, inner_size, inner_size)).clone();
            cv::Mat masked = inner.clone();
            cv::rectangle(masked, cv::Rect(5, 5, inner_size - 10, inner_size - 10), cv::Scalar(0, 0, 0), cv::FILLED);

            std::vector<float> img_tensor = build_image_input(inner, masked);

            std::vector<float> audio_window = gather_audio_window(audio_feats, static_cast<int>(i), 4);
            std::vector<int64_t> audio_shape;
            if (!variant_328) {
                if (mode == "hubert") audio_shape = {1, 16, 32, 32};
                else if (mode == "wenet") audio_shape = {1, 128, 16, 32};
                else throw std::runtime_error("AVE encoder requires --variant 328.");
            } else {
                if (mode == "hubert") audio_shape = {1, 32, 32, 32};
                else if (mode == "wenet") audio_shape = {1, 256, 16, 32};
                else if (mode == "ave") audio_shape = {1, 32, 16, 16};
                else throw std::runtime_error("Unsupported ASR mode: " + mode);
            }
            size_t expected_audio = static_cast<size_t>(audio_shape[1] * audio_shape[2] * audio_shape[3]);
            if (audio_window.size() != expected_audio) {
                throw std::runtime_error("Audio window size mismatch: got " +
                                         std::to_string(audio_window.size()) + ", expected " +
                                         std::to_string(expected_audio));
            }

            std::vector<int64_t> img_shape = {1, 6, inner_size, inner_size};

            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            Ort::Value input_img = Ort::Value::CreateTensor<float>(
                memory_info, img_tensor.data(), img_tensor.size(), img_shape.data(), img_shape.size());
            Ort::Value input_audio = Ort::Value::CreateTensor<float>(
                memory_info, audio_window.data(), audio_window.size(), audio_shape.data(), audio_shape.size());

            const char* input_names[] = {input_name_img.get(), input_name_audio.get()};
            const char* output_names[] = {output_name.get()};
            std::array<Ort::Value, 2> input_tensors = {
                std::move(input_img),
                std::move(input_audio)
            };
            auto outputs = session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                input_tensors.data(),
                input_tensors.size(),
                output_names,
                1);

            auto& out_tensor = outputs.front();
            float* out_data = out_tensor.GetTensorMutableData<float>();

            cv::Mat pred = tensor_to_bgr(out_data, 3, inner_size, inner_size);

            cv::Mat inner_orig = inner.clone();
            if (opt.use_soft_mask) {
                cv::Mat mask = create_soft_mask(pred.size());
                cv::Mat mask3;
                cv::merge(std::vector<cv::Mat>{mask, mask, mask}, mask3);
                cv::Mat blended;
                pred.convertTo(pred, CV_32F);
                inner_orig.convertTo(inner_orig, CV_32F);
                cv::multiply(pred, mask3, pred);
                cv::multiply(inner_orig, cv::Scalar(1.0f, 1.0f, 1.0f) - mask3, inner_orig);
                cv::add(pred, inner_orig, blended);
                blended.convertTo(pred, CV_8U);
            }

            pred.copyTo(crop(cv::Rect(border, border, inner_size, inner_size)));
            cv::Mat resized;
            cv::resize(crop, resized, roi.size(), 0, 0, cv::INTER_CUBIC);

            if (opt.use_seamless) {
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

            writer.write(img);
        }

        writer.release();
        std::cout << "Saved video to " << opt.output_path << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
