#include "npy_reader.hpp"

#include <fstream>
#include <stdexcept>

namespace {

uint16_t read_uint16(std::ifstream& f) {
    uint8_t b0 = static_cast<uint8_t>(f.get());
    uint8_t b1 = static_cast<uint8_t>(f.get());
    return static_cast<uint16_t>(b0 | (b1 << 8));
}

}  // namespace

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
    uint16_t header_len = read_uint16(f);

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
    if (shape.empty()) shape.push_back(1);

    size_t total = 1;
    for (auto v : shape) total *= v;

    std::vector<float> data(total);
    f.read(reinterpret_cast<char*>(data.data()), total * sizeof(float));
    if (!f) {
        throw std::runtime_error("Failed to read npy payload: " + path);
    }
    return {shape, std::move(data)};
}
