#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct NpyArray {
    std::vector<size_t> shape;
    std::vector<float> data;
};

NpyArray load_npy_float32(const std::string& path);
