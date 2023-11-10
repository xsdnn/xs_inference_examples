//
// Created by rozhin on 14.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XS_INFERENCE_EXAMPLES_MODELS_H
#define XS_INFERENCE_EXAMPLES_MODELS_H

#include <xsdnn/xsdnn.h>
#include <fstream>
using namespace xsdnn;

struct TensorHolder {
    explicit TensorHolder(std::string weight_filename) {
        std::ifstream ifs(weight_filename, std::ios_base::in | std::ios_base::binary);
        if (!ifs.is_open()) {
            std::string msg = "Error when opening \x1B[33m" + weight_filename;
            throw xs_error(msg);
        }
        if (!Graph.ParseFromIstream(&ifs)) {
            throw xs_error("Error when parse model.");
        }
    }

    const xs::TensorInfo* GetTensorByName(std::string TensorName) {
        for (size_t i = 0; i < Graph.tensors_size(); ++i) {
            if (Graph.tensors(i).name() == TensorName) {
                return &Graph.tensors(i);
            }
        }
        throw xs_error("Tensor Name '" + TensorName + "' not found.");
    }

    xs::GraphInfo Graph;
};

namespace models {

bool FP32SsdMobileNetV1_1_default_1(xsdnn::network* net);

} // models

#endif //XS_INFERENCE_EXAMPLES_MODELS_H
