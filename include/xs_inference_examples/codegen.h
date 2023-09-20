//
// Created by rozhin on 20.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

/*
 Генерация .h файла, в котором явно записаны значения весов модели onnx
 */

#ifndef XS_INFERENCE_EXAMPLES_CODEGEN_H
#define XS_INFERENCE_EXAMPLES_CODEGEN_H

#include <string>
#include "onnx.proto3.pb.h"

namespace codegen {

class OnnxModelHolder {
public:
    explicit OnnxModelHolder(const std::string model_path);

public:
    void GenerateHFile();

private:
    void ParseModel(std::string model_path);
    void GenerateHeader();
    void GenerateAllWeights();
    void GenerateFooter();
    std::string ReplaceToSpace(std::string s, char Symbol);
    bool IsSupportedDataType(int32_t type);
    bool IsFloatDataType(int32_t type);
    size_t GetTensorSize(const onnx::TensorProto& tensor);
    void SaveToOutputFile();
    void PrintSummary();

private:
    std::string model_path_;
    std::string output_filename_;
    onnx::ModelProto model_;
    std::stringstream output_buffer_;

    /*
     * Переменные для подсчета статистик
     */

    size_t BadTensorCount = 0; // кол-во не сохраненных тензоров
    size_t OutputFileSize;

};

} // codegen

#endif //XS_INFERENCE_EXAMPLES_CODEGEN_H
