//
// Created by rozhin on 20.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <xs_inference_examples/codegen.h>
#include <xs_inference_examples/onnx.proto3.pb.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <getopt.h>

namespace codegen {
    OnnxModelHolder::OnnxModelHolder(std::string model_path) {
        model_path_ = ReplaceToSpace(model_path, '.');
        output_filename_ = model_path_ + ".h";
        ParseModel(model_path);
    }

    void OnnxModelHolder::ParseModel(std::string model_path) {
        std::ifstream input(model_path, std::ios::ate | std::ios::binary);

        if (!input.is_open()) {
            throw std::runtime_error("Error when opening model file.");
        }

        std::streamsize size = input.tellg();
        input.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        input.read(buffer.data(), size);

        if (!model_.ParseFromArray(buffer.data(), size)) {
            throw std::runtime_error("Error when parse model.");
        }
        spdlog::info("Parse Model Completed");
    }

    void OnnxModelHolder::GenerateHFile() {
        GenerateHeader();
        GenerateAllWeights();
        GenerateFooter();
        SaveToOutputFile();
        PrintSummary();
    }

    void OnnxModelHolder::GenerateHeader() {
        output_buffer_ << "//\n"
                          "// IMPORTANT! \n"
                          "// IT'S AUTOMATICLY GENERATED WEIGHTS FILE. DON'T CHANGE IT! \n"
                          "//";
        output_buffer_ << "\n";
        output_buffer_ << "#pragma once";
        output_buffer_ << "\n";
        output_buffer_ << "#include <vector>";
        output_buffer_ << "\n";
        output_buffer_ << "namespace " << model_path_ << "_tensor_data {";
        output_buffer_ << "\n";
    }

    void OnnxModelHolder::GenerateAllWeights() {

        for (size_t i = 0; i < model_.graph().initializer_size(); ++i) {
            const auto& Initializer = model_.graph().initializer(i);
            if (!IsSupportedDataType(Initializer.data_type())) {
                spdlog::warn("Initializer: " + Initializer.name() + " Not Converted. Data Type Mismatch!");
                BadTensorCount++;
                continue;
            }

            if (IsFloatDataType(Initializer.data_type())) {
                output_buffer_ << "std::vector<float>" << " " << ReplaceToSpace(Initializer.name(), '/') << " = { ";
                output_buffer_ << "\n";

                size_t CountIn1Row = 0;
                float weight;
                const char* data = Initializer.raw_data().c_str();
                for (size_t s = 0; s < GetTensorSize(Initializer) * sizeof(float); s += sizeof(float)) {
                    char d[] = {data[s + 0], data[s + 1], data[s + 2], data[s + 3]};
                    memcpy(&weight, &d, sizeof(float));
                    if (CountIn1Row++ == 10) {
                        output_buffer_ << "\n";
                        CountIn1Row = 0;
                    }
                    if (s != GetTensorSize(Initializer) * sizeof(float) - sizeof(float)) output_buffer_ << weight << ", ";
                    else output_buffer_ << weight;
                }
                output_buffer_ << "\n";
                output_buffer_ << "};";
                output_buffer_ << "\n";
                output_buffer_ << "\n";
            } else {
                throw std::runtime_error("Not Implemented yet supporting for this data-type");
            }
        }
    }

    void OnnxModelHolder::GenerateFooter() {
        output_buffer_ << "\n";
        output_buffer_ << "} // " << model_path_;
    }

    std::string OnnxModelHolder::ReplaceToSpace(std::string s, char Symbol) {
        std::string S = s;
        for (auto& c : S) {
            if (c == Symbol) c = '_';
        }
        return S;
    }

    bool OnnxModelHolder::IsSupportedDataType(int32_t type) {
        return IsFloatDataType(type);
    }

    bool OnnxModelHolder::IsFloatDataType(int32_t type) {
        return type == onnx::TensorProto::DataType::TensorProto_DataType_FLOAT;
    }

    size_t OnnxModelHolder::GetTensorSize(const onnx::TensorProto& tensor) {
        size_t Size = 1;
        for (size_t i = 0; i < tensor.dims_size(); ++i) {
            Size *= tensor.dims(i);
        }
        return Size;
    }

    void OnnxModelHolder::SaveToOutputFile() {
        std::ofstream output(output_filename_);
        output << output_buffer_.rdbuf();
        OutputFileSize = output.tellp();
    }

    void OnnxModelHolder::PrintSummary() {
        spdlog::info("Output Weight File Created");

        spdlog::info("Num Tensor: " + std::to_string(model_.graph().initializer_size()));
        spdlog::info("Num Bad Tensor Writed: " + std::to_string(BadTensorCount));
        spdlog::info("Num Succesfully Tensor Writed: " + std::to_string(model_.graph().initializer_size() - BadTensorCount));
        spdlog::info("Total File Size: " + std::to_string(OutputFileSize) + " bytes");
    }
}

namespace cli {
    void display_usage() {
        std::cout
        << "ONNX Codegen Weights Tool\n"
        << "\t--model_name, -m Path to onnx model file"
        << "\n";
    }

    struct Settings {
        std::string model_name;
    };

    Settings ParseArgument(int argc, char** argv) {
        Settings s;

        int c;
        while (true) {
            static struct option long_options[] = {
                    {"model_name", required_argument, nullptr, 'm'},
                    {"help", no_argument, nullptr, 'h'},
                    {nullptr, 0, nullptr, 0}};

            /* getopt_long stores the option index here. */
            int option_index = 0;

            c = getopt_long(argc, argv, "m:h",
                            long_options, &option_index);

            /* Detect the end of the options. */
            if (c == -1) break;

            switch (c) {
                case 'm':
                    s.model_name = optarg;
                    break;
                case 'h':
                case '?':
                    /* getopt_long already printed an error message. */
                    display_usage();
                    exit(-1);
                default:
                    exit(-1);
            }
        }
        return s;
    }

}

int main(int argc, char** argv) {
    cli::Settings S = cli::ParseArgument(argc, argv);
    codegen::OnnxModelHolder MH(S.model_name);
    MH.GenerateHFile();
}