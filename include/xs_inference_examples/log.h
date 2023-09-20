//
// Created by rozhin on 20.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XS_INFERENCE_EXAMPLES_LOG_H
#define XS_INFERENCE_EXAMPLES_LOG_H

#include "iostream"
#include "sstream"

class Logger {
public:
    Logger(const char* msg) { stream_ << msg << ": "; }
    std::stringstream& stream() { return stream_; }
    ~Logger() { std::cerr << stream_.str() << std::endl; }
private:
    std::stringstream stream_;
};

#define LOG(msg) Logger(#msg).stream()

#endif //XS_INFERENCE_EXAMPLES_LOG_H
