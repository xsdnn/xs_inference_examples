
//
// Created by rozhin on 21.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <xsdnn/xsdnn.h>
#include <xsdnn/gsl/span>
#include <benchmark/benchmark.h>
#include <xs_inference_examples/models.h>

MmActivationType Int2Activation(int32_t Activation) {
    switch (Activation) {
        case 0:
            return MmActivationType::NotSet;
        case 1:
            return MmActivationType::Relu;
        case 2:
            return MmActivationType::HardSigmoid;
        default:
            throw xsdnn::xs_error("Unsupported activation type");
    }
}

xsdnn::core::backend_t Int2Engine(int32_t Engine) {
    switch (Engine) {
        case 0:
            return xsdnn::core::backend_t::xs;
        case 1:
            return xsdnn::core::backend_t::xnnpack;
        default:
            throw xsdnn::xs_error("Unsupported engine");
    }
}

void UniformRandom(xsdnn::mat_t& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = xsdnn::uniform_rand(-10000, 10000);
    }
}


void random_init_fp32(xsdnn::mat_t& X) {
    gsl::span<float> XSpan = xsdnn::GetMutableDataAsSpan<float>(&X);
    for (size_t i = 0; i < XSpan.size() / sizeof(float); ++i) {
        XSpan[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
}

void ComputeSsdMobilinet_v1_1_default_1_small(benchmark::State& state, const char*) {
    xsdnn::network net("FP32SsdMobileNetV1_1_default_1");
    if (!models::FP32SsdMobileNetV1_1_default_1(&net)) throw xs_error("Error");
    mat_t Input(300*300*3 * sizeof(float));
    random_init_fp32(Input);

    net.set_num_threads(1);
    net.configure();
    for (auto _ : state) {
        net.predict({Input});
    }
}

static void SsdTinyTest(benchmark::internal::Benchmark* b) {
    xnn_status status = xnn_initialize(nullptr);
    b->ArgNames({});


    b->Args({});
}

BENCHMARK_CAPTURE(ComputeSsdMobilinet_v1_1_default_1_small, SsdMobilenetV1_1_Default, "SsdMobilenetV1")->Apply(SsdTinyTest)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
