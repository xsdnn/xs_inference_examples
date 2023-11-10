//
// Created by rozhin on 21.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <xsdnn/xsdnn.h>
#include <xsdnn/gsl/span>
#include <benchmark/benchmark.h>

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

void fp32_convolution(benchmark::State& state, const char*) {
    size_t in_channel = state.range(0);
    size_t in_height = state.range(1);
    size_t in_width = state.range(2);
    size_t out_channel = state.range(3);
    size_t kernel_h = state.range(4);
    size_t kernel_w = state.range(5);
    size_t group_count = state.range(6);
    bool has_bias = state.range(7);
    size_t stride_h = state.range(8);
    size_t stride_w = state.range(9);
    size_t dilation_h = state.range(10);
    size_t dilation_w = state.range(11);
    size_t pads_h_top = state.range(12);
    size_t pads_w_left = state.range(13);
    size_t pads_h_bottom = state.range(14);
    size_t pads_w_right = state.range(15);
    MmActivationType activation_type = Int2Activation(state.range(16));
    xsdnn::core::backend_t engine = Int2Engine(state.range(17));

    xsdnn::shape3d in_shape(in_channel, in_height, in_width);
    xsdnn::mat_t input(in_shape.size() * sizeof(float));
    random_init_fp32(input);

    xsdnn::conv ConvOp(/*in_shape=*/in_shape, /*out_channel=*/out_channel, /*kernel_shape=*/{kernel_h, kernel_w},
                        /*group_count=*/group_count, /*has_bias=*/has_bias, /*stride_shape=*/{stride_h, stride_w},
                        /*dilation_shape=*/{dilation_h, dilation_w}, /*pad_type=*/xsdnn::padding_mode::notset,
                        /*pads=*/{pads_h_top, pads_w_left, pads_h_bottom, pads_w_right},
                        /*activation_type=*/activation_type, /*engine=*/engine);

    ConvOp.set_parallelize(false);
    ConvOp.set_num_threads(1);
    ConvOp.setup(true);
    ConvOp.set_in_data({{ input }});

    for (auto _ : state) {
        ConvOp.forward();
    }
}

static void SsdMobileNetV1_1_VerySmall(benchmark::internal::Benchmark* b) {
    xnn_status status = xnn_initialize(nullptr);
    b->ArgNames({"C", "H", "W", "Cout", "Kh", "Kw", "Gc", "Bias", "Sh", "Sw", "Dh", "Dw", "PHtop", "PWleft", "PHbottom", "PWright", "Activation", "Engine"});


    /*************************** Conv 0 **************************/
    /*       C    H    W    Cout    Kh    Kw     Gc     Bias     Sh    Sw    Dh   Dw   PHtop  PWleft   PHbottom    PWright    Activation    Engine */
    b->Args({3, 300, 300,     24,     3,    3,     1,       1,     2,   2,    1,   1,      0,      0,         1,         1,            1,        1});

    /*************************** Conv 1DP ************************/
    b->Args({24,150, 150,     24,     3,    3,    24,       1,     1,   1,    1,   1,      1,      1,         1,         1,            1,        1});
    b->Args({24,150, 150,     48,     1,    1,     1,       1,     1,   1,    1,   1,      0,      0,         0,         0,            1,        1});

    /*************************** Conv 2DP ************************/
    b->Args({48,150, 150,     48,     3,    3,    48,       1,     2,   2,    1,   1,      0,      0,         1,         1,            1,        1});
    b->Args({48, 75,  75,     96,     1,    1,     1,       1,     1,   1,    1,   1,      0,      0,         0,         0,            1,        1});
}

BENCHMARK_CAPTURE(fp32_convolution, SsdMobilenetV1_1_Default, "SsdMobilenetV1")->Apply(SsdMobileNetV1_1_VerySmall)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
