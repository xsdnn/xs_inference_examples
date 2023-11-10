//
// Created by rozhin on 14.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <xs_inference_examples/models.h>
#include <spdlog/spdlog.h>

size_t GetTensorSize(const xs::TensorInfo* tensor) {
    size_t size = 1;
    for (auto d : tensor->dims()) size *= d;
    return size;
}

bool InitModelTensor(const xs::TensorInfo* src, mat_t* dst) {
    if (GetTensorSize(src) * sizeof(float) != dst->size()) return 0; // FIXME: сделать это для более частного случая
    const std::string& DataRaw = src->raw_data();
    std::copy(DataRaw.begin(), DataRaw.end(), dst->data());
    return 1;
}

#define INIT_MODEL_TENSOR(src, dst)                                     \
src_name = src;                                                         \
if(!InitModelTensor(TH.GetTensorByName(src_name), dst)) {               \
    spdlog::error("Size Mismatch at Tensor Name: " + src_name);         \
    throw xs_error("");                                                 \
}                                                                       \
spdlog::info("Tensor Name: " + src_name + " initialized");

namespace models {

bool FP32SsdMobileNetV1_1_default_1(xsdnn::network* net) {
    TensorHolder TH("ssd_mobilenet_v1_1_default_1_dequantize_onnx.xs_tensor");

    /*
     * Define Layer
     */

    static conv Conv2d_0(/*in_shape=*/shape3d(3, 300, 300), /*out_channel=*/24, /*kernel_shape=*/{3, 3},
                  /*group_count=*/1, /*has_bias=*/true, /*stride_shape=*/{2, 2}, /*dilation_shape=*/{1, 1},
                  /*pad_type=*/padding_mode::notset, /*pads=*/{0, 0, 1, 1}, /*activation_type=*/mmpack::Relu,
                  /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_1_Depthwise(/*in_shape=*/Conv2d_0.out_shape()[0], /*out_channel=*/24, /*kernel_shape=*/{3, 3},
                            /*group_count=*/24, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{1, 1, 1, 1}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_1_Pointwise(/*in_shape=*/Conv2d_1_Depthwise.out_shape()[0], /*out_channel=*/48, /*kernel_shape=*/{1, 1},
                            /*group_count=*/1, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{0, 0, 0, 0}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_2_Depthwise(/*in_shape=*/Conv2d_1_Pointwise.out_shape()[0], /*out_channel=*/48, /*kernel_shape=*/{3, 3},
                            /*group_count=*/48, /*has_bias=*/true, /*stride_shape=*/{2, 2}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{0, 0, 1, 1}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_2_Pointwise(/*in_shape=*/Conv2d_2_Depthwise.out_shape()[0], /*out_channel=*/96, /*kernel_shape=*/{1, 1},
                            /*group_count=*/1, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{0, 0, 0, 0}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_3_Depthwise(/*in_shape=*/Conv2d_2_Pointwise.out_shape()[0], /*out_channel=*/96, /*kernel_shape=*/{3, 3},
                            /*group_count=*/96, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{1, 1, 1, 1}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_3_Pointwise(/*in_shape=*/Conv2d_3_Depthwise.out_shape()[0], /*out_channel=*/96, /*kernel_shape=*/{1, 1},
                            /*group_count=*/1, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{0, 0, 0, 0}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_4_Depthwise(/*in_shape=*/Conv2d_3_Pointwise.out_shape()[0], /*out_channel=*/96, /*kernel_shape=*/{3, 3},
                            /*group_count=*/96, /*has_bias=*/true, /*stride_shape=*/{2, 2}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{1, 1, 1, 1}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_4_Pointwise(/*in_shape=*/Conv2d_4_Depthwise.out_shape()[0], /*out_channel=*/192, /*kernel_shape=*/{1, 1},
                            /*group_count=*/1, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{0, 0, 0, 0}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_5_Depthwise(/*in_shape=*/Conv2d_4_Pointwise.out_shape()[0], /*out_channel=*/192, /*kernel_shape=*/{3, 3},
                            /*group_count=*/192, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{1, 1, 1, 1}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_5_Pointwise(/*in_shape=*/Conv2d_5_Depthwise.out_shape()[0], /*out_channel=*/192, /*kernel_shape=*/{1, 1},
                            /*group_count=*/1, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{0, 0, 0, 0}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_6_Depthwise(/*in_shape=*/Conv2d_5_Pointwise.out_shape()[0], /*out_channel=*/192, /*kernel_shape=*/{3, 3},
                            /*group_count=*/192, /*has_bias=*/true, /*stride_shape=*/{2, 2}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{0, 0, 1, 1}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_6_Pointwise(/*in_shape=*/Conv2d_6_Depthwise.out_shape()[0], /*out_channel=*/384, /*kernel_shape=*/{1, 1},
                            /*group_count=*/1, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{0, 0, 0, 0}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_7_Depthwise(/*in_shape=*/Conv2d_6_Pointwise.out_shape()[0], /*out_channel=*/384, /*kernel_shape=*/{3, 3},
                            /*group_count=*/384, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{1, 1, 1, 1}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_7_Pointwise(/*in_shape=*/Conv2d_7_Depthwise.out_shape()[0], /*out_channel=*/384, /*kernel_shape=*/{1, 1},
                            /*group_count=*/1, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{0, 0, 0, 0}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_8_Depthwise(/*in_shape=*/Conv2d_7_Pointwise.out_shape()[0], /*out_channel=*/384, /*kernel_shape=*/{3, 3},
                            /*group_count=*/384, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{1, 1, 1, 1}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_8_Pointwise(/*in_shape=*/Conv2d_8_Depthwise.out_shape()[0], /*out_channel=*/384, /*kernel_shape=*/{1, 1},
                            /*group_count=*/1, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{0, 0, 0, 0}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_9_Depthwise(/*in_shape=*/Conv2d_8_Pointwise.out_shape()[0], /*out_channel=*/384, /*kernel_shape=*/{3, 3},
                            /*group_count=*/384, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{1, 1, 1, 1}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_9_Pointwise(/*in_shape=*/Conv2d_9_Depthwise.out_shape()[0], /*out_channel=*/384, /*kernel_shape=*/{1, 1},
                            /*group_count=*/1, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{0, 0, 0, 0}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_10_Depthwise(/*in_shape=*/Conv2d_9_Pointwise.out_shape()[0], /*out_channel=*/384, /*kernel_shape=*/{3, 3},
                            /*group_count=*/384, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{1, 1, 1, 1}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_10_Pointwise(/*in_shape=*/Conv2d_10_Depthwise.out_shape()[0], /*out_channel=*/384, /*kernel_shape=*/{1, 1},
                            /*group_count=*/1, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{0, 0, 0, 0}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_11_Depthwise(/*in_shape=*/Conv2d_10_Pointwise.out_shape()[0], /*out_channel=*/384, /*kernel_shape=*/{3, 3},
                            /*group_count=*/384, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{1, 1, 1, 1}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    static conv Conv2d_11_Pointwise(/*in_shape=*/Conv2d_11_Depthwise.out_shape()[0], /*out_channel=*/384, /*kernel_shape=*/{1, 1},
                            /*group_count=*/1, /*has_bias=*/true, /*stride_shape=*/{1, 1}, /*dilation_shape=*/{1, 1},
                            /*pad_type=*/padding_mode::notset, /*pads=*/{0, 0, 0, 0}, /*activation_type=*/mmpack::Relu,
                            /*engine=*/core::backend_t::xnnpack);

    /*
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     *                                                   LAYER INITIALIZATION                                              *
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     */
    std::string src_name;
    INIT_MODEL_TENSOR("const_fold_opt__629", Conv2d_0.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D_Fold_bias_dequant__153", Conv2d_0.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__690", Conv2d_1_Depthwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise_Fold_bias_dequant__135", Conv2d_1_Depthwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__670", Conv2d_1_Pointwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D_Fold_bias_dequant__133", Conv2d_1_Pointwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__657", Conv2d_2_Depthwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise_Fold_bias_dequant__131", Conv2d_2_Depthwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__611", Conv2d_2_Pointwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D_Fold_bias_dequant__129", Conv2d_2_Pointwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__644", Conv2d_3_Depthwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise_Fold_bias_dequant__127", Conv2d_3_Depthwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__677", Conv2d_3_Pointwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D_Fold_bias_dequant__125", Conv2d_3_Pointwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__666", Conv2d_4_Depthwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise_Fold_bias_dequant__123", Conv2d_4_Depthwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__647", Conv2d_4_Pointwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D_Fold_bias_dequant__121", Conv2d_4_Pointwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__671", Conv2d_5_Depthwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise_Fold_bias_dequant__119", Conv2d_5_Depthwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__674", Conv2d_5_Pointwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D_Fold_bias_dequant__117", Conv2d_5_Pointwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__607", Conv2d_6_Depthwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise_Fold_bias_dequant__115", Conv2d_6_Depthwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__684", Conv2d_6_Pointwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D_Fold_bias_dequant__113", Conv2d_6_Pointwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__683", Conv2d_7_Depthwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise_Fold_bias_dequant__111", Conv2d_7_Depthwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__624", Conv2d_7_Pointwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D_Fold_bias_dequant__109", Conv2d_7_Pointwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__651", Conv2d_8_Depthwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise_Fold_bias_dequant__107", Conv2d_8_Depthwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__653", Conv2d_8_Pointwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D_Fold_bias_dequant__105", Conv2d_8_Pointwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__682", Conv2d_9_Depthwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise_Fold_bias_dequant__103", Conv2d_9_Depthwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__581", Conv2d_9_Pointwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D_Fold_bias_dequant__101", Conv2d_9_Pointwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__672", Conv2d_10_Depthwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise_Fold_bias_dequant__151", Conv2d_10_Depthwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__637", Conv2d_10_Pointwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D_Fold_bias_dequant__149", Conv2d_10_Pointwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__681", Conv2d_11_Depthwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise_Fold_bias_dequant__147", Conv2d_11_Depthwise.weights()[1])

    INIT_MODEL_TENSOR("const_fold_opt__642", Conv2d_11_Pointwise.weights()[0])
    INIT_MODEL_TENSOR("FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D_Fold_bias_dequant__145", Conv2d_11_Pointwise.weights()[1])


    /*
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     *                                                   NETWORK CONSTRUCTION                                              *
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     */

    connect_subgraph(Conv2d_1_Depthwise, Conv2d_0);
    connect_subgraph(Conv2d_1_Pointwise, Conv2d_1_Depthwise);
    connect_subgraph(Conv2d_2_Depthwise, Conv2d_1_Pointwise);
    connect_subgraph(Conv2d_2_Pointwise, Conv2d_2_Depthwise);
    connect_subgraph(Conv2d_3_Depthwise, Conv2d_2_Pointwise);
    connect_subgraph(Conv2d_3_Pointwise, Conv2d_3_Depthwise);
//    connect_subgraph(Conv2d_4_Depthwise, Conv2d_3_Pointwise);
//    connect_subgraph(Conv2d_4_Pointwise, Conv2d_4_Depthwise);
//    connect_subgraph(Conv2d_5_Depthwise, Conv2d_4_Pointwise);
//    connect_subgraph(Conv2d_5_Pointwise, Conv2d_5_Depthwise);
//    connect_subgraph(Conv2d_6_Depthwise, Conv2d_5_Pointwise);
//    connect_subgraph(Conv2d_6_Pointwise, Conv2d_6_Depthwise);
//    connect_subgraph(Conv2d_7_Depthwise, Conv2d_6_Pointwise);
//    connect_subgraph(Conv2d_7_Pointwise, Conv2d_7_Depthwise);
//    connect_subgraph(Conv2d_8_Depthwise, Conv2d_7_Pointwise);
//    connect_subgraph(Conv2d_8_Pointwise, Conv2d_8_Depthwise);
//    connect_subgraph(Conv2d_9_Depthwise, Conv2d_8_Pointwise);
//    connect_subgraph(Conv2d_9_Pointwise, Conv2d_9_Depthwise);
//    connect_subgraph(Conv2d_10_Depthwise, Conv2d_9_Pointwise);
//    connect_subgraph(Conv2d_10_Pointwise, Conv2d_10_Depthwise);
//    connect_subgraph(Conv2d_11_Depthwise, Conv2d_10_Pointwise);
//    connect_subgraph(Conv2d_11_Pointwise, Conv2d_11_Depthwise);

    construct_graph(*net, {&Conv2d_0}, {&Conv2d_3_Pointwise});
    return 1;
}

} // models
