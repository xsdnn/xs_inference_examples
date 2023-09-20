//
// Created by rozhin on 14.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <xs_inference_examples/models.h>

namespace models {

xsdnn::network<xsdnn::graph> FP32SsdMobileNetV1_1_default_1() {
    xsdnn::network<xsdnn::sequential> net;
    xsdnn::fully_connected fc(10, 10);


}

} // models