//
// Created by rozhin on 14.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//
#include <xsdnn/gsl/span>
#include <xs_inference_examples/models.h>
#include <sys/time.h>

mat_t GetUniformRandomData(size_t tensor_size) {
    mat_t Tensor(tensor_size * sizeof(float));
    for (size_t i = 0; i < tensor_size; ++i) {
        Tensor[i] = 'a';
    }
}

void random_init_fp32(xsdnn::mat_t& X) {
    gsl::span<float> XSpan = xsdnn::GetMutableDataAsSpan<float>(&X);
    for (size_t i = 0; i < XSpan.size() / sizeof(float); ++i) {
        XSpan[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
}

int main() {
    xsdnn::network net("FP32SsdMobileNetV1_1_default_1");
    if (!models::FP32SsdMobileNetV1_1_default_1(&net)) throw xs_error("Error");
    mat_t Input(300*300*3 * sizeof(float));
    random_init_fp32(Input);

    net.set_num_threads(1);
    net.configure();
    double sum_time_taken = 0;
    for (size_t i = 0; i < 100; ++i) {
        struct timeval start, end;
        // start timer.
        gettimeofday(&start, NULL);
        // unsync the I/O of C and C++.
        std::ios_base::sync_with_stdio(false);
        net.predict({Input});
        // stop timer.
        gettimeofday(&end, NULL);
        // Calculating total time taken by the program.
        double time_taken;
        time_taken = (end.tv_sec - start.tv_sec) * 1e6;
        time_taken = (time_taken + (end.tv_usec -
                                    start.tv_usec)) * 1e-6;
        sum_time_taken += time_taken;

    }

    std::cout << "Mean Time taken by program is : " << std::fixed
              << sum_time_taken / double(100.0);
    std::cout << " sec" << std::endl;


}
