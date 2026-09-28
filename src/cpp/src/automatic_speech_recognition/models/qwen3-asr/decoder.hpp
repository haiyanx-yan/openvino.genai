// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include <openvino/runtime/core.hpp>

#include "openvino/genai/automatic_speech_recognition/generation_config.hpp"
#include "openvino/genai/automatic_speech_recognition/perf_metrics.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "sampling/sampler.hpp"

namespace ov::genai {

class Qwen3ASRDecoder {
public:
    Qwen3ASRDecoder(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties);

    EncodedResults generate(const ov::Tensor& input_ids,
                            const ov::Tensor& encoder_hidden_state,
                            const ASRGenerationConfig& config,
                            RawPerfMetrics& raw_metrics,
                            ASRRawPerfMetrics& asr_raw_metrics,
                            const std::shared_ptr<StreamerBase>& streamer_ptr = nullptr);

    void set_seed(size_t seed);

private:
    InferRequest m_request;
    Sampler m_sampler;
    std::filesystem::path m_model_path;
    std::string m_device;
    ov::AnyMap m_properties;
    bool m_compile_for_npu = false;
    ov::Shape m_context_shape;
    size_t m_prompt_len = 0;

    void compile_for_context(const ov::Tensor& input_ids, const ov::Tensor& encoder_hidden_states);
};

}  // namespace ov::genai
