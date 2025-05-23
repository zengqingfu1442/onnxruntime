// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <vector>
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
using contrib::AttentionKernelType;
using contrib::AttentionMaskType;

namespace test {

struct BaseAttentionTestData {
  bool is_static_kv = true;
  bool buffer_share = false;
  int hidden_size;
  int v_hidden_size;
  int num_heads;
  int batch_size;
  int sequence_length;
  int kv_sequence_length;

  std::vector<float> query_data;
  std::vector<float> key_data;
  std::vector<float> value_data;

  std::vector<float> kv_data;
  std::vector<float> qkv_data;

  std::vector<float> bias_data;
  std::vector<float> attention_bias_data;
  bool broadcast_attention_bias;

  std::vector<float> past_key_data;
  std::vector<float> past_value_data;

  std::vector<int32_t> past_seq_len_data;
  std::vector<int32_t> cache_indir_data;
  int num_beams;
  int max_sequence_length;

  std::vector<float> fp32_output_data;
  std::vector<float> fp16_output_data;

  std::vector<float> present_key_data;
  std::vector<float> present_value_data;

  std::vector<float> fp32_output_qk_data;
  std::vector<float> fp16_output_qk_data;

  std::vector<AttentionKernelType> skip_kernel_types;  // skip some kernels if they do not supported this test case.
};

struct AttentionTestData : public BaseAttentionTestData {
  AttentionMaskType mask_type;
  std::vector<int> key_padding_mask_data;
};

struct PackedAttentionTestData : public BaseAttentionTestData {
  int token_count;
  std::vector<int32_t> token_offset;
  std::vector<int32_t> cumulative_sequence_length;
};

// Return packed weights and bias for input projection.
void GetAttentionWeight(std::vector<float>& weight_data, int elements = 64 * 3 * 64, int offset = 0, int step = 1);
void GetAttentionBias(std::vector<float>& bias_data, int elements = 3 * 64, int offset = 0, int step = 1);

void GetCrossAttentionData_HeadSize40(AttentionTestData& data);
void GetCrossAttentionData_HeadSize40_NoBias(AttentionTestData& data);
void GetCrossAttentionData_Batch2_HeadSize32_RightSidePadding(AttentionTestData& data, bool is_mask_1d);
void GetCrossAttentionData_Batch2_HeadSize32_RightSidePadding_NoBias(AttentionTestData& data, bool is_mask_1d);
void GetCrossAttentionData_Batch1_HeadSize32_LeftSidePadding(AttentionTestData& data);
void GetCrossAttentionData_Batch1_HeadSize32_LeftSidePadding_NoBias(AttentionTestData& data);

void GetCrossAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedKV(AttentionTestData& data);
void GetSelfAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedQKV(AttentionTestData& data);

void GetCrossAttentionData_HeadSize16_8(AttentionTestData& data);
void GetCrossAttentionData_HeadSize16_8_NoBias(AttentionTestData& data);
void GetCrossAttentionData_HeadSize16(AttentionTestData& data);
void GetCrossAttentionData_HeadSize16_NoBias(AttentionTestData& data);

void GetCrossAttentionData_HeadSize8(AttentionTestData& data);
void GetCrossAttentionData_HeadSize8_NoBias(AttentionTestData& data);

void GetCrossAttentionDataWithPast(AttentionTestData& data);
void GetSelfAttentionData_WithPast_WithAttnBias_ForT5(AttentionTestData& data);

void GetCrossAttentionData_DiffSequenceLengths(AttentionTestData& data);
void GetCrossAttentionData_DiffSequenceLengths_HeadSize8(AttentionTestData& data);
void GetCrossAttentionData_DiffSequenceLengths_HeadSize8_NoBias(AttentionTestData& data);
void GetSelfAttentionData_WithPastAndPresent_NoMask_NoAttnBias(AttentionTestData& data);
void GetSelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoAttnBias(AttentionTestData& data);
void GetSelfAttentionData_WithPastAndPresent_HeadSize8_NoMask_NoAttnBias_NoBias(AttentionTestData& data);
void GetCrossAttentionData_WithPastPassedInDirectly_NoMask(AttentionTestData& data);

void GetSelfAttention_PastPresentBufferShare_UsingDMMHAInsideMHA(AttentionTestData& data);
void GetCrossAttention_DiffSequenceLengths_UsingDMMHAInsideMHA(AttentionTestData& data);

void GetCausal_EmptyPastState(std::vector<float>& input, std::vector<float>& output, std::vector<float>& present);

void GetAttentionDataCutlassAttnBias(AttentionTestData& data);
void GetAttentionDataWithNeoXRotaryEmbedding(std::vector<float>& input,
                                             std::vector<float>& weights,
                                             std::vector<float>& bias,
                                             std::vector<float>& output);

void GetPackedMultiHeadAttentionData_Batch2_HeadSize32_NoAttnBias(PackedAttentionTestData& data);

void GetPackedMultiHeadAttentionData_Batch2_HeadSize8_AttnBias(PackedAttentionTestData& data);

void GetPackedMultiHeadAttentionData_Batch2_HeadSize8_BroadcastAttnBias(PackedAttentionTestData& data);

bool SkipAttentionKernel(AttentionTestData& data, AttentionKernelType kernel_type);
}  // namespace test
}  // namespace onnxruntime
