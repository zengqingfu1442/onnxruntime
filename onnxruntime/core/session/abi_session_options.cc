// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cstring>
#include <sstream>

#include "core/common/inlined_containers.h"
#include "core/framework/error_code_helper.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"
#include "core/session/utils.h"

OrtSessionOptions::~OrtSessionOptions() = default;

OrtSessionOptions& OrtSessionOptions::operator=(const OrtSessionOptions&) {
  ORT_THROW("not implemented");
}
OrtSessionOptions::OrtSessionOptions(const OrtSessionOptions& other)
    : value(other.value), custom_op_domains_(other.custom_op_domains_), provider_factories(other.provider_factories) {
}

const onnxruntime::ConfigOptions& OrtSessionOptions::GetConfigOptions() const noexcept {
  return value.config_options;
}

onnxruntime::Status OrtSessionOptions::AddProviderOptionsToConfigOptions(
    const std::unordered_map<std::string, std::string>& provider_options, const char* provider_name) {
  // Add provider options to the session config options.
  // Use a new key with the format: "ep.<lowercase_provider_name>.<PROVIDER_OPTION_KEY>"
  auto key_prefix = GetProviderOptionPrefix(provider_name);
  for (const auto& [ep_key, ep_value] : provider_options) {
    const std::string new_key = key_prefix + ep_key;
    ORT_RETURN_IF_ERROR(value.config_options.AddConfigEntry(new_key.c_str(), ep_value.c_str()));
  }
  return Status::OK();
}

// static
std::string OrtSessionOptions::GetProviderOptionPrefix(const char* provider_name) {
  std::string key_prefix = "ep.";
  key_prefix += onnxruntime::utils::GetLowercaseString(provider_name);
  key_prefix += ".";

  return key_prefix;
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
onnxruntime::Status OrtSessionOptions::RegisterCustomOpsLibrary(onnxruntime::PathString library_name) {
  const auto& platform_env = onnxruntime::Env::Default();
  void* library_handle = nullptr;

  ORT_RETURN_IF_ERROR(platform_env.LoadDynamicLibrary(library_name, false, &library_handle));
  if (!library_handle) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to load dynamic library ",
                           onnxruntime::PathToUTF8String(library_name));
  }

  OrtStatus*(ORT_API_CALL * RegisterCustomOps)(OrtSessionOptions * options, const OrtApiBase* api) = nullptr;
  ORT_RETURN_IF_ERROR(platform_env.GetSymbolFromLibrary(library_handle, "RegisterCustomOps",
                                                        (void**)&RegisterCustomOps));

  // Call the exported RegisterCustomOps function.
  auto status = onnxruntime::ToStatusAndRelease(RegisterCustomOps(this, OrtGetApiBase()));

  if (!status.IsOK()) {
    auto unload_status = platform_env.UnloadDynamicLibrary(library_handle);
    if (!unload_status.IsOK()) {
      LOGS_DEFAULT(WARNING) << "Failed to unload handle for dynamic library "
                            << onnxruntime::PathToUTF8String(library_name) << ": " << unload_status;
    }

    return status;
  }

  // The internal onnxruntime::SessionOptions will manage the lifetime of library handles.
  this->value.AddCustomOpLibraryHandle(std::move(library_name), library_handle);

  return onnxruntime::Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

ORT_API_STATUS_IMPL(OrtApis::CreateSessionOptions, OrtSessionOptions** out) {
  API_IMPL_BEGIN
  GSL_SUPPRESS(r.11)
  *out = new OrtSessionOptions();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseSessionOptions, _Frees_ptr_opt_ OrtSessionOptions* ptr) {
  delete ptr;
}

ORT_API_STATUS_IMPL(OrtApis::CloneSessionOptions, const OrtSessionOptions* input, OrtSessionOptions** out) {
  API_IMPL_BEGIN
  GSL_SUPPRESS(r.11)
  *out = new OrtSessionOptions(*input);
  return nullptr;
  API_IMPL_END
}

// Set execution_mode.
ORT_API_STATUS_IMPL(OrtApis::SetSessionExecutionMode, _In_ OrtSessionOptions* options,
                    ExecutionMode execution_mode) {
  switch (execution_mode) {
    case ORT_SEQUENTIAL:
    case ORT_PARALLEL:
      options->value.execution_mode = execution_mode;
      break;
    default:
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "execution_mode is not valid");
  }

  return nullptr;
}

// set filepath to save optimized onnx model.
ORT_API_STATUS_IMPL(OrtApis::SetOptimizedModelFilePath, _In_ OrtSessionOptions* options, _In_ const ORTCHAR_T* optimized_model_filepath) {
  options->value.optimized_model_filepath = optimized_model_filepath;
  return nullptr;
}

// enable profiling for this session.
ORT_API_STATUS_IMPL(OrtApis::EnableProfiling, _In_ OrtSessionOptions* options, _In_ const ORTCHAR_T* profile_file_prefix) {
  options->value.enable_profiling = true;
  options->value.profile_file_prefix = profile_file_prefix;
  return nullptr;
}
ORT_API_STATUS_IMPL(OrtApis::DisableProfiling, _In_ OrtSessionOptions* options) {
  options->value.enable_profiling = false;
  options->value.profile_file_prefix.clear();
  return nullptr;
}

// enable the memory pattern optimization.
// The idea is if the input shapes are the same, we could trace the internal memory allocation
// and generate a memory pattern for future request. So next time we could just do one allocation
// with a big chunk for all the internal memory allocation.
ORT_API_STATUS_IMPL(OrtApis::EnableMemPattern, _In_ OrtSessionOptions* options) {
  options->value.enable_mem_pattern = true;
  return nullptr;
}
ORT_API_STATUS_IMPL(OrtApis::DisableMemPattern, _In_ OrtSessionOptions* options) {
  options->value.enable_mem_pattern = false;
  return nullptr;
}

// enable the memory arena on CPU
// Arena may pre-allocate memory for future usage.
// set this option to false if you don't want it.
ORT_API_STATUS_IMPL(OrtApis::EnableCpuMemArena, _In_ OrtSessionOptions* options) {
  options->value.enable_cpu_mem_arena = true;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::DisableCpuMemArena, _In_ OrtSessionOptions* options) {
  options->value.enable_cpu_mem_arena = false;
  return nullptr;
}

///< logger id to use for session output
ORT_API_STATUS_IMPL(OrtApis::SetSessionLogId, _In_ OrtSessionOptions* options, const char* logid) {
  options->value.session_logid = logid;
  return nullptr;
}

///< logging function and optional logging param to use for session output
ORT_API_STATUS_IMPL(OrtApis::SetUserLoggingFunction, _In_ OrtSessionOptions* options,
                    _In_ OrtLoggingFunction user_logging_function, _In_opt_ void* user_logging_param) {
  options->value.user_logging_function = user_logging_function;
  options->value.user_logging_param = user_logging_param;
  return nullptr;
}

///< applies to session load, initialization, etc
ORT_API_STATUS_IMPL(OrtApis::SetSessionLogVerbosityLevel, _In_ OrtSessionOptions* options, int session_log_verbosity_level) {
  options->value.session_log_verbosity_level = session_log_verbosity_level;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::SetSessionLogSeverityLevel, _In_ OrtSessionOptions* options, int session_log_severity_level) {
  options->value.session_log_severity_level = session_log_severity_level;
  return nullptr;
}

// Set Graph optimization level.
ORT_API_STATUS_IMPL(OrtApis::SetSessionGraphOptimizationLevel, _In_ OrtSessionOptions* options,
                    GraphOptimizationLevel graph_optimization_level) {
  if (graph_optimization_level < 0) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "graph_optimization_level is not valid");
  }

  switch (graph_optimization_level) {
    case ORT_DISABLE_ALL:
      options->value.graph_optimization_level = onnxruntime::TransformerLevel::Default;
      break;
    case ORT_ENABLE_BASIC:
      options->value.graph_optimization_level = onnxruntime::TransformerLevel::Level1;
      break;
    case ORT_ENABLE_EXTENDED:
      options->value.graph_optimization_level = onnxruntime::TransformerLevel::Level2;
      break;
    case ORT_ENABLE_LAYOUT:
      options->value.graph_optimization_level = onnxruntime::TransformerLevel::Level3;
      break;
    case ORT_ENABLE_ALL:
      options->value.graph_optimization_level = onnxruntime::TransformerLevel::MaxLevel;
      break;
    default:
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "graph_optimization_level is not valid");
  }

  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::SetIntraOpNumThreads, _Inout_ OrtSessionOptions* options, int intra_op_num_threads) {
  options->value.intra_op_param.thread_pool_size = intra_op_num_threads;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::SetInterOpNumThreads, _Inout_ OrtSessionOptions* options, int inter_op_num_threads) {
  options->value.inter_op_param.thread_pool_size = inter_op_num_threads;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::AddFreeDimensionOverride, _Inout_ OrtSessionOptions* options,
                    _In_ const char* dim_denotation, _In_ int64_t dim_value) {
  options->value.free_dimension_overrides.push_back(
      onnxruntime::FreeDimensionOverride{dim_denotation, onnxruntime::FreeDimensionOverrideType::Denotation, dim_value});
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::AddFreeDimensionOverrideByName, _Inout_ OrtSessionOptions* options,
                    _In_ const char* dim_name, _In_ int64_t dim_value) {
  options->value.free_dimension_overrides.push_back(
      onnxruntime::FreeDimensionOverride{dim_name, onnxruntime::FreeDimensionOverrideType::Name, dim_value});
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::DisablePerSessionThreads, _In_ OrtSessionOptions* options) {
  options->value.use_per_session_threads = false;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::AddSessionConfigEntry, _Inout_ OrtSessionOptions* options,
                    _In_z_ const char* config_key, _In_z_ const char* config_value) {
  return onnxruntime::ToOrtStatus(options->value.config_options.AddConfigEntry(config_key, config_value));
}

ORT_API_STATUS_IMPL(OrtApis::HasSessionConfigEntry, _In_ const OrtSessionOptions* options,
                    _In_z_ const char* config_key, _Out_ int* out) {
  API_IMPL_BEGIN
  auto value_opt = options->value.config_options.GetConfigEntry(config_key);
  *out = static_cast<int>(value_opt.has_value());
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetSessionConfigEntry, _In_ const OrtSessionOptions* options,
                    _In_z_ const char* config_key, _Out_ char* config_value, _Inout_ size_t* size) {
  API_IMPL_BEGIN
  auto value_opt = options->value.config_options.GetConfigEntry(config_key);

  if (!value_opt) {
    std::ostringstream err_msg;
    err_msg << "Session config entry '" << config_key << "' was not found.";
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, err_msg.str().c_str());
  }

  auto status = CopyStringToOutputArg(*value_opt, "Output buffer is not large enough for session config entry", config_value,
                                      size);

  return onnxruntime::ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetSessionOptionsConfigEntries, _In_ const OrtSessionOptions* options, _Outptr_ OrtKeyValuePairs** out) {
  API_IMPL_BEGIN
  if (options == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "options is nullptr");
  }
  auto& config_options = options->value.config_options.GetConfigOptionsMap();
  auto kvps = std::make_unique<OrtKeyValuePairs>();
  for (auto& kv : config_options) {
    kvps->Add(kv.first.c_str(), kv.second.c_str());
  }
  *out = reinterpret_cast<OrtKeyValuePairs*>(kvps.release());
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::AddInitializer, _Inout_ OrtSessionOptions* options, _In_z_ const char* name,
                    _In_ const OrtValue* val) {
  API_IMPL_BEGIN
  auto st = options->value.AddInitializer(name, val);
  if (!st.IsOK()) {
    return onnxruntime::ToOrtStatus(st);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::AddExternalInitializers, _In_ OrtSessionOptions* options,
                    _In_reads_(initializers_num) const char* const* initializer_names,
                    _In_reads_(initializers_num) const OrtValue* const* initializers, size_t initializers_num) {
#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
  API_IMPL_BEGIN
  onnxruntime::InlinedVector<std::string> names;
  onnxruntime::InlinedVector<OrtValue> values;
  names.reserve(initializers_num);
  values.reserve(initializers_num);
  for (size_t i = 0; i < initializers_num; ++i) {
    if (initializer_names[i] == nullptr || initializers[i] == nullptr) {
      auto message = onnxruntime::MakeString("Input index: ", i, " contains null pointers");
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, message.c_str());
    }
    names.emplace_back(initializer_names[i]);
    values.emplace_back(*initializers[i]);
  }

  auto st = options->value.AddExternalInitializers(names, values);
  if (!st.IsOK()) {
    return onnxruntime::ToOrtStatus(st);
  }
  return nullptr;
  API_IMPL_END
#else
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(initializer_names);
  ORT_UNUSED_PARAMETER(initializers);
  ORT_UNUSED_PARAMETER(initializers_num);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "External initializers are not supported in this build");
#endif
}

ORT_API_STATUS_IMPL(OrtApis::AddExternalInitializersFromFilesInMemory, _In_ OrtSessionOptions* options,
                    _In_reads_(num_external_initializer_files) const ORTCHAR_T* const* file_names,
                    _In_reads_(num_external_initializer_files) char* const* buffer_array,
                    _In_reads_(num_external_initializer_files) const size_t* file_lengths,
                    size_t num_external_initializer_files) {
#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
  API_IMPL_BEGIN
  onnxruntime::InlinedVector<onnxruntime::PathString> names;
  onnxruntime::InlinedVector<std::pair<char*, const size_t>> buffers;
  onnxruntime::InlinedVector<size_t> lengths;
  names.reserve(num_external_initializer_files);
  buffers.reserve(num_external_initializer_files);
  lengths.reserve(num_external_initializer_files);
  for (size_t i = 0; i < num_external_initializer_files; ++i) {
    if (file_names[i] == nullptr || buffer_array[i] == nullptr) {
      auto message = onnxruntime::MakeString("Input index: ", i, " contains null pointers");
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, message.c_str());
    }
    names.emplace_back(file_names[i]);
    buffers.emplace_back(std::make_pair(buffer_array[i], file_lengths[i]));
  }

  auto st = options->value.AddExternalInitializersFromFilesInMemory(names, buffers);
  if (!st.IsOK()) {
    return onnxruntime::ToOrtStatus(st);
  }
  return nullptr;
  API_IMPL_END
#else
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(file_names);
  ORT_UNUSED_PARAMETER(buffer_array);
  ORT_UNUSED_PARAMETER(file_lengths);
  ORT_UNUSED_PARAMETER(num_external_initializer_files);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED,
                               "AddExternalInitializersFromFilesInMemory is not supported in this build");
#endif
}

ORT_API_STATUS_IMPL(OrtApis::SetDeterministicCompute, _Inout_ OrtSessionOptions* options, bool value) {
  API_IMPL_BEGIN
  options->value.use_deterministic_compute = value;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsSetEpSelectionPolicy, _In_ OrtSessionOptions* options,
                    _In_ OrtExecutionProviderDevicePolicy policy) {
  API_IMPL_BEGIN
  options->value.ep_selection_policy.enable = true;
  options->value.ep_selection_policy.policy = policy;
  options->value.ep_selection_policy.delegate = nullptr;
  options->value.ep_selection_policy.state = nullptr;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsSetEpSelectionPolicyDelegate, _In_ OrtSessionOptions* options,
                    _In_opt_ EpSelectionDelegate delegate,
                    _In_opt_ void* state) {
  API_IMPL_BEGIN
  options->value.ep_selection_policy.enable = true;
  options->value.ep_selection_policy.policy = OrtExecutionProviderDevicePolicy_DEFAULT;
  options->value.ep_selection_policy.delegate = delegate;
  options->value.ep_selection_policy.state = state;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsSetLoadCancellationFlag, _Inout_ OrtSessionOptions* options,
                    _In_ bool is_cancel) {
  API_IMPL_BEGIN
  options->value.SetLoadCancellationFlag(is_cancel);
  return nullptr;
  API_IMPL_END
}
