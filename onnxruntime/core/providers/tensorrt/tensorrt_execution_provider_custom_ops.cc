// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_set>

#include "core/framework/provider_options.h"
#include "tensorrt_execution_provider_custom_ops.h"
#include "tensorrt_execution_provider.h"

// The filename extension for a shared library is different per platform
#ifdef _WIN32
#define LIBRARY_PREFIX
#define LIBRARY_EXTENSION ORT_TSTR(".dll")
#elif defined(__APPLE__)
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".dylib"
#else
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".so"
#endif

#ifdef _WIN32
#define ORT_DEF2STR_HELPER(x) L#x
#else
#define ORT_DEF2STR_HELPER(X) #X
#endif
#define ORT_DEF2STR(x) ORT_DEF2STR_HELPER(x)

namespace onnxruntime {
extern TensorrtLogger& GetTensorrtLogger(bool verbose);

/*
 * Create custom op domain list for TRT plugins.
 *
 * Here, we collect all registered TRT plugins from TRT registry and create custom ops with "trt.plugins" domain.
 * Additionally, if users specify extra plugin libraries, TRT EP will load them at runtime which will register those
 * plugins to TRT plugin registry and later TRT EP can get them as well.
 *
 * There are several TRT plugins registered as onnx schema op through contrib op with ONNX domain in the past,
 * for example, EfficientNMS_TRT, MultilevelCropAndResize_TRT, PyramidROIAlign_TRT and DisentangledAttention_TRT.
 * In order not to break the old models using those TRT plugins which were registered with ONNX domain and maintain
 * backward compatible, we need to keep those legacy TRT plugins registered with ONNX domain with contrib ops.
 *
 * Note: Current TRT plugin doesn't have APIs to get number of inputs/outputs of the plugin.
 * So, TensorRTCustomOp uses variadic inputs/outputs to pass ONNX graph validation.
 */
common::Status CreateTensorRTCustomOpDomainList(std::vector<OrtCustomOpDomain*>& domain_list, const std::string extra_plugin_lib_paths) {
  static std::unique_ptr<OrtCustomOpDomain> custom_op_domain = std::make_unique<OrtCustomOpDomain>();
  static std::vector<std::unique_ptr<TensorRTCustomOp>> created_custom_op_list;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  if (custom_op_domain->domain_ != "" && custom_op_domain->custom_ops_.size() > 0) {
    domain_list.push_back(custom_op_domain.get());
    return Status::OK();
  }

  // Load any extra TRT plugin library if any.
  // When the TRT plugin library is loaded, the global static object is created and the plugin is registered to TRT registry.
  // This is done through macro, for example, REGISTER_TENSORRT_PLUGIN(VisionTransformerPluginCreator).
  // extra_plugin_lib_paths has the format of "path_1;path_2....;path_n"
  static bool is_loaded = false;
  if (!extra_plugin_lib_paths.empty() && !is_loaded) {
    std::stringstream extra_plugin_libs(extra_plugin_lib_paths);
    std::string lib;
    while (std::getline(extra_plugin_libs, lib, ';')) {
      auto status = LoadDynamicLibrary(ToPathString(lib));
      if (status == Status::OK()) {
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Successfully load " << lib;
      } else {
        LOGS_DEFAULT(WARNING) << "[TensorRT EP]" << status.ToString();
      }
    }
    is_loaded = true;
  }

  try {
    // Get all registered TRT plugins from registry
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Getting all registered TRT plugins from TRT plugin registry ...";
    TensorrtLogger trt_logger = GetTensorrtLogger(false);
    try {
      void* library_handle = nullptr;
      const auto& env = onnxruntime::GetDefaultEnv();
#if NV_TENSORRT_MAJOR < 10
      auto full_path = env.GetRuntimePath() +
                       PathString(LIBRARY_PREFIX ORT_TSTR("nvinfer_plugin") LIBRARY_EXTENSION);
#else
#ifdef _WIN32
      auto full_path = PathString(LIBRARY_PREFIX ORT_TSTR("nvinfer_plugin_" ORT_DEF2STR(NV_TENSORRT_MAJOR)) LIBRARY_EXTENSION);
#else
      auto full_path = PathString(LIBRARY_PREFIX ORT_TSTR("nvinfer_plugin") LIBRARY_EXTENSION ORT_TSTR("." ORT_DEF2STR(NV_TENSORRT_MAJOR)));
#endif
#endif

      ORT_THROW_IF_ERROR(env.LoadDynamicLibrary(full_path, false, &library_handle));

      bool (*dyn_initLibNvInferPlugins)(void* logger, char const* libNamespace);
      ORT_THROW_IF_ERROR(env.GetSymbolFromLibrary(library_handle, "initLibNvInferPlugins", (void**)&dyn_initLibNvInferPlugins));
      if (!dyn_initLibNvInferPlugins(&trt_logger, "")) {
        LOGS_DEFAULT(INFO) << "[TensorRT EP] Default plugin library was found but was not able to initialize default plugins.";
      }
      LOGS_DEFAULT(INFO) << "[TensorRT EP] Default plugins successfully loaded.";
    } catch (const std::exception&) {
      LOGS_DEFAULT(INFO) << "[TensorRT EP] Default plugin library is not on the path and is therefore ignored";
    }
    int num_plugin_creator = 0;
    auto plugin_creators = getPluginRegistry()->getAllCreators(&num_plugin_creator);
    std::unordered_set<std::string> registered_plugin_names;

    for (int i = 0; i < num_plugin_creator; i++) {
      auto plugin_creator = plugin_creators[i];
      nvinfer1::AsciiChar const* plugin_name = nullptr;
      if (std::strcmp(plugin_creators[i]->getInterfaceInfo().kind, "PLUGIN CREATOR_V1") == 0) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)  // Ignore warning C4996: 'nvinfer1::*' was declared deprecated
#endif
        auto plugin_creator_v1 = static_cast<nvinfer1::IPluginCreator const*>(plugin_creator);
        plugin_name = plugin_creator_v1->getPluginName();
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] " << plugin_name << ", version : " << plugin_creator_v1->getPluginVersion();
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
      } else if (std::strcmp(plugin_creators[i]->getInterfaceInfo().kind, "PLUGIN CREATOR_V3ONE") == 0) {
        auto plugin_creator_v3 = static_cast<nvinfer1::IPluginCreatorV3One const*>(plugin_creator);
        plugin_name = plugin_creator_v3->getPluginName();
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP][V3ONE] " << plugin_name << ", version : " << plugin_creator_v3->getPluginVersion();
      } else {
        ORT_THROW("Unknown plugin creator type");
      }

      // plugin has different versions and we only register once
      if (registered_plugin_names.find(plugin_name) != registered_plugin_names.end()) {
        continue;
      }

      created_custom_op_list.push_back(std::make_unique<TensorRTCustomOp>(onnxruntime::kTensorrtExecutionProvider, nullptr));  // Make sure TensorRTCustomOp object won't be cleaned up
      created_custom_op_list.back().get()->SetName(plugin_name);
      custom_op_domain->custom_ops_.push_back(created_custom_op_list.back().get());
      registered_plugin_names.insert(plugin_name);
    }

    custom_op_domain->domain_ = "trt.plugins";
    domain_list.push_back(custom_op_domain.get());
  } catch (const std::exception&) {
    LOGS_DEFAULT(WARNING) << "[TensorRT EP] Failed to get TRT plugins from TRT plugin registration. Therefore, TRT EP can't create custom ops for TRT plugins";
  }
  return Status::OK();
}

void ReleaseTensorRTCustomOpDomain(OrtCustomOpDomain* domain) {
  if (domain != nullptr) {
    for (auto ptr : domain->custom_ops_) {
      if (ptr != nullptr) {
        delete ptr;
      }
    }
    delete domain;
  }
}

void ReleaseTensorRTCustomOpDomainList(std::vector<OrtCustomOpDomain*>& custom_op_domain_list) {
  for (auto ptr : custom_op_domain_list) {
    ReleaseTensorRTCustomOpDomain(ptr);
  }
}

}  // namespace onnxruntime
