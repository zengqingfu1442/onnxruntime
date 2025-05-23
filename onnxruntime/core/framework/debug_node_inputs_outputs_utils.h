// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// to create a build with these enabled run the build script with:
//   --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1

// to enable redirect to sqlite database run the build script with:
//   --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1 onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB=1
//
// see orttraining/tools/scripts/sqldb_to_tensors.py for retrieval
//
// select data dump destination using
//   ORT_DEBUG_NODE_IO_DUMP_DATA_DESTINATION= one of {stdout, files, sqlite}

#ifdef DEBUG_NODE_INPUTS_OUTPUTS

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <string>

namespace onnxruntime {
namespace utils {

// environment variables that control debug node dumping behavior
namespace debug_node_inputs_outputs_env_vars {
// Tensor shape and Node placement is printed by default unless it's turned OFF
// by setting the respective environment variables to 0
// set to non-zero to dump tensor shape data
constexpr const char* kDumpShapeData = "ORT_DEBUG_NODE_IO_DUMP_SHAPE_DATA";
// set to non-zero to dump node placement data
constexpr const char* kDumpNodePlacement = "ORT_DEBUG_NODE_IO_DUMP_NODE_PLACEMENT";
// set to non-zero to dump node input data
constexpr const char* kDumpInputData = "ORT_DEBUG_NODE_IO_DUMP_INPUT_DATA";
// set to non-zero to dump node output data
constexpr const char* kDumpOutputData = "ORT_DEBUG_NODE_IO_DUMP_OUTPUT_DATA";
// Output statistics data like min, max, count of NaN, count of infinity etc.
constexpr const char* kDumpStatisticsData = "ORT_DEBUG_NODE_IO_DUMP_STATISTICS_DATA";
// Output node name when any float input or output exceeds a threshold for float16 conversion overflow.
constexpr const char* kDumpHalfConversionOverflow = "ORT_DEBUG_NODE_IO_DUMP_HALF_CONVERSION_OVERFLOW";

// specify a node name filter to limit the nodes that are dumped
// see NodeDumpOptions::FilterOptions
constexpr const char* kNameFilter = "ORT_DEBUG_NODE_IO_NAME_FILTER";
// specify a node op type filter to limit the nodes that are dumped
// see NodeDumpOptions::FilterOptions
constexpr const char* kOpTypeFilter = "ORT_DEBUG_NODE_IO_OP_TYPE_FILTER";
// set to "stdout" or "files" or "sqlite" to select dump destination
constexpr const char* kDumpDataDestination = "ORT_DEBUG_NODE_IO_DUMP_DATA_DESTINATION";
// set to non-zero to append OpenMPI world rank to filename
constexpr const char* kAppendRankToFileName = "ORT_DEBUG_NODE_IO_APPEND_RANK_TO_FILE_NAME";
// specify the output directory for any data files produced
constexpr const char* kOutputDir = "ORT_DEBUG_NODE_IO_OUTPUT_DIR";
// specify the file prefix for sqlite3 db (process id will be appended)
constexpr const char* kSqliteDbPrefix = "ORT_DEBUG_NODE_IO_SQLITE_DB_PREFIX";
// set to non-zero to confirm that dumping data files for all nodes is acceptable
constexpr const char* kDumpingDataToFilesForAllNodesIsOk =
    "ORT_DEBUG_NODE_IO_DUMPING_DATA_TO_FILES_FOR_ALL_NODES_IS_OK";

// Total number of elements which trigger snippet rather than full dump (default 200). Value 0 disables snippet.
constexpr const char* kSnippetThreshold = "ORT_DEBUG_NODE_IO_SNIPPET_THRESHOLD";
// Number of array items in snippet at beginning and end of each dimension (default 3)
constexpr const char* kSnippetEdgeItems = "ORT_DEBUG_NODE_IO_SNIPPET_EDGE_ITEMS";

// Threshold for float to float16 conversion overflow detection (default 50000).
// It is a positive integer that <= 65504, and it is recommended to add some margin for new inputs.
constexpr const char* kHalfOverflowThreshold = "ORT_DEBUG_NODE_IO_HALF_OVERFLOW_THRESHOLD";

}  // namespace debug_node_inputs_outputs_env_vars

constexpr char kFilterPatternDelimiter = ';';

struct NodeDumpOptions {
  enum DumpFlags {
    None = 0,
    Shape = 1 << 0,
    InputData = 1 << 1,
    OutputData = 1 << 2,
    NodePlacement = 1 << 3,
    StatisticsData = 1 << 4,
    HalfConversionOverflow = 1 << 5,
    AllData = Shape | InputData | OutputData | NodePlacement | StatisticsData | HalfConversionOverflow,
  };

  // specifies the information to dump per node
  // see DumpFlags
  // Note:
  // When dumping every node, dumping both input and output may be redundant.
  // Doing that may be more useful when dumping a subset of all nodes.
  int dump_flags{DumpFlags::Shape};

  // filters the nodes that are dumped
  // Note:
  // Pattern strings are substrings (individual patterns) delimited by ';'.
  // For a given pattern type (e.g. Name), a pattern string matches a value if one of the following is true:
  // - the pattern string is empty
  // - one of the delimited substrings matches exactly
  // For a node to be dumped, it must match each pattern type.
  // By default (with empty pattern strings), all nodes will match and be dumped.
  struct FilterOptions {
    std::string name_pattern{};
    std::string op_type_pattern{};
  } filter{};

  // the destination for dumped data
  enum class DataDestination {
    // write to stdout
    StdOut,
    // write to one file per tensor input/output as a TensorProto
    TensorProtoFiles,
    // write to one row per tensor input/output in Sqlite table
    SqliteDb
  } data_destination{DataDestination::StdOut};

  std::string file_suffix;
  // the output directory for dumped data files
  std::filesystem::path output_dir;
  // the sqlite3 db to append dumped data
  std::filesystem::path sqlite_db_prefix;

  // Total number of elements which trigger snippet rather than full array for Stdout. Value 0 disables snippet.
  int snippet_threshold;

  // Number of array items in snippet at beginning and end of each dimension for Stdout.
  int snippet_edge_items;

  // Threshold for float16 conversion overflow.
  float half_overflow_threshold;
};

struct NodeDumpContext {
  // which execution pass are we on?
  size_t iteration;
  // which node are we on?
  size_t program_counter;
};

// A session level analysis of node dumps. It can be used to collect some statistics or analysis during node dumps.
struct NodeDumpAnalysis {
  std::mutex set_mutex;
  std::unordered_set<std::string> half_overflow_nodes;
  std::unordered_map<std::string, int> half_overflow_ops;
  int counter{0};
  void Add(const std::string& node_name, const std::string& op_name, bool is_half_overflow);
  void PrintToStdOut(const std::string& model_path);
};

// gets NodeDumpOptions instance configured from environment variable values
const NodeDumpOptions& NodeDumpOptionsFromEnvironmentVariables();

// dumps inputs for a node
void DumpNodeInputs(
    const NodeDumpOptions& dump_options,
    const NodeDumpContext& dump_context,
    const OpKernelContext& context,
    const Node& node,
    const SessionState& session_state,
    NodeDumpAnalysis& dump_analysis);

void DumpNodeInputs(
    const NodeDumpContext& dump_context,
    const OpKernelContext& context,
    const Node& node,
    const SessionState& session_state,
    NodeDumpAnalysis& dump_analysis);

// dumps outputs for a node
void DumpNodeOutputs(
    const NodeDumpOptions& dump_options,
    const NodeDumpContext& dump_context,
    OpKernelContext& context,
    const Node& node,
    const SessionState& session_state,
    NodeDumpAnalysis& dump_analysis);

void DumpNodeOutputs(
    const NodeDumpContext& dump_context,
    OpKernelContext& context,
    const Node& node,
    const SessionState& session_state,
    NodeDumpAnalysis& dump_analysis);

}  // namespace utils
}  // namespace onnxruntime

#endif
