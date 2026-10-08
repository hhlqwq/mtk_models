// 在 Genio 720 上持久加载双 DLA 并批量评测 Whisper-Tiny.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "neuron/api/RuntimeAPI.h"

namespace fs = std::filesystem;

constexpr size_t kVocabularySize = 51865;
constexpr size_t kPaddedVocabularySize = 51872;
constexpr size_t kAudioFeatureElements = 1500 * 384;
constexpr size_t kPositionCount = 448;
constexpr size_t kCacheTokenCount = 200;
constexpr size_t kCacheElements = 6 * 64 * kCacheTokenCount;
constexpr size_t kCacheCount = 8;
constexpr float kMaskNegative = -10000.0F;

struct Sample {
  std::string id;
  fs::path mel_path;
  double duration_seconds;
};

struct DecodeConfig {
  std::vector<int> initial_tokens;
  std::unordered_set<int> suppressed;
  std::unordered_set<int> suppress_first;
  int eot;
  size_t max_generated_tokens;
};

struct DecodeResult {
  std::vector<int> tokens;
  double encoder_ms = 0.0;
  std::vector<double> decoder_times_ms;
  double first_token_ms = 0.0;
  bool reached_eot = false;
};

// 检查 Runtime 返回码并转换为异常.
void CheckRuntime(int code, const std::string& operation) {
  if (code != 0) {
    throw std::runtime_error(operation + " 失败,code=" +
                             std::to_string(code));
  }
}

// 将 FP32 转为 IEEE 754 binary16 位模式.
uint16_t FloatToHalf(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint32_t sign = (bits >> 16U) & 0x8000U;
  int32_t exponent = static_cast<int32_t>((bits >> 23U) & 0xffU) - 112;
  uint32_t mantissa = bits & 0x7fffffU;
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa = (mantissa | 0x800000U) >> (1 - exponent);
    return static_cast<uint16_t>(sign | ((mantissa + 0x1000U) >> 13U));
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00U);
  }
  return static_cast<uint16_t>(sign |
                               (static_cast<uint32_t>(exponent) << 10U) |
                               ((mantissa + 0x1000U) >> 13U));
}

// 将 IEEE 754 binary16 位模式转为 FP32.
float HalfToFloat(uint16_t value) {
  const uint32_t sign = (static_cast<uint32_t>(value & 0x8000U)) << 16U;
  uint32_t exponent = (value >> 10U) & 0x1fU;
  uint32_t mantissa = value & 0x03ffU;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int32_t shift = 0;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1U;
        ++shift;
      }
      mantissa &= 0x03ffU;
      bits = sign | (static_cast<uint32_t>(112 - shift) << 23U) |
             (mantissa << 13U);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000U | (mantissa << 13U);
  } else {
    bits = sign | ((exponent + 112) << 23U) | (mantissa << 13U);
  }
  return std::bit_cast<float>(bits);
}

// 读取固定长度二进制文件.
std::vector<uint8_t> ReadBinary(const fs::path& path, size_t expected_bytes) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("无法打开 Mel: " + path.string());
  }
  const size_t bytes = static_cast<size_t>(input.tellg());
  if (bytes != expected_bytes) {
    throw std::runtime_error("Mel 长度不匹配: " + path.string());
  }
  std::vector<uint8_t> data(bytes);
  input.seekg(0);
  input.read(reinterpret_cast<char*>(data.data()),
             static_cast<std::streamsize>(bytes));
  return data;
}

// 解析逗号分隔的 Token ID.
std::vector<int> ParseTokens(const std::string& value) {
  std::vector<int> tokens;
  size_t begin = 0;
  while (begin < value.size()) {
    const size_t end = value.find(',', begin);
    tokens.push_back(std::stoi(value.substr(begin, end - begin)));
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return tokens;
}

// 读取板端解码配置.
DecodeConfig ReadConfig(const fs::path& path) {
  std::ifstream input(path);
  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    const size_t separator = line.find('=');
    if (separator != std::string::npos) {
      values[line.substr(0, separator)] = line.substr(separator + 1);
    }
  }
  DecodeConfig config;
  config.initial_tokens = ParseTokens(values.at("initial"));
  const auto suppressed = ParseTokens(values.at("suppress"));
  config.suppressed.insert(suppressed.begin(), suppressed.end());
  const auto suppress_first = ParseTokens(values.at("suppress_first"));
  config.suppress_first.insert(suppress_first.begin(), suppress_first.end());
  config.eot = std::stoi(values.at("eot"));
  config.max_generated_tokens =
      static_cast<size_t>(std::stoul(values.at("max_generated_tokens")));
  if (config.initial_tokens.empty() ||
      config.initial_tokens.size() + config.max_generated_tokens >
          kCacheTokenCount) {
    throw std::runtime_error("解码 Token 数超过固定 Cache 上限.");
  }
  return config;
}

// 读取 sample_id、Mel 路径和音频时长 TSV.
std::vector<Sample> ReadManifest(const fs::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("无法打开板端清单: " + path.string());
  }
  std::vector<Sample> samples;
  std::string line;
  while (std::getline(input, line)) {
    const size_t first = line.find('\t');
    const size_t second = line.find('\t', first + 1);
    if (first == std::string::npos || second == std::string::npos) {
      throw std::runtime_error("板端清单格式错误: " + line);
    }
    samples.push_back({line.substr(0, first),
                       line.substr(first + 1, second - first - 1),
                       std::stod(line.substr(second + 1))});
  }
  return samples;
}

// 从已有 JSONL 提取成功样例 ID,用于断点续跑.
std::unordered_set<std::string> ReadCompleted(const fs::path& path) {
  std::unordered_set<std::string> completed;
  if (!fs::is_regular_file(path)) {
    return completed;
  }
  std::ifstream input(path);
  std::string line;
  constexpr char kPrefix[] = "\"sample_id\":\"";
  while (std::getline(input, line)) {
    if (line.find("\"status\":\"ok\"") == std::string::npos) {
      continue;
    }
    const size_t begin = line.find(kPrefix);
    if (begin != std::string::npos) {
      const size_t value_begin = begin + sizeof(kPrefix) - 1;
      const size_t end = line.find('"', value_begin);
      completed.insert(line.substr(value_begin, end - value_begin));
    }
  }
  return completed;
}

class NeuronModel {
 public:
  // 创建 Runtime、加载 DLA 并分配原生对齐缓冲区.
  NeuronModel(const fs::path& model_path, size_t expected_inputs,
              size_t expected_outputs) {
    const auto start = std::chrono::steady_clock::now();
    EnvOptions options{};
    options.deviceKind = kEnvOptHardware;
    options.MDLACoreOption = Auto;
    options.CPUThreadNum = 1;
    options.suppressInputConversion = true;
    options.suppressOutputConversion = true;
    CheckRuntime(NeuronRuntime_create(&options, &runtime_),
                 "NeuronRuntime_create");
    CheckRuntime(NeuronRuntime_loadNetworkFromFile(runtime_,
                                                   model_path.c_str()),
                 "NeuronRuntime_loadNetworkFromFile");
    Allocate(expected_inputs, expected_outputs);
    const auto end = std::chrono::steady_clock::now();
    load_ms_ = std::chrono::duration<double, std::milli>(end - start).count();
  }

  NeuronModel(const NeuronModel&) = delete;
  NeuronModel& operator=(const NeuronModel&) = delete;

  // 释放 Runtime 资源.
  ~NeuronModel() {
    if (runtime_ != nullptr) {
      NeuronRuntime_release(runtime_);
    }
  }

  // 返回模型加载毫秒数.
  double load_ms() const { return load_ms_; }

  // 返回可写输入缓冲区.
  std::vector<uint8_t>& input(size_t index) { return inputs_.at(index); }

  // 返回只读输出缓冲区.
  const std::vector<uint8_t>& output(size_t index) const {
    return outputs_.at(index);
  }

  // 执行一次 NPU 推理并返回墙钟毫秒数.
  double Infer() {
    const BufferAttribute attribute{NON_ION_FD};
    for (size_t index = 0; index < inputs_.size(); ++index) {
      CheckRuntime(NeuronRuntime_setInput(runtime_, index,
                                          inputs_[index].data(),
                                          inputs_[index].size(), attribute),
                   "NeuronRuntime_setInput");
    }
    for (size_t index = 0; index < outputs_.size(); ++index) {
      CheckRuntime(NeuronRuntime_setOutput(runtime_, index,
                                           outputs_[index].data(),
                                           outputs_[index].size(), attribute),
                   "NeuronRuntime_setOutput");
    }
    const auto start = std::chrono::steady_clock::now();
    CheckRuntime(NeuronRuntime_inference(runtime_), "NeuronRuntime_inference");
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

 private:
  // 校验 I/O 数量并读取每个原生缓冲区字节数.
  void Allocate(size_t expected_inputs, size_t expected_outputs) {
    size_t input_count = 0;
    size_t output_count = 0;
    CheckRuntime(NeuronRuntime_getInputNumber(runtime_, &input_count),
                 "NeuronRuntime_getInputNumber");
    CheckRuntime(NeuronRuntime_getOutputNumber(runtime_, &output_count),
                 "NeuronRuntime_getOutputNumber");
    if (input_count != expected_inputs || output_count != expected_outputs) {
      throw std::runtime_error("DLA I/O 数量不匹配.");
    }
    inputs_.resize(input_count);
    outputs_.resize(output_count);
    for (size_t index = 0; index < input_count; ++index) {
      size_t bytes = 0;
      CheckRuntime(NeuronRuntime_getInputPaddedSize(runtime_, index, &bytes),
                   "NeuronRuntime_getInputPaddedSize");
      inputs_[index].resize(bytes);
    }
    for (size_t index = 0; index < output_count; ++index) {
      size_t bytes = 0;
      CheckRuntime(NeuronRuntime_getOutputPaddedSize(runtime_, index, &bytes),
                   "NeuronRuntime_getOutputPaddedSize");
      outputs_[index].resize(bytes);
    }
  }

  void* runtime_ = nullptr;
  double load_ms_ = 0.0;
  std::vector<std::vector<uint8_t>> inputs_;
  std::vector<std::vector<uint8_t>> outputs_;
};

// 清零 FP16 缓冲区并设置一个 One-Hot 位置.
void SetOneHot(std::vector<uint8_t>* buffer, size_t index) {
  std::fill(buffer->begin(), buffer->end(), 0);
  if ((index + 1) * sizeof(uint16_t) > buffer->size()) {
    throw std::out_of_range("One-Hot 索引越界.");
  }
  reinterpret_cast<uint16_t*>(buffer->data())[index] = FloatToHalf(1.0F);
}

// 填充固定长度因果 Attention Mask.
void SetAttentionMask(std::vector<uint8_t>* buffer, size_t position) {
  auto* values = reinterpret_cast<uint16_t*>(buffer->data());
  for (size_t index = 0; index < kCacheTokenCount; ++index) {
    values[index] = FloatToHalf(index <= position ? 0.0F : kMaskNegative);
  }
}

// 应用 Token 过滤规则并返回最大 Logit Token.
int SelectToken(const std::vector<uint8_t>& logits,
                const std::unordered_set<int>& suppressed) {
  const auto* values = reinterpret_cast<const uint16_t*>(logits.data());
  int best_token = -1;
  float best_logit = -std::numeric_limits<float>::infinity();
  for (size_t index = 0; index < kVocabularySize; ++index) {
    if (!suppressed.contains(static_cast<int>(index))) {
      const float logit = HalfToFloat(values[index]);
      if (logit > best_logit) {
        best_logit = logit;
        best_token = static_cast<int>(index);
      }
    }
  }
  return best_token;
}

class WhisperRuntime {
 public:
  // 一次性加载 Encoder 与 Decoder DLA.
  WhisperRuntime(const fs::path& encoder_path, const fs::path& decoder_path,
                 DecodeConfig config)
      : encoder_(encoder_path, 1, 1),
        decoder_(decoder_path, 13, 9),
        config_(std::move(config)) {
    if (encoder_.input(0).size() != 80 * 3000 * sizeof(uint16_t) ||
        encoder_.output(0).size() !=
            kAudioFeatureElements * sizeof(uint16_t) ||
        decoder_.input(0).size() !=
            kPaddedVocabularySize * sizeof(uint16_t)) {
      throw std::runtime_error("DLA 原生 I/O 长度不匹配.");
    }
  }

  // 返回双 DLA 加载总耗时.
  double load_ms() const { return encoder_.load_ms() + decoder_.load_ms(); }

  // 清空 Cache 后执行一条 Mel 的完整 Greedy Search.
  DecodeResult Decode(const fs::path& mel_path) {
    DecodeResult result;
    encoder_.input(0) = ReadBinary(mel_path, 80 * 3000 * sizeof(uint16_t));
    result.encoder_ms = encoder_.Infer();
    decoder_.input(1) = encoder_.output(0);
    for (size_t index = 0; index < kCacheCount; ++index) {
      std::fill(decoder_.input(5 + index).begin(),
                decoder_.input(5 + index).end(), 0);
    }

    int current_token = config_.initial_tokens.front();
    size_t prompt_index = 0;
    size_t position = 0;
    while (position < kCacheTokenCount &&
           result.tokens.size() < config_.max_generated_tokens) {
      SetOneHot(&decoder_.input(0), static_cast<size_t>(current_token));
      SetOneHot(&decoder_.input(2), position);
      SetOneHot(&decoder_.input(3), position);
      SetAttentionMask(&decoder_.input(4), position);
      result.decoder_times_ms.push_back(decoder_.Infer());
      for (size_t index = 0; index < kCacheCount; ++index) {
        decoder_.input(5 + index) = decoder_.output(1 + index);
      }
      if (prompt_index + 1 < config_.initial_tokens.size()) {
        current_token = config_.initial_tokens[++prompt_index];
        ++position;
        continue;
      }
      std::unordered_set<int> suppressed = config_.suppressed;
      if (result.tokens.empty()) {
        suppressed.insert(config_.suppress_first.begin(),
                          config_.suppress_first.end());
        result.first_token_ms = result.encoder_ms + std::accumulate(
            result.decoder_times_ms.begin(), result.decoder_times_ms.end(),
            0.0);
      }
      const int next_token = SelectToken(decoder_.output(0), suppressed);
      if (next_token == config_.eot) {
        result.reached_eot = true;
        break;
      }
      result.tokens.push_back(next_token);
      current_token = next_token;
      ++position;
    }
    return result;
  }

 private:
  NeuronModel encoder_;
  NeuronModel decoder_;
  DecodeConfig config_;
};

// 将成功结果写为单行 JSON.
void WriteResult(std::ofstream* output, const Sample& sample,
                 const DecodeResult& result) {
  const double decoder_total = std::accumulate(
      result.decoder_times_ms.begin(), result.decoder_times_ms.end(), 0.0);
  const double npu_total = result.encoder_ms + decoder_total;
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  *output << "{\"sample_id\":\"" << sample.id
          << "\",\"status\":\"ok\",\"duration_seconds\":"
          << sample.duration_seconds << ",\"tokens\":[";
  for (size_t index = 0; index < result.tokens.size(); ++index) {
    if (index != 0) {
      *output << ',';
    }
    *output << result.tokens[index];
  }
  *output << "],\"reached_eot\":"
          << (result.reached_eot ? "true" : "false")
          << ",\"encoder_ms\":" << result.encoder_ms
          << ",\"first_token_ms\":" << result.first_token_ms
          << ",\"decoder_calls\":" << result.decoder_times_ms.size()
          << ",\"decoder_total_ms\":" << decoder_total
          << ",\"npu_total_ms\":" << npu_total
          << ",\"npu_rtf\":"
          << npu_total / (sample.duration_seconds * 1000.0)
          << ",\"tokens_per_second\":"
          << (npu_total > 0.0 ? result.tokens.size() * 1000.0 / npu_total
                              : 0.0)
          << ",\"peak_rss_kb\":" << usage.ru_maxrss << "}\n";
  output->flush();
}

// 批量运行、断点续跑并隔离单样例错误.
int RunBatch(const fs::path& encoder, const fs::path& decoder,
             const fs::path& manifest, const fs::path& config_path,
             const fs::path& output_path) {
  const auto samples = ReadManifest(manifest);
  const auto completed = ReadCompleted(output_path);
  WhisperRuntime runtime(encoder, decoder, ReadConfig(config_path));
  std::ofstream output(output_path, std::ios::app);
  size_t processed = 0;
  size_t failures = 0;
  std::cout << "[INFO] samples=" << samples.size()
            << " completed=" << completed.size()
            << " model_load_ms=" << runtime.load_ms() << '\n';
  for (const auto& sample : samples) {
    if (completed.contains(sample.id)) {
      continue;
    }
    try {
      WriteResult(&output, sample, runtime.Decode(sample.mel_path));
    } catch (const std::exception& error) {
      output << "{\"sample_id\":\"" << sample.id
             << "\",\"status\":\"error\",\"error\":\""
             << error.what() << "\"}\n";
      output.flush();
      ++failures;
    }
    ++processed;
    if (processed % 25 == 0 || processed + completed.size() == samples.size()) {
      std::cout << "[PROGRESS] " << processed + completed.size() << '/'
                << samples.size() << " failures=" << failures << '\n';
    }
  }
  std::cout << "[OK] processed=" << processed << " failures=" << failures
            << '\n';
  return failures == 0 ? 0 : 1;
}

// 解析固定位置参数并执行批量评测.
int main(int argc, char** argv) {
  try {
    if (argc != 6) {
      throw std::invalid_argument(
          "用法: whisper_board_eval <encoder.dla> <decoder.dla> "
          "<board_manifest.tsv> <decode_config.txt> <output.jsonl>");
    }
    return RunBatch(argv[1], argv[2], argv[3], argv[4], argv[5]);
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << '\n';
    return 2;
  }
}
