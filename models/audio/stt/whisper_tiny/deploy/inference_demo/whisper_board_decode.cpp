// 在 Genio 720 上执行 Whisper-Tiny Encoder 和逐 Token Decoder.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
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
constexpr size_t kCacheElements = 6 * 1 * 64 * kCacheTokenCount;
constexpr size_t kCacheCount = 8;
constexpr float kMaskNegative = -10000.0F;

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
  int32_t exponent = static_cast<int32_t>((bits >> 23U) & 0xffU) - 127 + 15;
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
      bits = sign | (static_cast<uint32_t>(127 - 15 - shift) << 23U) |
             (mantissa << 13U);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000U | (mantissa << 13U);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23U) | (mantissa << 13U);
  }
  return std::bit_cast<float>(bits);
}

// 读取完整二进制文件并校验长度.
std::vector<uint8_t> ReadBinary(const fs::path& path, size_t expected_bytes) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("无法打开文件: " + path.string());
  }
  const size_t bytes = static_cast<size_t>(input.tellg());
  if (bytes != expected_bytes) {
    throw std::runtime_error("文件长度不匹配: " + path.string() + ",actual=" +
                             std::to_string(bytes) + ",expected=" +
                             std::to_string(expected_bytes));
  }
  std::vector<uint8_t> data(bytes);
  input.seekg(0);
  input.read(reinterpret_cast<char*>(data.data()),
             static_cast<std::streamsize>(data.size()));
  return data;
}

// 解析逗号分隔的 Token ID.
std::vector<int> ParseTokenList(const std::string& value) {
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

// 读取 key=value 格式的解码配置.
std::unordered_map<std::string, std::string> ReadConfig(const fs::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("无法打开配置: " + path.string());
  }
  std::unordered_map<std::string, std::string> config;
  std::string line;
  while (std::getline(input, line)) {
    const size_t separator = line.find('=');
    if (separator != std::string::npos) {
      config[line.substr(0, separator)] = line.substr(separator + 1);
    }
  }
  return config;
}

class NeuronModel {
 public:
  // 创建硬件 Runtime、加载 DLA 并分配对齐 I/O 缓冲区.
  NeuronModel(const fs::path& model_path, size_t expected_inputs,
              size_t expected_outputs) {
    EnvOptions options{};
    options.deviceKind = kEnvOptHardware;
    options.MDLACoreOption = Auto;
    options.CPUThreadNum = 1;
    options.suppressInputConversion = true;
    options.suppressOutputConversion = true;
    CheckRuntime(NeuronRuntime_create(&options, &runtime_),
                 "NeuronRuntime_create");
    try {
      CheckRuntime(NeuronRuntime_loadNetworkFromFile(runtime_,
                                                     model_path.c_str()),
                   "NeuronRuntime_loadNetworkFromFile");
      AllocateBuffers(expected_inputs, expected_outputs);
    } catch (...) {
      NeuronRuntime_release(runtime_);
      runtime_ = nullptr;
      throw;
    }
  }

  NeuronModel(const NeuronModel&) = delete;
  NeuronModel& operator=(const NeuronModel&) = delete;

  // 释放 Runtime 资源.
  ~NeuronModel() {
    if (runtime_ != nullptr) {
      NeuronRuntime_release(runtime_);
    }
  }

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
                   "NeuronRuntime_setInput(" + std::to_string(index) + ")");
    }
    for (size_t index = 0; index < outputs_.size(); ++index) {
      CheckRuntime(NeuronRuntime_setOutput(runtime_, index,
                                           outputs_[index].data(),
                                           outputs_[index].size(), attribute),
                   "NeuronRuntime_setOutput(" + std::to_string(index) + ")");
    }
    const auto start = std::chrono::steady_clock::now();
    CheckRuntime(NeuronRuntime_inference(runtime_), "NeuronRuntime_inference");
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

 private:
  // 读取 Runtime I/O 数量与硬件对齐字节数.
  void AllocateBuffers(size_t expected_inputs, size_t expected_outputs) {
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
  std::vector<std::vector<uint8_t>> inputs_;
  std::vector<std::vector<uint8_t>> outputs_;
};

// 将一个 FP16 缓冲区清零并设置指定位置为 1.
void SetOneHot(std::vector<uint8_t>* buffer, size_t index) {
  std::fill(buffer->begin(), buffer->end(), 0);
  if ((index + 1) * sizeof(uint16_t) > buffer->size()) {
    throw std::out_of_range("One-Hot 索引越界.");
  }
  auto* values = reinterpret_cast<uint16_t*>(buffer->data());
  values[index] = FloatToHalf(1.0F);
}

// 填充因果 Attention Mask.
void SetAttentionMask(std::vector<uint8_t>* buffer, size_t position) {
  if (buffer->size() != kCacheTokenCount * sizeof(uint16_t)) {
    throw std::runtime_error("Attention Mask 长度不匹配.");
  }
  auto* values = reinterpret_cast<uint16_t*>(buffer->data());
  for (size_t index = 0; index < kCacheTokenCount; ++index) {
    values[index] = FloatToHalf(index <= position ? 0.0F : kMaskNegative);
  }
}

// 在 FP16 Logits 中应用过滤并返回最大值 Token.
int SelectToken(const std::vector<uint8_t>& logits,
                const std::unordered_set<int>& suppressed) {
  if (logits.size() != kVocabularySize * sizeof(uint16_t)) {
    throw std::runtime_error("Logits 长度不匹配.");
  }
  const auto* values = reinterpret_cast<const uint16_t*>(logits.data());
  int best_token = -1;
  float best_logit = -std::numeric_limits<float>::infinity();
  for (size_t index = 0; index < kVocabularySize; ++index) {
    if (suppressed.contains(static_cast<int>(index))) {
      continue;
    }
    const float logit = HalfToFloat(values[index]);
    if (logit > best_logit) {
      best_logit = logit;
      best_token = static_cast<int>(index);
    }
  }
  return best_token;
}

// 将 Token ID 每行一个写入文件.
void WriteTokens(const fs::path& path, const std::vector<int>& tokens) {
  std::ofstream output(path);
  for (int token : tokens) {
    output << token << '\n';
  }
}

// 执行双 DLA 自回归解码并写出 Token 与性能数据.
void RunDecode(const fs::path& encoder_path, const fs::path& decoder_path,
               const fs::path& mel_path, const fs::path& config_path,
               const fs::path& output_dir) {
  const auto config = ReadConfig(config_path);
  const std::vector<int> initial_tokens = ParseTokenList(config.at("initial"));
  const std::vector<int> suppress_list = ParseTokenList(config.at("suppress"));
  const std::vector<int> suppress_first_list =
      ParseTokenList(config.at("suppress_first"));
  const int eot = std::stoi(config.at("eot"));
  const size_t max_generated_tokens =
      static_cast<size_t>(std::stoul(config.at("max_generated_tokens")));
  if (initial_tokens.empty()) {
    throw std::runtime_error("初始 Token 不能为空.");
  }

  std::cout << "[1/4] 加载 Encoder DLA 并执行 30 秒 Mel.\n";
  NeuronModel encoder(encoder_path, 1, 1);
  encoder.input(0) = ReadBinary(mel_path, 80 * 3000 * sizeof(uint16_t));
  const double encoder_ms = encoder.Infer();
  const std::vector<uint8_t> audio_features = encoder.output(0);
  if (audio_features.size() != kAudioFeatureElements * sizeof(uint16_t)) {
    throw std::runtime_error("Encoder 输出长度不匹配.");
  }

  std::cout << "[2/4] 加载 Decoder DLA 并初始化 KV Cache.\n";
  NeuronModel decoder(decoder_path, 13, 9);
  if (decoder.input(0).size() != kPaddedVocabularySize * sizeof(uint16_t) ||
      decoder.input(1).size() != audio_features.size() ||
      decoder.input(2).size() != kPositionCount * sizeof(uint16_t)) {
    throw std::runtime_error("Decoder 固定输入长度不匹配.");
  }
  decoder.input(1) = audio_features;
  for (size_t index = 0; index < kCacheCount; ++index) {
    if (decoder.input(5 + index).size() !=
        kCacheElements * sizeof(uint16_t)) {
      throw std::runtime_error("Decoder Cache 输入长度不匹配.");
    }
    std::fill(decoder.input(5 + index).begin(),
              decoder.input(5 + index).end(), 0);
  }

  std::unordered_set<int> suppressed(suppress_list.begin(),
                                     suppress_list.end());
  const std::unordered_set<int> suppress_first(suppress_first_list.begin(),
                                               suppress_first_list.end());
  std::vector<int> generated_tokens;
  std::vector<double> decoder_times;
  int current_token = initial_tokens.front();
  size_t prompt_index = 0;
  size_t position = 0;

  std::cout << "[3/4] 执行固定 Cache Greedy Search.\n";
  while (position < kCacheTokenCount &&
         generated_tokens.size() < max_generated_tokens) {
    SetOneHot(&decoder.input(0), static_cast<size_t>(current_token));
    SetOneHot(&decoder.input(2), position);
    SetOneHot(&decoder.input(3), position);
    SetAttentionMask(&decoder.input(4), position);
    const double decoder_ms = decoder.Infer();
    decoder_times.push_back(decoder_ms);
    for (size_t index = 0; index < kCacheCount; ++index) {
      decoder.input(5 + index) = decoder.output(1 + index);
    }

    if (prompt_index + 1 < initial_tokens.size()) {
      ++prompt_index;
      current_token = initial_tokens[prompt_index];
      ++position;
      continue;
    }
    std::unordered_set<int> step_suppressed = suppressed;
    if (generated_tokens.empty()) {
      step_suppressed.insert(suppress_first.begin(), suppress_first.end());
    }
    const int next_token = SelectToken(decoder.output(0), step_suppressed);
    if (next_token == eot) {
      break;
    }
    generated_tokens.push_back(next_token);
    current_token = next_token;
    ++position;
    std::cout << "[TOKEN] " << generated_tokens.size() << '/'
              << max_generated_tokens << " id=" << next_token
              << " decoder_ms=" << decoder_ms << '\n';
  }

  std::cout << "[4/4] 写出 Token 和性能记录.\n";
  fs::create_directories(output_dir);
  WriteTokens(output_dir / "board_tokens.txt", generated_tokens);
  const double decoder_total_ms =
      std::accumulate(decoder_times.begin(), decoder_times.end(), 0.0);
  const double decoder_mean_ms = decoder_times.empty()
                                     ? 0.0
                                     : decoder_total_ms / decoder_times.size();
  std::ofstream performance(output_dir / "board_performance.json");
  performance << "{\n"
              << "  \"encoder_ms\": " << encoder_ms << ",\n"
              << "  \"decoder_calls\": " << decoder_times.size() << ",\n"
              << "  \"decoder_total_ms\": " << decoder_total_ms << ",\n"
              << "  \"decoder_mean_ms\": " << decoder_mean_ms << ",\n"
              << "  \"generated_tokens\": " << generated_tokens.size()
              << "\n}\n";
  std::cout << "[OK] encoder_ms=" << encoder_ms
            << " decoder_mean_ms=" << decoder_mean_ms
            << " generated_tokens=" << generated_tokens.size() << '\n';
}

// 解析固定位置参数并执行板端解码.
int main(int argc, char** argv) {
  try {
    if (argc != 7) {
      throw std::invalid_argument(
          "用法: whisper_board_decode <encoder.dla> <decoder.dla> "
          "<mel_fp16.bin> <decode_config.txt> <output_dir> <run_id>");
    }
    const fs::path output_dir = fs::path(argv[5]) / argv[6];
    RunDecode(argv[1], argv[2], argv[3], argv[4], output_dir);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << '\n';
    return 1;
  }
}
