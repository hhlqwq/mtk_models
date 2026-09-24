// 在 Genio 720 板端以 C++ 完成 LibriSpeech 清单、FLAC 解码和 Whisper Mel.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace fs = std::filesystem;

struct Sample {
  std::string id;
  fs::path audio;
  std::string reference;
};

// 解析数据集、静态滤波器和本次运行目录.
std::array<fs::path, 3> ParseArgs(int argc, char** argv) {
  std::array<fs::path, 3> values;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) throw std::invalid_argument("参数缺少值.");
    const std::string key = argv[index];
    if (key == "--dataset-root") values[0] = argv[index + 1];
    else if (key == "--filters") values[1] = argv[index + 1];
    else if (key == "--output-dir") values[2] = argv[index + 1];
    else throw std::invalid_argument("未知参数: " + key);
  }
  if (values[0].empty() || values[1].empty() || values[2].empty()) {
    throw std::invalid_argument("需要 --dataset-root、--filters、--output-dir.");
  }
  return values;
}

// 严格读取官方 LibriSpeech test-clean 转录与 FLAC 文件.
std::vector<Sample> ListSamples(const fs::path& root) {
  std::vector<fs::path> transcripts;
  for (const fs::directory_entry& entry : fs::recursive_directory_iterator(root)) {
    if (entry.is_regular_file() &&
        entry.path().filename().string().ends_with(".trans.txt")) {
      transcripts.push_back(entry.path());
    }
  }
  std::sort(transcripts.begin(), transcripts.end());
  std::vector<Sample> samples;
  for (const fs::path& transcript : transcripts) {
    std::ifstream input(transcript);
    if (!input) throw std::runtime_error("无法读取转录: " + transcript.string());
    std::string line;
    while (std::getline(input, line)) {
      const size_t separator = line.find(' ');
      if (separator == std::string::npos || separator == 0 ||
          separator + 1 >= line.size()) {
        throw std::runtime_error("转录行无效: " + transcript.string());
      }
      Sample sample{line.substr(0, separator), {}, line.substr(separator + 1)};
      sample.audio = transcript.parent_path() / (sample.id + ".flac");
      if (!fs::is_regular_file(sample.audio)) {
        throw std::runtime_error("缺少音频: " + sample.audio.string());
      }
      samples.push_back(std::move(sample));
    }
  }
  if (samples.size() != 2620) {
    throw std::runtime_error("LibriSpeech test-clean 必须恰好 2620 条.");
  }
  return samples;
}

// 为受控本地路径构造安全的单引号 shell 参数.
std::string ShellQuote(const fs::path& path) {
  std::string result = "'";
  for (const char value : path.string()) {
    if (value == '\'') result += "'\\''";
    else result.push_back(value);
  }
  return result + "'";
}

// 用系统 ffmpeg 解码为 16 kHz 单声道 PCM16,保持原始 FLAC 不变.
std::pair<std::vector<float>, size_t> DecodeAudio(const fs::path& path) {
  const std::string command =
      "ffmpeg -hide_banner -loglevel error -nostdin -i " + ShellQuote(path) +
      " -ar 16000 -ac 1 -f s16le -acodec pcm_s16le pipe:1";
  FILE* process = popen(command.c_str(), "r");
  if (process == nullptr) throw std::runtime_error("ffmpeg 启动失败.");
  std::vector<int16_t> pcm;
  std::array<int16_t, 8192> buffer{};
  while (true) {
    const size_t count = fread(buffer.data(), sizeof(int16_t), buffer.size(),
                               process);
    pcm.insert(pcm.end(), buffer.begin(), buffer.begin() + count);
    if (count < buffer.size()) {
      if (ferror(process)) {
        pclose(process);
        throw std::runtime_error("读取 ffmpeg 输出失败.");
      }
      break;
    }
  }
  if (pclose(process) != 0 || pcm.empty() || pcm.size() > 480000) {
    throw std::runtime_error("FLAC 解码失败或音频超过 30 秒: " + path.string());
  }
  std::vector<float> audio(480000, 0.0F);
  for (size_t index = 0; index < pcm.size(); ++index) {
    audio[index] = static_cast<float>(pcm[index]) / 32768.0F;
  }
  return {std::move(audio), pcm.size()};
}

// 将数组外索引按 PyTorch reflect padding 规则映射到源音频.
size_t ReflectIndex(int index, size_t length) {
  if (index < 0) return static_cast<size_t>(-index);
  if (index >= static_cast<int>(length)) {
    return static_cast<size_t>(2 * static_cast<int>(length) - 2 - index);
  }
  return static_cast<size_t>(index);
}

// 读取 80x201 的官方固定 Mel 滤波器矩阵.
cv::Mat LoadFilters(const fs::path& path) {
  if (fs::file_size(path) != 80 * 201 * sizeof(float)) {
    throw std::runtime_error("Mel 滤波器大小不匹配.");
  }
  cv::Mat filters(80, 201, CV_32F);
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(filters.data),
             static_cast<std::streamsize>(filters.total() * sizeof(float)));
  if (!input) throw std::runtime_error("读取 Mel 滤波器失败.");
  return filters;
}

// 复现 Whisper 400 点 Hann STFT、80 Mel 和 log10 归一化.
cv::Mat ComputeMel(const std::vector<float>& audio, const cv::Mat& filters) {
  cv::Mat power(201, 3000, CV_32F);
  cv::Mat windowed(1, 400, CV_32F);
  cv::Mat spectrum;
  std::array<float, 400> hann{};
  constexpr double kPi = 3.14159265358979323846;
  for (size_t index = 0; index < hann.size(); ++index) {
    hann[index] = static_cast<float>(0.5 - 0.5 *
        std::cos(2.0 * kPi * index / hann.size()));
  }
  for (int frame = 0; frame < 3000; ++frame) {
    float* values = windowed.ptr<float>();
    for (int index = 0; index < 400; ++index) {
      const int audio_index = frame * 160 + index - 200;
      values[index] = audio[ReflectIndex(audio_index, audio.size())] *
                      hann[index];
    }
    cv::dft(windowed, spectrum, cv::DFT_COMPLEX_OUTPUT);
    const auto* frequencies = spectrum.ptr<cv::Vec2f>();
    for (int frequency = 0; frequency < 201; ++frequency) {
      const float real = frequencies[frequency][0];
      const float imaginary = frequencies[frequency][1];
      power.at<float>(frequency, frame) = real * real + imaginary * imaginary;
    }
    if ((frame + 1) % 1000 == 0) {
      std::cout << "[MEL] " << frame + 1 << "/3000 帧.\r" << std::flush;
    }
  }
  cv::Mat mel;
  cv::gemm(filters, power, 1.0, cv::Mat(), 0.0, mel);
  float maximum = -std::numeric_limits<float>::infinity();
  for (int row = 0; row < mel.rows; ++row) {
    float* values = mel.ptr<float>(row);
    for (int column = 0; column < mel.cols; ++column) {
      values[column] = std::log10(std::max(values[column], 1e-10F));
      maximum = std::max(maximum, values[column]);
    }
  }
  for (int row = 0; row < mel.rows; ++row) {
    float* values = mel.ptr<float>(row);
    for (int column = 0; column < mel.cols; ++column) {
      values[column] = (std::max(values[column], maximum - 8.0F) + 4.0F) / 4.0F;
    }
  }
  cv::Mat half;
  mel.convertTo(half, CV_16F);
  return half;
}

// 对 JSONL 文本字段执行最低限度的标准转义.
std::string JsonEscape(const std::string& input) {
  std::string output;
  for (const unsigned char value : input) {
    if (value == '"' || value == '\\') output.push_back('\\');
    if (value < 0x20) {
      throw std::runtime_error("转录含不支持的控制字符.");
    }
    output.push_back(static_cast<char>(value));
  }
  return output;
}

// 输出模型输入和 WER 计算所需清单,每百条显示进度.
void Prepare(const fs::path& root, const fs::path& filters_path,
             const fs::path& output_dir) {
  const std::vector<Sample> samples = ListSamples(root);
  const cv::Mat filters = LoadFilters(filters_path);
  const fs::path mel_dir = output_dir / "mels";
  fs::create_directories(mel_dir);
  std::ofstream sources(output_dir / "source_manifest.jsonl");
  std::ofstream board(output_dir / "board_manifest.tsv");
  std::ofstream metrics(output_dir / "preprocess_metrics.jsonl");
  if (!sources || !board || !metrics) {
    throw std::runtime_error("无法写入板端音频清单.");
  }
  for (size_t index = 0; index < samples.size(); ++index) {
    const Sample& sample = samples[index];
    const auto decode_start = std::chrono::steady_clock::now();
    auto [audio, sample_count] = DecodeAudio(sample.audio);
    const double decode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - decode_start).count();
    const double duration = static_cast<double>(sample_count) / 16000.0;
    const auto mel_start = std::chrono::steady_clock::now();
    const cv::Mat mel = ComputeMel(audio, filters);
    const double mel_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - mel_start).count();
    const fs::path mel_path = mel_dir / (sample.id + ".bin");
    const auto write_start = std::chrono::steady_clock::now();
    std::ofstream mel_output(mel_path, std::ios::binary);
    mel_output.write(reinterpret_cast<const char*>(mel.data),
                     static_cast<std::streamsize>(mel.total() * mel.elemSize()));
    if (!mel_output) throw std::runtime_error("写入 Mel 失败: " + mel_path.string());
    mel_output.close();
    const double write_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - write_start).count();
    sources << "{\"sample_id\":\"" << JsonEscape(sample.id)
            << "\",\"audio_path\":\"" << JsonEscape(sample.audio.string())
            << "\",\"duration_seconds\":" << duration
            << ",\"reference_text\":\"" << JsonEscape(sample.reference)
            << "\"}\n";
    board << sample.id << '\t' << mel_path.string() << '\t' << duration << '\n';
    metrics << "{\"sample_id\":\"" << sample.id
            << "\",\"duration_seconds\":" << duration
            << ",\"audio_decode_ms\":" << decode_ms
            << ",\"log_mel_ms\":" << mel_ms
            << ",\"mel_write_ms\":" << write_ms
            << ",\"cache_hit\":false}\n";
    if ((index + 1) % 100 == 0 || index + 1 == samples.size()) {
      std::cout << "[PROGRESS] LibriSpeech " << index + 1 << '/'
                << samples.size() << " 条.\n";
    }
  }
  if (!sources || !board || !metrics) {
    throw std::runtime_error("音频清单写入失败.");
  }
}

// 运行完整输入准备并将异常作为非零退出码返回.
int main(int argc, char** argv) {
  try {
    const auto [root, filters, output] = ParseArgs(argc, argv);
    Prepare(root, filters, output);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << '\n';
    return 1;
  }
}
