# Qwen2.5-7B

## 1. 创建软链接

```bash
mkdir -p /usr/share/llm

ln -sfn \
/root/hailong.he/MTK_G720_DLA/02_Generative/LLM/qwen2.5-7b \
/usr/share/llm/qwen2.5-7b
```

## 2. 开启 Performance Mode

```bash
cd /root/hailong.he/MTK_G720_DLA/02_Generative/LLM/qwen2.5-7b
bash scripts/set_performance.sh
```

## 3. 运行 10 Prompts Benchmark

```bash
llm_cmdline_tool \
  scripts/config-yocto_qwen2.5_7b_instruct.yaml \
  -i scripts/prompts/sample_prompt-introduction.txt \
  -i scripts/prompts/p01.txt \
  -i scripts/prompts/p02.txt \
  -i scripts/prompts/p03.txt \
  -i scripts/prompts/p04.txt \
  -i scripts/prompts/p05.txt \
  -i scripts/prompts/p06.txt \
  -i scripts/prompts/p07.txt \
  -i scripts/prompts/p08.txt \
  -i scripts/prompts/p09.txt \
  --preformatter QwenNoInput \
  -m 512 \
  2>&1 | tee benchmark_qwen2.5-7b.log
```

## 4. 查看平均性能

```bash
grep -A3 "Average Performance" benchmark_qwen2.5-7b.log
```
