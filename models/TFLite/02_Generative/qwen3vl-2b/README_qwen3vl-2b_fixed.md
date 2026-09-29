# Qwen3VL-2B

## 1. 创建软链接

```bash
mkdir -p /usr/share/llm

ln -sfn \
/root/hailong.he/MTK_G720_DLA/02_Generative/VLM/qwen3vl-2b \
/usr/share/llm/qwen3vl-2b
```

## 2. 开启 Performance Mode

```bash
cd /root/hailong.he/MTK_G720_DLA/02_Generative/VLM/qwen3vl-2b
bash scripts/set_performance.sh
```

## 3. 运行图文推理

```bash
mllm_llava \
  scripts/config-yocto_qwen3vl-2b.yaml \
  --preformatter Qwen3VLNoInput \
  -i scripts/sample_prompt.txt \
  -im scripts/demo.jpeg \
  -m 128 \
  2>&1 | tee benchmark_qwen3vl-2b.log
```

## 4. 查看性能结果

```bash
grep -E "Prompt Mode|Generative Mode|Average|Time taken" benchmark_qwen3vl-2b.log
```
