# 输入样例

`public/` 固定保存三张项目生成的 CC0-1.0 图片:金毛犬、校车和浓缩咖啡.
执行 `bash deploy/generate_examples.sh` 会按当前 TFLite 量化参数生成中间输入,并把三张图
逐张送到 Genio 720 NPU.公开原图纳入 Git;`generated/` 中间文件不提交.

正式 ImageNet 50,000 张评测仍使用外部数据集,不把这三张展示图片计入精度指标.
