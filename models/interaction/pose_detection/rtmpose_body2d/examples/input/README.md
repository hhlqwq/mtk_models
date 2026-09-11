# 输入样例

`public/` 固定保存三张项目生成的 CC0-1.0 单人全身图片和 `annotations.json` 人体框.
执行 `bash deploy/generate_examples.sh` 会生成量化输入,并把三张图逐张送到 Genio 720
NPU.公开原图和人体框纳入 Git;`generated/` 中间文件不提交.
