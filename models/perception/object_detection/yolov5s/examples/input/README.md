# 输入样例

`public/` 固定保存三张项目生成的 CC0-1.0 图片,覆盖室外交通、室内家具和公园人物场景.
执行 `bash deploy/generate_examples.sh` 会交叉编译推理器,再把三张图送到 Genio 720,
逐张显示推理进度.图片纳入 Git,不再复制 COCO val2017 原图.
