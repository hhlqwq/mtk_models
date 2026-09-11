# 输入样例

放置公开授权的人体姿态图片和对应人体框信息,并记录来源.

`DEMO_COUNT=5 bash deploy/deploy_board.sh` 默认从 COCO val2017 选择五个独立 person
框,复制原图并生成量化输入到 `generated/`.原图默认不提交 Git.
