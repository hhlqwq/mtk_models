# Project Instructions

## 项目说明

这是一个基于 MTK G720 and G5100 的 模型移植项目，直接对标https://huggingface.co/qualcomm/models。

## 开发环境

### 编译环境

编译在 Ubuntu 环境执行。SSH:192.168.0.89 
登录方式为  ssh ubuntu89
工作目录：/data/users/hailong.he/github/mtk_models

## 测试环境

测试必须在开发板执行。SoC: MT8189 / Genio 720   SSH:192.168.0.92
账号名root无密码
测试目录：/root/hailong.he
如果板端缺少什么资源或者内部的系统资源是旧版本导致的可以升级开发板的系统或者文件(允许官网最新的系统移植)