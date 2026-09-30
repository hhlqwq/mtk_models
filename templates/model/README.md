# MODEL_NAME

本目录由 `tools/create_model.py` 生成.完成开源上游来源锁定、原始框架推理、自行导出、转换、
板端验证、精度和性能报告后,才能将注册表状态改为 `complete`.交付参考页面不能替代模型来源.
仅保留 `model_card.md` 记录模型信息,并将 `deploy/run.sh` 实现为编译主机与开发板共用的单一入口.
