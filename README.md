# 说明



一个基于qwen2-vl实现的简易VLM-Token丢失模型

- 直接丢失visual token的前N个token
- 可以用来作为和自己压缩方法相比的Lower-Bound





### TodoList

- [ ] 在LLava上也实现一下
- [ ] 配合huggingface eval，提供VLM指标测试脚本
- [ ] 实现random丢失