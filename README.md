# dify-plugin-txt2img
调用第三方API实现文生图, 兼容Flux, RecraftV3, OpenAI DALL·E 3

## Dify 使用指南

### 实际使用参考图
![Dify 文生图插件使用界面示例](files/guide1.png)

### 示例配置
我们提供完整的应用配置样例文件 [example-txt2img.yml](files/example-txt2img.yml)，该文件包含：
- 完整的LLM调用链配置
- 图片尺寸参数传递逻辑
- Flux模型的调用参数设置
- 输出结果处理流程

使用时请：
1. 将yml文件导入Dify应用编排界面
2. 确保已安装对应版本的txt2img插件
3. 根据实际需求调整图片尺寸参数列表
