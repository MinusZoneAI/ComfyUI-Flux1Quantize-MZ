![image](./examples/workflow1.png)

## Flux1的非官方量化模型

量化工具来自 https://github.com/casper-hansen/AutoAWQ 和 https://github.com/IST-DASLab/marlin

仅适用sm_80以上的显卡(30系列及以上)

 
需要先安装marlin依赖
```shell
pip install git+https://github.com/IST-DASLab/marlin
```
 
模型下载地址:

https://www.modelscope.cn/models/wailovet/flux1-quantize/resolve/master/flux1-dev-unet-marlin-int4.safetensors

在examples文件夹可以找到一个简单的使用示例


## The unofficial quantized model of Flux1

The quantization tools come from https://github.com/casper-hansen/AutoAWQ and https://github.com/IST-DASLab/marlin

Only suitable for GPUs above sm_80 (30 series and above)


You need to install the marlin dependency first
```shell
pip install git+https://github.com/IST-DASLab/marlin
```

Model download link:

https://huggingface.co/MinusZoneAI/flux1-quantize/resolve/main/flux1-dev-unet-marlin-int4.safetensors

You can find a simple example in the examples folder