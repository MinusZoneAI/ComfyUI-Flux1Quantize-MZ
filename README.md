![image](./examples/workflow1.png)

## Flux1的非官方量化模型

量化工具来自 https://github.com/casper-hansen/AutoAWQ 和 https://github.com/IST-DASLab/marlin

仅适用sm_80以上的显卡(30系列及以上)

 
需要先安装marlin依赖
```shell
pip install git+https://github.com/IST-DASLab/marlin
```
 
模型下载地址:

https://www.modelscope.cn/models/wailovet/flux1-quantize/resolve/master/flux1-unet-marlin-int4.safetensors