# Car Plate Segmentation

> 本项目使用opencv进行对车牌的预处理以及分割操作，然后使用tensorrt部署省份字符以及字母数字字符的分类模型，进而
> 实现对车牌各个字符的分割以及识别。界面使用qt进行实现。

项目依赖：

- opencv

- tensorrt

## 目录结构
```python
├── assets # 存放测试图片 
│   ├── a.jpg
│   ├── b.jpg
│   ├── blank.png
│   ├── image.png
│   ├── q.jpg
│   ├── wai.png
│   └── yy.png
├── charseg # 车牌字符分割api
│   ├── CMakeLists.txt
│   ├── seg.cpp
│   └── seg.hpp
├── CMakeLists.txt
├── Infer # 车牌识别api
│   ├── CMakeLists.txt
│   ├── FindTensorRT.cmake
│   ├── Format_Print.hpp
│   ├── TRTFrame.cpp
│   └── TRTFrame.hpp
├── main.cpp # 主函数
├── model # 存放模型文件
│   ├── numalpha.onnx
│   ├── plate.engine
│   ├── plate.onnx
│   └── seq.txt
├── MvCamera # 相机api
│   ├── CMakeLists.txt
│   ├── MvCamera.cpp
│   └── MvCamera.h
├── qui.cpp # qt界面
├── qui.hpp # qt界面头文件
└── README.md # 说明文档

```