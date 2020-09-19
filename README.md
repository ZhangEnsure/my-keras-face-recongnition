## Face-Recognition：人脸识别算法在Keras当中的实现
---

### 目录
1. [简单介绍](#简单介绍)
2. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [使用方法 Usage](#使用方法)
5. [效果 Performance](#Reference)

### 简单介绍
这是一个基于mtcnn和facenet的人脸识别模型，可实现在线人脸识别

### 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

### 文件下载
训练所需的facenet_keras.h5可以在Release里面下载。  
也可以去百度网盘下载  
链接: https://pan.baidu.com/s/1A9jCJa_sQ4D3ejelgXX2RQ 提取码: tkhg  
### 使用方法
1、先将整个仓库download下来。  
2、下载完之后解压，同时下载facenet_keras.h5文件。  
3、将facenet_keras.h5放入model_data中。  
4、将自己想要识别的人脸放入到face_dataset中。  
5、运行face_recognize.py即可。  
6、align.py可以查看人脸对齐的效果。  
### 效果
face_recognize.py的运行结果：  
![result](/result/result.png))  
