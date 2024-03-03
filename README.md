

# DAPart

此开源代码是论文《DAPart: An Online DRL-based Adaptive Partition Framework for DNN Models in Edge Computing》中的实际实验搭建的真实测试平台。

<!-- PROJECT SHIELDS -->

[//]: #
[//]: # ([![Contributors][contributors-shield]][contributors-url])

[//]: # ([![Forks][forks-shield]][forks-url])

[//]: # ([![Stargazers][stars-shield]][stars-url])

[//]: # ([![Issues][issues-shield]][issues-url])

[//]: # ([![MIT License][license-shield]][license-url])

[//]: # ([![LinkedIn][linkedin-shield]][linkedin-url])

<!-- PROJECT LOGO -->
<br />
<h3 align="center">DAPart</h3>
  <p align="center">
    <a href="https://github.com/Jma512/DAPart"><strong>探索本项目的文档 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Jma512/DAPart">查看Demo</a>
    ·
    <a href="https://github.com/Jma512/DAPart/blob/main/README.md">中文</a>
    ·
    <a href="https://github.com/Jma512/DAPart/blob/main/README_en.md">English</a>
  </p>




 本篇README.md面向开发者，

## 目录

- [上手指南](#上手指南)
  - [开发前的配置要求](#开发前的配置要求)
    - [用户端设备](#用户端设备)
    - [服务端设备](#服务端设备)
  - [安装步骤](#安装步骤)
- [文件目录说明](#文件目录说明)
- [部署和运行](#部署和运行)
- [贡献者](#贡献者)
- [版本控制](#版本控制)
- [作者](#作者)

### 上手指南
该开源代码分为用户端设备和服务端设备两部分，用户端设备采用的是Jetson Nano，服务端设备采用的是具有Linux系统的电脑。本论文中实验的设备如下表格展示。

| Hardware           | User Equipment Device<br>(Jetson Nano) | Edge Server                         |
|--------------------|----------------------------------------|-------------------------------------|
| System             | Ubuntu 18.04.6 LTS                     | Ubuntu 22.04.2 LTS                  |
| CPU                | 4-core ARM A57@1.43GHz                 | Intel(R) Core(TM) i9-10940X@3.30GHz |
| GPU                | 128-core Maxwell@921MHz                | GeForce GTX 3090 24GB               |
| Memory             | 4GB LPDDR4 25.6GB/s                    | 4*16GB LPDDR4 3200 MT/s             |
| Hard Disk          | 64GB microSDXC 140M/s(max)             | 1*1T SSD + 4*2T HDD                 |
| Network Connection | WiFi 2.4G:300Mbps 5G:867Mbps           | Ethernet 1000Mbps                   |



##### 开发前的主要环境配置要求

###### 用户端设备（Jetson Nano只允许该环境版本）
1. python==3.6.15
2. torch==1.4.0
3. torchvision==0.5.0
4. tegrastats
5. jtop

注：Jetson Nano的环境安装详细过程详见官方参考文档

###### 服务端设备
1. python>=3.7
2. torch==1.13.1
3. torchvision==0.13.1

###### **安装步骤**

1. 克隆仓库的源代码

```sh
git clone https://github.com/Jma512/DAPart.git
```

2. 安装环境配置必要的包


### 文件目录说明

```
DAPart 
├── /data/
│  ├── /test/
│  │  └── ...
│  └── ...      //实验时模拟任务所需的图像
├── /model
│  ├── /mobilenetv2/
│  │  │  └── /logs
│  │  ├── downloadmobilenetv2.py
│  │  └── mobilenetv2_pretrained_imagenet.pth
│  ├── /resnet50/
│  │  │  └── /logs
│  │  ├── downloadresnet50.py
│  │  └── resnet50_pretrained_imagenet.pth
│  ├── /vgg16/
│  │  │  └── /logs
│  │  ├── downloadvgg16.py
│  │  └── vgg16_pretrained_imagenet.pth
├── DAPart_Edge_Server.py
├── DAPart_User_Equipment.py
├── experiment_neuro.py
├── mobilenetv2.py
├── resnet50.py
├── vgg16.py
└── README.md

```

### 部署和运行

将该代码分别部署在用户端和服务器端即可，服务器端运行[DAPart_Edge_Server.py](https://github.com/Jma512/DAPart/blob/main/DAPart_Edge_Server.py),用户端设备运行[DAPart_User_Equipment.py](https://github.com/Jma512/DAPart/blob/main/DAPart_User_Equipment.py)


### 贡献者

xxx@xxxx（暂不公开）

### 版本控制

该项目使用Git进行版本管理。

### 作者

xxx@xxxx（暂不公开）


<!-- links -->
[your-project-path]:shaojintian/Best_README_template
[contributors-shield]: https://img.shields.io/github/contributors/shaojintian/Best_README_template.svg?style=flat-square
[contributors-url]: https://github.com/shaojintian/Best_README_template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/shaojintian/Best_README_template.svg?style=flat-square
[forks-url]: https://github.com/shaojintian/Best_README_template/network/members
[stars-shield]: https://img.shields.io/github/stars/shaojintian/Best_README_template.svg?style=flat-square
[stars-url]: https://github.com/shaojintian/Best_README_template/stargazers
[issues-shield]: https://img.shields.io/github/issues/shaojintian/Best_README_template.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/shaojintian/Best_README_template.svg
[license-shield]: https://img.shields.io/github/license/shaojintian/Best_README_template.svg?style=flat-square
[license-url]: https://github.com/shaojintian/Best_README_template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian



