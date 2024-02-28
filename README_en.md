

# DAPart

This open source code is a real test platform built by the actual experiment in the paper "DAPart: An Online DRL-based Adaptive Partition Framework for DNN Models in Edge Computing".
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
    <a href="https://github.com/Jma512/DAPart"><strong>Explore the documentation for this project »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Jma512/DAPart">View Demo</a>
    ·
    <a href="https://github.com/Jma512/DAPart/README.md">中文</a>
    ·
    <a href="https://github.com/Jma512/DAPart/README_en.md">English</a>
  </p>




This README.md is for developers.

## Catalogue

- [Getting started guide](#getting-started-guide)
  - [Configuration requirements before development](#major-environment-configuration-requirements-before-development)
    - [Client devices](#client-devices-jetson-nano-only-allows-this-version-of-the-environment)
    - [Server device](#server-device)
  - [Installation procedure](#installation-procedure)
- [File directory description](#file-directory-description)
- [Deployment and operation](#deployment-and-operation)
- [Contributors](#contributors)
- [Version control](#version-control)
- [Author](#author)

### Getting started guide
The open source code is divided into two parts: the client device uses Jetson Nano, and the server device uses a computer with a Linux system. The experimental equipment in this paper is shown in the table below.

| Hardware           | User Equipment Device<br>(Jetson Nano) | Edge Server                         |
|--------------------|----------------------------------------|-------------------------------------|
| System             | Ubuntu 18.04.6 LTS                     | Ubuntu 22.04.2 LTS                  |
| CPU                | 4-core ARM A57@1.43GHz                 | Intel(R) Core(TM) i9-10940X@3.30GHz |
| GPU                | 128-core Maxwell@921MHz                | GeForce GTX 3090 24GB               |
| Memory             | 4GB LPDDR4 25.6GB/s                    | 4*16GB LPDDR4 3200 MT/s             |
| Hard Disk          | 64GB microSDXC 140M/s(max)             | 1*1T SSD + 4*2T HDD                 |
| Network Connection | WiFi 2.4G:300Mbps 5G:867Mbps           | Ethernet 1000Mbps                   |



##### Major environment configuration requirements before development

###### Client devices (Jetson Nano only allows this version of the environment)
1. python==3.6.15
2. torch==1.4.0
3. torchvision==0.5.0
4. tegrastats
5. jtop

Note: The Jetson Nano environment installation process is detailed in the official reference documentation

###### Server device
1. python>=3.7
2. torch==1.13.1
3. torchvision==0.13.1

###### **Installation procedure**

1. Clone the source code of the repository

```sh
git clone https://github.com/Jma512/DAPart.git
```

2. Installation environment Configure the necessary packages


### File directory description

```
DAPart 
├── /data/
│  ├── /test/
│  │  └── ...
│  └── ...      //The images needed to simulate the task during the experiment
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

### Deployment and operation

The code can be deployed on the client side and the server side respectively, and the server side runs [DAPart_Edge_Server.py](https://github.com/Jma512/DAPart/blob/main/DAPart_Edge_Server.py), the client device runs [DAPart_User_Equipment.py](https://github.com/Jma512/DAPart/blob/main/DAPart_User_Equipment.py)


### Contributors

xxx@xxxx(Keep private)

### Version control

The project uses Git for version management.

### Author

xxx@xxxx(Keep private)


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
