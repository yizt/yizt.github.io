[TOC]

## 分类

### Imagenet

http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar

http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar

http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar

http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar

训练集类别id:https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57

## 目标检测

### Clipart1K，Comic2K，Watercolor2K

卡通图像



### MSCOCO

下载地址：http://cocodataset.org/#download

```shell
wget -c -t 0 http://images.cocodataset.org/zips/train2017.zip
http://images.cocodataset.org/zips/val2017.zip
http://images.cocodataset.org/zips/test2017.zip
http://images.cocodataset.org/annotations/annotations_trainval2017.zip
http://images.cocodataset.org/annotations/image_info_test2017.zip


wget -c -t 0 http://images.cocodataset.org/zips/val2014.zip
wget -c -t 0 http://images.cocodataset.org/zips/test2014.zip
wget -c -t 0 http://images.cocodataset.org/annotations/annotations_trainval2014.zip

```


## 人脸识别

### CASIA-WebFace

​        官网地址<http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html>

通过百度网络下载；链接: <https://pan.baidu.com/s/1qZlQLU8> 密码: q84w

### youtube face db

​        需要注册申请，官网地址:<http://www.cs.tau.ac.il/~wolf/ytfaces/>

```shell
wget -c -t 0 --user=wolftau --password=wtal997 http://www.cslab.openu.ac.il/personal/Hassner/wolftau/YouTubeFaces.tar.gz

```



### megaface

​         需要注册申请，官网地址: <http://megaface.cs.washington.edu/dataset/download.html>

```shell
wget -c -t 0 --user=csuyzt@gmail.com --password=xSYRnxhOs http://megaface.cs.washington.edu/dataset/download/content/MegaFace_dataset.tar.gz

```



### 人脸识别数据库

引用自：<https://blog.csdn.net/u012374174/article/details/71420766?locationNum=12&fps=1>

（2004年发布）CASPEAL：约1000个人,共约3万幅人脸图像

<http://www.jdl.ac.cn/peal/index.html>

（2008年发布）Multi-PIE：337个人,共约75万图像

<http://www.flintbox.com/public/project/4742/>

（2007年发布）LFW ：5749个人,共13233幅人脸图像

<http://vis-www.cs.umass.edu/lfw/>

（2009年发布）PubFig ：200个人,共58797幅人脸图像

<http://www.cs.columbia.edu/CAVE/databases/pubfig/>

（2014年发布）CASIAWebFace ：10575个人,共49414幅人脸图像

<http://www.cbsr.ia.ac.cn/english/CASIAWebFace-Database.html>

（2014年发布）FaceScrub ：530个人,共106863幅人脸图像

<http://vintage.winklerbros.net/facescrub.html>

（2016年发布）MegaFace ：约69万个人,共约100万幅人脸图像

<http://megaface.cs.washington.edu/>

（2016年发布）MS-Celeb-1M数据集：100w个人，1000w+图像

https://www.msceleb.org/

介绍一下MS-Celeb-1M数据集： 
MSR IRC是目前世界上规模最大、水平最高的图像识别赛事之一，由MSRA（微软亚洲研究院）图像分析、大数据挖掘研究组组长张磊发起，每年定期举办。参赛队伍被要求基于微软云服务，搭建包括人脸检测、对齐、识别的完整人脸识别系统，而且识别系统必须先通过远程实验评估.

reference paper：MS-Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition published at ECCV 2016.

Training dataset, contains 10M images in version 1, is the largest publicly available one in the world 



目前比较常用的数据集是LFW，CASIAWebFace，MegaFace



### Task : 识别 1M 个明星 from their face images.
作者：_小马奔腾 
来源：CSDN 
原文：https://blog.csdn.net/dongfang1984/article/details/53283729 
版权声明：本文为博主原创文章，转载请附上博文链接！



## 姿态估计/动作识别

### **Violent-Flows Dataset**

http://www.cslab.openu.ac.il/download/violentflow/

```shell
wget -c -t 0 --user=violentflow --password=violentflow http://www.cslab.openu.ac.il/download/violentflow/21VideosForDetection.rar

wget -c -t 0 --user=violentflow --password=violentflow http://www.cslab.openu.ac.il/download/violentflow/movies.rar
```



### **Hockey Fight Dataset** 

http://visilab.etsii.uclm.es/?page_id=1249



### UCF101

```shell
wget -c -t 0 https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
```





**LSP**
地址：http://sam.johnson.io/research/lsp.htm
样本数：2K
关节点个数：14
全身，单人

**FLIC**
地址：https://bensapp.github.io/flic-dataset.html
样本数：2W
关节点个数：9
全身，单人

**MPII**
地址：http://human-pose.mpi-inf.mpg.de/
样本数：25K
关节点个数：16
全身，单人/多人，40K people，410 human activities

**MSCOCO**
地址：http://cocodataset.org/#download
样本数：>=30W
关节点个数：18
全身，多人，keypoints on 10W people

**AI Challenge**
地址：https://challenger.ai/competition/keypoint/subject
样本数：21W Training, 3W Validation, 3W Testing
关节点个数：14
全身，多人，38W people

**PoseTrack**
地址：https://posetrack.net/
来源：CVPR2018
关键数据：>500 video sequences,>20K frames, >150K body pose annotations



## 医学影像

### Luna16

下载地址：

<https://zenodo.org/record/2604219#.XLWS7OgzY2w>

<https://zenodo.org/record/2596479#.XLWTwugzY2w>



## ocr

### Synth80k

- paper：[Synthetic Data for Text Localisation in Natural Images](http://www.robots.ox.ac.uk/~ankush/textloc.pdf)
- [数据库下载](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
- [代码下载](https://github.com/ankush-me/SynthText)
- 地址:http://www.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip



### Synth90k

- paper：[Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition](https://arxiv.org/pdf/1406.2227.pdf)
- [数据库下载](http://www.robots.ox.ac.uk/~vgg/data/text/)
- MJSynth: http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz 



### SVT

- paper：[Word Spotting in the Wild](http://vision.ucsd.edu/~kai/pubs/wang_eccv2010.pdf)
- [数据库下载](http://vision.ucsd.edu/~kai/grocr/)
- 地址：http://www.iapr-tc11.org/dataset/SVT/svt.zip



### IIIT5K-Word

- paper：[Scene Text Recognition using Higher Order Language Priors](http://www.di.ens.fr/willow/pdfscurrent/mishra12a.pdf)
- [数据库下载](http://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset)
- 地址：http://cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz