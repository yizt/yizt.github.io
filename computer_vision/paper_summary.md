[TOC]





## ML

### xgboost

#### 特性

正则化

二阶导数近似

尺寸收缩(类似学习率)

列下采样

近似算法(贪婪学习)

带权百分比摘要(Weighted Quantile Sketch)

稀疏模型处理



#### 系统实现方法

列快并行(column block) 预计算：近似时多个块(每个块包含部分行)，对local proposal algorithms；每个列的统计可以并行

Cache-aware获取

堆外计算(块压缩和分片) 



> 优点 

- XGB利用了二阶梯度来对节点进行划分，相对其他GBM来说，精度更加高。
- 利用局部近似算法对分裂节点的贪心算法优化，取适当的eps时，可以保持算法的性能且提高算法的运算速度。
- 在损失函数中加入了L1/L2项，控制模型的复杂度，提高模型的鲁棒性。
- 提供并行计算能力，主要是在树节点求不同的候选的分裂点的Gain Infomation（分裂后，损失函数的差值）
- Tree Shrinkage，column subsampling等不同的处理细节。

> 缺点

- 需要pre-sorted，这个会耗掉很多的内存空间（2 * #data * # features）
- 数据分割点上，由于XGB对不同的数据特征使用pre-sorted算法而不同特征其排序顺序是不同的，所以分裂时需要对每个特征单独做依次分割，遍历次数为#data * #features来将数据分裂到左右子节点上。
- 尽管使用了局部近似计算，但是处理粒度还是太细了
- 由于pre-sorted处理数据，在寻找特征分裂点时（level-wise），会产生大量的cache随机访问。

> 因此LightGBM针对这些缺点进行了相应的改进。

1. LightGBM基于histogram算法代替pre-sorted所构建的数据结构，利用histogram后，会有很多有用的tricks。例如histogram做差，提高了cache命中率（主要是因为使用了leaf-wise）。
2. 在机器学习当中，我们面对大数据量时候都会使用采样的方式（根据样本权值）来提高训练速度。又或者在训练的时候赋予样本权值来关于于某一类样本（如Adaboost）。LightGBM利用了GOSS来做采样算法。
3. 由于histogram算法对稀疏数据的处理时间复杂度没有pre-sorted好。因为histogram并不管特征值是否为0。因此我们采用了EFB来预处理稀疏数据。



#### 参考：

https://zhuanlan.zhihu.com/p/38516467









## 基础网络

### VGG VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION

<https://arxiv.org/pdf/1409.1556.pdf>





### Network In Network



参考：https://www.cnblogs.com/yinheyi/p/6978223.html



### ResNet v1

<https://arxiv.org/pdf/1512.03385.pdf>



#### 1. 介绍

网络深度非常重要；

深度导致难以训练已经较大程度解决:  Normalized 初始化、  Batch Normalization

深度带来另一个问题：网络退化，并非过拟合(训练和测试误差都增大)

更深的网络不应该比更浅的网络差(假定最后几层就学习一个恒等映射)；说明学习恒等映射很难

考虑学习一个残差(极致情况，权重全为0就可以了)；而且不会增加计算量和参数。

实验结果表明：更容易优化、性能随深度单调增加

#### 3. 深度残差学习

存在一个假说:多层非线性函数可以逼近复杂的函数; 那么同样可以逼近我们的残差

如果恒等映射是最优的，那么残差网络将权重学习为0就可以了

实验表明恒等映射通常有很小的反应

**网络结构**

a)相同的feature map大小,相同的channel, b) feature map 减半，深度加倍

c) 维度变化使用一个1*1的卷积来解决



#### 4. 实验

普通深度网络深度加深到一定程度后训练和验证误差都增加，**推测**是由于**收敛速率指数级降低** 

恒等映射 vs 投影映射 ；后者效果稍好，但会增加参数

Bottleneck架构：层数增加，参数没有增加

残差网络的相应更小

1202层和152层训练误差一样，但验证误差更高



### ResNet V2

<https://arxiv.org/pdf/1603.05027.pdf>

​          探索传播方式，前向和后向信号可以直接从一个block传到另一个block。

#### 介绍

​         聚焦直接传递信息，不仅在残差单元内，而是整个网络；

​         预激活思想设计一个全新的残差单元



#### 分析残差网络

​        如果xl+1 ≡ yl; 信息可以传递到任何浅层单元

#### Skip Connections的影响

​       shortcut connections是信息传递的最直接路径



#### 激活分析

​      预激活有两个优点：容易优化、防止过拟合





### PolyNet

<https://arxiv.org/pdf/1611.05725.pdf>



### Non-local Neural Networks

地址: https://arxiv.org/pdf/1711.07971.pdf



### Deformable Convolutional Networks

https://arxiv.org/pdf/1703.06211.pdf

 

由尺度、姿态、视角和部分形变等因素引起的几何变化是目标识别和检测的主要挑战；目前解决方法分为两类

1.数据增广: 覆盖尽可能多的变换(如: 仿射变换)；缺点：无法应对不可预知的变换前款

2.transformation-invariant 的特征和算法(如：SIFT)；缺点：手工设计的变换无关特征和算法很困难，且无法应对复杂的变换

本文提出可形变卷积和可形变RoI Pooling; 通过添加位移来丰富空间采样，位移值通过任务学习。





### GCNet

https://arxiv.org/pdf/1904.11492



可以应用到所有的残差块；不像NL由于计算量大，只用于少数层

兼具NL的性能和SENet的轻量；比NL轻量，比NL、SENet效果都要好；

实验分析NL



### VoVNet

https://arxiv.org/pdf/1904.09730.pdf

本文是DenseNet的变种。DenseNet的shortcut是密集的，即当前层的feature会传到后面的所有层。虽然DenseNet因为shortcut的原因，模型参数变少，但是其对GPU的资源消耗人仍旧非常的大，并且因为密集的shortcut，会增加后层特征channel，增大计算量。作者提出了One-short Aggregation（文中称为VoVNet），其实就是改变了shortcut的连接方式，从原来的前层feature与后面每一层都连接，变成了前层feature只与最后一层的连接。大大减少了连接数量。作者声称同样的结构，性能差不多，但是比DenseNet快2倍，energy consumption也减少了1.6~4.1倍。

 



## 目标检测

### R-CNN

​          PASCAL VOC目标检测进入停滞期, mAP 58.3; 两个见解: CNN用到region proposal中来定位对象；监督预训练和领域精调。

#### 简介

​        进展缓慢； 2012 AlexNet 在ImageNet分类引起关注，CNN分类任务多大程度能够泛化到目标检测。

​        关注两个问题：CNN定位对象，小量标注，大容量模型。

​        解决方法：region proposal和监督预训练(以前是无监督预训练)；



#### R-CNN目标检测

​         三个模块 region proposal，cnn, svm分类

​         高效：特征少，共享计算

**训练过程**

​        监督预训练

​        领域精调: 1:3 正负样本比，



### SSD



低分辨率获取高精度，进一步促进速度提升

不同feature maps预测不同的尺寸

使用小的卷积过滤器到feature maps上



### CBNet: A Novel Composite Backbone Network Architecture for Object Detection

https://arxiv.org/pdf/1909.03625.pdf



### Retinanet：Focal Loss for Dense Object Detection

https://arxiv.org/pdf/1708.02002.pdf

引用数：4760



### RefineNet: Single-Shot Refifinement Neural Network for Object Detection 

https://arxiv.org/pdf/1711.06897.pdf

引用数：504

两阶段网络精度高，单阶段网络速度快，RefineNet继承两者优点同时克服两者缺点，精度优于两阶段同时计算量与单阶段相当。

我们认为两阶段相对于一阶段有三个优点：

a) 处理类别不平衡

b) 级联回归边框参数

c) 两阶段特征描述对象（RPN:二分类，RCNN：多分类）

通过两个内联的模块：anchor refifinement module (ARM)和 object detect module(ODM)



### Cascade R-CNN: Delving into High Quality Object Detection 

https://arxiv.org/abs/1712.00726



### Cascade R-CNN: High Quality Object Detection and Instance Segmentation

https://arxiv.org/pdf/1906.09756v1.pdf





### PLN:Point Linking Network for Object Detection 

 https://arxiv.org/pdf/1706.03646.pdf



不同于以往将边框看做一个整体，我们使用顶点(points)和连接(links)来表示边框；

优点：对于遮挡、尺寸变换、长宽比变化 更加鲁棒





### CornerNet: Detecting Objects as Paired Keypoints 

https://arxiv.org/pdf/1808.01244.pdf

引用数：582

将检测对象边框变为检测一对关键点(左上角和右下角)；提出corner池化，更好的定位边框的角；

基于anchor框的检测器有两个缺点：

1. 大量anchor框；通常20k~100k个anchor边框，同时引发正负样本不均衡
2. 引入了大量相关超参数和设计选择；边框个数、尺寸、长宽比以及在多尺寸feature map上分配等等



为什么corner方式优于边框中心？推测边框中心更难定位，依赖4个边；二corner只依赖两个边；

为什么corner方式优于anchor框？wh个corner可以代表${w^2h^2}$个anchor 边框；是一个更加有效的离散表示



Group corner ?

参考:

Associative embedding: End-to-end learning for joint detection and grouping:<https://arxiv.org/pdf/1611.05424.pdf>



### ExtremeNet：Bottom-up Object Detection by Grouping Extreme and Center Points 

https://arxiv.org/abs/1901.08043.pdf



近两年来，有很多基于关键点预测的目标检测方法，比如CornerNet（在我的专栏中也有详细解读），通过预测左上角和右下角的角点，构成bounding box。在角点分类的方法上，借鉴了人体姿态估计的方法，进行向量嵌入进行分组。与CornerNet相比，本文提出的方法有两大不同：**关键点的定义和分组方式**。角点是bounding box的一种特殊形式，比如CornerNet，知道左上角和右下角的点后就形成了一个bounding box。再比如CenterNet，在CornerNet的基础上加了一个中间点，利用三个点形成一个bounding box。但是这种bounding box的生成方式是有**缺陷**的，因为这种点通常**在真正物体的外部**，并**没有很强的外观特征**，甚至说，它本不应该属于物体，只是因为我们要用bounding box表示而已。但是本文提出的**极值点是位于物体上的**，因此**在视觉上是可区分的**，具有**一致的局部外观特征**。



性能：相同backbone和分辨率情况下小幅优于CornerNet

### FCOS

https://arxiv.org/pdf/1904.01355.pdf

anchor free，单阶段，类似语义分割逐像素做检测

基于anchor的检测器的缺点

a) 对anchor尺寸、长宽比、数量敏感;需要小心的调试优化超参数

b) 无法适应尺寸的剧烈变化、小物体检测困难；且预定的anchor边框丢失通用性

c) 引起正负anchor样本数的不平衡问题

d) 大量的anchor边框iou计算耗时且占用内存



1. 对于重叠的GT 边框，像素到底预测哪一个GT box;FPN可以缓解这个问题(对于歧义的像素，关联面积小的GT box)。
2. center-ness 预测像素偏离中心的程度，该分支可以过滤低质量的检测边框
3. fcos可以利用尽可能多的前景样本，不像基于anchor的检测器，只利用IoU较高的样本





### CenterNet: Keypoint Triplets for Object Detection

https://arxiv.org/pdf/1904.08189.pdf



CornerNet受限于较弱的全局信息能力(对关键点检测敏感，对哪些关键点应该是一对不敏感)；

CenterNet增加一个关键点感知region的中心；同时增加两个策略center pooling,cascade corner pooling



### EfficientDet

https://arxiv.org/pdf/1911.09070.pdf



## 实时目标检测

### CenterNet: Objects as Points

https://arxiv.org/pdf/1904.07850.pdf

关键词: anchor-free,没有分组,没有后处理(如NMS);

对应目标检测keypiont就是GT 边框中心点

建模过程

1. Keypoint: 计算GT边框的k的Keypoint在步长R为预测FeatureMap(H/R,W/R,C)上的坐标(x,y)，将通道维Ck通道(x,y)赋值为1，然后使用高斯核函数均匀化；使用FocalLoss回归
2. Keypoint坐标偏移: 使用L1 loss计算由于步长R带来的坐标偏移
3. Keypoint尺寸预测：使用L1 loss计算Keypoint尺寸

预测过程

1. 通过Keypoint预测结果找到最大的n个峰值，所谓峰值就是不小于领域所有值
2. 将峰值坐标偏移，再根据尺寸即可得到边框坐标





### TTF:Training-Time-Friendly Network for Real-Time Object Detection

<https://arxiv.org/pdf/1909.00700.pdf>



### CSPNet: A New Backbone that can Enhance Learning Capability of CNN

<https://arxiv.org/pdf/1911.11929.pdf>





## 小目标检测

### HRDNet: High-resolution Detection Network for Small Objects

https://arxiv.org/pdf/2006.07607.pdf







## 语义分割

RefineNet、PSPNet

### DeepLab v1

<https://arxiv.org/pdf/1412.7062v3.pdf>





#### 摘要

​        三个贡献: 

a)空洞卷积:控制分辨率、增大感受野、保持参数不变

b) ASPP-空洞金字塔池化: 鲁棒分割多尺寸对象，多重采样比例和视野，来捕捉不同尺寸的对象

c) 组合DCNN和PGM: 提升边界定位，卷积网络的下采样影响定位精度，通过组合DCNN最后一层响应和全连接CRF，提升定位性能



#### 简介

​        DCNN在视觉识别任务(图像分类、目标检测)取得很大成功，关键的因素是DCNN对于局部图像变换的内在不变性；可以学习数据的抽象表示，对于分类很好；但会妨碍像语义分割这种密集预测任务，抽象的空间信息在这里不合适。

​         DCNN遇到的三个挑战: 分辨率的降低、目标的多尺寸、	不变性带来的定位精度降低

第一个挑战: 去除最后几个池化层，使用滤波器上采样(就是空洞卷积)；组合多个空洞卷积，接一个双线性插值到原图分辨率。

第二个挑战：不同于多尺寸输入，使用多个并行的不同采样率的空洞卷积，称之为ASPP.

第三个挑战：一种方式使用skip-layers，最终用多层来预测最终分割结果。我们使用全连接CRF，捕获边缘细节，满足长距离依赖

​          DeepLab的三个优点：

a)速度：8FPS, 全连接CRF 0.5秒   b) 

b)精度：VOC 2012 79.7%

c) 简单：级联两个固定的模块DCNN和CRF





#### 相关工作

​         像素分类，完全抛弃分割；组合DCNN和局部CRF，将superpixs作为顶点对待，忽略远程依赖。我们的方法将像素作为顶点，捕获远程依赖，服从快速平均场推断。

​        其它论文的重要和有价值的

a) 端到端的训练，结构化预测：我们的CRF是后处理步骤，已经有人提出的端到端联合训练 ；使用一个卷积层近似密集CRF平均场推断的一个迭代；另一个方向是使用DCNN学习CRF的成对项 

b) 弱监督：无需整张图都有像素级别标注



#### 方法

##### 空洞卷积抽取特征和增大视野

​        一种修复分辨率减小的方法是使用反卷积，这需要额外的内存和耗时。我们使用空洞卷积，可以应用到任意层，满足任意分辨率，并无缝集成。

​         可以在任意高分辨率上计算DCNN最终的响应；例如：为了加倍特征响应的空间密度；将最后一个池化层的补充设置为1，接下来所有的卷积层的采样比设置为2。在所有层上使用这种方法，可以获得原始图像分辨率的响应，但是这计算量太大。采用混合方式，平滑效率/精度;最后两层使用空洞卷积, 然后8倍双线性插值，恢复到原图分辨率。双线性插值不需要学习任何参数，速度更快

​         空洞卷积可以在任意层，任意增大视野，计算和参数保持不变。大的采样率，对性能有提升。

##### ASPP表示多尺寸图像

​         DCNN有隐式代表图像尺寸的能力，明确说明对象尺寸可以提高DCNN成功处理大尺寸和小尺寸对象的能力。

​         第一种是多尺寸处理。

​         第二种是重采样单尺寸的卷积层特征；我们使用ASPP，泛化了DeepLab-LargeFOV。



##### 全连接CRF结构化预测

​         DCNN顶层节点只有很平滑的响应，只能预测对象的粗略位置；以前有两种方式来处理：一是多层预测，而是采用超像素表示，使用低级分割方法。

​         我们组合DCNN和全连接CRF，传统的CRF用于弱分类器的平滑，相邻像素倾向于相同类别。DCNN的特征图原本就很平滑，我们的目标不是平滑而是恢复局部结构的细节

​         第一项依赖像素位置和RGB颜色，第二项只依赖像素。第一项迫使相似颜色和位置的像素有相同的标签；第二项仅考虑像素位置。



#### 实验结果

​         ImageNet预训练模型、交叉熵损失函数、在8倍下采样的每个空间计算损失、DCNN和CRF分开训练。先介绍会议版本，接下来是最近结果



##### PASCAL VOC 2012

​         20个目标对象，一个背景类; 1, 464 (train)、1, 449 (val), and 1, 456 (test) 像素级别标注图像；性能度量是21个类别上像素级别IoU

**会议版本结果**

​         使用VGG16预训练模型，mini-batch 20，学习率0.001，每2000个迭代学习率缩减10倍，权重衰减0.0005、动量大小0.9。

​          最终使用3*3的卷积核、大的采样率12、fc6和fc7层神经元改为1000个，形成DeepLab-LargeFOV版本。CRF可以提升3~5个百分点。



**会议版本后的改进**

a)学习率策略

​        使用poly学习率，比sgd好1.17%

b)采用ASPP

​        多个并行的fc6-fc7-fc8分支，fc6 是3*3卷积，ASPP-S和 ASPP-L的采样率分别为{2, 4, 8, 12}和{6, 12, 18, 24}

CRF前ASPP-S比LargeFOV好1.22%；经过CRF后基本一样(也就是ASPP没有效果)。ASPP-L还是有提升的。

c)更深的网络和多尺寸处理

​      使用ResNet101; 分别输入scale = {0.5, 0.75,1}；在score map层融合最大响应；在MS-COCO上预训练，训练时随机缩放0.5~1.5; 最终79.7%的精度。

​     

##### PASCAL-Context

​          语义标注了整个图像包括对象和stuff；在最常见的59类+背景类上评估；训练数据4998；验证集5105；最好精度为45.7%



##### PASCAL-Person-Part

​       包含更多的尺寸和姿态，标注了人体的每一部分；融合标注为Head, Torso,Upper/Lower Arms and Upper/Lower Legs 6类+1个背景类；1716个训练集，1817个验证集。最好精度为64.94%。

​        但是LargeFOV or ASPP在这个数据集上没有作用。



##### Cityscapes

​       高质量的像素级别标注，来自50个城市的5000张街景图像。19个类别属于7个大类： ground, construction,object, nature, sky, human, and vehicle；

​        training, validation, and test 分别为 2975, 500, 1525 ；最好精度71.4。

​        图像原始分辨率为2048×1024；没有使用多尺寸。



##### 失败案例

​       不能捕获纤细物体的边界，如自行车、椅子



### DeeplabV2

<https://arxiv.org/pdf/1606.00915.pdf>

语义分割面临的三大问题：a)分辨率降低；b)多尺寸；c) 	平移不变带来的定位精度降低



a)空洞卷积：

下采样8倍，空洞卷积提升4倍，最后使用双向性插值放到到原始图像尺寸，优点没有需要学习的参数

k×k 空洞卷积感受野：ke = k + (k − 1)(r − 1) ；

空洞卷积的两种实现方式：普通空洞卷积; 下采样形成r2个patch,接普通的卷积，然后编织成原始图像大小



b)空洞空间金字塔池化：

DCNN有隐含的多尺寸能力，这是通过训练样本中的不同尺寸对象获得的。显示的考虑对象尺寸可以提升DCNN处理大物体和小物体的检测能力。

多尺寸一般有两种方法：多个尺寸的图像处理和预测；ASPP



c)FC-CRF精准边界恢复

以前的两种方式：多层featere map预测(如：FCN)；超像素代表

DCNN预测结果很平滑；短距离CRF不合适，需要长距离CRF



### DeeplabV3

<https://arxiv.org/pdf/1706.05587.pdf>

图像级特征编码全局上下文，去除CRF后处理；



全局信息有利于像素分类



当sample rate 变大时，有效的卷积权重降低；当sample rate过大时，3 × 3 的卷积退化为1×1的卷积



### DeeplabV3+

<https://arxiv.org/pdf/1802.02611.pdf>

空间金字塔池化和编码-解码结构在语义分割中经常用到；

主要贡献：

1. 组合空间金字塔池化和编码-解码结构
2. 引入Xception的DW Conv,速度更快



### 总结

​        空洞卷积、ASPP、组合DCNN和全连接CRF。





参考：https://blog.csdn.net/u013580397/article/details/78508392

[DeepLab(1,2,3)系列总结](https://blog.csdn.net/u011974639/article/details/79148719)

[DeepLab v3+](https://www.paperweekly.site/papers/notes/326)





### MNC Instance-aware semantic segmentation via Multi-task Network Cascades

https://arxiv.org/pdf/1512.04412.pdf

COCO 2015 语义分割冠军



### InstanceFCN Instance-sensitive Fully Convolutional Networks

<https://arxiv.org/abs/1603.08678>

不同于FCN只有一个score map,产生少量instance-sensitive score maps

动机：如果能够区分左边和右边，就能够使用score map区分实例；

使用一个实例相对位置分类器

assembling 模块是复制粘贴操作

局部连贯性：





### FCIS

https://arxiv.org/pdf/1611.07709.pdf

COCO 2016 语义分割冠军

第一个instance-aware的语义分割FCN;传统FCN需要检测和分割来做到instance-aware;之前的instance-aware语义分割使用FC实现，形变和固定尺寸表示损害分割精度，参数过多，计算量大，耗时长

position-sensitive score maps 可以实现平移变化

在提议框上执行ROI操作，而不是滑动窗口



### FCN Fully Convolutional Networks for Semantic Segmentation

<https://arxiv.org/pdf/1411.4038.pdf>

FC网络可以看做卷积网络的一种；

227 × 227 耗时1.2ms产生一个点输出;500 × 500耗时22ms,产生10 × 10网格输出；对应的反向传播2.4 ms和37ms；

分类网络产生的分割结果是粗糙的

Sampling in patchwise training can correct class imbalance [27, 8, 2] and mitigate the spatial correlation of dense patches [28, 16].

Whole image fully convolutional training is identical to patchwise training where each batch consists of all the receptive fields of the units below the loss for an image (or collection of images).

If the kept patches still have significant overlap, fully convolutional computation will still speed up training.





### Learning deconvolution network for semantic segmentation 

<https://arxiv.org/pdf/1505.04366>



1. FCN由于固定的感受野，对应于大大超出感受野或者远小于感受野的物体分割不好；或分割不完整或误标记.
2. 由于下采样,对象的精细结构在最终label map上丢失或者平滑，而且双线性插值上采样的过程太过简单；最近的网络使用CRF

解决方法：

1. 由反卷积、上池化和ReLU组成多层反卷积网络
2. 对每个实例proposal单独分割，然后融合成整个图像的分割
3. 





fixed-size receptive field ,大对象分类不一致，小对象误分类或漏标



- a multi-layer deconvolution network 
- The trained network is applied to individual object proposals to obtain instance-wise segmentations 

### 

deconvolution 是形状生成器

unpooling 捕获example-specific 结构，跟踪原始定位；

deconvolutional 捕获class-specific 形状

Instance-wise segmentation 可以处理不同的尺寸



训练细节

BN

两阶段训练：参数多，样本少； we employ a two-stage training method to address this issue, where we train the network with easy examples first and fine-tune the trained network with more challenging examples later ；

容易样本，crop GT; 困难样本使用proposals 

预测：top50 proposal 以max方式融合

FCN擅长整体形状，DeconvNet 擅长细节







### ParseNet: Looking Wider to See Better

 <https://arxiv.org/pdf/1506.04579.pdf>





### Segnet

https://arxiv.org/pdf/1511.00561.pdf

encoder-decoder结构、non-linear upsampling 

non-linear upsampling: 提升边框刻画能力，减少参数，移植到其它encoder-decoder网络中

FCN encoder参数过多，decoder参数过少，训练困难；预测耗时；同时因为encoder的feature map需要重用，导致预测时内存占用高；



使用CRF是因为解码器不够好

Therefore, it is necessary to capture and store boundary information in the encoder feature maps before sub-sampling is performed 

改进点

1. 去掉了VGG的FC层，大大降低参数量
2. 在encoder中捕获和存储边界信息



### U-Net: Convolutional Networks for Biomedical Image Segmentation

 <https://arxiv.org/pdf/1505.04597.pdf>

训练深度神经网络通常需要数千张标注图像，我们通过数据增强提升标注图像的使用效率，使用少量标注图像即可训练一个深度网络

通过灵活的形变做数据增广











### RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation

https://arxiv.org/pdf/1611.06612.pdf



DeConv不能还原有下采样丢失的低级视觉特征，无法提供高分辨率下精准的预测；

DeepLab的空洞卷积计算量大；内存/显存占用高；空洞卷积的稀疏下采样也可能丢失细节信息



很多方法使用中间层做分割预测，怎样有效利用中间层feature map是一个开放的问题



a)通过递归方式将高层语义信息融合到低层feature map生成高分辨率的feature map  b)采用短距离和长距离残差连接，可以高效的e2e训练；b) 链式池化获取全局上下文信息



语义分割可以看做是密集的分类问题；

分类网络：分辨率逐步下降的原因： 逐步增加感受野，捕获全局上下文依赖；计算和存储资源受限；

空洞卷积内存占用高







### PSPNet Pyramid Scene Parsing Network

<https://arxiv.org/pdf/1612.01105.pdf>

通过不同区域上下文集成，层次全局先验



1. 上下文关系普遍存在又非常重要，特别是对于复杂的场景理解，往往存在共现视觉关系。

2. 没有上下文关系会导致：a)预测不匹配的关系;b)混淆类别;c)不显眼的类别被忽略

3. 普通全局平均池化上下文，无法在复杂场景中覆盖足够信息

4. 本文提出全局先验表征，通过金字塔池化集成不同区域上下文，能够精准理解复杂场景

   

通过金字塔池化集成不同区域上下文，形成层次的全局先验表征；



感受野可以大概表面上下文范围；理论感受野大，但实际感受野小很多；



其它的全局描述子代表性不足，全局平均池化上下文无法在复杂场景中覆盖足够信息





### DANet:Dual attention network for scene segmentation

<https://arxiv.org/pdf/1809.02983.pdf>

需要考虑易混淆的类别，如：田野、草地；

上下文融合不能以全局视角利用对象之间的关系

提出一个双注意力网络，使用自注意力机制捕获空间和通道的依赖关系；

1. 依据特征相似度计算权重，两个相似的特征可以互相提升，不管他们距离多远；可以防止不显眼的物体被支配物体影响
2. 需要不同尺寸的上下文信息；不同尺寸的特征需要同等对待



### ExFuse: Enhancing Feature Fusion for Semantic Segmentation

<https://arxiv.org/pdf/1804.03821.pdf>

1. 直接融合高层语义与底层空间信息并不是最有效，因为高层信息与底层信息有鸿沟
2. U-Net很成功，但是其工作机制并不完全清楚，值得的研究



引入更多语义信息到底层





### Object-Contextual Representations for Semantic Segmentation

<https://arxiv.org/pdf/1909.11065.pdf>

探索像素点和它的上下文关系；像素的类别是像素所属对象的类别；通过对象区域的类别表征来增强像素的表征



1. 将上下文分割为soft object regions ，每个region关联一个类别；
2. 集成region中像素表征来估计regions的表征
3. 通过对象上下文表征来增强像素的表征



OCR 区分对象类别，ASPP 值区分空间位置；之前的上下文关系是像素之间的关系，OCR是像素与对象区域的关系；

使用coarse segmentation map 来生成上下文表征







## 实时语义分割

### HarDNet: A Low Memory Traffic Network

https://arxiv.org/pdf/1909.00948.pdf



### ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation

https://arxiv.org/pdf/1803.06815.pdf



### ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation

<https://arxiv.org/pdf/1906.09826.pdf>







### ShelfNet for Fast Semantic Segmentation

https://arxiv.org/pdf/1811.11254.pdf





### Fast-SCNN

https://arxiv.org/pdf/1902.04502.pdf

大感受野，空间细节

Encoder-decoder、多分支

ICNet [36], ContextNet [21], BiSeNet [34] and GUN [17]



DSConv + Inverted ResBlock + PPM 

### CAS

### DF1-Seg-d8



### FasterSeg: Searching for Faster Real-time Semantic Segmentation

https://arxiv.org/pdf/1912.10917.pdf

NAS

 “zoomed conv.”: bilinear downsampling + 3×3 conv. + bilinear upsampling

“zoomed conv. ×2”: bilinear downsampling + 3×3 conv. ×2 + bilinear upsampling

知识蒸馏



### BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation

https://arxiv.org/pdf/1808.00897.pdf



### LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation

https://arxiv.org/pdf/1905.02423.pdf





## 实例分割

### Hybrid Task Cascade for Instance Segmentation

https://arxiv.org/pdf/1901.07518.pdf





三点改进：

Interleaved Execution

Cascade R-CNN 虽然强行在每一个 stage 里面塞下了两个分支，但是这两个分支之间在训练过程中没有任何交互，它们是并行执行的。所以我们提出 Interleaved Execution，也即在每个 stage 里，先执行 box 分支，将回归过的框再交由 mask 分支来预测 mask，如上图（b）所示。这样既增加了每个 stage 内不同分支之间的交互，也消除了训练和测试流程的 gap。

Mask Information Flow

 Cascade Mask R-CNN 中，不同 stage 之间的 mask 分支是没有任何直接的信息流的，Mi+1 只和当前 Bi 通过 RoI Align 有关联而与 Mi 没有任何联系。多个 stage 的 mask 分支更像用不同分布的数据进行训练然后在测试的时候进行 ensemble，而没有起到 stage 间逐渐调整和增强的作用。为了解决这一问题，我们在相邻的 stage 的 mask 分支之间增加一条连接，提供 mask 分支的信息流，让 Mi+1能知道 Mi 的特征。

Semantic Feature Fusion

这一步是我们尝试将语义分割引入到实例分割框架中，以获得更好的 spatial context。因为语义分割需要对全图进行精细的像素级的分类，所以它的特征是具有很强的空间位置信息，同时对前景和背景有很强的辨别能力。通过将这个分支的语义信息再融合到 box 和 mask 分支中，这两个分支的性能可以得到较大提升。





参考：https://baijiahao.baidu.com/s?id=1627161484427243571&wfr=spider&for=pc



### YOLACT



1.两阶段分割网络过于依赖特征定位来生成mask；一阶段如FCIS在定位后有大量后处理，无法实时

2.YOLACT放弃显示的定位，将实例分割分为两个任务：a)生成non-local prototype masks ;b)	为每个实例预测一组linear combination coefficients 



### YOLACT++

https://arxiv.org/pdf/1912.06218.pdf

### PANet

https://arxiv.org/abs/1803.01534

https://github.com/ShuLiu1993/PANet

COCO 2017 实例分割第一名。

信息传递方式很重要(fpn原本自底向上路径太长)；较低层的精准定位通过bottom-up路径增强，提升了整个feature层次结构。



较低层的特征对尺寸大的对象预测也是很重要的

每个边框只通过某一层feature预测不是最优的，其它层丢弃的信息对最后预测可能很重要

mask基于单个视野预测丢弃的多样性信息，增加一个FC来辅助FCN;FC是位置敏感的，且有全局信息







### MS R-CNN

Mask Scoring R-CNN 

https://arxiv.org/pdf/1903.00241.pdf

实例分割中使用分类的分数作为mask的质量得分；mask的得分由IoU量化

smask = scls · siou 



### PolarMask: Single Shot Instance Segmentation with Polar Representation

https://arxiv.org/pdf/1909.13226.pdf



Smooth L1忽略的距离的关系；

IoU loss计算复杂，无法并行





### SOLO

https://arxiv.org/pdf/1912.04488.pdf



### CenterMask:Real-Time Anchor-Free Instance Segmentation

https://arxiv.org/pdf/1911.06667.pdf

https://github.com/youngwanLEE/CenterMask

基于FCOS+VovNet,增加SAM做mask;  SAM参考CBAM；

传统的RoiAlign没有考虑Roi尺寸

VovNet做了改进，增加残差和SE的attention



## 人脸识别

### Docface+

#### DWI-AMSoftmax Loss

$$
w_j= \frac {w_j^*}  {||w_j^*||_2}  \tag 4
$$

$$
w_j^*=(1-\alpha)w_j + \alpha w_j^{batch}  \tag 5
$$

$w_j^{batch}$ 是根据当前mini-batch计算出的目标权重向量; 就是嵌入特征 $f$  的L2归一化值；注意$w_j$ 也只更新当前mini-batch中的权重。

margin m 为5.0,最初使用



Face-ResNet架构

一般的方法到ID-自拍数据集上迁移效果不好：收敛慢，陷入局部极小值；

由于数据集浅，造成欠拟合

**数据采样** 

​           随机采样$B/2$ 各类别；$B$ 是batch-size; 然后从每个类别各采集一个ID图像和自拍图像





### 总结

https://www.jianshu.com/p/1dd8c0364710



## OCR/场景文本检测

### CTPN

<https://arxiv.org/pdf/1609.03605.pdf>

固定宽度、垂直的anchor

CNN+RNN组合,in-network rnn

end-to-end

#### 简介

之前的检测器不鲁棒、不可靠；使用低级特征，区别单个笔画或字符，没有使用上下文信息。

faster r-cnn很在目标检测上成功;但无法直接用到这里，因为需要更高的定位精度，这是细粒度的识别

定位准、in-network rnn、适用多尺寸、多语言、end-to-end

#### 相关工作

Connected-componets:快速过滤器区分像素文本/非文本; 然后使用低级特征分组为笔画或字符候选

sliding-window:多尺寸密集滑窗检测字符候选; 计算量大

这两种通用方法受限于字符检测性能

#### Connectionist Text Proposal Network

细粒度的提议框、recurrent connectionist text proposals、side-refinement

细粒度的提议框：text line可以较好的区分文本，固定宽度,预测垂直的效果更好；anchor垂直数量10,范围11~273; 同时预测评分和垂直的边框位置;细粒度的的检测提供了更加精细的监督信息，导致更精准的定位

Recurrent Connectionist Text Proposals: 独立的检测proposals，导致类文本的噪声;上下文信息保障可靠

Side-Refinement: 水平距离小于50像素(垂直IoU>0.7)的proposals连起来; 精调两侧的边框水平方向位置



### Canny Text Detector: Fast and Robust Scene Text Localization Algorithm

<https://zpascal.net/cvpr2016/Cho_Canny_Text_Detector_CVPR_2016_paper.pdf>

引用数：78

使用ER提取候选

NMS过滤

双阈值分类

文本分组

输出结果



### TextBoxes: A Fast Text Detector with a Single Deep Neural Network

<https://arxiv.org/pdf/1611.06779>





### **EAST: An Efficient and Accurate Scene Text Detector** 

<https://arxiv.org/pdf/1704.03155.pdf>

引用数：431



### TextBoxes++: A Single-Shot Oriented Scene Text Detector

<https://arxiv.org/pdf/1801.02765.pdf>

引用数：174

设计长卷积核，多层预测

4方面

 

### PSENet：Shape Robust Text Detection with Progressive Scale Expansion Network

<https://arxiv.org/pdf/1806.02559.pdf>

引用数：27



### TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes

<https://arxiv.org/pdf/1807.01544.pdf>





### Pixel-Anchor：A Fast Oriented Scene Text Detector with Combined Networks

<https://arxiv.org/pdf/1811.07432.pdf>

引用数：10



### TextField: Learning A Deep Direction Field for Irregular Scene Text Detection

<https://arxiv.org/pdf/1812.01393.pdf>

引用数：53



### CRAFT: Character-Region Awareness For Text detection

https://arxiv.org/abs/1904.01941.pdf

引用数：36

CRAFT for Character Region Awareness For Text detec- tion

character region score 和 character affinity score ；弱监督生成虚拟字符标注；

使用字符和字符的亲和度检测文本，利用合成的数据集的字符级别标注及真实数据集模型评估的字符位置



region score : 字符的中心概率

affinity score ：相邻两个字符的中心概率



置信度得分：检测字符数/GT字符数	；得分小于0.5时，字符宽度使用边框宽度/字符数，然后置信度得分设置为0.5



### PAN Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network

<https://arxiv.org/pdf/1908.05900.pdf>

引用数：12





### DB:Real-time Scene Text Detection with Differentiable Binarization

<https://arxiv.org/abs/1911.08947>

引用数：7

产生：probability map (F)和 threshold map(T) ；然后 根据F和T计算approximate binary map

threshold map用于二值化

### 总结

https://blog.csdn.net/xwukefr2tnh4/article/details/80589198



## OCR/文本识别

### Strokelets: A Learned Multi-Scale Representation for Scene Text Recognition

<https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Yao_Strokelets_A_Learned_2014_CVPR_paper.pdf>



### CRNN：An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition

<https://arxiv.org/pdf/1507.05717.pdf>

类序列识别，长度变化大；CNN不能可变

end-to-end、不需要字符级别标注、不受限于固定词典、序列长度

引用数：731

### RARE: Robust Scene Text Recognition with Automatic Rectifification 

<https://arxiv.org/pdf/1603.03915.pdf>

引用数：216



### R2AM: Recursive Recurrent Nets with Attention Modeling for OCR in the Wild

<https://arxiv.org/pdf/1603.03101.pdf>





### ASTER: An Attentional Scene Text Recognizer with Flexible Rectification

<https://www.vlrlab.net/admin/uploads/avatars/ASTER_An_Attentional_Scene_Text_Recognizer_with_Flexible_Rectification.pdf>

引用数：101



### Drawing and Recognizing Chinese Characters with Recurrent Neural Network

<https://arxiv.org/pdf/1606.06539.pdf>

引用数：138 



### Learning Spatial-Semantic Context with Fully Convolutional Recurrent Network for Online Handwritten Chinese Text Recognition

<https://arxiv.org/pdf/1610.02616.pdf>

 引用数：47





### Building Fast and Compact Convolutional Neural Networks for Offline Handwritten Chinese Character Recognition

<https://arxiv.org/pdf/1702.07975.pdf>

引用数：69





### Attention-based Extraction of Structured Information from Street View Imagery

<https://arxiv.org/pdf/1704.03549.pdf>





### Learning to read irregular text with attention mechanisms.

<http://clgiles.ist.psu.edu/pubs/IJCAI2017.pdf>



### Focusing Attention: Towards Accurate Text Recognition in Natural Images

<https://arxiv.org/pdf/1709.02054.pdf>



### AON: Towards Arbitrarily-Oriented Text Recognition

<https://arxiv.org/pdf/1711.04226.pdf>

引用数：83



### Edit Probability for Scene Text Recognition

<https://arxiv.org/pdf/1805.03384.pdf>

引用数：40

大多数文字识别网络的架构是encoder-decoder；encoder为cnn或crnn,decoder为rnn或ctc或attention机制

名词：probability distribution (pd) ， frame-wise probability (FP) ；attention drift 问题



### MORAN: A Multi-Object Rectified Attention Network for Scene Text Recognition

<https://arxiv.org/pdf/1901.03003.pdf>

 引用数：6

不规则文本识别难以收敛；

MORN 比仿射变换好，仿射变换不能捕捉弯曲等复杂形变；比RARE(ASTER)好，当图像宽度大时无法捕获文本形状细节



### Aggregation Cross-Entropy for Sequence Recognition 



CTC计算量大，无法应用到2D场景；

Attention机制依赖于标签对齐，导致额外存储和计算，由于错位问题基于Attention的识别模型很难从头开始训练，特别是长输入序列



ACE包括三个简单步骤

1. 每个类别延时间维度集成
2. 标准化预测和GT的累积分布
3. 使用交叉熵计算式两个累积分布的loss



### FACLSTM: ConvLSTM with Focused Attention for Scene Text Recognition

引用数：3



### A Simple and Robust Convolutional-Attention Network for Irregular Text Recognition

<https://arxiv.org/pdf/1904.01375v1.pdf>



### A Holistic Representation Guided Attention Network for Scene Text Recognition

 <https://arxiv.org/pdf/1904.01375v4.pdf>





### TextScanner: Reading Characters in Order for Robust Scene Text Recognition

<https://arxiv.org/pdf/1912.12422.pdf>

引用数：0





 





## OCR/E2E检测识别

### FOTS: Fast Oriented Text Spotting with a Unified Network

https://arxiv.org/pdf/1801.01671.pdf

引用数：133



### An end-to-end TextSpotter with Explicit Alignment and Attention

<https://arxiv.org/pdf/1803.03474.pdf>

 引用数：61



### Verisimilar Image Synthesis for Accurate Detection and Recognition of Texts in Scenes

<https://arxiv.org/pdf/1807.03021.pdf>

引用数：19 



### A Novel Integrated Framework for Learning both Text Detection and Recognition

https://arxiv.org/pdf/1811.08611.pdf

引用数：4



### TextDragon: An End-to-End Framework for Arbitrary Shaped Text Spotting

<https://openaccess.thecvf.com/content_ICCV_2019/papers/Feng_TextDragon_An_End-to-End_Framework_for_Arbitrary_Shaped_Text_Spotting_ICCV_2019_paper.pdf>



### You Only Recognize Once: Towards Fast Video Text Spotting

<https://arxiv.org/pdf/1903.03299.pdf>

 引用数：3



### GA-DAN: Geometry-Aware Domain Adaptation Network for Scene Text Detection and Recognition

<https://arxiv.org/pdf/1907.09653.pdf>

引用数：2 







### Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes

<https://arxiv.org/pdf/1807.02242.pdf>



### Mask TextSpotter v2: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes

<https://arxiv.org/pdf/1908.08207.pdf>



### Towards Unconstrained End-to-End Text Spotting

<https://arxiv.org/pdf/1908.09231.pdf>

引用数：13





### Convolutional Character Networks

<https://arxiv.org/pdf/1910.07954.pdf>



### ABCNet：Real-time Scene Text Spotting with Adaptive Bezier-Curve Network

<https://arxiv.org/pdf/2002.10200.pdf>

 引用数：4



### Mask TextSpotter v3: Segmentation Proposal Network for Robust Scene Text Spotting

<https://arxiv.org/pdf/2007.09482.pdf>



### Text Recognition in the Wild: A Survey

<https://arxiv.org/pdf/2005.03492v1.pdf>



## OCR/字符分割

### Character Segmentation in Asian Collector’s Seal Imprints: An Attempt to Retrieval Based on Ancient Character Typeface

<https://arxiv.org/pdf/2003.00831.pdf>

https://github.com/timcanby/collector-s_seal-ImageProcessing.git



### Word and character segmentation directly in run-length compressed handwritten document images

<https://arxiv.org/pdf/1909.05146.pdf>



### Chinese/English mixed Character Segmentation as Semantic Segmentation

<https://arxiv.org/pdf/1611.01982.pdf>



### AUTOMATIC TEXT EXTRACTION AND CHARACTER SEGMENTATION USING MAXIMALLY STABLE EXTREMAL REGIONS

<https://arxiv.org/pdf/1608.03374.pdf>





<https://arxiv.org/pdf/1707.00800.pdf>



## ocr/图像合成



### Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition 



### Synthetic Data for Text Localisation in Natural Images

https://arxiv.org/pdf/1604.06646.pdf



### Verisimilar image synthesis for accurate detection and recognition of texts in scenes 

在计算机图形学研究中，以逼真方式将对象插入图像中已被广泛研究为图像合成的一种手段[7]。 目标是通过控制物体的尺寸，物体的视角（或方向），环境照明等来实现插入的逼真度，即合成图像的真实相似度。 [8]开发了一种半自动技术，该技术可将物体插入具有真实感照明和透视效果的旧照片中。

近年来，当只有有限数量的带注释的图像可用时，图像合成已被研究为一种数据增强方法，用于训练准确而健壮的DNN模型。 例如，Jaderberg等。 [9]创建一个单词生成器，并使用合成图像来训练文本识别网络。

8 Karsch, K., Hedau, V., Forsyth, D., Hoiem, D.: Rendering synthetic objects into legacy photographs. ACM Transactions on Graphics (6) (2011) 157:1–157:12 

9 Jaderberg, M., Simonyan, K., Vedaldi, A., Zisserman, A.: Synthetic data and artificial neural networks for natural scene text recognition. arXiv preprint arXiv:1406.2227 (2014) 



### Spatial Fusion GAN for Image Synthesis

https://arxiv.org/pdf/1812.05840.pdf



### Generating Text Sequence Images for Recognition

https://arxiv.org/pdf/1901.06782.pdf

Jaderberg等人[1，5]提出的方法是基于字体目录的。
对通过字体,边框和阴影渲染合成的文字图像进行着色和投影形变，然后将经过处理的图像添加到具有某些噪声的背景场景图像中。 Gupta等人[6]提出首先在场景图像上应用语义分割。然后将处理后的单词图像粘贴到其连续区域上，以确保单词不会出现在不同距离的对象上。基于[6]，Zhan等人[7]提出了一种实现语义连贯综合的方法。通过利用在先前的语义分割研究中创建的对象和图像区域的语义注释，在合成文本图像时已经达到了文本和背景之间的语义一致性。这些方法是有效的，但通常需要复杂的先前或后续步骤，例如收集背景图像，对单词进行着色和添加噪声以提高鲁棒性，这需要更多的手动工程。



## OCR/其它

### Separating Content from Style Using Adversarial Learning for Recognizing Text in the Wild

https://arxiv.org/pdf/2001.04189.pdf





## Image Dewarping

### RectiNet:A Gated and Bifurcated Stacked U-Net Module for Document Image Dewarping

https://arxiv.org/pdf/2007.09824.pdf



## 模型解释

### CAM Learning Deep Features for Discriminative Localization

https://arxiv.org/pdf/1512.04150.pdf

1.本文重新审视全局平均池化(GAP)，阐明了GAP赋能卷积神经网络可观的定位能力，尽管网络是在图像级别的标注上训练的；

2.卷积网络的神经元实际上充当目标检测器的角色；但是分类网络中的全连接层使得网络丧失的空间定位能力

3.简单改变一下分类网络,使用GAP替换FC层，即可使得分类网络有弱监督定位能力。



### Grad-CAM:  Visual Explanations from Deep Networks via Gradient-based Localization 

https://arxiv.org/pdf/1610.02391.pdf

1.本文提出的方法Gradient-weighted Class Activation Mapping (Grad-CAM)通过对任意目标类别梯度反传到最后一个卷积层；在图像中突出对预测类别重要的区域；

2.相对于CAM适用更多网络(image classification、image captioning、VQA)；同时也不需要更改网络结构；

3.通过Grad-CAM可以轻松分别哪个网络更强哪个网络更弱。



### Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks

https://arxiv.org/pdf/1710.11063.pdf





## 域适应

### Domain Adaptive Faster R-CNN for Object Detection in the Wild

<https://arxiv.org/pdf/1803.03243>

测试与训练分布不一致有两方面：

a) the image-level shift, such as image style, illumination, 

b) the instance-level shift, such as object appearance, size



### SCL: Towards Accurate Domain Adaptive Object Detection via Gradient Detach Based Stacked Complementary Losses

<https://arxiv.org/pdf/1911.02559v3.pdf>

一般两种方法处理域偏移：a) 在在源域训练,在目标域fine tune;b) 无监督跨域表征

无监督有两个挑战：a) source and target domain data should be embedded into a common space  b) 特征对齐



与COL的的关系：不是交替更新，是同时更新；

Stacked Complementary Losses (SCL) ：辅助损失混合不同域，为检测的域分类器



引用：

Complement objective training：<https://arxiv.org/pdf/1903.01182v1.pdf>







### Unsupervised Domain Adaptation for Object Detection via Cross-Domain Semi-Supervised Learning

<https://arxiv.org/pdf/1911.07158.pdf>





### Self-Guided Adaptation: Progressive Representation Alignment for Domain Adaptive Object Detection

<https://arxiv.org/pdf/2003.08777.pdf>





### Cross-domain Object Detection through Coarse-to-Fine Feature Adaptation

<https://arxiv.org/pdf/2003.10275.pdf>



### WebSOD

<https://arxiv.org/pdf/2003.09790.pdf>



Base class from target domain with bounding box annotations

Novel class from target domain and web with image-level annotations



Object Detector as Object Region Estimator: 在base class上训练，用于检测可能对象

Attentive Classification Loss：网络图像一般单个对象，直接将图像类别赋予边框，来训练faster r-cnn; 使用cam来过滤类别预测错误的边框

Residual Feature Refinement：固定所有层，增加RFR 块，精调target domain和web domain图像。



总结：整体比较复杂，分三个阶段训练，且网络结构有差异





## Few Shot Learning-小样本学习

### LSTD: A Low-Shot Transfer Detector for Object Detection

<https://arxiv.org/pdf/1803.01529.pdf>

当目标域样本很少时，迁移学习有三个挑战

1. 普通的迁移策略不合适，无法消除检测与分类的差异
2. 检测器相对分类更容易过拟合
3. 简单fine tune忽略来自源域和目标域的对象知识，降低可迁移性



创新点：

1：组合SSD和Faster R-CNN；SSD做类别无关的边框回归；Faster R-CNN做coarse-to-fine的分类

2：精调时使用背景抑制正则化，减少复杂背景干扰

3：精调时最后的分类使用源域到目标域的知识蒸馏



第一个小样本迁移学习解决方案；三个创新点都聚焦于减低过拟合。



### Few-shot Object Detection via Feature Reweighting

<https://arxiv.org/pdf/1812.01866.pdf>

meta feature learner：元特征学习器

feature reweighting module：特征重新加权模块；类似通道基本的注意力机制；只在支持图像上训练；输入图像增加mask通道，指示对象区域，多个对象只使用一个

​          This is achieved by setting the reweighting vector for a target class to the average one predicted by the model after taking the k-shot samples as input. 

​          ,we include an additional “mask” channel (Mi) that has only binary values: on the position within the bounding box of an object of interest, the value is 1, otherwise it is 0 (see left-bottom of Fig. 2). If multiple target objects are present on the image, only one object is used



base classes：有大量标注样本

novel classes: 少量标注样本

首先学习base classes；然后在novel classes上精调；

一个问题，reweighting 模块没有显示学习？



### Frustratingly Simple Few-Shot Object Detection

<https://arxiv.org/abs/2003.06957.pdf>

在base classes上训练；在平衡的base classes和novel classes上精调最后几层；精调时采用实例级特征标准化(余弦相似性)

简单的难以置信



### Context-Transformer: Tackling Object Confusion for Few-Shot Detection

<https://arxiv.org/pdf/2003.07304v1.pdf>

小样本迁移学习通常定位好，但分类困难；定位是类别无关的；

通过环境信息可以帮助辨别类别；Context-Transformer 包括两部分：affinity discovery 和context aggregation 



affinity discovery：先验框和环境亲和度

context aggregation : 亲和度矩阵与环境向量相乘；获取先验框的环境向量加权结果作为环境信息



## 模型压缩、量化

### Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM

https://arxiv.org/pdf/1707.09870.pdf

mixed integer programs 

extragradient 

ADMM 

BWN 

These results suggest that we should quantize different parts of the networks with different bit width in practice 





## 边缘检测

### HED Holistically-Nested Edge Detection

<https://arxiv.org/pdf/1504.06375.pdf>

速度太慢：e.g., 0.4 seconds using GPU and 12 seconds using CPU



### STEAL Devil is in the Edges: Learning Semantic Boundaries from Noisy Annotations

<https://arxiv.org/pdf/1904.07934.pdf>

### 

### 



## 表格提取

### GFTE: Graph-based Financial Table Extraction

<https://arxiv.org/pdf/2003.07560.pdf>





## 半监督

### Semi-Supervised Learning with Ladder Networks

<https://arxiv.org/pdf/1507.02672.pdf>



### TEMPORAL ENSEMBLING FOR SEMI-SUPERVISED LEARNING

<https://arxiv.org/pdf/1610.02242.pdf>



### Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results

<https://arxiv.org/pdf/1703.01780.pdf>





## 风格迁移

### Texture Synthesis Using Convolutional Neural Networks



### A Neural Algorithm of Artistic Style

<https://arxiv.org/pdf/1508.06576.pdf>



### Image Style Transfer Using Convolutional Neural Networks

<https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf>



### INCORPORATING LONG-RANGE CONSISTENCY IN CNN-BASED TEXTURE GENERATION

<https://arxiv.org/pdf/1606.01286.pdf>



### Neural Style Transfer: A Review

<https://arxiv.org/pdf/1705.04058v7.pdf>



### Demystifying Neural Style Transfer

<https://arxiv.org/pdf/1701.01036.pdf>





### Laplacian-Steered Neural Style Transfer

<https://arxiv.org/pdf/1707.01253.pdf>





### Arbitrary Style Transfer with Deep Feature Reshuffle

<https://arxiv.org/pdf/1805.04103.pdf>



### Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis

<https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Li_Combining_Markov_Random_CVPR_2016_paper.pdf>



### Perceptual Losses for Real-Time Style Transfer and Super-Resolution

<https://arxiv.org/pdf/1603.08155.pdf>





### Texture Networks: Feed-forward Synthesis of Textures and Stylized Images

<http://proceedings.mlr.press/v48/ulyanov16.pdf>



### Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks

<https://arxiv.org/pdf/1604.04382.pdf>



### **Image-to-Image Translation with Conditional Adversarial Networks** 



### A Learned Representation for Artistic Style



### StyleBank: An Explicit Representation for Neural Image Style Transfer



### Multi-style Generative Network for Real-time Transfer



### Diversified Texture Synthesis With Feed-Forward Networks



### Fast Patch-based Style Transfer of Arbitrary Style



### Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization



### Exploring the Structure of a Real-time, Arbitrary Neural Artistic Stylization Network



### Universal Style Transfer via Feature Transforms



### Meta Networks for Neural Style Transfer

<https://arxiv.org/pdf/1709.04111.pdf>







## 其它

视觉会议基本参考

PAMI/IJCV/JMLR 4 分；

TIP/TNNLS/TCSVT 2 分；

CVPR/ICCV/ECCV/NIPS/ICML 1.5 分

PR/CVIU/PRL/neurocomputing/AAAI/IJCAI 1 分