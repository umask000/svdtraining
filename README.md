# $0$ 序言

最终笔者决定抽空把最近的工作做一个回顾总结。本文是对以下两篇博客工作的进一步延申探讨：

1. [【数值分析×机器学习】以SVD的分解形式进行深度神经网络的训练](https://caoyang.blog.csdn.net/article/details/116707855)
2. [【论文实现】以SVD的分解形式进行深度神经网络的训练（PyTorch）](https://caoyang.blog.csdn.net/article/details/117391702)

前面的工作笔者不再赘述，有兴趣地可以仔细阅读以上两文，其中第一篇是论文笔注，第二篇则是对该文的一个初步实现，后续的工作基本围绕第二篇博客中的实现展开，完整代码已上传[$\text{GitHub@SVDtraining}$](https://github.com/umask000/svdtraining)，由于目前原论文作者仍然没有提供其论文实现的源码，因此笔者在实现上可能仍然存在一些问题，详细的实现思路在第二篇博客中已经阐述，本文将不再重述代码细节。

第$1,2,4$节中则是对笔者在使用$\text{PyTorch}$对论文实现中遇到的一些小$\text{trick}$做一个记录，笔者觉得这些技巧应该还是对初学者比较有帮助的。<font color=red>本文的重点内容在第$3$节</font>，在这一部分中，笔者将着重于解析笔者在原文基础上提出的**在线剪枝**方法的实现，以及基于笔者在实验的结果，提出一些可能算是创新点的愚见，权当抛砖引玉，愿诸君不吝赐教。

----

[toc]

----

# $1$ $\rm PyTorch$模型参数分组

## $1.1$ 为什么需要模型参数分组？

可能一些朋友还不太理解为什么需要对神经网络模型中的参数进行分组，不妨思考以下几种场景：

1. 需要对不同的参数使用不同的优化策略，如可以为不同参数设置不同的学习率（步长），典型例子是对敏感性较高的模型参数可以赋予较小的学习率（步长），如在$\text{SVD training}$中的因为**左（右）奇异向量矩阵**（$\text{left(right) singular vector matrix}$）理论上要求严格满足正交约束，这类参数的敏感性就非常高，需要较小的学习率（步长）进行谨慎地微调。
2. 需要对不同的参数应用不同的损失函数，典型的例子是损失函数中带有与参数相关的正则项，而不同参数的正则项的权重是不相同。尤其当这个正则项并非传统的$\text{L1}$或$\text{L2}$正则项（因为这些正则项的权重可以直接在$\text{PyTorch}$优化器中设置参数$\text{weight_decay}$即可），而是自定义的正则项（此时需要重写损失函数），如原论文中的**奇异向量的正交正则器**（$\text{Singular vectors orthogonality regularizer}$）与**奇异值稀疏导出正则器**（$\text{Singular values sparsity-inducing regularizer}$）。
3. 在模型训练过程中可能需要基于模型参数的实际情况，引入一些自定义的逻辑对不同类型的模型参数进行人工调整。这就是本文第$3$节中**在线剪枝**中所提出的方案。

$\rm OK$，笔者想应该对模型参数分组已经有了一个初步的认识，接下来通过举例说明上述第$1$点（优化器中的参数分组）与第$2$点（损失函数中的参数分组），其中$\rm PyTorch$优化器的参数就提供了分组功能而相对容易，但是$\rm PyTorch$损失函数中的参数分组其实存在一些比较难以察觉的陷阱。

## $1.2$ $\rm PyTorch$优化器中的模型参数分组

关于$\rm PyTorch$优化器中的模型参数分组可以参考[官方文档](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/)，以`torch.optim.SGD`随机梯度下降优化器为例，我们首先可以通过调用`model.named_parameters()`生成器来获取一个$\rm PyTorch$模型中的所有参数及其名称，以`torchvision.models.resnet`模块中自带的$\text{ResNet18}$模型为例：

```python
from torchvision.models import resnet

model = resnet.resnet18()
for name, parameter in model.named_parameters():
    print(name, parameter.shape)
```

输出结果：

```shell
conv1.weight torch.Size([64, 3, 7, 7])
bn1.weight torch.Size([64])
bn1.bias torch.Size([64])
layer1.0.conv1.weight torch.Size([64, 64, 3, 3])
layer1.0.bn1.weight torch.Size([64])
layer1.0.bn1.bias torch.Size([64])
layer1.0.conv2.weight torch.Size([64, 64, 3, 3])
layer1.0.bn2.weight torch.Size([64])
layer1.0.bn2.bias torch.Size([64])
layer1.1.conv1.weight torch.Size([64, 64, 3, 3])
layer1.1.bn1.weight torch.Size([64])
layer1.1.bn1.bias torch.Size([64])
layer1.1.conv2.weight torch.Size([64, 64, 3, 3])
layer1.1.bn2.weight torch.Size([64])
layer1.1.bn2.bias torch.Size([64])
layer2.0.conv1.weight torch.Size([128, 64, 3, 3])
layer2.0.bn1.weight torch.Size([128])
layer2.0.bn1.bias torch.Size([128])
layer2.0.conv2.weight torch.Size([128, 128, 3, 3])
layer2.0.bn2.weight torch.Size([128])
layer2.0.bn2.bias torch.Size([128])
layer2.0.downsample.0.weight torch.Size([128, 64, 1, 1])
layer2.0.downsample.1.weight torch.Size([128])
layer2.0.downsample.1.bias torch.Size([128])
layer2.1.conv1.weight torch.Size([128, 128, 3, 3])
layer2.1.bn1.weight torch.Size([128])
layer2.1.bn1.bias torch.Size([128])
layer2.1.conv2.weight torch.Size([128, 128, 3, 3])
layer2.1.bn2.weight torch.Size([128])
layer2.1.bn2.bias torch.Size([128])
layer3.0.conv1.weight torch.Size([256, 128, 3, 3])
layer3.0.bn1.weight torch.Size([256])
layer3.0.bn1.bias torch.Size([256])
layer3.0.conv2.weight torch.Size([256, 256, 3, 3])
layer3.0.bn2.weight torch.Size([256])
layer3.0.bn2.bias torch.Size([256])
layer3.0.downsample.0.weight torch.Size([256, 128, 1, 1])
layer3.0.downsample.1.weight torch.Size([256])
layer3.0.downsample.1.bias torch.Size([256])
layer3.1.conv1.weight torch.Size([256, 256, 3, 3])
layer3.1.bn1.weight torch.Size([256])
layer3.1.bn1.bias torch.Size([256])
layer3.1.conv2.weight torch.Size([256, 256, 3, 3])
layer3.1.bn2.weight torch.Size([256])
layer3.1.bn2.bias torch.Size([256])
layer4.0.conv1.weight torch.Size([512, 256, 3, 3])
layer4.0.bn1.weight torch.Size([512])
layer4.0.bn1.bias torch.Size([512])
layer4.0.conv2.weight torch.Size([512, 512, 3, 3])
layer4.0.bn2.weight torch.Size([512])
layer4.0.bn2.bias torch.Size([512])
layer4.0.downsample.0.weight torch.Size([512, 256, 1, 1])
layer4.0.downsample.1.weight torch.Size([512])
layer4.0.downsample.1.bias torch.Size([512])
layer4.1.conv1.weight torch.Size([512, 512, 3, 3])
layer4.1.bn1.weight torch.Size([512])
layer4.1.bn1.bias torch.Size([512])
layer4.1.conv2.weight torch.Size([512, 512, 3, 3])
layer4.1.bn2.weight torch.Size([512])
layer4.1.bn2.bias torch.Size([512])
fc.weight torch.Size([1000, 512])
fc.bias torch.Size([1000])
```

确实是有许多参数，比如现在笔者想将上述这些参数按照名称以`weight`结尾或`bias`结尾分为两组，然后赋予他们不同的随机梯度下降优化的策略：

```python
from torch import optim
from torchvision.models import resnet

weight_params = []
bias_params = []

model = resnet.resnet18()
for name, parameter in model.named_parameters():
    suffix = name.split('.')[-1]
    if suffix == 'weight':
        weight_params.append(parameter)
    elif suffix == 'bias':
        bias_params.append(parameter)
        
optimizer = optim.SGD([{'params': weight_params, 'lr': 1e-2, 'momentum':.9}, 
                       {'params': bias_params, 'lr': 1e-3, 'weight_decay':.1}],
                      lr=1e-2,
                      momentum=.5,
                      weight_decay=1e-2)
```

注意在`optim.SGD`中除了为每个组别的模型参数赋予了不同的学习率`lr`，动量因子`momentum`，以及权重衰减`weight_decay`外，仍然设置学习率`lr=1e-2`，动量因子`momentum=.5`，以及权重衰减`weight_decay=1e-2`。这些可以理解为缺省值：

- `weight_params`没有设置`weight_decay`，则默认为`1e-2`；

- `bias_params`没有设置`momentum`，则默认为`.5`；

- 倘若`weight_params`与`bias_params`未能覆盖所有模型参数（当然上例中模型参数不存在未分组的情形），剩余模型参数的优化默认值即`lr=1e-2, momentum=.5, weight_decay=1e-2`。

其实上述这些参数名称其实都是$\text{PyTorch}$中默认给出的，比如：

1. `conv1.weight`其实对应模块`model.conv1.weight1`；

2. `layer1.0.conv1.weight`对应模块`model.layer1[0].conv1.weight`；

3. `layer3.0.downsample.0.weight`对应模块`model.layer3[0].downsample[0].weight`；

根据这种默认的命名规则，笔者当然可以写一个函数来根据参数名称直接调用到对应的参数变量（<font color=red>详见第$3$节</font>），可能有人会疑惑使用`for name, parameter in model.named_parameters(): ...`时，不就已经可以通过`parameter`调用参数名称为`name`的变量了嘛，为什么不能这样调用的原因笔者在第$3$节中将具体说明。

<font color=red>笔者想吐槽的问题是</font>，为什么$\rm PyTorch$没有开放给模型参数自主命名，然后通过自主命名直接调用到模型参数的方法呢？明明$\text{PyTorch}$里都可以对`torch.tensor`的张量型变量命名，实话说有点无语。

本小节告一段落，接下来让我们来看看非常具有迷惑性的$\text{PyTorch}$损失函数中的模型参数分组。

## $1.3$ $\text{PyTorch}$损失函数中的模型参数分组

$\text{PyTorch}$损失函数中的模型参数分组仍然可以用相同的方法，即通过`model.named_parameters()`生成器来实现，以这里以$\text{SVD training}$实现中的带**奇异向量的正交正则器**（$\text{Singular vectors orthogonality regularizer}$）与**奇异值稀疏导出正则器**（$\text{Singular values sparsity-inducing regularizer}$）的交叉熵损失函数代码为例，第一版的实现如下：

```python
# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn

import torch
from torch.nn import functional as F

class CrossEntropyLossSVD(torch.nn.CrossEntropyLoss):

	def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
		super(CrossEntropyLossSVD, self).__init__(
			weight=weight,
			size_average=size_average,
			ignore_index=ignore_index,
			reduce=reduce,
			reduction=reduction,
		)

	def forward(self, input, target, model=None, regularizer_weights=[1, 1], orthogonal_suffix='svd_weight_matrix', sparse_suffix='svd_weight_vector', mode='lh') -> torch.FloatTensor:
		cross_entropy_loss = F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
		if model is None:
			return cross_entropy_loss
		
		# 正交正则器
		def _orthogonality_regularizer(x):		 		 				 # x应当是一个2D张量（矩阵）且高大于宽
			return torch.norm(torch.mm(x.t(), x) - torch.eye(x.shape[1]).cuda(), p='fro') / x.shape[1] / x.shape[1]
		
		# 稀疏导出正则器
		def _sparsity_inducing_regularizer(x, mode='lh'):				 # x应当是一个1D张量（向量）
			mode = mode.lower()
			if mode == 'lh':
				return torch.norm(x, 1) / torch.norm(x, 2)	
			elif model == 'l1':
				return torch.norm(x, 1)
			raise Exception(f'Unknown mode: {mode}')
	
		regularizer = torch.zeros(1, ).cuda()
		for name, parameter in model.named_parameters():
			lastname = name.split('.')[-1]
			if lastname.startswith(orthogonal_suffix):					 # 奇异向量矩阵参数：添加正交正则项
				regularizer += _orthogonality_regularizer(parameter) * regularizer_weights[0]
			elif lastname.startswith(sparse_suffix):					 # 奇异值向量参数：添加稀疏导出正则项
				regularizer += _sparsity_inducing_regularizer(parameter, mode) * regularizer_weights[1]
		return cross_entropy_loss + regularizer
```

可以看到笔者是在`forward`函数中进行模型参数分组的，并将`model`作为参数传入了`forward`函数。然后笔者就发现这样写会导致损失函数的计算特别地慢，后来发现其实与优化器相同，也可以直接将两类参数存成`list`传入，这样就不需要将`model`作为参数传入且无需每次计算损失函数时都执行一次模型参数分组逻辑。

改良后的写法如下所示：

```python
# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn

import torch
from torch import nn
from torch.nn import functional as F

class CrossEntropyLossSVD(nn.CrossEntropyLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(CrossEntropyLossSVD, self).__init__(weight=weight,
                                                  size_average=size_average,
                                                  ignore_index=ignore_index,
                                                  reduce=reduce,
                                                  reduction=reduction)

    def forward(self,
                input,
                target,
                orthogonal_params,
                sparsity_params,
                orthogonal_regularizer_weight=1.,
                sparsity_regularizer_weight=1.,
                device='cuda',
                mode='lh'):
        cross_entropy_loss = F.cross_entropy(input,
                                             target,
                                             weight=self.weight,
                                             ignore_index=self.ignore_index,
                                             reduction=self.reduction)

        # Regularizer calculation
        regularizer = torch.zeros(1, ).to(device)

        if not orthogonal_regularizer_weight == .0:

            def _orthogonality_regularizer(x):
                r = x.shape[1]
                return torch.norm(torch.mm(x.t(), x) - torch.eye(r).to(device), p='fro') / r / r

            for orthogonal_param in orthogonal_params:
                regularizer += _orthogonality_regularizer(orthogonal_param) * orthogonal_regularizer_weight

        if not sparsity_regularizer_weight == .0:

            def _sparsity_inducing_regularizer(x, mode='lh'):
                if mode == 'lh':
                    return torch.norm(x, 1) / torch.norm(x, 2)
                elif model == 'l1':
                    return torch.norm(x, 1)
                elif model == 'l2':
                    return torch.norm(x, 2)
                raise Exception(f'Unknown mode: {mode}')

            for sparsity_param in sparsity_params:
                regularizer += _sparsity_inducing_regularizer(sparsity_param, mode=mode) * sparsity_regularizer_weight

        return cross_entropy_loss + regularizer
```

其实就是在`forward`函数中添加两个模型参数分组的列表（`orthogonal_params`与`sparse_params`）。

有人可能会觉得`orthogonal_params`与`sparse_params`中存储的模型参数是不是已经和原模型没有关系了（类似地，上面生成`weight_params`与`bias_params`时是否也有疑惑？），因为这些参数在训练中是会发生变化的，那么我们在模型训练前就生成好的`orthogonal_params`与`sparse_params`（同理`weight_params`与`bias_params`）中的变量值也会随之一起发生变化吗？

答案是肯定的，具体可以参考下面这个简单的示例，本质他们还是共享地址，只是可能写$\text{Python}$时的人都比较懒，不太能注意得到这些细节（<font color=red>说的就是笔者自己</font>）：

```python
class Graph:
    def __init__(self):
        self.vertex = [0, 1, 2, 3]
        self.edges = [(0, 1), (0, 2), (0, 3)]
        self.matrix = [
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
graph = Graph()

# 生产参数列表
params = []
params.append(graph.vertex)
params.append(graph.edges)
params.append(graph.matrix)

# 修改列表中的变量，类成员变量值会发生变化
params[0].append(4)
params[1].append((0, 4))
params[2][0].append(1)
params[2][1].insert(0, 0)
params[2][2].insert(0, 0)
params[2][3].insert(0, 0)
params[2].append([0, 0, 0, 0, 0])

print(graph.vertex)
print(graph.edges)
print(graph.matrix)

# 修改类成员变量，列表中的变量值同样会发生变化
graph.vertex[-1] = 5
print(params[0])

# 输出结果
"""
[0, 1, 2, 3, 4]
[(0, 1), (0, 2), (0, 3), (0, 4)]
[[0, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
[0, 1, 2, 3, 5]
"""
```

但是如果将上述代码倒数第二行`graph.vertex[-1] = 5`直接改成`graph.vertex = graph.vertex[:-1] + [5]`，则无法输出得到`[0, 1, 2, 3, 5]`，而仍为`[0, 1, 2, 3, 4]`，这其实也就是为什么在第$3$节里笔者没有使用`for name, parameter in model.named_parameters(): ...`的方法来进行**在线剪枝**的原因了。

# $2$ $\rm PyTorch$优化器的进阶用法

不知道为什么官方文档中关于`torch.optim`的内容写得特别少，很多有用的模块都没有提到。这里笔者先提一个点，以后如果还遇到什么其他用法可能会更新在本节中。

比如我们想要使用带衰减的学习率（步长），这时候就需要用到`lr_scheduler`，一个简单的使用方法如下所示：

```python
from torchvision.models import resnet
from torch.optim.lr_scheduler import LambdaLR

initial_lr = 0.1

model = resnet.resnet18()
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
scheduler = LambdaLR(optimizer_1, lr_lambda=lambda epoch: 1 / (epoch + 1))
print(f'Initial learning rate: {optimizer_1.defaults["lr"]}')

for epoch in range(1, 11):
    optimizer.zero_grad()
    optimizer.step()
    print(f'Learning rate of epoch {epoch}: {optimizer.param_groups[0]["lr"]}')
    scheduler.step()
```

输出结果：

```shell
Initial learning rate: 0.1
Learning rate of epoch 1: 0.1
Learning rate of epoch 2: 0.05
Learning rate of epoch 3: 0.03333333333333333
Learning rate of epoch 4: 0.025
Learning rate of epoch 5: 0.020000000000000004
Learning rate of epoch 6: 0.016666666666666666
Learning rate of epoch 7: 0.014285714285714285
Learning rate of epoch 8: 0.0125
Learning rate of epoch 9: 0.011111111111111112
Learning rate of epoch 10: 0.010000000000000002
```

其实笔者之前一直以为`weight_decay`是学习率的衰减因子，真是贻笑大方了。另外提一个非常经典的$\text{shuffleNet}$系列中使用的学习率变化策略如下所示：

```python
import torch 

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1.0-step/args.total_iters) if step <= args.total_iters else 0, last_epoch=-1)
```

注意不管是`optimizer`还是`scheduler`，都和`model`一样可以调用`.state_dict()`方法得到其所包的含的状态参数，并可以一起随模型参数存储到外部的`.pth`文件，然后通过`load_state_dict()`方法即可复原`optimizer`，`scheduler`的状态，从而可以实现无信息损失的断点训练，这个将在本文第$4$节中详述。

# $3$ $\text{SVD training}$在线剪枝及一些思考

<font color=red>前排提示：</font>==本节内容与序言中的两篇博客关系紧密，并且很多都是笔者的**胡言乱语**，可能并不是那么容易看懂。==

## $3.1$ 关于在线剪枝的逻辑

事实上在[【论文实现】以SVD的分解形式进行深度神经网络的训练（PyTorch）](https://caoyang.blog.csdn.net/article/details/117391702)中笔者改写的是他人$\text{DIY}$的$\rm ResNet18$，后来笔者发现`torchvision.models.resnet`源码里的写法与上文里摘用的他人代码是有区别的，而且训练结果也有很大差异，所以在[$\text{GitHub@SVDtraining}$](https://github.com/umask000/svdtraining)里笔者是把``torchvision.models.resnet`源码按照这篇博客里的方法改写成$\text{SVD training}$的形式了（见`src/models/svd_resnet.py`文件）。

在上文中，笔者遇到一件似乎有些反常识的问题，<font color=red>即$\text{SVD training}$的训练速度要比原模型慢得多</font>。

直觉上降秩的模型应该会更快，这就是思维定式了。以全连接层的权重参数$W\in\R^{m\times n }$为例，其$\rm SVD$形式：
$$
W=U\text{diag}(s)V^\top\tag{1}
$$
其中$U\in\R^{m\times r},s\in\R^r,V\in\R^{n\times r}$，参数总数将会由$mn$变为$r(m+n+1)$，在满秩$\text{SVD}$的情况下有$r=\min\{m,n\}$，显然参数总数增加了一倍不止，模型训练自然会变慢。这还不算损失函数中正则项的计算时间，事实上损失函数里计算正则项的时间是不容忽视的，对比实验结果显示这部分时间占用甚至会占到$\text{50%}$以上的训练总耗时，当然这不排除在第$2.2$节中提到的带正则项的交叉熵损失函数的写法仍有优化空间，这是后话。

所以原论文提出的$\text{SVD training}$的具体流程就是先满秩训练，然后在后处理中对模型剪枝微调，所以模型训练会更加耗时，但是使用模型进行预测的速度会加快（因为剪完枝后的模型参数量大大减少）。即$\text{SVD training}$优化的是<font color=red>模型部署速度</font>，而非<font color=red>模型构建速度</font>。

于是笔者指出是否可以不使用满秩训练，即定义模型时忽略$r=\min\{m,n\}$，而是在训练时就设置一个较小的$r$，如$r=\min\{m,n\}/16$或直接定义一个较小的常量值？在上个月底汇报$\rm DDL$紧张的情况下笔者就这两种情形做了一些测试，结果如下（所用代码是上文的版本）：

| $\lambda_o$ | $\lambda_s$ | $r$                     | 平均每个训练$\text{epoch}$耗时 | 训练终止$\text{epoch}$数 | 测试集精确度 |
| ----------- | ----------- | ----------------------- | ------------------------------ | ------------------------ | ------------ |
| $0.01$      | $0.01$      | $4$                     | $232.46$                       | $53$                     | $35.11\%$    |
| $0.01$      | $0.01$      | $8$                     | $277.43$                       | $50$                     | $45.14\%$    |
| $0.01$      | $0.01$      | $16$                    | $300.11$                       | $45$                     | $55.72\%$    |
| $0.01$      | $0.01$      | $32$                    | $423.36$                       | $40$                     | $66.75\%$    |
| $0.01$      | $0.01$      | $\min\{m,n\}/16$        | $400.43$                       | $40$                     | $63.25\%$    |
| $0.1$       | $0.1$       | $4$                     | $235.11$ | $60$ | $34.21\%$    |
| $0.1$       | $0.1$       | $8$                     | $265.59$ | $55$ | $40.55\%$    |
| $0.1$       | $0.1$       | $16$                    | $296.71$ | $52$ | $57.79\%$    |
| $0.1$       | $0.1$       | $32$                    | $430.31$ | $50$ | $61.47\%$    |
| $0.1$       | $0.1$       | $\min\{m,n\}/16$        | $401.54$ | $50$ | $61.22\%$    |
| $1$         | $1$         | $4$                     | $239.44$ | $70$ | $35.11\%$    |
| $1$         | $1$         | $8$                     | $265.14$ | $65$ | $45.14\%$    |
| $1$         | $1$         | $16$                    | $302.46$ | $60$ | $55.72\%$    |
| $1$         | $1$         | $32$                    | $431.62$ | $55$ | $66.75\%$         |
| $1$         | $1$         | $\min\{m,n\}/16$        | $412.44$ | $55$ | $63.25\%$    |
| $0$         | $0$         | $\text{Baseline model}$ | $318.25$                       | $35$                     | $80.25\%$    |

会发现其实模型评估的减损非常大，且训练耗时在$r$取到$16$时基本上就和原模型（$\text{baseline model}$）在训练耗时上相仿了。

于是后来笔者借鉴学习率衰减的思路，<font color=red>试图隔一段$\text{epoch}$后，人工对模型的秩进行减约</font>，具体逻辑如下：

1. 每隔一段$\rm epoch$，检查模型中所有的奇异值向量参数（即式$(1)$中的$s$），按照如下三种可能的策略进行剪枝：

   - 选择一个常数$c$，找到参数$s$中绝对值最小的$c$个元素删除；
   - 选择一个阈值$\rm threshold$，删除参数$s$中绝对值小于$\text{threshold}$的所有元素；
   - 选择一个固定比率$p$，删除参数$s$中绝对值最小的占比为$p$的若干元素；

   为了防止剪枝时参数$s$被减得过小，可以设置下限值。

2. 根据参数$s$中删除元素的位置，对应删除参数$U$和$V$中对应的列向量。

这一段逻辑的实现在[$\text{GitHub@SVDtraining}$](https://github.com/umask000/svdtraining)中的`src/utils.py`中的`svd_layer_prune`函数中：

```python
def svd_layer_prune(layer,
                    prune_by=2,
                    threshold=1e-6,
                    reduce_by=.98,
                    min_rank=1,
                    min_decay=.0):
    current_rank = layer.singular_value_vector.shape[0]
    min_rank = max(min_rank, math.ceil(current_rank * min_decay))                                                       # minimum rank reduced is determined by the maximum between ratio and absolute value

    # Determine the index remained after pruning
    remaining_index = []
    prune_to = current_rank
    if prune_by is not None:
        prune_to = max(current_rank - prune_by, min_rank)
    elif threshold is not None:
        for i, singular_value in enumerate(layer.singular_value_vector):
            if abs(singular_value) <= threshold:
                remaining_index.append(i)
        if len(remaining_index) < min_rank:
            warnings.warn(f'Threshold strategy leads to rank less than {min_rank}!')
            prune_to = min_rank
            remaining_index = []
    elif reduce_by is not None:
        prune_to = max(math.floor(current_rank * reduce_by), min_rank)
    else:
        raise Exception('Pruning strategy is not specified!')

    if not remaining_index:
        sorted_index, _ = get_sorted_index(torch.abs(layer.singular_value_vector))
        for i in range(current_rank):
            if sorted_index[i] < prune_to:
                remaining_index.append(i)

    # Pruning
    layer.singular_value_vector = nn.Parameter(layer.singular_value_vector[remaining_index])
    layer.left_singular_matrix = nn.Parameter(layer.left_singular_matrix[:, remaining_index])
    layer.right_singular_matrix = nn.Parameter(layer.right_singular_matrix[:, remaining_index])
```

## $3.2$ 关于在线剪枝中的实现细节

这一块的细节非常非常的多，笔者面面俱到，首先提剪枝会造成的问题，考察`src/train.py`中的代码：

```python
# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

if __name__ == '__main__':
    import sys
    sys.path.append('../')


import time
import torch
import logging

from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision.models import resnet


from config import ModelConfig
from src.data import load_cifar
from src.models import svd_resnet
from src.svd_loss import CrossEntropyLossSVD
from src.svd_layer import Conv2dSVD, LinearSVD
from src.utils import save_args, summary_detail, initialize_logging, load_args, svd_layer_prune

def train(args):
    if __name__ == '__main__':
        ckpt_root = '../ckpt/'                                                                                          # model checkpoint path
        logging_root = '../logging/'                                                                                    # logging saving path
        data_root = '../data/'                                                                                          # dataset saving path
    else:
        ckpt_root = 'ckpt/'                                                                                             # model checkpoint path
        logging_root = 'logging/'                                                                                       # logging saving path
        data_root = 'data/'                                                                                             # dataset saving path

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, model_name = svd_resnet.resnet18()
    model = model.to(device)

    # Initialize logging
    initialize_logging(filename=f'{logging_root}{model_name}.log', filemode='w')
    save_args(args, logging_root + model_name + '.json')

    # Group model parameters
    orthogonal_params = []
    sparsity_params = []
    for name, parameter in model.named_parameters():
        lastname = name.split('.')[-1]
        if lastname == 'left_singular_matrix' or lastname == 'right_singular_matrix':
            orthogonal_params.append(parameter)
        elif lastname == 'singular_value_vector':
            sparsity_params.append(parameter)

    # Group modules
    svd_module_names = []
    svd_module_expressions = []
    for name, modules in model.named_modules():
        if isinstance(modules, Conv2dSVD) or isinstance(modules, LinearSVD):
            svd_module_names.append(name)
            expression = 'model'
            for character in name.split('.'):
                if character.isdigit():
                    expression += f'[{character}]'
                else:
                    expression += f'.{character}'
            svd_module_expressions.append(expression)

    # Define loss function and optimizer
    loss = CrossEntropyLossSVD()
    optimizer = args.optimizer([{'params': orthogonal_params, 'lr': args.orthogonal_learning_rate, 'momentum': args.orthogonal_momentum, 'weight_decay': args.orthogonal_weight_decay},
                           {'params': sparsity_params, 'lr': args.sparsity_learning_rate, 'momentum': args.sparsity_momentum, 'weight_decay': args.sparsity_weight_decay}],
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    # Load dataset
    trainloader, testloader = load_cifar(root=data_root, download=False, batch_size=args.batch_size)
    num_batches = len(trainloader)
    if args.summary:
        for i, data in enumerate(trainloader, 0):
            input_size = data[0].shape
            summary_detail(model, input_size=input_size)
            break

    for epoch in range(args.max_epoch):
        # Train
        epoch_start_time = time.time()
        model.train()
        total_losses = 0.
        correct_count = 0
        total_count = 0
        for i, data in enumerate(trainloader, 0):
            X_train, y_train = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            y_prob = model(X_train)
            loss_value = loss(y_prob,
                              y_train,
                              orthogonal_params,
                              sparsity_params,
                              orthogonal_regularizer_weight=args.orthogonal_regularizer_weight,
                              sparsity_regularizer_weight=args.sparsity_regularizer_weight,
                              device=device)
            loss_value.backward()
            optimizer.step()
            total_losses += loss_value.item()
            _, y_pred = torch.max(y_prob.data, 1)
            total_count += y_train.size(0)
            correct_count += (y_pred == y_train).sum()
            train_accuracy = 100. * correct_count / total_count
            logging.debug('[Epoch:%d, Iteration:%d] Loss: %.03f | Train accuarcy: %.3f%%' % (epoch + 1,
                                                                                             i + 1 + epoch * num_batches,
                                                                                             total_losses / (i + 1),
                                                                                             train_accuracy))
        epoch_end_time = time.time()

        # Prune
        logging.info('Pruning ...')

        if args.svd_prune and ((epoch + 1) % args.svd_prune_cycle == 0):
            with torch.no_grad():
                for expression in svd_module_expressions:
                    svd_layer_prune(eval(expression),
                                    prune_by=args.svd_prune_rank_each_time,
                                    threshold=args.svd_prune_threshold,
                                    reduce_by=args.svd_prune_decay,
                                    min_rank=args.svd_prune_min_rank,
                                    min_decay=args.svd_prune_min_decay)
                orthogonal_params = []
                sparsity_params = []
                for name, parameter in model.named_parameters():
                    lastname = name.split('.')[-1]
                    if lastname == 'left_singular_matrix' or lastname == 'right_singular_matrix':
                        orthogonal_params.append(parameter)
                    elif lastname == 'singular_value_vector':
                        sparsity_params.append(parameter)
                    optimizer = args.optimizer([{'params': orthogonal_params, 'lr': args.orthogonal_learning_rate,
                                                 'momentum': args.orthogonal_momentum,
                                                 'weight_decay': args.orthogonal_weight_decay},
                                                {'params': sparsity_params, 'lr': args.sparsity_learning_rate,
                                                 'momentum': args.sparsity_momentum, 'weight_decay': args.sparsity_weight_decay}],
                                               lr=args.learning_rate,
                                               momentum=args.momentum,
                                               weight_decay=args.weight_decay)
        # Test
        logging.info('Waiting Test ...')
        model.eval()
        with torch.no_grad():
            correct_count = 0
            total_count = 0
            for data in testloader:
                X_test, y_test = data[0].to(device), data[1].to(device)
                y_prob = model(X_test)
                _, y_pred = torch.max(y_prob.data, 1)
                total_count += y_test.size(0)
                correct_count += (y_pred == y_test).sum()
            test_accuracy = 100. * correct_count / total_count
            logging.info('EPOCH=%03d | Accuracy=%.3f%%, Time=%.3f' % (epoch + 1, test_accuracy, epoch_end_time - epoch_start_time))

            # Save model to checkpoints
            if (epoch + 1) % args.ckpt_cycle == 0:
                logging.info('Saving model ...')
                torch.save(model.state_dict(), ckpt_root + model_name + '_%03d.pth' % (epoch + 1))


if __name__ == '__main__':
    args = load_args(ModelConfig)
    args.orthogonal_regularizer_weight = 1.0
    args.sparsity_regularizer_weight = 1.0
    train(args)
```

上面代码中有如下几个点值得注意：

1. 第$58$行道$68$行笔者做了一次**模块分组**，即本文第$1.2$节中提到的根据参数（或模块）名称直接调用到对应参数（或模块）的方法，即实现下面的逻辑：

   - `conv1.weight`其实对应模块`model.conv1.weight1`；

   - `layer1.0.conv1.weight`对应模块`model.layer1[0].conv1.weight`；

   - `layer3.0.downsample.0.weight`对应模块`model.layer3[0].downsample[0].weight`；

   然后通过`eval`函数**转译**字符串将对应的`layer`传入本文第$3.1$节中提到的`svd_layer_prune`函数中。

   为什么要做这么繁琐的工作呢？还记得本文第$1.3$节最后提到的那个例子嘛，剪枝本身是要改变参数形状的！所以如果在`model.named_modules()`中提供的`module`上直接修改参数，结果就是什么事情也不会发生，就跟第$1.3$节最后`graph.vertex[-1] = 5`直接改成`graph.vertex = graph.vertex[:-1] + [5]`，则无法输出得到`[0, 1, 2, 3, 5]`，而仍为`[0, 1, 2, 3, 4]`的道理是一样的。

2. 第$119$行道$146$行中是剪枝的逻辑，注意这里每次剪完枝，都要重新对模型参数分一次组，然后重置`optimizer`，否则你就会发现从此剪完枝的模型参数再也不会发生变动了。原因很简单，剪完枝后模型参数已经发生了硬件地址上的变化，所以`orthogonal_params`和`sparsity_params`中存储的地址已经失效了，从此优化器和模型就半毛钱关系都没有了。这个亏坑了笔者很久，因为一开始确实没有注意到这个问题，也很难发现模型参数没有发生变化（因为只是剪枝了与$\rm SVD$相关的参数，其他参数还是在变的，如果不调试就很难察觉得到这个问题）。

3. 笔者在写本文时突然发现优化器里只放了`orthogonal_params`和`sparsity_params`，没有放其他参数，但是最近的一个测试基本上达到了和原模型同等的效果，笔者之后改了再试试，晕到死，真是醉了。

暂且先提这些问题，感觉还有一些工作可以精进一下。


## $3.3$  关于笔者的一些其他思考

其实笔者遇到很多很奇怪的问题，最困扰的就是撤了损失函数里的正则项后，模型跑得又快又好，似乎原论文中提到的两个正则项并没有起到实际的意义。

笔者测试的是第一种固定常数减秩的策略，选取的默认值为$2$，每隔$4$个$\text{epoch}$会做一次剪枝，总$\text{epoch}$为$128$，结果显示其实精确度到后期并没有太大减损，所以在线剪枝是一个可走的路，目前限于时间没有做另外两种策略的研究，有空可以试试。

笔者其实想说的是和其他一些方法的结合，因为前一阵子看了另一篇论文[【论文阅读】训练数据稀疏？我们可以做得更好！（稀疏输入×神经网络 on PyTorch）](https://caoyang.blog.csdn.net/article/details/118199209)，其实这与$\text{SVD training}$关系不算很大，该文说的是输入稀疏，$\text{SVD training}$强调的是模型低秩，但是笔者认为两者有可以结合的点，因为目前笔者在实现卷积层的$\rm SVD$形式时，采用的方法还是非常低级的，仍然是继承原`Conv2d`类来改写，感觉上有很多冗余，似乎卷积核改写为$\rm SVD$后应该可以有更快的卷积运算方法，这是值得思考的点，不过笔者觉得这很难。

其他一些点可能就是训练上的技巧了，除了在线剪枝，是否有其他在线的一些训练策略值得去探究，学习率可以衰减，模型参数也可以衰减，或者是否即便不是在$\text{SVD training}$的框架下，一般的神经网络是否可以采用在线剪枝策略，这本身也是一种降秩的直接手段，不得不说很有意思。

<font color=red>其实笔者一直觉得如果只是把人工智能的研究停留在调参、标注数据上就太过于浅然了，必须引入更多的理论研究。会写几行代码本质上跟机器人并没有太大区别，你取代不了机器就早晚有一天会被取代。多思考，少吹逼才是正道。</font>

# $4$ $\rm PyTorch$模型保存与加载问题

最后为什么笔者还要画蛇添足地写这个章节呢？原因非常简单，因为在保存与加载$\text{SVD training}$模型遇到问题了。

$\rm PyTorch$中模型保存有两种形式，第一种是存整个模型，第二个是只存模型的参数。

先来看第一种方法：

```python
torch.save(model, 'model.h5')
model = torch.load('model.h5')
```

第二种方法：

```python
def save_checkpoint(model,
                    save_path,
                    optimizer=None,
                    scheduler=None,
                    epoch=None,
                    iteration=None):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model, save_path, optimizer=None, scheduler=None):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, checkpoint['epoch'], checkpoint['iteration']
```

其实两种方法的存储大小区别不大，第二种除了可以保存`model`的参数，还可以一起保存`optimizer`和`scheduler`的参数，但是在加载的时候需要定义好一个初始的`model`以及`optimizer`和`scheduler`，然后`load_state_dict()`才能得到保存的结果。

<font color=red>发现问题了吗？</font>没错$\text{SVD training}$模型在训练中被剪枝了，模型中参数的形状都变了！所以如果用第二个方法，定义一个空的初始模型，就会发现根本`load_state_dict()`不了，因为模型参数形状都不一样，你怎么`load`得了呢？

其实这里倒是感觉跟$\rm TensorFlow$有点区别，笔者记得如果要加载保存好的$\rm TensorFlow$模型，一些自定义的模块是需要写成字典作为参数传入的，好像$\rm PyTorch$里保存的模型如果带有自定义模块，用第一种方法保存的模型在加载时不会有问题的，这是一件好事。
