- Paper: https://arxiv.org/abs/1512.03385

Increase in layers in NNs causes vanishing/exploding gradient problems (think of chain rule, more multiplications of the partial derivatives can lead to very small gradients being updated or very large gradients). Paper finds thats deeper networks have higher training-error that is not caused by overfitting, also known as the **degrading problem**. 

Given $H(x)$ being the desired mapping/output of network, we can instead learn $H(x) = F(x) + x,$ where the $+ x$ term is referred to identify mapping. The intuition behind this is deeper networks should perform equally or better than shallower networks, and in the extreme case where adding more layers has no effect, the added layers should form a identity mapping of $H(x) = x$. Instead of the solver tyring to learn the identify mapping directly (which is difficult as layers are non-linear), resnets can just push the non-linear layers to 0 and use the $+ x$ term as identify mappings.

Formally, the resnet blocks are defined as : $y = F(x, \{W_{i}\}) + x$, where $F(\cdot)$ has the same dimension as $x$. If not, the paper performs a linear projection $W_{x}x$ to match the dimensions. 