- Paper: https://arxiv.org/pdf/1502.03167

### Paper Summary

- **Internal Covariate shifts**, defined as *change in the distribution of network activations due to the change in network parameters during training*, are an issue  as after each layer, the input distributions can change and it can slow down training. 
- Let us define a 2-layer network: $L = F_{2}(F_{1}(u, \theta_{1}), \theta_{2})$. For $F_{2}$, its inputs can be seen as $x=F_{1}(u, \theta_{1})$ then $L = F_{2}(x, \theta_{2})$. Therefore, if the distribution of the input $x$ changes at the layer, the parameters $\theta_{2}$ has to adjust according to the shift of $x$ at each layer. 
- The issues arise in layers outside of networks. For a layer $z = g(Wu + b)$ with sigmoid activation function $g(x) = \frac{1}{1 + \exp(-x)}$, if the absolute value of $x$ increases, the gradients are close to zero except for small absolute values of $x$. 
- To demonstrate this, say we want to update the gradient with respect to the loss $L$. Then, due to the chain rule of partial derivatives, $\frac{dL}{du} = \frac{dL}{dz} \times \frac{dz}{dx} \times \frac{dx}{du} = \frac{dL}{dz} \times g^{'}(x) \times W$. But the $g^{'}(x)$ could likely be close to zero for larger absolute values, which suggests that (1) $u$ will receive very small signals to update (2) $W$ and $b$ will also update slowly due to the small gradients.
- Therefore using batchnorm can ensure that the distribution of $x$ remains in the more stable region, the training can accelerate.   

### Working towards Reducing Internal Covariance Shifts
- The gradient descent optimization needs to take into account that normalization takes place.
- For example, assume that we normalize the inputs ($\hat{x} = x - E[x]$) by subtracting the mean over the training data, where $x = u + b$. 
- If the gradient is calculated while ignoring the normalization step; $b \leftarrow b + \Delta b$, where $\frac{\delta L}{\delta b} = \frac{\delta L}{\delta \hat{x}} \times \frac{\delta \hat{x}}{\delta x}  \times \frac{\delta x}{\delta b} = \frac{\delta L}{\delta \hat{x}} \times 1 \times 1 = \frac{\delta L}{\delta \hat{x}}$. So $\Delta b \propto - \frac{\delta L}{\delta \hat{x}}$, and we are ignoring the fact that the $E[x]$ term in $\hat{x}$ contains $b$.
- Then, $x_{updated} = u + (b + \Delta b)$ and $E[x_{updated}] = E[u + (b + \Delta b)] = E[u + b] + \Delta b$. 
- This results in $\hat{x}_{updated} = x - E[x] = (u +b + \Delta b) - E[u +b] + \Delta = u +b  - E[u +b]$
- In summary, $\hat{x}_{updated} = u + b - E[u +b] = x - E[x]$ suggesting that the bias $b$ was added to $x$ but the output of the layer $\hat{x}$ did not change as the bias is added then subtracted by the normalization part.
- If we go back to see $\Delta b \propto - \frac{\delta L}{\delta \hat{x}}$, the optimizer will see try to increase $\Delta b$ to decrease the loss but because $\hat{x}$ remains unchanged, $\frac{\delta L}{\delta \hat{x}} $ remains unchanged, so we can confirm that the loss remains unchanged as well.
- The paper mentions that this leads to $b$ growing indefinitely and blowing up.

**Key point**
- Instead we can ensure that the gradients of the loss take account of the normalization by fixing a normal distribution for the activation functions. We can go back to the previous example here as well.
- Back to this step, here, we include that $E[x]$ contains $b$, then again $\frac{\delta L}{\delta b} = \frac{\delta L}{\delta \hat{x}} \times \frac{\delta \hat{x}}{\delta x} \times \frac{\delta x}{\delta b}$
- But this time taking into account $E[x]$ is a some function of $x$ or more specifically the batch mean that approximates $x$, $\frac{1}{\delta x}(x - E[x]) = 1 - 1 = 0$. 
- Now, $\frac{\delta L}{\delta b} = \frac{\delta L}{\delta \hat{x}} \times 0 \times 1 = 0$. So the optimizer does not have to adjust $\Delta b$ as it doesn't affect the loss.
- Therefore, this ensures that the optimizer focuses on adjusting the weights of the model instead of parameters that affect internal covariance shifts.
- Furthremore, a normal distribution of mean zero and variance 1 ensures the issue with the signals for non-linear functions such as the sigmoid function.

### Mini-Batch Normalization
- The paper normalizes the features independently instead of across all features by $\hat{x}^{k} = \frac{x^{k} - E[x^{k}]}{\sqrt{Var[x^{k}]}}​$, where $k$ represents the independent features.
- Also note that because normalizing features can change the representations of the layers, the transformation has to be able to represent the identity transform (affine transform), which is accomplished through: $y^{(k)} = \gamma^{k}\hat{x}^{k} + \beta^{k}$.
- What it means to represent the identity transform is from $y^{(k)} = \gamma^{k}\frac{x^{k} - E[x^{k}]}{\sqrt{Var[x^{k}]}} + \beta^{k}$, we can adjust both the parameters $\gamma^{k}$, $\beta^{k}$  to cancel out the $Var[x^{k}]$ part and the $- E[x^{k}]$ part such that $y^{k} = x^{k}$ in order to be able to recover the original activations before normalization if needed.
- The new parameters $\gamma$ and $\beta$ are learned and not canceled out because it is outide of the layer as well.
- During inference, we use $\hat{x}$ from the population rather than the mini batch as the normalization should be independent of the batch itself. In the case of the implementation, a moving average is used to track the population mean and variance.

#### Effects of Batch-Normalization
- Batch normalization enables higher learning rates as it normalizes the activations by forcing the gradients to be in the flat parts of the activations (sigmoid).
- Furthermore, batch norm is scale invariant for its parameters as $BN((aW)u) = \frac{aWu - E[aWu]}{\sqrt{Var[aWu]}} = \frac{a(Wu - E[Wu])}{a\sqrt{Var[Wu]}} = BN(Wu)$. 

### Implementation Details
- We add an $\epsilon$ for stability following the paper : $\hat{x} = \frac{x_{i} - \mu_{\beta}}{\sqrt{\sigma^{2}_{\beta} + \epsilon}}​$
- We calculate the mean across the channels (features) and initialize the parameters $\gamma$ and $\beta$ as $1$ and $0$, which if you recall, makes $y^{(k)} = \gamma^{k}\hat{x}^{k} + \beta^{k} = \hat{x}^{k}$, so just the normalized input. 
- Recall that the batch normalization layer can act as the identiy transform if needed, so we can initialize it as an affine transform but the model can also learn the parameters $\gamma$ and $\beta$ during back propogation to better fit the activations as well.
- Note the dimensions were matched during the normalization part by using ```keepdim=True```
```py
class BatchNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x):
        # batchnorm across the number of features (channels) -> want to flatten other dims
        # x: (Batch, Channel, Width, Height) => (Batch, Channel, Feature_Size)
        init_shape = x.shape
        x = x.flatten(2)

        # take mean across features
        # mean, mean_sq : (Channel) -> keepdim=True -> (1, Channel, 1)
        mean = x.mean(dim=(0, 2), keepdim=True)
        mean_sq = (x**2).mean(dim=(0, 2), keepdim=True)

        # Var = E[x**2] - E[x]**2
        var = mean_sq - mean**2

        x_norm = (x - mean) / (torch.sqrt(var + self.eps))

        print(f"dim(x_norm) {x_norm.shape}")

        output = self.gamma * x_norm + self.beta

        return output.view(init_shape)
```
- If we set a random input ```x = torch.rand(32, 3, 64, 64)```, we can observe that the input after being flattened will have a shape of ```(32, 3, 64 * 64)``` and the mean of ```x``` across ```dim=(0,2)``` (in other words, mean across the features/channels) will have a shape of ```(1, 3, 1)```, so one for each feature.
- Lastly we reshape it to the original ```(32, 3, 64, 64)``` as the input to the activation. 

#### Training v.s Inference
- We noted above that during inference, we use the statistics on the population rather than the batches. The paper suggests a more efficient way, which is to take a running average of the mean and variances of the mini-batches during training.
- Here we add ```register_buffer()```, which are parameters updated during training but are not learnable.
- We also introduce ```momentum```, which is a hyperparameter used in PyTorch's ```BatchNorm2d``` with a default value of ```0.1```, which we followed and will update the exponential moving averages for the running mean and variances.
```py
class BatchNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5, train=True, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))

        self.train = train
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(1, num_channels, 1))
        self.register_buffer('running_variance', torch.ones(1, num_channels, 1))

    def forward(self, x):
        # batchnorm across the number of features (channels) -> want to flatten other dims
        # x: (Batch, Channel, Width, Height) => (Batch, Channel, Feature_Size)
        init_shape = x.shape
        x = x.flatten(2)

        if self.train:
            # take mean across features
            # mean, mean_sq : (Channel) -> keepdim=True -> (1, Channel, 1)
            mean = x.mean(dim=(0, 2), keepdim=True)
            mean_sq = (x**2).mean(dim=(0, 2), keepdim=True)

            # Var = E[x**2] - E[x]**2
            var = mean_sq - mean**2

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_variance = (1 - self.momentum) * self.running_variance + self.momentum * var
        
            mean = mean
            var = var
        else:
            mean = self.running_mean
            var = self.running_variance

        x_norm = (x - mean) / (torch.sqrt(var + self.eps))

        output = self.gamma * x_norm + self.beta

        return output.view(init_shape)
```