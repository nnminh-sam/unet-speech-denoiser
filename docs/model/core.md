# U-NET MODEL CORE CONSTUCTION

> *This note references to the model's code (`core.py`) file.*

## U-Net model

U-Net model visualization:

![alt text](/docs/assets/unet-model.png)

---

## Residual Block with Pre-activation

```py
class PreactResBlock(nn.Sequential):
    """
    Residual Block with Pre-activation
    """

    def __init__(self, dim):
        super().__init__(
            nn.GroupNorm(dim // 16, dim), # Normalization
            nn.GELU(), # Activation
            nn.Conv2d(dim, dim, 3, padding=1), # Convolution
            nn.GroupNorm(dim // 16, dim), # Nomalization
            nn.GELU(), # Activation
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + super().forward(x) # Residual connection
```

- **Pre-activation**: Normalization + Activation before convolution.
- **Residual Connection**: Adds input `x` into the processed output.
- **Purpose**: Improves gradient flow and prevents vanishing gradients.

### Group Normalization

- Helps stabilize training by reducing internal covariate shift.
- Makes optimization easier by ensuring consistent feature distributions.
- Allows for faster convergence.
- Works well even with small batch sizes (unlike BatchNorm).

### Residual Connection

- Helps gradient flow in deep networks, preventing vanishing gradients.
- Enables identity mapping, meaning if a layer is unnecessary, it can be skipped.
- Makes optimization easier by allowing the network to learn modifications rather than full transformations.

### Activation Function - GELU

- Introduces non-linearity, allowing the network to learn complex patterns.
- Prevents the network from collapsing into a simple linear transformation.
- GELU (Gaussian Error Linear Unit) is used instead of ReLU because:
    - Smoother than ReLU.
    - Helps with better gradient flow in deeper networks.

### Advantage of Pre-activation first

1. **Better Gradient Flow**:
    - When using BatchNorm/GroupNorm before Conv, gradients can flow through the network more easily, reducing vanishing gradients.
2. **Regularization Effect**:
    - Applying normalization first ensures that activations stay well-conditioned before entering convolutions.
3. **Improved Training Stability**:
    - If we applied activation after convolution, we could get high-variance activations that destabilize training.
    - Applying Norm first ensures stable inputs before applying non-linearity.

### Using Residual connection at the end

```py
def forward(self, x):
    return x + super().forward(x)
```

- The residual connection bypasses the transformations, ensuring the original information is still available.
- If the convolution layers fail to learn something useful, the network can still pass identity information.

---

## UNet Block - Core Block for Encoder & Decoder

```py
class UNetBlock(nn.Module):
    def __init__(self, input_dim, output_dim=None, scale_factor=1.0):
        super().__init__()
        
        if output_dim is None:
            output_dim = input_dim

        self.pre_conv = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.res_block1 = PreactResBlock(output_dim)
        self.res_block2 = PreactResBlock(output_dim)

        # Downsampling or Upsampling
        self.downsample = self.upsample = nn.Identity()
        if scale_factor > 1:
            # Perfoming upsampling
            self.upsample = nn.Upsample(scale_factor=scale_factor)
        elif scale_factor < 1:
            # Perfoming Downsampling
            self.downsample = nn.Upsample(scale_factor=scale_factor)

    def forward(self, x, h=None):
        """
        Args:
            x: (b c h w), last output
            h: (b c h w), skip output
        Returns:
            o: (b c h w), output
            s: (b c h w), skip output
        """
        x = self.upsample(x)
        if h is not None:
            assert x.shape == h.shape, f"{x.shape} != {h.shape}"
            x = x + h
        x = self.pre_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.downsample(x), x
```

- **Convolution + Residual blocks**: Process features.
- **Upsampling/Downsampling**: Controls scaling for encoder/decoder.
- **Skip Connections (`h`)**: Improve information flow in U-Net.
- **Purpose**: a building block for both the encoder (downsampling) and decoder (upsampling) parts of the U-Net architecture.
    - **Downsampling** (encoder): Extracts higher-level features by reducing spatial dimensions.
    - **Upsampling** (decoder): Reconstructs the original resolution by increasing spatial dimensions.
    - **Skip connections**: Pass features from encoder to decoder to recover details lost in downsampling.

**Params:**

- `input_dim`: Number of channels in the input feature map.
- `output_dim`: Number of channels in the output feature map. Defaults to input_dim if not specified.
- `scale_factor`: Determines if this block is downsampling (scale < 1) or upsampling (scale > 1).

### Convolutional Layer

```py
self.pre_conv = nn.Conv2d(input_dim, output_dim, 3, padding=1)
```

- **Applies a 3x3 convolution** to transform `input_dim → output_dim`.
- Preserves spatial resolution (`padding=1` ensures output size remains the same).


### Two Residual Blocks (PreactResBlock)

```py
self.res_block1 = PreactResBlock(output_dim)
self.res_block2 = PreactResBlock(output_dim)
```

- These two blocks **refine features** at each stage.
- Each `PreactResBlock` consists of **GroupNorm → Activation (GELU) → Conv2D → GroupNorm → Activation → Conv2D**.
- Using **two residual blocks** improves expressiveness without making the model too deep.

**Why Two Residual Blocks?**
- A **single block might not learn enough complex features** at each scale.
- U-Net's **contracting and expanding paths** benefit from multiple convolutions before downsampling/upsampling.
- Helps **better feature extraction** at different levels.


### Downsampling and Upsampling

```py
self.downsample = self.upsample = nn.Identity()
if scale_factor > 1:
    self.upsample = nn.Upsample(scale_factor=scale_factor)
elif scale_factor < 1:
    self.downsample = nn.Upsample(scale_factor=scale_factor)
```

- If `scale_factor > 1`, we **upsample** (increase spatial size).
- If `scale_factor < 1`, we **downsample** (reduce spatial size).
- If `scale_factor == 1`, no resizing is performed (default to `Identity`).


### Forward Pass (`forward` method)

```py
def forward(self, x, h=None):
```
- `x`: The input feature map.
- `h`: The **skip connection** (from encoder to decoder in U-Net).


### Upsample if needed

```py
x = self.upsample(x)
```

- **Decoder case**: If `scale_factor > 1`, increase spatial size before processing.


### Add Skip Connection (`h`)

```py
if h is not None:
    assert x.shape == h.shape, f"{x.shape} != {h.shape}"
    x = x + h
```

- In the **decoder**, the skip connection from the encoder is added.
- **Why?** This helps the model **recover spatial details** lost during downsampling.


### Apply Pre-convolution and Residual Blocks

```py
x = self.pre_conv(x)
x = self.res_block1(x)
x = self.res_block2(x)
```

- **1st Conv (`pre_conv`)**: Adjusts feature dimensions.
- **Residual Blocks (`res_block1` & `res_block2`)**: Learn transformations.

### Downsample if needed

```py
return self.downsample(x), x
```

- **Encoder case**: `scale_factor < 1`, so apply downsampling.
- **Decoder case**: `scale_factor > 1`, no further changes.

**U-Net Design Philosophy**
- U-Net **repeatedly applies convolutions before downsampling and upsampling**.  
- This ensures **better feature representation** and **smooth gradients**.

**Deeper Feature Extraction**
- **One block might be too shallow** to capture complex patterns.
- **Two blocks** allow **multi-step feature transformation** before moving to the next scale.

**Stability and Expressiveness**
- **More non-linearity** means **richer features**.
- Helps **mitigate vanishing gradients** in deeper networks.

---

## U-Net: Model configuration

```py
class UNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16, num_blocks=4, num_middle_blocks=2):
        super().__init__()
```

Prams:

- `input_dim`: Number of input channels.
- `output_dim`: Number of output channels.
- `hidden_dim`: Base feature dimension.
- `num_blocks`: Number of encoder/decoder blocks.
- `num_middle_blocks`: Number of middle layers (bottleneck).

### Encoder - Downsampling process

```py
self.encoder_blocks = nn.ModuleList(
    [
        UNetBlock(input_dim=hidden_dim * 2**i, output_dim=hidden_dim * 2 ** (i + 1), scale_factor=0.5)
        for i in range(num_blocks)
    ]
)
```

- **Reduces spatial size** while increasing feature channels.
- Uses `scale_factor=0.5` for downsampling.

### Middle block - Bottle neck

```py
self.middle_blocks = nn.ModuleList(
    [UNetBlock(input_dim=hidden_dim * 2**num_blocks) for _ in range(num_middle_blocks)]
)
```

- Acts as the **deepest layer** of U-Net.
- Extracts complex features.

### Decoder - Upsamling process

```py
self.decoder_blocks = nn.ModuleList(
    [
        UNetBlock(input_dim=hidden_dim * 2 ** (i + 1), output_dim=hidden_dim * 2**i, scale_factor=2)
        for i in reversed(range(num_blocks))
    ]
)
```

- Upsamples feature maps back to the original size.
- Uses `scale_factor=2` to double **spatial** dimensions.

### Output layer

```py
self.head = nn.Sequential(
    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
    nn.GELU(),
    nn.Conv2d(hidden_dim, output_dim, 1),
)
```

- Converts the output into the desired format.

### Padding to fit U-Net scaling

```py
    def pad_to_fit(self, x):
        """
        Args:
            x: (b c h w), input
        Returns:
            x: (b c h' w'), padded input
        """
        hpad = (self.scale_factor - x.shape[2] % self.scale_factor) % self.scale_factor
        wpad = (self.scale_factor - x.shape[3] % self.scale_factor) % self.scale_factor
        return F.pad(x, (0, wpad, 0, hpad))
```

- Ensures that the input **height & width** are **divisible** by the U-Net scale factor.
- Uses `F.pad` to add necessary padding.

### Foward pass - Data flow in U-Net

```py
def forward(self, x):
    shape = x.shape

    x = self.pad_to_fit(x)
    x = self.input_proj(x)

    s_list = []
    for block in self.encoder_blocks:
        x, s = block(x)
        s_list.append(s)

    for block in self.middle_blocks:
        x, _ = block(x)

    for block, s in zip(self.decoder_blocks, reversed(s_list)):
        x, _ = block(x, s)

    x = self.head(x)
    x = x[..., : shape[2], : shape[3]]

    return x
```

**Step-by-step Execution:**

1. Padding to make input dimensions compatible.
2. Pass input through convolutional projection (`self.input_proj`).
3. Encoder Blocks:
    - Extract features while storing skip connections in `s_list`.
4. Middle Blocks:
    - Process deep features.
5. Decoder Blocks: 
    - Use skip connections from the encoder.
    - Upsample back to the original resolution.
6. Final Output Projection (`self.head`).

### Model complexity analysis

```py
def test(self, shape=(3, 512, 256)):
    import ptflops

    macs, params = ptflops.get_model_complexity_info(
        self,
        shape,
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True,
    )

    print(f"macs: {macs}")
    print(f"params: {params}")
```

- Uses `ptflops` to measure FLOPs (computational cost) and number of parameters.
