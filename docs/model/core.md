# U-NET MODEL CORE CONSTUCTION

> *This note references to the model's code (`core.py`) file.*

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
