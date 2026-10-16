# Assignment 2 - Part 3: Theory: Arithmetic of CNNs

For standard convolution, the output size along each dimension is
`floor((input + 2 * padding - kernel) / stride) + 1`.

**Q1.** Input 32×32×3, eight 5×5 filters, stride 1, no padding:
`floor((32 - 5) / 1) + 1 = 28`. Therefore the output is **28×28×8** (8 channels because there are 8 filters).

**Q2.** With `same` padding and stride 1, the spatial size is preserved, so the output becomes **32×32×8**.

**Q3.** Input 64×64, kernel 3×3, stride 2, no padding:
`floor((64 - 3) / 2) + 1 = floor(30.5) + 1 = 31`. Output spatial size: **31×31**.

**Q4.** Pooling on 16×16 with 2×2 kernel and stride 2:
`floor((16 - 2) / 2) + 1 = 8`. Output spatial size: **8×8** (channels unchanged).

**Q5.** Each 3×3 convolution uses stride 1 and `same` padding, preserving height and width. After two such layers the output spatial shape is **128×128**. The number of channels depends on the number of filters, which is not specified.

**Q6.** `model.train()` enables training mode for layers such as dropout and batch normalization. If the model was just created, PyTorch modules already start in training mode, so removing this line usually does not affect the first training loop. However, if `model.eval()` was called earlier, omitting `model.train()` leaves the model in evaluation mode: dropout stays disabled and batch normalization uses stored statistics. Gradients still work, but training behavior could be wrong. This CNN has no dropout or batch normalization, so the mode switch does not change its layer outputs.
