# 3D Gaussian Splatting Avatar Generation

A novel framework combining deformable 3D Gaussian splatting with geometry-aware generative adversarial networks for high-fidelity avatar synthesis with improved 3D consistency and rendering efficiency.

## Project Overview

This project presents a unified framework for 3D-aware avatar generation that leverages advances in both 3D Gaussian splatting and geometry-aware GAN techniques. The architecture enables the creation of animatable avatars with high visual quality while maintaining strong 3D consistency across viewpoints and poses.

## Framework Architecture

The integrated framework consists of the following key components:

1. **Latent Space & Mapping Network**: A mapping network transforms input latent vectors (z) into style vectors (w) that guide the generation process.
2. **3D Gaussian Initialization**:
   - 50k Gaussians with 11-dimensional features each
   - Initial positions (xyz) sampled from canonical SMPL mesh
   - Features include: 3D position, 1D opacity, 3D scaling, and 4D rotation
3. **Deformation Pipeline**:
   - Canonical 3D Gaussians in standard space
   - Non-rigid deformation using SMPL parameters
   - Rigid transformation for pose adjustment
   - Optimizable SMPL parameters for fine-grained control
4. **Rendering Pipeline**:
   - Color prediction via MLP based on ray direction
   - Differentiable Gaussian rasterization for final image synthesis

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Rainiver/3DGS_Gen.git
cd 3DGS_Gen

# Install dependencies
pip install -r requirements.txt

# Run training
python train.sh

# Run testing
python run.sh
```

## Acknowledgements

This project builds upon the excellent work of:

- [3DGS-Avatar](https://github.com/mikeqzy/3dgs-avatar-release) by Qianyi Zhou et al.
- [EG3D](https://github.com/NVlabs/eg3d) by NVLabs

