# Report notes

## Best setting per prompt

- **bangs**: s0.30_g7.5 | CLIP=0.2303, LPIPS=0.2016, SSIM=0.7692, Tradeoff=0.1799
- **glasses**: s0.50_g5.0 | CLIP=0.2736, LPIPS=0.3123, SSIM=0.6784, Tradeoff=0.1955
- **smile**: s0.30_g5.0 | CLIP=0.2453, LPIPS=0.2017, SSIM=0.7687, Tradeoff=0.1949

## Short interpretation

- Mild settings work well when the edit must remain close to the original face.
- Glasses often needs stronger intervention than smile or bangs.
- Higher strength usually increases CLIP and LPIPS at the same time.