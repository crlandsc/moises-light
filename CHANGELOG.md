# Changelog

## 0.1.4 (2026-03-20)

- **BREAKING**: Architecture fixes — pretrained weights from v0.1.3 and earlier are incompatible
- Fix final conv: remove BN+ReLU that clamped output to non-negative (broke masking phase and direct generation)
- Fix double residual in dual-path RoPE blocks (outer + inner residuals doubled identity path vs BS-RoFormer)
- Fix dual-path ordering: time→freq to match BS-RoFormer (was freq→time)
- Add final RMSNorm after all RoPE blocks (matches BS-RoFormer)
- Remove zero-init on transformer output projection (no reference support)
- Change `bn_factor` default from 4 to 8 (matches DTTNet for vocals/drums/other)
- Updated param counts across all presets and documentation

## 0.1.3 (2026-03-19)

- Fix broken external links: lucidrains repos moved to [GitLab](https://gitlab.com/lucidrains), corrected DTTNet repo to official implementation at [junyuchen-cjy/DTTNet-Pytorch](https://github.com/junyuchen-cjy/DTTNet-Pytorch)

## 0.1.2 (2026-03-19)

- Remove contrib/ from package distribution

## 0.1.1 (2026-03-19)

- Initial release
