# Phase 2 vs Phase 3.1 Training Comparison

## Summary

| Metric | Phase 2 (Baseline) | Phase 3.1 (Data Aug + Dropout↑) | Δ |
|--------|-------------------|----------------------------------|---|
| **Best val mIoU** | **0.5040** | **0.5065** | **+0.0025 (+0.5%)** |
| Best epoch | 10 | 45 | +35 epochs |
| Early stop epoch | 60 | 60 | Same |
| Final train loss | 0.2442 | 0.6155 | +0.3713 |
| Train loss @ epoch 10 | ~0.35 | ~0.64 | - |

---

## Key Findings

### ✅ Improvements
1. **Best mIoU improved slightly**: 0.5040 → 0.5065 (+0.5%)
2. **Overfitting significantly reduced**: 
   - Phase 2: Best at epoch 10, then degraded for 50 epochs
   - Phase 3.1: Best at epoch 45, much more stable
3. **Better generalization**: Model took 4.5× longer to reach peak performance

### ⚠️ Issues
1. **Improvement is marginal**: Only +0.0025 mIoU gain
2. **Train loss did not decrease**: 
   - Phase 2: 0.6484 → 0.2442 (62% drop)
   - Phase 3.1: 0.6600 → 0.6155 (6.7% drop)
3. **Data augmentation may be too aggressive**: Preventing model from fitting training data

---

## Training Curves Comparison

### Phase 2 (Baseline)
- Initial train loss: 0.6484
- Final train loss: 0.2442 (**↓ 62.3%**)
- Best val mIoU: 0.5040 (epoch 10)
- Severe overfitting after epoch 10

### Phase 3.1 (Data Aug + Dropout 0.2)
- Initial train loss: ~0.66
- Final train loss: 0.6155 (**↓ 6.7%**)
- Best val mIoU: 0.5065 (epoch 45)
- Much slower convergence, less overfitting

---

## Conclusion

**Phase 3.1 has minimal improvement over Phase 2.**

The data augmentation successfully reduced overfitting (best epoch 10→45), but the actual performance gain is negligible (+0.5%). The train loss plateau suggests the augmentation is too strong, preventing the model from learning effectively.

---

## Next Steps

### Option A: Reduce augmentation strength
- Lower flip/rotation probability from 0.5 to 0.3
- Reduce color jitter intensity
- Goal: Allow faster convergence while maintaining generalization

### Option B: Increase model capacity (Phase 3.2)
- LoRA rank 4 → 8 or 16
- More prompts: 5 → 10
- Goal: Give model more capacity to learn from augmented data

### Option C: Move to complete dataset
- Current: 200 samples
- Full LoveDA Train: 2522 samples
- With data augmentation, more data may help significantly

**Recommendation**: Try Option B (increase capacity) first, as we already have augmentation in place. If train loss decreases but val mIoU improves, it confirms capacity was the bottleneck.
