# Qwen Image on Intel Arc B70 - Performance Notes

## VAE Decode Bottleneck

**Problem**: Qwen Image VAE decode at 512×512 takes 5+ minutes on Intel Arc B70, often hanging indefinitely. Diffusion steps complete in ~3 seconds, but VAE decode never progresses.

**Root Cause**: 
- The Qwen Image VAE uses many small elementwise operations
- Neither XPU nor CPU has optimized kernels for this workload
- Memory fragmentation after each generation causes DEVICE_LOST crashes

## Workarounds

### 1. Reduce Resolution (Recommended)
- **256×256**: ~40 seconds total
- **384×384**: ~90 seconds
- **512×512**: Avoid - 5+ minutes, often fails

### 2. CPU Offload VAE (Implemented in Mix Studio)
```python
# In QwenImageTextToImageLocal custom node
pipe.vae = pipe.vae.to("cpu")
# Monkey-patch decode to ensure latents are on CPU
original_decode = pipe.vae.decode
def cpu_decode(latents, *args, **kwargs):
    return original_decode(latents.to("cpu"), *args, **kwargs)
pipe.vae.decode = cpu_decode
```

### 3. VRAM Cleanup After Each Generation
```python
# Prevents memory fragmentation crashes
if hasattr(torch, "xpu") and torch.xpu.is_available():
    torch.xpu.empty_cache()
elif torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 4. Disable Unnecessary Features
- Set `watermark: ''` (avoids PIL font loading delay)
- Set `prompt_extend: false` (prevents 400+ sec wait for unavailable text-gen model)

## Model Configuration for Mix Studio
```javascript
{
  "prompt_extend": false,  // Critical - disables missing text-gen model wait
  "watermark": "",         // Critical - avoids slow PIL font loading
  "steps": 4,             // Lightning scheduler
  "cfg": 1.0,             // Lower CFG = faster
  "width": 256,           // Use 256 for speed
  "height": 256           // Or 384 for better quality
}
```

## Verification Commands
```bash
# Check GPU status
curl -s http://127.0.0.1:49000/system_stats | jq '.system.devices[0]'

# Check queue
curl -s http://127.0.0.1:49000/queue | jq '.queue_running'

# Check history for completion
curl -s "http://127.0.0.1:49000/history" | jq 'keys'
```

## Alternative Engines
For faster generation on Arc B70, consider:
- **Krea 2**: Better optimized, supports editing modes
- **Flux (Klein 4B/9B)**: Good quality/speed ratio
- **Qwen Image Edit**: Direct editing with mask support (but same VAE issues)