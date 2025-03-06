# Cloud Platform Setup Guide

This guide provides detailed instructions for running the Earth system simulation on various cloud platforms, with practical examples and troubleshooting tips.

## Quick Start

Generate cloud platform links for your fork:

```bash
python scripts/generate_cloud_links.py --github-user your-username
```

This will create:
- Binder launch links
- Google Colab links
- Kaggle metadata
- README badges

## Platform-Specific Instructions

### 1. Google Colab

#### Setup Steps
1. Open [examples/colab_simulation.ipynb](examples/colab_simulation.ipynb) in Colab
2. Select "Runtime" -> "Change runtime type" -> "GPU"
3. Run all cells

#### Tips for Colab
- Save long-running results to Google Drive
- Use `@torch.jit.script` for faster execution
- Clear output cells to save memory
- Reconnect every 12 hours to maintain GPU access

#### Troubleshooting
- If OOM errors occur, reduce `batch_size` in configuration
- For disconnections, enable "Settings -> Site -> Keep awake"
- Save checkpoints frequently using Drive mount

### 2. Kaggle Kernels

#### Setup Steps
1. Fork the repository
2. Add as a Kaggle dataset:
   - Go to "Datasets" -> "New Dataset"
   - Select "GitHub" as source
   - Enter your repository URL
3. Create new notebook using the dataset

#### Tips for Kaggle
- Enable GPU: "Settings -> Accelerator -> GPU"
- Use persistent storage with Datasets
- Take advantage of TPU when available
- Enable internet for data downloads

#### Configuration
```yaml
# Add to kernel-metadata.json
{
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["your-username/earth-system-sim"]
}
```

### 3. Binder

#### Launch Options
1. Basic simulation:
```
https://mybinder.org/v2/gh/your-username/earth_system_sim/main?filepath=examples/basic_simulation.ipynb
```

2. Full environment:
```
https://mybinder.org/v2/gh/your-username/earth_system_sim/main
```

#### Tips for Binder
- Use smaller datasets for demos
- Cache intermediate results
- Keep sessions under 1 hour
- Download results frequently

### 4. Gradient (Paperspace)

#### Setup Steps
1. Create new Notebook
2. Select free GPU instance
3. Clone repository:
```bash
git clone https://github.com/your-username/earth_system_sim.git
cd earth_system_sim
pip install -r requirements.txt
```

#### Tips for Gradient
- Use persistent storage
- Enable public IP for monitoring
- Schedule notebooks for longer runs

## Resource Management

### Memory Optimization

1. Reduce model size:
```yaml
# config/model_config.yaml
physical_system:
  hidden_dims: [32, 32]  # Smaller architecture
  grid_height: 16        # Reduced grid size
  grid_width: 16
```

2. Enable gradient checkpointing:
```python
model.use_checkpointing = True
```

3. Use mixed precision:
```python
from torch.cuda.amp import autocast

with autocast():
    predictions, physics_losses = model(input_data)
```

### Storage Management

1. Compress results:
```python
# Save compressed arrays
np.savez_compressed('results.npz', **trajectory)
```

2. Clean up temporary files:
```python
# Clear old results
!rm -rf /tmp/simulation_*
```

3. Use efficient formats:
```python
# Save in chunks
for i, chunk in enumerate(chunks):
    np.save(f'chunk_{i}.npy', chunk)
```

## Platform Comparison

| Feature          | Colab         | Kaggle        | Binder        | Gradient      |
|------------------|---------------|---------------|---------------|---------------|
| GPU Access      | K80/T4/P100   | P100          | No            | Variable      |
| Runtime Limit   | 12 hours      | 30 hours      | 1 hour        | 6 hours       |
| Storage         | 15GB+Drive    | 20GB          | Temporary     | 5GB           |
| Setup Difficulty| Easy          | Medium        | Very Easy     | Medium        |
| Persistence     | Drive Mount   | Datasets      | None          | Workspace     |

## Best Practices

### 1. Code Optimization
- Use `@torch.jit.script` for computational kernels
- Implement data prefetching
- Optimize memory usage
- Profile code regularly

### 2. Data Management
- Cache preprocessed data
- Use efficient formats
- Implement checkpointing
- Clean up temporary files

### 3. Resource Monitoring
- Track GPU memory usage
- Monitor execution time
- Log system metrics
- Handle interruptions

## Common Issues

### 1. Memory Errors
```python
# Monitor GPU memory
print(torch.cuda.memory_summary())

# Clear cache if needed
torch.cuda.empty_cache()
```

### 2. Performance Issues
```python
# Profile execution
from torch.profiler import profile
with profile(activities=[ProfilerActivity.CPU]) as prof:
    model(input_data)
print(prof.key_averages())
```

### 3. Storage Problems
```python
# Check disk usage
!df -h

# Clean up old results
!find . -name "*.npy" -mtime +1 -delete
```

## Getting Help

- Open issues on GitHub
- Check platform-specific forums
- Review documentation
- Join community discussions

For more information:
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Kaggle Documentation](https://www.kaggle.com/docs)
- [Binder Documentation](https://mybinder.readthedocs.io/)
- [Gradient Documentation](https://docs.paperspace.com/gradient/)