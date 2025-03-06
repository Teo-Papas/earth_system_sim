# Earth System Simulation

A hybrid Earth system simulation framework that combines Physics-Informed Neural Networks (PINNs), Policy Gradient methods, and reinforcement learning to model interactions between physical (atmosphere/ocean), biosphere, and geosphere components.

## Features

- **Physical System**
  - PINN-based atmosphere/ocean modeling
  - Conservation law enforcement
  - ConvLSTM for spatiotemporal patterns
  - Multi-scale dynamics

- **Biosphere System**
  - Policy gradient-based ecosystem modeling
  - Adaptive parameter adjustment
  - Seasonal to interannual timescales
  - Vegetation and carbon flux dynamics

- **Geosphere System**
  - Slow-paced reinforcement learning
  - Long-term geological processes
  - Episodic updates
  - Topographical evolution

- **Integration Framework**
  - Multi-timescale synchronization
  - Component data flow management
  - Physical constraint validation
  - State transformations

- **Visualization System**
  - Static plots and animations
  - Interactive dashboards
  - 3D visualization support
  - Cross-component analysis tools

## Requirements

- Python 3.8 or higher
- PyTorch 1.8.0 or higher
- Additional dependencies in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/earth_system_sim.git
cd earth_system_sim
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify setup:
```bash
python scripts/verify_setup.py
```

## Quick Start

1. Run the example notebook:
```bash
jupyter notebook examples/basic_simulation.ipynb
```

2. Or run a simulation from command line:
```bash
python scripts/run_simulation.py --config config/model_config.yaml
```

3. Visualize results:
```bash
python scripts/visualize_results.py \
    --data simulation_output.npz \
    --config config/visualization_config.yaml
```

## Project Structure

```
earth_system_sim/
├── src/
│   ├── models/
│   │   ├── physical/      # PINN implementation
│   │   ├── biosphere/     # Policy gradient networks
│   │   └── geosphere/     # Slow-paced RL
│   ├── integration/
│   │   ├── data_flow.py   # Component interaction
│   │   └── temporal_sync.py # Time synchronization
│   └── visualization/     # Visualization components
├── scripts/
│   ├── run_simulation.py  # Main simulation runner
│   └── visualize_results.py # Visualization tools
├── config/
│   ├── model_config.yaml  # Model configuration
│   └── visualization_config.yaml # Visualization settings
├── tests/
│   ├── test_basic.py     # Component tests
│   └── test_integration.py # Integration tests
└── examples/
    └── basic_simulation.ipynb # Example notebook
```

## Configuration

### Model Configuration
Edit `config/model_config.yaml` to modify:
- Grid dimensions
- Model architectures
- Training parameters
- Physical constraints
- Component interactions

### Visualization Configuration
Edit `config/visualization_config.yaml` to customize:
- Plot styles
- Animation parameters
- Interactive features
- Output formats

## Testing

Refer to `TESTING.md` for detailed testing instructions. Quick test:

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_integration.py
```

## Examples

1. **Basic Simulation**
   - See `examples/basic_simulation.ipynb`
   - Demonstrates core functionality
   - Includes visualization examples
   - Shows component interactions

2. **Command Line Usage**
```bash
# Run simulation
python scripts/run_simulation.py \
    --config config/model_config.yaml \
    --steps 1000 \
    --output sim_results.npz

# Create visualizations
python scripts/visualize_results.py \
    --data sim_results.npz \
    --config config/visualization_config.yaml
```

## Component Details

### Physical System (PINN)
- Conservation of mass, energy, momentum
- Spatiotemporal coherence
- Physical constraint enforcement
- ConvLSTM-based architecture

### Biosphere System
- Policy gradient optimization
- Ecosystem parameter adaptation
- Vegetation dynamics modeling
- Carbon flux tracking

### Geosphere System
- Long-term process modeling
- Episodic reinforcement learning
- Geological constraint satisfaction
- Memory-based policy updates

## Visualization Options

1. **Static Plots**
   - Temperature fields
   - Vegetation patterns
   - Topographical maps
   - Time series plots

2. **Animations**
   - Wind field evolution
   - Vegetation growth
   - Erosion processes
   - System dynamics

3. **Interactive Visualizations**
   - 3D topography viewer
   - Ecosystem dashboard
   - Component interactions
   - Time series explorer

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{earth_system_sim,
  title = {Earth System Simulation},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/Teo-Papas/earth_system_sim}
}
```

## Contact

For questions or collaboration opportunities, please contact:
- Email: Papastertheof@gmail.com
- GitHub: 166628481+Teo-Papas@users.noreply.github.com

## Acknowledgments

This project builds on various research works in:
- Earth system modeling
- Physics-informed neural networks
- Reinforcement learning
- Deep learning for climate science
