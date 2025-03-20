# Earth System Simulation - Project Architecture

## Project Structure

```
earth_system_sim/
├── src/
│   ├── models/
│   │   ├── physical/
│   │   │   ├── conv_lstm.py
│   │   │   └── pinn_module.py
│   │   ├── biosphere/
│   │   │   └── policy_network.py
│   │   └── geosphere/
│   │       └── slow_policy.py
│   ├── integration/
│   │   ├── temporal_sync.py
│   │   └── data_flow.py
│   └── visualization/
│       ├── base.py
│       ├── physical_vis.py
│       ├── biosphere_vis.py
│       └── geosphere_vis.py
├── scripts/
│   ├── run_simulation.py
│   └── visualize_results.py
└── config/
    ├── model_config.yaml
    └── visualization_config.yaml
```

## Component Interactions

### 1. Data Flow Architecture
```
                      ┌─────────────────┐
                      │  DataFlowManager│
                      └────────┬────────┘
                               │
                  ┌────────────┼────────────┐
                  │            │            │
          ┌───────▼───┐  ┌────▼─────┐ ┌────▼────┐
          │  Physical │  │ Biosphere│ │Geosphere│
          │   (PINN)  │  │ (Policy) │ │(Policy) │
          └───────────┘  └──────────┘ └─────────┘

```

### 2. State Dimensions and Transformations

```
Physical State (PINN)
[batch_size, channels=5, height=16, width=16]
│
├──> Biosphere Input: [batch_size, state_dim=4]
│    - Vegetation cover
│    - Soil moisture
│    - Temperature (mean)
│    - Pressure (mean)
│
└──> Geosphere Input: [batch_size, state_dim=3]
     - Elevation
     - Erosion rate
     - Soil composition
```

### 3. Component Dependencies

#### Physical System (PINN)
- **Input**: [batch, channels, height, width]
  - Channels: [density, temperature, pressure, u_velocity, v_velocity]
- **Dependencies**: None
- **Outputs**: Updated physical state

#### Biosphere System (Policy Network)
- **Input**: [batch, state_dim]
  - Current vegetation state
  - Physical system metrics (averaged)
- **Dependencies**: Physical system state
- **Outputs**: Growth actions

#### Geosphere System (Policy Network)
- **Input**: [batch, state_dim]
  - Current elevation state
  - Physical system ground metrics
- **Dependencies**: Physical system state
- **Outputs**: Erosion/deposition actions

## Data Flow in run_simulation.py

1. **Initialization**
```python
physical_state = [1, 5, 16, 16]  # [batch, channels, height, width]
biosphere_state = [1, 4]         # [batch, state_dim]
geosphere_state = [1, 3]         # [batch, state_dim]
```

2. **Timestep Execution**
```
1. Physical Update (PINN)
   - Input: physical_state
   - Output: new_physical_state

2. Biosphere Update (if scheduled)
   - Get physical metrics
   - Update vegetation state
   - Apply feedback to physical system

3. Geosphere Update (if scheduled)
   - Get physical ground state
   - Update elevation
   - Apply feedback to physical system
```

## Key Integration Points

### 1. Temporal Synchronization
- Physical system: Every timestep
- Biosphere: Every 10 timesteps
- Geosphere: Every 100 timesteps

### 2. State Transformations
```python
# Physical to Biosphere
bio_input = _prepare_biosphere_input(physical_state)  # [1, 4]

# Physical to Geosphere
geo_input = _prepare_geosphere_input(physical_state)  # [1, 3]
```

### 3. Feedback Integration
```python
# Biosphere feedback affects:
- Temperature distribution
- Moisture levels

# Geosphere feedback affects:
- Wind patterns
- Temperature gradients
```

## Common Issues and Solutions

1. **Dimension Mismatches**
   - Always verify input shapes match policy network expectations
   - Use debug mode to track tensor transformations
   - Check state buffers after updates

2. **Policy Network Inputs**
   - Biosphere needs 4-dimensional state
   - Geosphere needs 3-dimensional state
   - Use _prepare_X_input functions for proper formatting

3. **Integration Timing**
   - Components update at different frequencies
   - Use synchronizer to manage updates
   - Check timescales in config

## Debug Tools

1. print_tensor_info:
```python
def print_tensor_info(name, tensor):
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Device: {tensor.device}")
    print(f"  Range: [{tensor.min()}, {tensor.max()}]")
```

2. Shape Validation:
```python
if state.shape != expected_shape:
    raise ValueError(f"Expected shape {expected_shape}, got {state.shape}")
```

This architecture document should help in understanding how components interact and where to look for potential issues.