# Testing and Verification Guide

This guide explains how to verify the connectivity and proper functioning of the Earth system simulation components.

## Quick Verification

To quickly verify that all components are properly connected and functioning:

```bash
python scripts/verify_setup.py
```

This script will:
1. Check all required dependencies
2. Verify CUDA availability
3. Run integration tests
4. Create a sample simulation
5. Generate test visualizations

## Detailed Testing

### 1. Run Integration Tests

To run the full suite of integration tests:

```bash
pytest tests/test_integration.py -v
```

This will test:
- Component initialization
- Data flow between components
- Temporal synchronization
- Physical conservation laws
- End-to-end simulation
- Visualization pipeline

### 2. Test Individual Components

#### Physical System
```bash
pytest tests/test_basic.py -k "test_physical"
```
Tests:
- ConvLSTM functionality
- PINN implementation
- Conservation laws
- Physical constraints

#### Biosphere System
```bash
pytest tests/test_basic.py -k "test_biosphere"
```
Tests:
- Policy network
- Action sampling
- State transitions
- Reward computation

#### Geosphere System
```bash
pytest tests/test_basic.py -k "test_geosphere"
```
Tests:
- Slow policy implementation
- Long-term memory
- State evolution
- Action generation

### 3. Test Integration Components

#### Temporal Synchronization
```bash
pytest tests/test_basic.py -k "test_temporal"
```
Verifies:
- Proper timescale handling
- Component synchronization
- Update scheduling
- Time tracking

#### Data Flow
```bash
pytest tests/test_basic.py -k "test_data_flow"
```
Checks:
- State transformations
- Data validation
- Component interactions
- Buffer management

### 4. Test Visualization System

```bash
pytest tests/test_basic.py -k "test_visualization"
```
Validates:
- Plot generation
- Animation creation
- Interactive visualizations
- Output file handling

## Continuous Integration

The project includes GitHub Actions workflows for automated testing:

1. **Basic Tests**: Run on every push and pull request
   ```yaml
   - name: Run basic tests
     run: pytest tests/test_basic.py -v
   ```

2. **Integration Tests**: Run on main branch updates
   ```yaml
   - name: Run integration tests
     run: pytest tests/test_integration.py -v
   ```

## Performance Testing

To test system performance:

```bash
python scripts/run_simulation.py --config config/model_config.yaml --profile
```

This will output:
- Component execution times
- Memory usage
- GPU utilization (if available)
- Overall simulation performance

## Common Issues and Solutions

### 1. CUDA Out of Memory
If you encounter CUDA out of memory errors:
- Reduce batch size in model_config.yaml
- Decrease grid resolution
- Use gradient checkpointing

### 2. Component Synchronization
If components are not properly synchronized:
- Check timescale settings in model_config.yaml
- Verify update frequencies
- Inspect data flow connections

### 3. Visualization Errors
If visualizations fail:
- Check output directory permissions
- Verify data dimensions
- Ensure proper backend selection

## Best Practices

1. **Regular Testing**
   - Run quick verification daily
   - Run full integration tests before major changes
   - Monitor performance metrics

2. **Data Validation**
   - Use test data generators
   - Verify physical constraints
   - Check state bounds

3. **Performance Optimization**
   - Profile bottlenecks
   - Monitor memory usage
   - Optimize critical paths

## Adding New Tests

When adding new components or features:

1. Add unit tests in `tests/test_basic.py`
2. Add integration tests in `tests/test_integration.py`
3. Update verification script if needed
4. Add performance benchmarks

Example test structure:
```python
def test_new_feature():
    # Setup
    component = NewComponent()
    
    # Test functionality
    result = component.process()
    
    # Verify results
    assert result.shape == expected_shape
    assert check_constraints(result)
    
    # Test integration
    integrated_result = system.integrate(result)
    assert verify_integration(integrated_result)
```

## Troubleshooting

If verification fails:

1. Check dependency versions
2. Verify CUDA setup (if using GPU)
3. Inspect log files
4. Run individual component tests
5. Check data flow connections

For detailed error analysis:
```bash
pytest tests/test_integration.py -v --pdb
```

This will drop into a debugger on test failures.