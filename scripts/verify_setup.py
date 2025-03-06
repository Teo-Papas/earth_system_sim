"""
Quick-start script to verify project setup and connectivity.
Checks dependencies, runs tests, and creates a sample simulation.
"""

import sys
import subprocess
import importlib
import torch
import numpy as np
from pathlib import Path
import yaml
import tempfile
import time
from typing import List, Dict

def check_dependency(package: str, min_version: str = None) -> bool:
    """Check if a package is installed and optionally verify its version."""
    try:
        module = importlib.import_module(package)
        if min_version:
            version = getattr(module, '__version__', '0.0.0')
            from packaging import version as version_parser
            return version_parser.parse(version) >= version_parser.parse(min_version)
        return True
    except ImportError:
        return False

def check_dependencies() -> List[str]:
    """Check all required dependencies."""
    required = {
        'torch': '1.8.0',
        'numpy': '1.19.0',
        'pyyaml': '5.4.0',
        'matplotlib': '3.3.0',
        'plotly': '4.14.0',
        'holoviews': '1.14.0',
        'pytest': '6.2.0'
    }
    
    missing = []
    for package, version in required.items():
        if not check_dependency(package, version):
            missing.append(f"{package}>={version}")
    
    return missing

def run_tests() -> bool:
    """Run integration tests."""
    result = subprocess.run(
        ['pytest', 'tests/test_integration.py', '-v'],
        capture_output=True,
        text=True
    )
    
    print("\nTest Results:")
    print(result.stdout)
    
    if result.stderr:
        print("\nErrors:")
        print(result.stderr)
        
    return result.returncode == 0

def create_sample_simulation() -> Dict:
    """Create and run a small sample simulation."""
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    from scripts.run_simulation import EarthSystemSimulation
    
    print("\nRunning sample simulation...")
    
    # Load configuration
    config_path = project_root / 'config' / 'model_config.yaml'
    
    # Create simulation
    sim = EarthSystemSimulation(
        config_path=str(config_path),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Run short simulation
    trajectory = sim.run_simulation(
        num_steps=20,
        save_frequency=5
    )
    
    return trajectory

def create_sample_visualizations(trajectory: Dict):
    """Create sample visualizations from simulation results."""
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    from scripts.visualize_results import create_visualizations
    
    print("\nCreating sample visualizations...")
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save trajectory data
        data_path = Path(tmpdir) / 'sample_data.npz'
        np.savez(str(data_path), **trajectory)
        
        # Load visualization config
        viz_config_path = project_root / 'config' / 'visualization_config.yaml'
        with open(viz_config_path, 'r') as f:
            viz_config = yaml.safe_load(f)
        
        # Create visualizations
        output_dir = project_root / 'sample_outputs'
        output_dir.mkdir(exist_ok=True)
        
        create_visualizations(
            trajectory,
            viz_config,
            str(output_dir),
            make_animations=True
        )
        
        print(f"\nSample visualizations created in: {output_dir}")

def main():
    """Run all verification steps."""
    print("Earth System Simulation - Setup Verification")
    print("===========================================")
    
    # Check Python version
    print("\nChecking Python version...")
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        return False
    
    # Check dependencies
    print("\nChecking dependencies...")
    missing = check_dependencies()
    if missing:
        print("ERROR: Missing required packages:")
        for package in missing:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing)}")
        return False
    print("All dependencies satisfied")
    
    # Check CUDA availability
    print("\nChecking CUDA availability...")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
    
    # Run integration tests
    print("\nRunning integration tests...")
    if not run_tests():
        print("ERROR: Integration tests failed")
        return False
    print("Integration tests passed")
    
    # Create sample simulation and visualizations
    try:
        trajectory = create_sample_simulation()
        create_sample_visualizations(trajectory)
    except Exception as e:
        print(f"\nERROR: Sample simulation failed: {str(e)}")
        return False
    
    print("\nSetup verification complete!")
    print("All components are properly connected and functioning")
    print("\nNext steps:")
    print("1. Explore the sample outputs in the 'sample_outputs' directory")
    print("2. Modify the configuration files to customize the simulation")
    print("3. Run your own simulation with: python scripts/run_simulation.py")
    return True

if __name__ == "__main__":
    start_time = time.time()
    success = main()
    end_time = time.time()
    
    print(f"\nVerification completed in {end_time - start_time:.2f} seconds")
    sys.exit(0 if success else 1)