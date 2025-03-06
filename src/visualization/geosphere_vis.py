"""
Visualization components for the geosphere system.
Implements specialized plots and animations for geological processes.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
from .base import BaseVisualizer

class GeosphereVisualizer(BaseVisualizer):
    """Visualizer for geosphere system components."""
    
    def plot_topography_3d(
        self,
        topography: np.ndarray,
        time_idx: int = -1,
        title: str = "Topographic Surface",
        **kwargs
    ):
        """
        Create 3D surface plot of topography.
        
        Args:
            topography: Topography data array (time, height, width)
            time_idx: Time index to plot
            title: Plot title
            **kwargs: Additional arguments for plot_surface
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mesh grid
        x, y = np.meshgrid(
            np.arange(topography.shape[2]),
            np.arange(topography.shape[1])
        )
        
        # Create surface plot
        surf = ax.plot_surface(
            x, y, topography[time_idx],
            cmap=self.elevation_cmap,
            **kwargs
        )
        
        # Add labels and colorbar
        ax.set_title(title)
        ax.set_xlabel('X (grid points)')
        ax.set_ylabel('Y (grid points)')
        ax.set_zlabel('Elevation')
        
        fig.colorbar(surf, ax=ax, label='Elevation (m)')
        
        return fig, ax
    
    def create_erosion_animation(
        self,
        topography: np.ndarray,
        erosion_rate: np.ndarray,
        interval: int = 500,
        title: str = "Erosion Process"
    ) -> animation.FuncAnimation:
        """
        Create animation of erosion process.
        
        Args:
            topography: Topography data array (time, height, width)
            erosion_rate: Erosion rate data array (time, height, width)
            interval: Animation interval in milliseconds
            title: Animation title
            
        Returns:
            Animation object
        """
        fig = plt.figure(figsize=(15, 5))
        
        # Create subplots for topography and erosion rate
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # Initialize plots
        im1 = ax1.imshow(
            topography[0],
            cmap=self.elevation_cmap
        )
        im2 = ax2.imshow(
            erosion_rate[0],
            cmap='RdYlBu_r'
        )
        
        # Add colorbars
        plt.colorbar(im1, ax=ax1, label='Elevation (m)')
        plt.colorbar(im2, ax=ax2, label='Erosion Rate')
        
        ax1.set_title('Topography')
        ax2.set_title('Erosion Rate')
        
        def update(frame):
            """Update function for animation."""
            im1.set_array(topography[frame])
            im2.set_array(erosion_rate[frame])
            return [im1, im2]
        
        # Create animation
        anim = self.create_animation(
            None, update, frames=len(topography),
            interval=interval, blit=True
        )
        
        fig.suptitle(title)
        
        return anim
    
    def plot_geological_cross_section(
        self,
        topography: np.ndarray,
        soil_types: np.ndarray,
        y_idx: int,
        time_idx: int = -1,
        title: str = "Geological Cross Section"
    ):
        """
        Plot geological cross section showing topography and soil types.
        
        Args:
            topography: Topography data array (time, height, width)
            soil_types: Soil type data array (time, height, width)
            y_idx: Y-index for cross section
            time_idx: Time index to plot
            title: Plot title
        """
        fig, ax = self.create_figure()
        
        # Plot topography profile
        ax.plot(
            topography[time_idx, y_idx, :],
            'k-',
            label='Surface'
        )
        
        # Plot soil types as colored regions
        for i in range(soil_types.shape[-1]):
            ax.fill_between(
                range(soil_types.shape[2]),
                0,
                topography[time_idx, y_idx, :],
                where=soil_types[time_idx, y_idx, :] == i,
                alpha=0.5,
                label=f'Soil Type {i}'
            )
        
        # Add labels and legend
        self.set_labels(
            ax, title, "X (grid points)", "Elevation (m)"
        )
        ax.legend()
        
        return fig, ax
    
    def create_interactive_topography_viewer(
        self,
        topography: np.ndarray,
        times: np.ndarray
    ) -> go.Figure:
        """
        Create interactive 3D topography viewer.
        
        Args:
            topography: Topography data array (time, height, width)
            times: Time points array
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add surface for initial topography
        fig.add_trace(
            go.Surface(
                z=topography[0],
                colorscale='earth',
                name='t=0'
            )
        )
        
        # Create sliders for time control
        steps = []
        for i in range(len(times)):
            step = dict(
                method="update",
                args=[{"z": [topography[i]]}],
                label=f"t={times[i]:.1f}"
            )
            steps.append(step)
        
        sliders = [dict(
            active=0,
            steps=steps
        )]
        
        # Update layout
        fig.update_layout(
            title='Interactive Topography Evolution',
            scene=dict(
                xaxis_title='X (grid points)',
                yaxis_title='Y (grid points)',
                zaxis_title='Elevation (m)'
            ),
            sliders=sliders
        )
        
        return fig
    
    def plot_long_term_trends(
        self,
        mean_elevation: np.ndarray,
        erosion_total: np.ndarray,
        times: np.ndarray,
        title: str = "Long-term Geological Trends"
    ):
        """
        Plot long-term trends in geological processes.
        
        Args:
            mean_elevation: Mean elevation data array (time)
            erosion_total: Total erosion data array (time)
            times: Time points array
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot mean elevation
        ax1.plot(times, mean_elevation, 'b-', label='Mean Elevation')
        ax1.set_ylabel('Elevation (m)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot cumulative erosion
        ax2.plot(times, erosion_total, 'r-', label='Cumulative Erosion')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Erosion')
        ax2.legend()
        ax2.grid(True)
        
        fig.suptitle(title)
        
        return fig, (ax1, ax2)
    
    def plot_policy_performance(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        title: str = "Geosphere Policy Performance"
    ):
        """
        Plot policy performance metrics.
        
        Args:
            states: State data array (time, state_dim)
            actions: Action data array (time, action_dim)
            rewards: Reward data array (time)
            title: Plot title
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot state evolution
        for i in range(states.shape[1]):
            axes[0].plot(states[:, i], label=f'State {i}')
        axes[0].set_ylabel('State Values')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot actions
        for i in range(actions.shape[1]):
            axes[1].plot(actions[:, i], label=f'Action {i}')
        axes[1].set_ylabel('Action Values')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot rewards
        axes[2].plot(rewards, 'g-', label='Rewards')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Reward')
        axes[2].legend()
        axes[2].grid(True)
        
        fig.suptitle(title)
        
        return fig, axes