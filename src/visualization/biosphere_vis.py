"""
Visualization components for the biosphere system.
Implements specialized plots and animations for ecosystem dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from .base import BaseVisualizer

class BiosphereVisualizer(BaseVisualizer):
    """Visualizer for biosphere system components."""
    
    def plot_vegetation_distribution(
        self,
        vegetation: np.ndarray,
        time_idx: int = -1,
        title: str = "Vegetation Distribution",
        **kwargs
    ):
        """
        Plot vegetation distribution as a heatmap.
        
        Args:
            vegetation: Vegetation data array (time, height, width)
            time_idx: Time index to plot
            title: Plot title
            **kwargs: Additional arguments for imshow
        """
        fig, ax = self.create_figure()
        
        # Create vegetation heatmap
        im = ax.imshow(
            vegetation[time_idx],
            cmap=self.veg_cmap,
            aspect='equal',
            **kwargs
        )
        
        # Add labels and colorbar
        self.set_labels(
            ax, title, "X (grid points)", "Y (grid points)"
        )
        self.add_colorbar(fig, im, "Vegetation Density")
        
        return fig, ax
    
    def create_vegetation_animation(
        self,
        vegetation: np.ndarray,
        interval: int = 200,
        title: str = "Vegetation Evolution"
    ) -> animation.FuncAnimation:
        """
        Create animation of vegetation pattern evolution.
        
        Args:
            vegetation: Vegetation data array (time, height, width)
            interval: Animation interval in milliseconds
            title: Animation title
            
        Returns:
            Animation object
        """
        fig, ax = self.create_figure()
        ax.set_title(title)
        
        # Initialize plot
        im = ax.imshow(
            vegetation[0],
            cmap=self.veg_cmap,
            aspect='equal',
            animated=True
        )
        
        self.add_colorbar(fig, im, "Vegetation Density")
        
        def update(frame):
            """Update function for animation."""
            im.set_array(vegetation[frame])
            return [im]
        
        # Create animation
        anim = self.create_animation(
            None, update, frames=len(vegetation),
            interval=interval, blit=True
        )
        
        return anim
    
    def plot_carbon_flux_timeseries(
        self,
        carbon_flux: np.ndarray,
        times: np.ndarray,
        title: str = "Carbon Flux Evolution"
    ):
        """
        Plot carbon flux time series.
        
        Args:
            carbon_flux: Carbon flux data array (time)
            times: Time points array
            title: Plot title
        """
        fig, ax = self.create_figure()
        
        # Plot carbon flux
        ax.plot(times, carbon_flux, 'g-', label='Carbon Flux')
        ax.fill_between(
            times,
            carbon_flux,
            alpha=0.3,
            color='green'
        )
        
        # Add labels and legend
        self.set_labels(
            ax, title, "Time", "Carbon Flux"
        )
        ax.legend()
        ax.grid(True)
        
        return fig, ax
    
    def create_interactive_ecosystem_dashboard(
        self,
        vegetation: np.ndarray,
        carbon_flux: np.ndarray,
        soil_moisture: np.ndarray,
        times: np.ndarray
    ) -> go.Figure:
        """
        Create interactive dashboard for ecosystem variables.
        
        Args:
            vegetation: Vegetation data array (time)
            carbon_flux: Carbon flux data array (time)
            soil_moisture: Soil moisture data array (time)
            times: Time points array
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = go.Figure()
        
        # Add vegetation trace
        fig.add_trace(
            go.Scatter(
                x=times,
                y=np.mean(vegetation, axis=(1, 2)),
                name='Mean Vegetation',
                line=dict(color='green')
            )
        )
        
        # Add carbon flux trace
        fig.add_trace(
            go.Scatter(
                x=times,
                y=carbon_flux,
                name='Carbon Flux',
                line=dict(color='blue')
            )
        )
        
        # Add soil moisture trace
        fig.add_trace(
            go.Scatter(
                x=times,
                y=soil_moisture,
                name='Soil Moisture',
                line=dict(color='brown')
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Ecosystem Variables Evolution',
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_vegetation_policy_actions(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        title: str = "Vegetation Policy Actions"
    ):
        """
        Plot vegetation policy actions against states.
        
        Args:
            states: State data array (time, state_dim)
            actions: Action data array (time, action_dim)
            title: Plot title
        """
        fig, ax = self.create_figure()
        
        # Create scatter plot of actions vs states
        scatter = ax.scatter(
            states[:, 0],  # Assuming first state dim is most relevant
            actions[:, 0],  # Assuming first action dim is most relevant
            c=np.arange(len(states)),  # Color by time
            cmap='viridis',
            alpha=0.6
        )
        
        # Add labels and colorbar
        self.set_labels(
            ax, title, "State", "Action"
        )
        self.add_colorbar(fig, scatter, "Time Step")
        
        return fig, ax
    
    def create_spatial_correlation_plot(
        self,
        vegetation: np.ndarray,
        temperature: np.ndarray,
        precipitation: np.ndarray,
        time_idx: int = -1,
        title: str = "Ecosystem-Climate Correlations"
    ):
        """
        Create spatial correlation plot between vegetation and climate variables.
        
        Args:
            vegetation: Vegetation data array (time, height, width)
            temperature: Temperature data array (time, height, width)
            precipitation: Precipitation data array (time, height, width)
            time_idx: Time index to plot
            title: Plot title
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Compute correlations
        temp_corr = np.corrcoef(
            vegetation[time_idx].flatten(),
            temperature[time_idx].flatten()
        )[0, 1]
        
        precip_corr = np.corrcoef(
            vegetation[time_idx].flatten(),
            precipitation[time_idx].flatten()
        )[0, 1]
        
        # Create scatter plots
        axes[0].scatter(
            temperature[time_idx].flatten(),
            vegetation[time_idx].flatten(),
            alpha=0.5,
            label=f'Correlation: {temp_corr:.2f}'
        )
        axes[0].set_xlabel('Temperature')
        axes[0].set_ylabel('Vegetation Density')
        axes[0].legend()
        
        axes[1].scatter(
            precipitation[time_idx].flatten(),
            vegetation[time_idx].flatten(),
            alpha=0.5,
            label=f'Correlation: {precip_corr:.2f}'
        )
        axes[1].set_xlabel('Precipitation')
        axes[1].set_ylabel('Vegetation Density')
        axes[1].legend()
        
        fig.suptitle(title)
        
        return fig, axes