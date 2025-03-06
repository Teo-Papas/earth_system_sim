"""
Visualization components for the physical system (atmosphere/ocean).
Implements specialized plots and animations for physical variables.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
from .base import BaseVisualizer

class PhysicalSystemVisualizer(BaseVisualizer):
    """Visualizer for physical system components."""
    
    def plot_temperature_field(
        self,
        temperature: np.ndarray,
        time_idx: int = -1,
        title: str = "Temperature Field",
        **kwargs
    ):
        """
        Plot temperature field as a contour map.
        
        Args:
            temperature: Temperature data array (time, height, width)
            time_idx: Time index to plot
            title: Plot title
            **kwargs: Additional arguments for contourf
        """
        fig, ax = self.create_figure()
        
        # Create contour plot
        im = ax.contourf(
            temperature[time_idx],
            cmap=self.temp_cmap,
            levels=20,
            **kwargs
        )
        
        # Add labels and colorbar
        self.set_labels(
            ax, title, "X (grid points)", "Y (grid points)"
        )
        self.add_colorbar(fig, im, "Temperature (K)")
        
        return fig, ax
    
    def create_wind_field_animation(
        self,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        interval: int = 100,
        title: str = "Wind Field Evolution"
    ) -> animation.FuncAnimation:
        """
        Create animation of wind field evolution.
        
        Args:
            u_wind: U-component of wind (time, height, width)
            v_wind: V-component of wind (time, height, width)
            interval: Animation interval in milliseconds
            title: Animation title
            
        Returns:
            Animation object
        """
        fig, ax = self.create_figure()
        ax.set_title(title)
        
        # Create grid for quiver plot
        x, y = np.meshgrid(
            np.arange(u_wind.shape[2]),
            np.arange(u_wind.shape[1])
        )
        
        # Initialize quiver plot
        Q = ax.quiver(
            x[::2, ::2], y[::2, ::2],
            u_wind[0, ::2, ::2], v_wind[0, ::2, ::2],
            scale=50
        )
        
        def update(frame):
            """Update function for animation."""
            Q.set_UVC(
                u_wind[frame, ::2, ::2],
                v_wind[frame, ::2, ::2]
            )
            return Q,
        
        # Create animation
        anim = self.create_animation(
            None, update, frames=len(u_wind),
            interval=interval, blit=True
        )
        
        return anim
    
    def plot_pressure_field(
        self,
        pressure: np.ndarray,
        time_idx: int = -1,
        with_streamlines: bool = True,
        title: str = "Pressure Field",
        **kwargs
    ):
        """
        Plot pressure field with optional streamlines.
        
        Args:
            pressure: Pressure data array (time, height, width)
            time_idx: Time index to plot
            with_streamlines: Whether to add streamlines
            title: Plot title
            **kwargs: Additional arguments for contourf
        """
        fig, ax = self.create_figure()
        
        # Create pressure contour plot
        im = ax.contourf(
            pressure[time_idx],
            cmap='viridis',
            levels=20,
            **kwargs
        )
        
        if with_streamlines:
            # Compute pressure gradients for streamlines
            dy, dx = np.gradient(pressure[time_idx])
            ax.streamplot(
                np.arange(pressure.shape[2]),
                np.arange(pressure.shape[1]),
                dx, dy, color='white', linewidth=0.5
            )
        
        # Add labels and colorbar
        self.set_labels(
            ax, title, "X (grid points)", "Y (grid points)"
        )
        self.add_colorbar(fig, im, "Pressure (Pa)")
        
        return fig, ax
    
    def create_interactive_temperature_plot(
        self,
        temperature: np.ndarray,
        times: np.ndarray
    ) -> go.Figure:
        """
        Create interactive temperature evolution plot.
        
        Args:
            temperature: Temperature data array (time, height, width)
            times: Time points array
            
        Returns:
            Plotly figure object
        """
        # Compute spatial mean temperature
        mean_temp = np.mean(temperature, axis=(1, 2))
        
        # Create interactive plot
        fig = self.create_interactive_figure(
            {
                'time': times,
                'temperature': mean_temp
            },
            plot_type='line',
            x='time',
            y='temperature',
            title='Mean Temperature Evolution',
            labels={'temperature': 'Temperature (K)'}
        )
        
        return fig
    
    def plot_conservation_metrics(
        self,
        metrics: Dict[str, np.ndarray],
        times: np.ndarray,
        title: str = "Conservation Law Metrics"
    ):
        """
        Plot conservation law metrics over time.
        
        Args:
            metrics: Dictionary of conservation metrics
            times: Time points array
            title: Plot title
        """
        fig, ax = self.create_figure()
        
        # Plot each conservation metric
        for name, values in metrics.items():
            ax.plot(times, values, label=name)
        
        # Add labels and legend
        self.set_labels(
            ax, title, "Time", "Conservation Error"
        )
        ax.legend()
        ax.grid(True)
        
        return fig, ax
    
    def create_hovmoller_diagram(
        self,
        data: np.ndarray,
        times: np.ndarray,
        spatial_coord: np.ndarray,
        title: str,
        **kwargs
    ):
        """
        Create Hovmöller diagram for variable evolution.
        
        Args:
            data: Data array (time, space)
            times: Time points array
            spatial_coord: Spatial coordinates array
            title: Plot title
            **kwargs: Additional arguments for pcolormesh
        """
        fig, ax = self.create_figure()
        
        # Create Hovmöller plot
        im = ax.pcolormesh(
            spatial_coord,
            times,
            data,
            shading='auto',
            **kwargs
        )
        
        # Add labels and colorbar
        self.set_labels(
            ax, title, "Space", "Time"
        )
        self.add_colorbar(fig, im, title)
        
        return fig, ax
    
    def create_energy_spectrum_animation(
        self,
        spectrum: np.ndarray,
        wavenumbers: np.ndarray,
        interval: int = 200,
        title: str = "Energy Spectrum Evolution"
    ) -> animation.FuncAnimation:
        """
        Create animation of energy spectrum evolution.
        
        Args:
            spectrum: Energy spectrum data (time, wavenumbers)
            wavenumbers: Wavenumber array
            interval: Animation interval in milliseconds
            title: Animation title
            
        Returns:
            Animation object
        """
        fig, ax = self.create_figure()
        ax.set_title(title)
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Energy")
        ax.set_yscale('log')
        ax.set_xscale('log')
        
        # Initialize line plot
        line, = ax.plot(wavenumbers, spectrum[0])
        
        def update(frame):
            """Update function for animation."""
            line.set_ydata(spectrum[frame])
            return line,
        
        # Create animation
        anim = self.create_animation(
            None, update, frames=len(spectrum),
            interval=interval, blit=True
        )
        
        return anim