"""
Base visualization components for Earth system simulation.
Provides core functionality for static, dynamic, and interactive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Union, Any
import imageio
from pathlib import Path
import holoviews as hv
hv.extension('bokeh')

class BaseVisualizer:
    """Base class for visualization components."""
    
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        config: Dict[str, Any],
        output_dir: Optional[str] = None
    ):
        """
        Initialize visualizer.
        
        Args:
            data: Dictionary containing simulation data
            config: Visualization configuration
            output_dir: Directory for saving outputs
        """
        self.data = data
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / 'visualizations'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up color maps
        self.setup_colormaps()
        
    def setup_colormaps(self):
        """Setup custom colormaps for different variables."""
        # Temperature colormap (blue to red)
        self.temp_cmap = LinearSegmentedColormap.from_list(
            'temperature',
            ['#313695', '#4575b4', '#74add1', '#abd9e9',
             '#fee090', '#fdae61', '#f46d43', '#d73027']
        )
        
        # Vegetation colormap (brown to green)
        self.veg_cmap = LinearSegmentedColormap.from_list(
            'vegetation',
            ['#8c510a', '#bf812d', '#dfc27d', '#f6e8c3',
             '#c7eae5', '#80cdc1', '#35978f', '#01665e']
        )
        
        # Elevation colormap (green to brown to white)
        self.elevation_cmap = LinearSegmentedColormap.from_list(
            'elevation',
            ['#276419', '#4d9221', '#7fbc41', '#b8e186',
             '#e6f5d0', '#f6e8c3', '#dfc27d', '#bf812d',
             '#8c510a', '#543005']
        )
        
    def create_figure(
        self,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a new figure and axes.
        
        Args:
            figsize: Figure size in inches
            dpi: Dots per inch
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        return fig, ax
    
    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        **kwargs
    ):
        """
        Save figure to file.
        
        Args:
            fig: Figure to save
            filename: Output filename
            **kwargs: Additional arguments for savefig
        """
        output_path = self.output_dir / filename
        fig.savefig(output_path, **kwargs)
        plt.close(fig)
        
    def create_animation(
        self,
        data: np.ndarray,
        update_func: callable,
        frames: int,
        interval: int = 50,
        blit: bool = True,
        **kwargs
    ) -> animation.FuncAnimation:
        """
        Create animation using matplotlib.
        
        Args:
            data: Data to animate
            update_func: Function to update animation frame
            frames: Number of frames
            interval: Delay between frames in milliseconds
            blit: Whether to use blitting
            **kwargs: Additional arguments for FuncAnimation
            
        Returns:
            Animation object
        """
        fig, ax = self.create_figure()
        anim = animation.FuncAnimation(
            fig, update_func, frames=frames,
            interval=interval, blit=blit, **kwargs
        )
        return anim
    
    def save_animation(
        self,
        anim: animation.FuncAnimation,
        filename: str,
        fps: int = 30,
        **kwargs
    ):
        """
        Save animation to file.
        
        Args:
            anim: Animation to save
            filename: Output filename
            fps: Frames per second
            **kwargs: Additional arguments for save
        """
        output_path = self.output_dir / filename
        if filename.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps, **kwargs)
        else:
            anim.save(output_path, fps=fps, **kwargs)
            
    def create_interactive_figure(
        self,
        data: Dict[str, np.ndarray],
        plot_type: str = 'scatter',
        **kwargs
    ) -> go.Figure:
        """
        Create interactive figure using plotly.
        
        Args:
            data: Data to plot
            plot_type: Type of plot
            **kwargs: Additional arguments for plot
            
        Returns:
            Plotly figure object
        """
        if plot_type == 'scatter':
            fig = px.scatter(data, **kwargs)
        elif plot_type == 'line':
            fig = px.line(data, **kwargs)
        elif plot_type == 'heatmap':
            fig = px.imshow(data, **kwargs)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
            
        return fig
    
    def save_interactive_figure(
        self,
        fig: go.Figure,
        filename: str,
        **kwargs
    ):
        """
        Save interactive figure to HTML file.
        
        Args:
            fig: Figure to save
            filename: Output filename
            **kwargs: Additional arguments for write_html
        """
        output_path = self.output_dir / filename
        fig.write_html(output_path, **kwargs)
        
    def create_holoviews_plot(
        self,
        data: Union[np.ndarray, Dict[str, np.ndarray]],
        plot_type: str,
        **kwargs
    ) -> hv.core.ViewableElement:
        """
        Create interactive plot using holoviews.
        
        Args:
            data: Data to plot
            plot_type: Type of plot
            **kwargs: Additional arguments for plot
            
        Returns:
            Holoviews plot object
        """
        if plot_type == 'image':
            plot = hv.Image(data, **kwargs)
        elif plot_type == 'curve':
            plot = hv.Curve(data, **kwargs)
        elif plot_type == 'scatter':
            plot = hv.Scatter(data, **kwargs)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
            
        return plot
    
    def save_holoviews_plot(
        self,
        plot: hv.core.ViewableElement,
        filename: str,
        **kwargs
    ):
        """
        Save holoviews plot to file.
        
        Args:
            plot: Plot to save
            filename: Output filename
            **kwargs: Additional arguments for save
        """
        output_path = self.output_dir / filename
        hv.save(plot, output_path, **kwargs)
        
    @staticmethod
    def add_colorbar(
        fig: plt.Figure,
        mappable: plt.cm.ScalarMappable,
        label: str,
        **kwargs
    ):
        """
        Add colorbar to figure.
        
        Args:
            fig: Figure to add colorbar to
            mappable: ScalarMappable for colorbar
            label: Colorbar label
            **kwargs: Additional arguments for colorbar
        """
        cbar = fig.colorbar(mappable, **kwargs)
        cbar.set_label(label)
        
    @staticmethod
    def set_labels(
        ax: plt.Axes,
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Set plot labels.
        
        Args:
            ax: Axes to set labels on
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            **kwargs: Additional arguments for labels
        """
        ax.set_title(title, **kwargs.get('title_kwargs', {}))
        ax.set_xlabel(xlabel, **kwargs.get('xlabel_kwargs', {}))
        ax.set_ylabel(ylabel, **kwargs.get('ylabel_kwargs', {}))