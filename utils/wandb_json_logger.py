import os
import json
import time
from typing import Dict, Any, Optional, Union
from collections import defaultdict
import threading
import queue
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class WandbJSONLogger:
    """
    A JSON logger that mimics wandb's interface for drop-in replacement.
    Saves metrics to JSONL files and creates dynamic plots.
    """
    
    # Class-level variables to mimic wandb's global state
    _instance = None
    _initialized = False
    
    def __init__(self, project_name: str = None, config: Dict[str, Any] = None, 
                 save_dir: str = None, experiment_name: str = None, 
                 update_freq: int = 10, max_metrics_in_memory: int = 10000):
        """Initialize the logger."""
        self._project_name = project_name or "default_project"
        self._experiment_name = experiment_name or "experiment"
        self._save_dir = save_dir or "./logs"
        self._update_freq = update_freq
        self._max_metrics_in_memory = max_metrics_in_memory
        
        # Create experiment directory structure
        self._experiment_dir = os.path.join(self._save_dir, self._project_name, self._experiment_name)
        os.makedirs(self._experiment_dir, exist_ok=True)
        
        # Log files
        self._metrics_log_file = os.path.join(self._experiment_dir, "metrics.jsonl")
        self._config_file = os.path.join(self._experiment_dir, "config.json")
        self._plot_dir = os.path.join(self._experiment_dir, "plots")
        os.makedirs(self._plot_dir, exist_ok=True)
        
        # Initialize metrics storage
        self._metrics = defaultdict(list)
        self._step_count = 0
        self._last_plot_update = 0
        
        # Thread-safe plotting
        self._plot_queue = queue.Queue()
        self._plot_thread = None
        self._stop_plotting = threading.Event()
        
        # Initialize plots
        self._setup_plots()
        self._start_plot_thread()
        
        # Log initial config if provided
        if config is not None:
            self.config = config
            self._log_config(config)
        else:
            self.config = {}
            
        print(f"JSON Logger initialized. Logs will be saved to: {self._experiment_dir}")
    
    def _setup_plots(self):
        """Initialize matplotlib plots."""
        self._fig, self._axes = plt.subplots(2, 3, figsize=(15, 10))
        self._fig.suptitle(f'Training Metrics - {self._experiment_name}')
        plt.tight_layout()
        
    def _start_plot_thread(self):
        """Start background thread for plot updates."""
        self._plot_thread = threading.Thread(target=self._plot_worker, daemon=True)
        self._plot_thread.start()
        
    def _plot_worker(self):
        """Background worker for updating plots."""
        while not self._stop_plotting.is_set():
            try:
                self._plot_queue.get(timeout=1.0)
                self._update_plots_impl()
                # Clear queue
                while not self._plot_queue.empty():
                    try:
                        self._plot_queue.get_nowait()
                    except queue.Empty:
                        break
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Warning: Plot worker error: {e}")
                
    def _request_plot_update(self):
        """Request a plot update (non-blocking)."""
        try:
            self._plot_queue.put_nowait(True)
        except queue.Full:
            pass
    
    def log(self, metrics: Dict[str, Union[float, int, torch.Tensor]], step: Optional[int] = None):
        """
        Log metrics (mimics wandb.log interface).
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step (optional)
        """
        if step is None:
            step = self._step_count
            self._step_count += 1
        
        timestamp = time.time()
        
        # Process metrics
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                processed_metrics[key] = float(value.item())
            elif isinstance(value, (int, float, np.number)):
                processed_metrics[key] = float(value)
            else:
                processed_metrics[key] = str(value)
        
        # Log to JSONL file
        log_entry = {
            'step': step,
            'timestamp': timestamp,
            'metrics': processed_metrics
        }
        
        with open(self._metrics_log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
        
        # Update internal storage for plotting
        for key, value in processed_metrics.items():
            self._metrics[key].append((step, value))
            # Memory management
            if len(self._metrics[key]) > self._max_metrics_in_memory:
                self._metrics[key] = self._metrics[key][-self._max_metrics_in_memory:]
        
        # Update plots more frequently to handle individual metric logging
        if step % max(1, self._update_freq // 5) == 0 or step - self._last_plot_update >= max(1, self._update_freq // 5):
            self._last_plot_update = step
            self._request_plot_update()
    
    def _update_plots_impl(self):
        """Internal plot update implementation."""
        try:
            if not self._metrics:
                return
            
            # Clear all axes
            for ax in self._axes.flat:
                ax.clear()
            
            # Group metrics by category (train/val/other)
            train_metrics = {}
            val_metrics = {}
            other_metrics = {}
            
            for key in self._metrics.keys():
                if '/train_loader' in key:
                    clean_key = key.replace('/train_loader', '')
                    train_metrics[clean_key] = key
                elif '/valid_loader' in key:
                    clean_key = key.replace('/valid_loader', '')
                    val_metrics[clean_key] = key
                elif key.startswith('train/'):
                    clean_key = key.replace('train/', '')
                    train_metrics[clean_key] = key
                elif key.startswith('val/'):
                    clean_key = key.replace('val/', '')
                    val_metrics[clean_key] = key
                elif key.startswith('statistics/'):
                    # Keep statistics separate
                    other_metrics[key] = key
                else:
                    other_metrics[key] = key
            
            # Get unique metric names
            all_metric_names = set()
            all_metric_names.update(train_metrics.keys())
            all_metric_names.update(val_metrics.keys())
            all_metric_names.update(other_metrics.keys())
            
            metric_names = sorted(list(all_metric_names))[:6]  # Limit to 6 plots
            
            # Plot metrics
            for i, metric_name in enumerate(metric_names):
                row, col = divmod(i, 3)
                ax = self._axes[row, col]
                
                has_data = False
                
                # Plot training data
                if metric_name in train_metrics:
                    train_key = train_metrics[metric_name]
                    if train_key in self._metrics and len(self._metrics[train_key]) > 0:
                        steps, values = zip(*self._metrics[train_key])
                        ax.plot(steps, values, label='Train', color='blue', alpha=0.7, linewidth=1.5)
                        has_data = True
                
                # Plot validation data
                if metric_name in val_metrics:
                    val_key = val_metrics[metric_name]
                    if val_key in self._metrics and len(self._metrics[val_key]) > 0:
                        steps, values = zip(*self._metrics[val_key])
                        ax.plot(steps, values, label='Validation', color='red', alpha=0.7, linewidth=1.5)
                        has_data = True
                
                # Plot other data
                if metric_name in other_metrics:
                    other_key = other_metrics[metric_name]
                    if other_key in self._metrics and len(self._metrics[other_key]) > 0:
                        steps, values = zip(*self._metrics[other_key])
                        ax.plot(steps, values, label='Other', color='green', alpha=0.7, linewidth=1.5)
                        has_data = True
                
                if has_data:
                    ax.set_title(metric_name.replace('_', ' ').title(), fontsize=10)
                    ax.set_xlabel('Step', fontsize=9)
                    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=9)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=8)
                else:
                    ax.set_visible(False)
            
            # Hide unused subplots
            for i in range(len(metric_names), 6):
                row, col = divmod(i, 3)
                self._axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self._plot_dir, "training_metrics.png")
            self._fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            
            plot_path_svg = plot_path.replace('.png', '.svg')
            self._fig.savefig(plot_path_svg, bbox_inches='tight')
            
            # Save timestamped version periodically
            # if self._step_count % 100 == 0:
            #     timestamp = int(time.time())
            #     plot_path_timestamped = os.path.join(self._plot_dir, f"training_metrics_{timestamp}.png")
            #     self._fig.savefig(plot_path_timestamped, dpi=150, bbox_inches='tight')
                
        except Exception as e:
            print(f"Warning: Failed to update plots: {e}")
    
    def _log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        try:
            with open(self._config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save config: {e}")
    
    def finish(self):
        """Finish logging (mimics wandb.finish)."""
        try:
            # Stop plot thread
            self._stop_plotting.set()
            if self._plot_thread and self._plot_thread.is_alive():
                self._plot_thread.join(timeout=5.0)
            
            # Final plot update
            self._update_plots_impl()
            final_plot_path = os.path.join(self._plot_dir, "final_training_metrics.png")
            self._fig.savefig(final_plot_path, dpi=300, bbox_inches='tight')
            
            final_plot_path_svg = final_plot_path.replace('.png', '.svg')
            self._fig.savefig(final_plot_path_svg, bbox_inches='tight')
            
            
            # Save summary
            self._save_summary()
            
            plt.close(self._fig)
            print(f"Training complete. Logs saved to: {self._experiment_dir}")
        except Exception as e:
            print(f"Warning: Failed to finalize logger: {e}")
    
    def _save_summary(self):
        """Save summary statistics."""
        try:
            summary = {}
            for key, values in self._metrics.items():
                if values:
                    steps, vals = zip(*values)
                    summary[key] = {
                        'final_value': vals[-1],
                        'min_value': min(vals),
                        'max_value': max(vals),
                        'mean_value': sum(vals) / len(vals),
                        'total_steps': len(vals)
                    }
            
            summary_file = os.path.join(self._experiment_dir, "summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save summary: {e}")
    
    def __del__(self):
        """Cleanup when logger is destroyed."""
        try:
            if hasattr(self, '_stop_plotting'):
                self._stop_plotting.set()
            if hasattr(self, '_plot_thread') and self._plot_thread and self._plot_thread.is_alive():
                self._plot_thread.join(timeout=1.0)
            if hasattr(self, '_fig'):
                plt.close(self._fig)
        except:
            pass


# Global instance to mimic wandb's global state
_global_logger = None

def init(project: str = None, config: Dict[str, Any] = None, save_dir: str = None, 
         experiment_name: str = None, **kwargs):
    """Initialize the global logger (mimics wandb.init)."""
    global _global_logger
    _global_logger = WandbJSONLogger(
        project_name=project,
        config=config,
        save_dir=save_dir,
        experiment_name=experiment_name,
        **kwargs
    )
    return _global_logger

def log(metrics: Dict[str, Union[float, int, torch.Tensor]], step: Optional[int] = None):
    """Log metrics using the global logger (mimics wandb.log)."""
    if _global_logger is None:
        raise RuntimeError("Logger not initialized. Call init() first.")
    _global_logger.log(metrics, step)

def finish():
    """Finish logging (mimics wandb.finish)."""
    global _global_logger
    if _global_logger is not None:
        _global_logger.finish()
        _global_logger = None

# For backward compatibility, create a config object
config = {}

def set_config(cfg: Dict[str, Any]):
    """Set global config."""
    global config
    config.update(cfg)
    if _global_logger is not None:
        _global_logger._log_config(config)

# Image class placeholder (for compatibility)
class Image:
    def __init__(self, data):
        self.data = data
