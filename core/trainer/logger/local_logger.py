import torch
import logging
import time
from pathlib import Path
import json
from typing import Dict, Any, Optional
from datetime import datetime

class TrainingLogger:
    def __init__(
        self,
        output_dir: str,
        project_name: str,
        log_every_n_steps: int = 100,
        save_every_n_steps: int = 1000,
        save_best_only: bool = True
    ):
        """
        Initialize training logger with various logging options.

        Args:
            output_dir: Directory to save checkpoints and logs
            project_name: Name of the project
            use_wandb: Whether to use Weights & Biases logging
            log_every_n_steps: How often to log metrics
            save_every_n_steps: How often to save checkpoints
            save_best_only: Whether to save only the best model
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        self.log_file = self.output_dir / 'training.log'
        self.setup_logging()

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.start_time = time.time()
        self.last_log_time = self.start_time

        # Configuration
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_steps = save_every_n_steps

        # Save configuration
        self.save_config({
            'output_dir': str(output_dir),
            'project_name': project_name,
            'log_every_n_steps': log_every_n_steps,
            'save_every_n_steps': save_every_n_steps,
            'save_best_only': save_best_only,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )

    def save_config(self, config: Dict[str, Any]):
        """Save configuration to JSON file."""
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        force_log: bool = False
    ):
        """
        Log metrics to all configured outputs.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (uses global_step if not provided)
            force_log: Whether to log regardless of log_every_n_steps
        """
        if step is not None:
            self.global_step = step

        # Check if we should log
        if not force_log and self.global_step % self.log_every_n_steps != 0:
            return

        # Calculate time statistics
        current_time = time.time()
        elapsed = current_time - self.start_time
        elapsed_since_last = current_time - self.last_log_time
        steps_since_last = self.log_every_n_steps
        steps_per_second = steps_since_last / elapsed_since_last

        # Add timing metrics
        metrics.update({
            'elapsed_time': elapsed,
            'steps_per_second': steps_per_second
        })

        # Log to terminal and file
        log_str = f'Step {self.global_step}: ' + ', '.join(
            f'{k}: {v}' for k, v in metrics.items()
        )
        logging.info(log_str)

        self.last_log_time = current_time

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: float,
        extra_data: Optional[Dict[str, Any]] = None,
        force_save: bool = False
    ):
        """
        Save model checkpoint.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            loss: Current loss value
            extra_data: Additional data to save in checkpoint
            force_save: Whether to save regardless of save_every_n_steps
        """
        # Check if we should save
        should_save = (
            force_save or
            self.global_step % self.save_every_n_steps == 0 or
            (self.save_best_only and loss < self.best_loss)
        )

        if not should_save:
            return

        # Update best loss if needed
        if loss < self.best_loss:
            self.best_loss = loss

        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': self.global_step,
            'loss': loss,
            'best_loss': self.best_loss
        }

        if extra_data:
            checkpoint.update(extra_data)

        # Save checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_{self.global_step}.pt'
        torch.save(checkpoint, checkpoint_path)

        logging.info(f'Saved checkpoint at step {self.global_step}')

    def finish(self):
        """Cleanup and final logging."""
        total_time = time.time() - self.start_time
        logging.info(f'Training finished. Total time: {total_time:.2f}s')