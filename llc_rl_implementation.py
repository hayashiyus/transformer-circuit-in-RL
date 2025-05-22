"""
Local Learning Coefficient (LLC) and Refined LLC Implementation
for Reinforcement Learning Fine-tuning of Transformer Models

This implementation is based on the research papers:
1. "The Local Learning Coefficient: A Singularity-Aware Complexity Measure" (2308.12108)
2. "Differentiation and Specialization of Attention Heads via the Refined Local Learning Coefficient" (2410.02984)
3. "Reinforcement Learning Finetunes Small Subnetworks in Large Language Models" (2505.11711)
4. "Loss Landscape Degeneracy Drives Stagewise Development in Transformers" (2402.02364)

And the devinterp library: https://github.com/timaeus-research/devinterp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging
from collections import defaultdict
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLCConfig:
    """Configuration for LLC estimation"""
    temperature: float = 1.0  # Inverse temperature (beta)
    sgld_steps: int = 1000    # Number of SGLD sampling steps
    sgld_lr: float = 1e-6     # SGLD learning rate
    burn_in_ratio: float = 0.9  # Fraction of steps to discard as burn-in
    batch_size: int = 32      # Batch size for SGLD sampling
    epsilon: float = 1e-8     # Small constant for numerical stability
    n_chains: int = 1         # Number of independent SGLD chains

class SGLDSampler:
    """
    Stochastic Gradient Langevin Dynamics sampler for LLC estimation
    Based on Welling & Teh (2011) and adapted for LLC calculation
    """
    
    def __init__(self, model: nn.Module, config: LLCConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
    def sample(self, dataloader: DataLoader, 
               loss_fn: Callable, 
               center_params: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Sample parameters using SGLD around center_params
        
        Args:
            dataloader: DataLoader for the dataset
            loss_fn: Loss function
            center_params: Center point for sampling (θ*)
            
        Returns:
            List of sampled parameter dictionaries
        """
        samples = []
        
        # Initialize model at center point
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in center_params:
                    param.copy_(center_params[name])
        
        # SGLD sampling loop
        for step in range(self.config.sgld_steps):
            # Forward pass on random batch
            batch = next(iter(dataloader))
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs, targets = batch.to(self.device), batch.to(self.device)
            
            # Compute loss and gradients
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Add prior term (Gaussian centered at center_params)
            prior_loss = 0.0
            for name, param in self.model.named_parameters():
                if name in center_params and param.requires_grad:
                    prior_loss += 0.5 * self.config.temperature * torch.sum(
                        (param - center_params[name])**2
                    )
            
            total_loss = loss + prior_loss
            total_loss.backward()
            
            # SGLD update with noise injection
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        # Add gradient descent step
                        param.add_(param.grad, alpha=-self.config.sgld_lr)
                        
                        # Add Gaussian noise
                        noise_std = np.sqrt(2 * self.config.sgld_lr / self.config.temperature)
                        noise = torch.randn_like(param) * noise_std
                        param.add_(noise)
            
            # Store sample (after burn-in)
            if step >= int(self.config.sgld_steps * self.config.burn_in_ratio):
                sample = {}
                for name, param in self.model.named_parameters():
                    sample[name] = param.clone().detach()
                samples.append(sample)
        
        return samples

class LLCEstimator:
    """
    Local Learning Coefficient estimator
    Based on "The Local Learning Coefficient: A Singularity-Aware Complexity Measure"
    """
    
    def __init__(self, model: nn.Module, config: LLCConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
    def estimate_llc(self, dataloader: DataLoader, 
                    loss_fn: Callable,
                    center_params: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Estimate the Local Learning Coefficient
        
        Args:
            dataloader: DataLoader for the dataset
            loss_fn: Loss function
            center_params: Center point for LLC estimation (if None, uses current params)
            
        Returns:
            Dictionary containing LLC estimate and related metrics
        """
        if center_params is None:
            center_params = {name: param.clone().detach() 
                           for name, param in self.model.named_parameters()}
        
        # Sample using SGLD
        sampler = SGLDSampler(self.model, self.config)
        samples = sampler.sample(dataloader, loss_fn, center_params)
        
        if not samples:
            logger.warning("No samples collected after burn-in")
            return {"llc_mean": 0.0, "llc_std": 0.0, "n_samples": 0}
        
        # Compute loss at center point
        loss_center = self._compute_loss(dataloader, loss_fn, center_params)
        
        # Compute expected loss under perturbation
        loss_values = []
        for sample in samples:
            loss_sample = self._compute_loss(dataloader, loss_fn, sample)
            loss_values.append(loss_sample.item())
        
        loss_mean = np.mean(loss_values)
        loss_std = np.std(loss_values)
        
        # Compute LLC estimate using formula from paper
        # LLC_hat = (E[L(θ)] - L(θ*)) / log(n)
        n_samples = len(dataloader.dataset)
        if n_samples <= 1:
            logger.warning("Dataset too small for reliable LLC estimation")
            return {"llc_mean": 0.0, "llc_std": 0.0, "n_samples": len(samples)}
        
        llc_estimate = (loss_mean - loss_center.item()) / np.log(n_samples)
        
        # Estimate uncertainty in LLC
        loss_sem = loss_std / np.sqrt(len(loss_values))
        llc_std = loss_sem / np.log(n_samples)
        
        return {
            "llc_mean": llc_estimate,
            "llc_std": llc_std,
            "loss_center": loss_center.item(),
            "loss_perturbed_mean": loss_mean,
            "loss_perturbed_std": loss_std,
            "n_samples": len(samples)
        }
    
    def _compute_loss(self, dataloader: DataLoader, 
                     loss_fn: Callable,
                     params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss with given parameters"""
        # Set model parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.copy_(params[name])
        
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    inputs, targets = batch.to(self.device), batch.to(self.device)
                
                outputs = self.model(inputs)
                
                # Handle different output formats
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_loss += loss.item()
                n_batches += 1
        
        return torch.tensor(total_loss / n_batches if n_batches > 0 else 0.0)

class RefinedLLCEstimator:
    """
    Refined Local Learning Coefficient estimator for component-wise analysis
    Based on "Differentiation and Specialization of Attention Heads via the Refined Local Learning Coefficient"
    """
    
    def __init__(self, model: nn.Module, config: LLCConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
    def estimate_refined_llc(self, dataloader: DataLoader,
                           loss_fn: Callable,
                           component_groups: Dict[str, List[str]],
                           center_params: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Dict[str, float]]:
        """
        Estimate refined LLC for different model components
        
        Args:
            dataloader: DataLoader for the dataset
            loss_fn: Loss function
            component_groups: Dictionary mapping component names to parameter name patterns
            center_params: Center point for LLC estimation
            
        Returns:
            Dictionary mapping component names to their LLC estimates
        """
        if center_params is None:
            center_params = {name: param.clone().detach() 
                           for name, param in self.model.named_parameters()}
        
        results = {}
        
        for component_name, param_patterns in component_groups.items():
            logger.info(f"Estimating refined LLC for component: {component_name}")
            
            # Select parameters for this component
            component_params = self._select_component_params(param_patterns, center_params)
            
            if not component_params:
                logger.warning(f"No parameters found for component {component_name}")
                continue
            
            # Estimate LLC for this component
            llc_result = self._estimate_component_llc(
                dataloader, loss_fn, component_params, center_params
            )
            results[component_name] = llc_result
        
        return results
    
    def _select_component_params(self, param_patterns: List[str], 
                               all_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Select parameters matching the given patterns"""
        component_params = {}
        
        for param_name in all_params:
            for pattern in param_patterns:
                if pattern in param_name:
                    component_params[param_name] = all_params[param_name]
                    break
        
        return component_params
    
    def _estimate_component_llc(self, dataloader: DataLoader,
                              loss_fn: Callable,
                              component_params: Dict[str, torch.Tensor],
                              center_params: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Estimate LLC for a specific component"""
        # Create modified config for component-specific sampling
        component_config = copy.deepcopy(self.config)
        
        # Sample only the component parameters, keeping others fixed
        sampler = ComponentSGLDSampler(self.model, component_config, component_params)
        samples = sampler.sample(dataloader, loss_fn, center_params)
        
        if not samples:
            return {"llc_mean": 0.0, "llc_std": 0.0, "n_samples": 0}
        
        # Compute loss at center point
        loss_center = self._compute_loss(dataloader, loss_fn, center_params)
        
        # Compute expected loss under component perturbation
        loss_values = []
        for sample in samples:
            loss_sample = self._compute_loss(dataloader, loss_fn, sample)
            loss_values.append(loss_sample.item())
        
        loss_mean = np.mean(loss_values)
        loss_std = np.std(loss_values)
        
        # Compute refined LLC
        n_samples = len(dataloader.dataset)
        if n_samples <= 1:
            return {"llc_mean": 0.0, "llc_std": 0.0, "n_samples": len(samples)}
        
        llc_estimate = (loss_mean - loss_center.item()) / np.log(n_samples)
        loss_sem = loss_std / np.sqrt(len(loss_values))
        llc_std = loss_sem / np.log(n_samples)
        
        return {
            "llc_mean": llc_estimate,
            "llc_std": llc_std,
            "loss_center": loss_center.item(),
            "loss_perturbed_mean": loss_mean,
            "loss_perturbed_std": loss_std,
            "n_samples": len(samples),
            "n_component_params": len(component_params)
        }
    
    def _compute_loss(self, dataloader: DataLoader,
                     loss_fn: Callable,
                     params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss with given parameters"""
        # Set model parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.copy_(params[name])
        
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    inputs, targets = batch.to(self.device), batch.to(self.device)
                
                outputs = self.model(inputs)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_loss += loss.item()
                n_batches += 1
        
        return torch.tensor(total_loss / n_batches if n_batches > 0 else 0.0)

class ComponentSGLDSampler:
    """
    SGLD sampler that only perturbs specific component parameters
    """
    
    def __init__(self, model: nn.Module, config: LLCConfig, component_params: Dict[str, torch.Tensor]):
        self.model = model
        self.config = config
        self.component_params = set(component_params.keys())
        self.device = next(model.parameters()).device
    
    def sample(self, dataloader: DataLoader,
               loss_fn: Callable,
               center_params: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Sample only component parameters while keeping others fixed"""
        samples = []
        
        # Initialize model at center point
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in center_params:
                    param.copy_(center_params[name])
        
        # SGLD sampling loop
        for step in range(self.config.sgld_steps):
            # Forward pass
            batch = next(iter(dataloader))
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs, targets = batch.to(self.device), batch.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Add prior term only for component parameters
            prior_loss = 0.0
            for name, param in self.model.named_parameters():
                if name in self.component_params and name in center_params:
                    prior_loss += 0.5 * self.config.temperature * torch.sum(
                        (param - center_params[name])**2
                    )
            
            total_loss = loss + prior_loss
            total_loss.backward()
            
            # SGLD update only for component parameters
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in self.component_params and param.grad is not None:
                        # Gradient descent step
                        param.add_(param.grad, alpha=-self.config.sgld_lr)
                        
                        # Add Gaussian noise
                        noise_std = np.sqrt(2 * self.config.sgld_lr / self.config.temperature)
                        noise = torch.randn_like(param) * noise_std
                        param.add_(noise)
            
            # Store sample after burn-in
            if step >= int(self.config.sgld_steps * self.config.burn_in_ratio):
                sample = {name: param.clone().detach() 
                         for name, param in self.model.named_parameters()}
                samples.append(sample)
        
        return samples

class RLLLCTracker:
    """
    Main class for tracking LLC during reinforcement learning fine-tuning
    Based on "Reinforcement Learning Finetunes Small Subnetworks in Large Language Models"
    """
    
    def __init__(self, model: nn.Module, config: LLCConfig):
        self.model = model
        self.config = config
        self.llc_estimator = LLCEstimator(model, config)
        self.refined_llc_estimator = RefinedLLCEstimator(model, config)
        
        # Define standard transformer component groups
        self.component_groups = {
            "input_embeddings": ["embed", "wte", "token_emb"],
            "position_embeddings": ["pos", "wpe", "position_emb"],
            "attention_query": ["q_proj", "query", "attn.c_attn"],
            "attention_key": ["k_proj", "key"],
            "attention_value": ["v_proj", "value"],
            "attention_output": ["o_proj", "out_proj", "attn.c_proj"],
            "mlp_intermediate": ["fc1", "c_fc", "intermediate"],
            "mlp_output": ["fc2", "c_proj", "output"],
            "layer_norm": ["norm", "ln"],
            "output_layer": ["lm_head", "output", "classifier"]
        }
        
        self.episode_history = []
    
    def track_episode(self, episode: int, 
                     dataloader: DataLoader,
                     loss_fn: Callable,
                     track_components: bool = True) -> Dict[str, any]:
        """
        Track LLC metrics for a single RL episode
        
        Args:
            episode: Episode number
            dataloader: DataLoader for current data batch
            loss_fn: Loss function
            track_components: Whether to compute refined LLC for components
            
        Returns:
            Dictionary containing LLC metrics for this episode
        """
        logger.info(f"Computing LLC for RL episode {episode}")
        
        # Get current model parameters as center point
        center_params = {name: param.clone().detach() 
                        for name, param in self.model.named_parameters() 
                        if param.requires_grad}
        
        # Compute overall LLC
        overall_llc = self.llc_estimator.estimate_llc(dataloader, loss_fn, center_params)
        
        results = {
            "episode": episode,
            "overall_llc": overall_llc,
            "timestamp": torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        }
        
        # Compute component-wise refined LLC
        if track_components:
            refined_llc = self.refined_llc_estimator.estimate_refined_llc(
                dataloader, loss_fn, self.component_groups, center_params
            )
            results["refined_llc"] = refined_llc
        
        # Store in history
        self.episode_history.append(results)
        
        return results
    
    def get_llc_evolution(self) -> Dict[str, List[float]]:
        """Get the evolution of LLC values across episodes"""
        if not self.episode_history:
            return {}
        
        evolution = {
            "episodes": [h["episode"] for h in self.episode_history],
            "overall_llc_mean": [h["overall_llc"]["llc_mean"] for h in self.episode_history],
            "overall_llc_std": [h["overall_llc"]["llc_std"] for h in self.episode_history]
        }
        
        # Add component evolution if available
        if "refined_llc" in self.episode_history[0]:
            for component in self.component_groups.keys():
                evolution[f"{component}_llc_mean"] = []
                evolution[f"{component}_llc_std"] = []
                
                for h in self.episode_history:
                    if component in h["refined_llc"]:
                        evolution[f"{component}_llc_mean"].append(
                            h["refined_llc"][component]["llc_mean"]
                        )
                        evolution[f"{component}_llc_std"].append(
                            h["refined_llc"][component]["llc_std"]
                        )
                    else:
                        evolution[f"{component}_llc_mean"].append(0.0)
                        evolution[f"{component}_llc_std"].append(0.0)
        
        return evolution
    
    def detect_phase_transitions(self, threshold: float = 0.1) -> List[Dict]:
        """
        Detect phase transitions in LLC evolution
        Based on "Loss Landscape Degeneracy Drives Stagewise Development in Transformers"
        """
        if len(self.episode_history) < 3:
            return []
        
        llc_values = [h["overall_llc"]["llc_mean"] for h in self.episode_history]
        episodes = [h["episode"] for h in self.episode_history]
        
        transitions = []
        
        for i in range(1, len(llc_values) - 1):
            # Compute local rate of change
            prev_change = llc_values[i] - llc_values[i-1]
            next_change = llc_values[i+1] - llc_values[i]
            
            # Detect significant change in derivative (potential phase transition)
            if abs(next_change - prev_change) > threshold:
                transitions.append({
                    "episode": episodes[i],
                    "llc_value": llc_values[i],
                    "change_magnitude": abs(next_change - prev_change),
                    "transition_type": "increase" if next_change > prev_change else "decrease"
                })
        
        return transitions
    
    def analyze_subnetwork_activity(self) -> Dict[str, any]:
        """
        Analyze which subnetworks are most active during RL fine-tuning
        Based on "Reinforcement Learning Finetunes Small Subnetworks in Large Language Models"
        """
        if not self.episode_history or "refined_llc" not in self.episode_history[0]:
            logger.warning("No refined LLC data available for subnetwork analysis")
            return {}
        
        # Compute variance of LLC across episodes for each component
        component_variance = {}
        component_mean_llc = {}
        
        for component in self.component_groups.keys():
            llc_values = []
            for h in self.episode_history:
                if component in h["refined_llc"]:
                    llc_values.append(h["refined_llc"][component]["llc_mean"])
            
            if llc_values:
                component_variance[component] = np.var(llc_values)
                component_mean_llc[component] = np.mean(llc_values)
        
        # Rank components by variance (higher variance = more active during training)
        active_components = sorted(component_variance.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        return {
            "component_variance": component_variance,
            "component_mean_llc": component_mean_llc,
            "most_active_components": active_components[:5],
            "least_active_components": active_components[-5:] if len(active_components) >= 5 else []
        }

def example_usage():
    """Example usage of the LLC tracking system"""
    
    # Create a simple transformer model (replace with your actual model)
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size: int = 1000, d_model: int = 128, nhead: int = 4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
                num_layers=2
            )
            self.lm_head = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            seq_len = x.size(1)
            x = self.embedding(x) + self.pos_encoding[:seq_len]
            x = self.transformer(x)
            return self.lm_head(x)
    
    # Initialize model and tracker
    model = SimpleTransformer()
    config = LLCConfig(
        temperature=1.0,
        sgld_steps=500,
        sgld_lr=1e-6,
        burn_in_ratio=0.8,
        batch_size=16
    )
    
    tracker = RLLLCTracker(model, config)
    
    # Create dummy data (replace with your actual RL data)
    dummy_data = torch.randint(0, 1000, (100, 20))  # 100 sequences of length 20
    dummy_targets = torch.randint(0, 1000, (100, 20))
    dataset = torch.utils.data.TensorDataset(dummy_data, dummy_targets)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Track LLC over several RL episodes
    for episode in range(5):
        # Simulate some parameter updates (replace with your actual RL updates)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.01)
        
        # Track LLC for this episode
        results = tracker.track_episode(episode, dataloader, loss_fn, track_components=True)
        
        print(f"Episode {episode}:")
        print(f"  Overall LLC: {results['overall_llc']['llc_mean']:.4f} ± {results['overall_llc']['llc_std']:.4f}")
        
        if "refined_llc" in results:
            print("  Component LLC:")
            for component, llc_data in results["refined_llc"].items():
                print(f"    {component}: {llc_data['llc_mean']:.4f}")
    
    # Analyze results
    evolution = tracker.get_llc_evolution()
    transitions = tracker.detect_phase_transitions()
    subnetwork_analysis = tracker.analyze_subnetwork_activity()
    
    print("\nPhase transitions detected:", len(transitions))
    print("Most active components during RL:", 
          [comp[0] for comp in subnetwork_analysis.get("most_active_components", [])])

if __name__ == "__main__":
    example_usage()
