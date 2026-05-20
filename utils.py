import os
import torch
import numpy as np
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
import scipy.optimize as opt
import torch.distributions as dist
from sklearn.metrics import accuracy_score
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict

# import torch.optim as optim
# from torch.distributions.kl import kl_divergence
# import matplotlib.pyplot as plt

# ============================================================================
# MMD (Maximum Mean Discrepancy) for Drift Detection
# ============================================================================

def gaussian_kernel(x, y, sigma=1.0):
    """
    Compute Gaussian RBF kernel between two sets of samples.
    
    Args:
        x: Tensor of shape (n_samples_x, n_features) or (n_samples_x,)
        y: Tensor of shape (n_samples_y, n_features) or (n_samples_y,)
        sigma: Bandwidth parameter for the Gaussian kernel (must be > 0)
    
    Returns:
        Kernel matrix of shape (n_samples_x, n_samples_y)
    """
    # Ensure inputs are 2D tensors
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    
    # Validate sigma
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    
    x = x.unsqueeze(1)  # (n_x, 1, n_features)
    y = y.unsqueeze(0)  # (1, n_y, n_features)
    
    # Compute squared Euclidean distance
    dist_sq = torch.sum((x - y) ** 2, dim=2)
    
    # Apply Gaussian kernel with numerical stability
    kernel = torch.exp(-dist_sq / (2 * sigma ** 2 + 1e-8))
    return kernel

def mmd_statistic(x, y, sigma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) statistic between two samples.
    
    Args:
        x: Tensor or numpy array of shape (n_x, n_features) or (n_x,) - reference distribution
        y: Tensor or numpy array of shape (n_y, n_features) or (n_y,) - test distribution
        sigma: Bandwidth parameter for the Gaussian kernel
    
    Returns:
        MMD statistic (scalar, non-negative)
    """
    # Input validation
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Input samples cannot be empty")
    
    # Convert to tensors if needed
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float32)
    
    # Ensure tensors are on the same device
    if x.device != y.device:
        y = y.to(x.device)
    
    # Handle 1D inputs (scalar values)
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    
    # Compute kernel matrices
    kxx = gaussian_kernel(x, x, sigma)
    kyy = gaussian_kernel(y, y, sigma)
    kxy = gaussian_kernel(x, y, sigma)
    
    # Compute MMD^2 statistic
    # Remove diagonal for kxx and kyy to avoid self-similarity bias
    n_x = kxx.shape[0]
    n_y = kyy.shape[0]
    if n_x > 1:
        kxx_no_diag = kxx - torch.diag(torch.diag(kxx))
        mean_kxx = kxx_no_diag.sum() / (n_x * (n_x - 1))
    else:
        mean_kxx = torch.tensor(0.0, device=x.device)
    
    if n_y > 1:
        kyy_no_diag = kyy - torch.diag(torch.diag(kyy))
        mean_kyy = kyy_no_diag.sum() / (n_y * (n_y - 1))
    else:
        mean_kyy = torch.tensor(0.0, device=x.device)
    
    mean_kxy = torch.mean(kxy)
    
    # Compute MMD^2 statistic
    mmd_sq = mean_kxx + mean_kyy - 2 * mean_kxy
    
    # Ensure non-negative (should be, but numerical errors might occur)
    mmd_sq = torch.clamp(mmd_sq, min=0.0)
    
    return mmd_sq.item()

def detect_drift_mmd(new_data, control_data, window_size, drift_threshold, sigma=1.0):
    """
    Detect concept drift using Maximum Mean Discrepancy (MMD).
    
    Args:
        new_data: New data samples (tensor or numpy array)
        control_data: Reference/control data samples (tensor or numpy array)
        window_size: Size of sliding window (must be > 0)
        drift_threshold: Threshold for MMD statistic to detect drift (must be >= 0)
        sigma: Bandwidth parameter for Gaussian kernel (must be > 0)
    
    Returns:
        Tuple[bool, float]: (drift_detected, max_mmd_value)
    """
    # Input validation
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if drift_threshold < 0:
        raise ValueError(f"drift_threshold must be non-negative, got {drift_threshold}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    
    # Convert to tensors if needed
    if isinstance(new_data, np.ndarray):
        new_data = torch.tensor(new_data, dtype=torch.float32)
    if isinstance(control_data, np.ndarray):
        control_data = torch.tensor(control_data, dtype=torch.float32)
    
    # Validate inputs
    if len(new_data) == 0:
        print("Warning: new_data is empty, no drift detected")
        return False, 0.0
    if len(control_data) == 0:
        print("Warning: control_data is empty, cannot detect drift")
        return False, 0.0
    
    # Ensure tensors are on the same device
    if new_data.device != control_data.device:
        control_data = control_data.to(new_data.device)
    
    # Use a subset of control data if it's too large (for efficiency)
    max_control_size = min(len(control_data), 1000)
    if len(control_data) > max_control_size:
        indices = torch.randperm(len(control_data))[:max_control_size]
        control_data = control_data[indices]
    
    # Process data in windows
    max_mmd_value = 0.0
    for i in range(0, len(new_data), window_size):
        window_data = new_data[i:i + window_size]
        if len(window_data) < window_size:
            break
        
        try:
            # Compute MMD statistic
            mmd_value = mmd_statistic(control_data, window_data, sigma=sigma)
            max_mmd_value = max(max_mmd_value, mmd_value)
            
            if mmd_value > drift_threshold:
                print(f"!!!!!!!!!!!!!!!!!!!!! Drift detected in window {i // window_size + 1} (MMD: {mmd_value:.6f})")
                return True, mmd_value
            else:
                print(f"No drift detected in window {i // window_size + 1} (MMD: {mmd_value:.6f})")
        except Exception as e:
            print(f"Error computing MMD for window {i // window_size + 1}: {e}")
            continue
    
    return False, max_mmd_value

# ============================================================================
# Gradient-Matching Coreset Selection (GCR / CRAIG)
# ============================================================================

def compute_gradients(model, inputs, labels, criterion, device, dataset='nsl'):
    """
    Compute gradients for each sample in the batch.
    
    Args:
        model: The neural network model
        inputs: Input samples (tensor)
        labels: Labels (tensor)
        criterion: Loss function
        device: Computing device
        dataset: Dataset type ('nsl' or 'unsw')
    
    Returns:
        gradients: List of gradient vectors for each sample (flattened)
    """
    model.eval()
    gradients = []
    
    # Compute full dataset gradient first (for reference)
    model.zero_grad()
    if dataset == 'nsl':
        features, recon_vec = model(inputs)
        loss = criterion(recon_vec, labels).mean()
    else:
        features, recon_vec, classifications = model(inputs)
        con_loss = criterion(recon_vec, labels).mean()
        classification_loss = nn.BCELoss()(classifications.squeeze(), labels.float())
        loss = con_loss + classification_loss
    
    loss.backward()
    
    # Get full gradient
    full_grad = []
    for param in model.parameters():
        if param.grad is not None:
            full_grad.append(param.grad.data.clone().flatten())
    full_gradient = torch.cat(full_grad) if full_grad else torch.tensor([], device=device)
    
    # Compute per-sample gradients
    for i in range(len(inputs)):
        model.zero_grad()
        single_input = inputs[i:i+1]
        single_label = labels[i:i+1]
        
        if dataset == 'nsl':
            features, recon_vec = model(single_input)
            loss = criterion(recon_vec, single_label).mean()
        else:
            features, recon_vec, classifications = model(single_input)
            con_loss = criterion(recon_vec, single_label).mean()
            classification_loss = nn.BCELoss()(classifications.squeeze(), single_label.float())
            loss = con_loss + classification_loss
        
        loss.backward()
        
        # Get gradient for this sample
        sample_grad = []
        for param in model.parameters():
            if param.grad is not None:
                sample_grad.append(param.grad.data.clone().flatten())
        sample_gradient = torch.cat(sample_grad) if sample_grad else torch.tensor([], device=device)
        gradients.append(sample_gradient)
    
    return gradients, full_gradient

def gradient_matching_coreset(model, candidate_x, candidate_y, buffer_size, criterion, device, dataset='nsl', method='greedy', max_candidates=5000, batch_size=64):
    """
    Select coreset using gradient matching (GCR / CRAIG method).
    
    Selects samples such that their gradients best match the full dataset gradient.
    
    Args:
        model: The neural network model
        candidate_x: Candidate samples (tensor, shape: [n_candidates, features])
        candidate_y: Candidate labels (tensor, shape: [n_candidates])
        buffer_size: Desired buffer size (number of samples to select)
        criterion: Loss function
        device: Computing device
        dataset: Dataset type ('nsl' or 'unsw')
        method: Selection method ('greedy' or 'random')
        max_candidates: Maximum number of candidates to consider (for efficiency)
        batch_size: Batch size for gradient computation
    
    Returns:
        selected_indices: Indices of selected samples
        selected_x: Selected samples
        selected_y: Selected labels
    """
    # Sample candidates if too many (for efficiency)
    # Use stratified sampling to maintain class balance
    original_candidate_x = candidate_x.clone()  # Save original for final selection
    original_candidate_y = candidate_y.clone()
    original_indices = None
    if len(candidate_x) > max_candidates:
        # print(f"Too many candidates ({len(candidate_x)}), sampling {max_candidates} for efficiency...")
        # Stratified sampling: maintain class balance
        unique_labels = torch.unique(candidate_y)
        sample_indices_list = []
        samples_per_class = max_candidates // len(unique_labels)
        remaining_samples = max_candidates % len(unique_labels)
        
        for i, label in enumerate(unique_labels):
            label_indices = torch.where(candidate_y == label)[0]
            n_samples = samples_per_class + (1 if i < remaining_samples else 0)
            n_samples = min(n_samples, len(label_indices))
            if n_samples > 0:
                perm = torch.randperm(len(label_indices), device=device)[:n_samples]
                sample_indices_list.append(label_indices[perm])
        
        sample_indices = torch.cat(sample_indices_list)
        original_indices = sample_indices.clone()
        candidate_x = candidate_x[sample_indices]
        candidate_y = candidate_y[sample_indices]
        # print(f"Computing gradients for {len(candidate_x)} sampled candidate samples (stratified sampling)...")
    else:
        print(f"Computing gradients for {len(candidate_x)} candidate samples...")

    if len(candidate_x) <= buffer_size:
        # If we have fewer candidates than buffer size, return all available
        if original_indices is not None:
            selected_indices = original_indices
            selected_x = original_candidate_x[selected_indices]
            selected_y = original_candidate_y[selected_indices]
        else:
            selected_indices = torch.arange(len(candidate_x), device=device)
            selected_x = candidate_x
            selected_y = candidate_y
        # print(f"Selected all {len(selected_x)} candidates (<= buffer size); skipping greedy selection.")
        return selected_indices, selected_x, selected_y
    
    # Compute full dataset gradient
    model.train()  # Enable gradient computation
    model.zero_grad()
    if dataset == 'nsl':
        _, recon_vec = model(candidate_x)
        full_loss = criterion(recon_vec, candidate_y).mean()
    else:
        _, recon_vec, classifications = model(candidate_x)
        con_loss = criterion(recon_vec, candidate_y).mean()
        classification_loss = nn.BCELoss()(classifications.squeeze(), candidate_y.float())
        full_loss = con_loss + classification_loss
    
    full_loss.backward()
    
    # Get full gradient
    full_grad_list = []
    for param in model.parameters():
        if param.grad is not None:
            full_grad_list.append(param.grad.data.clone().flatten())
    full_gradient = torch.cat(full_grad_list) if full_grad_list else torch.tensor([], device=device)
    full_gradient = full_gradient / len(candidate_x)  # Normalize by number of samples
    
    # Compute per-sample gradients in batches (much faster than one-by-one)
    all_gradients = []
    n_samples = len(candidate_x)
    
    # Use batch processing for gradient computation
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_x = candidate_x[batch_start:batch_end]
        batch_y = candidate_y[batch_start:batch_end]
        
        # Compute gradients for each sample in the batch
        batch_gradients = []
        for i in range(batch_start, batch_end):
            model.zero_grad()
            single_x = candidate_x[i:i+1]
            single_y = candidate_y[i:i+1]
            
            if dataset == 'nsl':
                _, recon_vec = model(single_x)
                loss = criterion(recon_vec, single_y).mean()
            else:
                _, recon_vec, classifications = model(single_x)
                con_loss = criterion(recon_vec, single_y).mean()
                classification_loss = nn.BCELoss()(classifications.squeeze(), single_y.float())
                loss = con_loss + classification_loss
            
            loss.backward()
            
            # Get gradient
            grad_list = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_list.append(param.grad.data.clone().flatten())
            grad = torch.cat(grad_list) if grad_list else torch.tensor([], device=device)
            batch_gradients.append(grad)
        
        all_gradients.extend(batch_gradients)
        
        # Progress indicator
        if batch_end % 500 == 0 or batch_end == n_samples:
            print(f"  Computed gradients for {batch_end}/{n_samples} samples...")
    
    all_gradients = torch.stack(all_gradients)  # Shape: [n_candidates, grad_dim]
    
    # Optimized greedy selection: use inner product approximation for speed
    if method == 'greedy':
        # Pre-compute gradient similarities to full gradient (inner product)
        # Samples with higher inner product are more aligned with full gradient
        grad_similarities = torch.matmul(all_gradients, full_gradient)  # Shape: [n_candidates]
        
        # Greedy selection: iteratively select samples
        selected_indices = []
        selected_grad_sum = torch.zeros_like(full_gradient)
        remaining_indices = set(range(len(candidate_x)))
        total_steps = min(buffer_size, len(candidate_x))
        
        # Pre-normalize all gradients for faster diversity computation
        grad_norms = torch.norm(all_gradients, dim=1, keepdim=True) + 1e-8
        normalized_gradients = all_gradients / grad_norms
        
        # Keep only recent selected gradients for diversity check (much faster)
        recent_selected_limit = 50  # Only check diversity against last 50 selected
        
        for step in range(total_steps):
            # Progress indicator
            if (step + 1) % 500 == 0 or step == 0:
                print(f"  Selecting samples: {step + 1}/{total_steps}...")
            
            if not remaining_indices:
                break
            
            remaining_tensor = torch.tensor(list(remaining_indices), device=device)
            
            # Limit search to top candidates by similarity for efficiency
            if len(remaining_indices) > 2000:
                top_k = min(2000, len(remaining_indices))
                remaining_similarities = grad_similarities[remaining_tensor]
                _, top_idx = torch.topk(remaining_similarities, k=top_k)
                candidates_tensor = remaining_tensor[top_idx]
            else:
                candidates_tensor = remaining_tensor
            
            candidate_grads = all_gradients[candidates_tensor]
            new_grad_sums = selected_grad_sum.unsqueeze(0) + candidate_grads
            new_grad_means = new_grad_sums / (step + 1)
            
            grad_diffs = torch.norm(new_grad_means - full_gradient.unsqueeze(0), dim=1)
            
            if len(selected_indices) > 0:
                recent_indices = selected_indices[-recent_selected_limit:]
                if len(recent_indices) > 0:
                    recent_norms = normalized_gradients[recent_indices]
                    candidate_norms = normalized_gradients[candidates_tensor]
                    similarities = torch.matmul(candidate_norms, recent_norms.t())
                    diversity_penalties = similarities.mean(dim=1) * 0.03
                else:
                    diversity_penalties = torch.zeros_like(grad_diffs)
            else:
                diversity_penalties = torch.zeros_like(grad_diffs)
            
            combined_scores = grad_diffs + diversity_penalties
            best_pos = torch.argmin(combined_scores)
            best_idx = candidates_tensor[best_pos].item()
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            selected_grad_sum += all_gradients[best_idx]
        
        selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=device)
        
        # Map back to original indices if we sampled
        if original_indices is not None:
            # Map selected indices (in sampled space) back to original space
            original_selected_indices = original_indices[selected_indices]
            # Get the actual samples from original candidate set
            selected_x = original_candidate_x[original_selected_indices]
            selected_y = original_candidate_y[original_selected_indices]
            selected_indices = original_selected_indices
        else:
            selected_x = candidate_x[selected_indices]
            selected_y = candidate_y[selected_indices]
    else:
        # Random selection (fallback)
        if original_indices is not None:
            sampled_rand_indices = torch.randperm(len(candidate_x), device=device)[:buffer_size]
            original_selected_indices = original_indices[sampled_rand_indices]
            selected_x = original_candidate_x[original_selected_indices]
            selected_y = original_candidate_y[original_selected_indices]
            selected_indices = original_selected_indices
        else:
            selected_indices = torch.randperm(len(candidate_x), device=device)[:buffer_size]
            selected_x = candidate_x[selected_indices]
            selected_y = candidate_y[selected_indices]
    
    print(f"Selected {len(selected_indices)} samples using {'random selection' if method == 'random' else 'gradient matching'}")
    
    return selected_indices, selected_x, selected_y

def select_coreset_gradient_matching(
    x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, y_test_this_epoch,
    buffer_size, model, criterion, device, dataset='nsl', num_new_samples=None, max_candidates=8000,
    drift_detected=False, drift_score=None, drift_threshold=None,
    old_scores=None, new_scores=None,
    coreset_method='greedy',
    no_strategic_forgetting=False
):
    """
    Select coreset using gradient matching for online continual learning.
    
    This function combines old buffer samples and new samples, then selects
    the best subset using gradient matching.
    
    Args:
        x_train_this_epoch: Current buffer samples (old samples)
        y_train_this_epoch: Current buffer labels
        x_test_this_epoch: New incoming samples
        y_test_this_epoch: New incoming labels
        buffer_size: Desired buffer size
        model: The neural network model
        criterion: Loss function
        device: Computing device
        dataset: Dataset type ('nsl' or 'unsw')
        num_new_samples: Baseline minimum number of new samples to include
        drift_detected: Whether drift is detected for the current window
        drift_score: Drift statistic (e.g., MMD value) if available
        drift_threshold: Drift threshold used (for scaling severity)
        old_scores: Optional tensor with importance scores for old samples (same order as x_train_this_epoch)
        new_scores: Optional tensor with importance scores for new samples (same order as x_test_this_epoch)
        no_strategic_forgetting: If True (ablation 3), ignore drift for buffer update: fixed base_ratio and min_new_samples.
    
    Returns:
        x_train_this_epoch: Updated buffer samples
        y_train_this_epoch: Updated buffer labels
        labeled_indices_current: Indices of selected new samples in x_test_this_epoch
        new_mask: Mask indicating which samples are new (1 for new, 0 for old)
    """
    # Convert optional scores to tensors on device
    old_scores_tensor = None
    new_scores_tensor = None
    if old_scores is not None:
        old_scores_tensor = old_scores.detach().to(device)
    if new_scores is not None:
        new_scores_tensor = new_scores.detach().to(device)

    # Combine old buffer and new samples as candidates
    candidate_x = torch.cat([x_train_this_epoch, x_test_this_epoch], dim=0)
    candidate_y = torch.cat([y_train_this_epoch, y_test_this_epoch], dim=0)
    
    n_old = len(x_train_this_epoch)
    n_new = len(x_test_this_epoch)
    
    # Ensure buffer size doesn't exceed memory limit
    # If current buffer + new samples exceed buffer_size, we need to select
    actual_buffer_size = min(buffer_size, len(candidate_x))
    
    # Determine target ratio of new samples (ablation: w/o strategic forgetting uses fixed ratio)
    base_ratio = 0.18  # default minimum ratio when no drift
    if not no_strategic_forgetting and drift_detected:
        if drift_threshold and drift_threshold > 0 and drift_score is not None:
            severity = min(2.0, drift_score / drift_threshold)
        else:
            severity = 1.0
        # Increase ratio with drift severity, capped at 50%
        base_ratio = min(0.5, 0.25 + 0.12 * severity)
    
    min_new_samples_target = int(actual_buffer_size * base_ratio)
    if num_new_samples is not None:
        min_new_samples_target = max(min_new_samples_target, num_new_samples)
    
    min_new_samples = min(n_new, max(1, min_new_samples_target))
    
    if not no_strategic_forgetting and drift_detected and n_new > 0:
        ratio = min_new_samples / max(1, actual_buffer_size)
        print(f"Drift-aware selection: requiring at least {min_new_samples} new samples (~{ratio:.1%}) based on severity.")
    
    # Determine effective candidate cap and selection budget (ablation: no strategic forgetting does not expand for drift)
    effective_max_candidates = min(len(candidate_x), max_candidates)
    if not no_strategic_forgetting and drift_detected:
        effective_max_candidates = min(len(candidate_x), max(int(effective_max_candidates * 1.5), effective_max_candidates + 2000))
    
    candidate_limit = min(len(candidate_x), effective_max_candidates)
    selection_budget = min(candidate_limit, actual_buffer_size)
    selection_budget = max(selection_budget, min_new_samples)
    
    # Select coreset using gradient matching or random (subset size = selection_budget)
    selected_indices, selected_x, selected_y = gradient_matching_coreset(
        model, candidate_x, candidate_y, selection_budget, criterion, device, dataset,
        method=coreset_method, max_candidates=effective_max_candidates
    )
    
    # Determine which selected samples are new vs old
    selected_indices_np = selected_indices.cpu().numpy()
    new_mask = torch.zeros(len(selected_x), dtype=torch.float32, device=device)
    
    new_sample_indices_in_original = []
    selected_set = set(selected_indices_np.tolist())
    
    # Identify which selected indices correspond to new samples
    for idx in selected_indices_np:
        if idx >= n_old:
            new_sample_indices_in_original.append(idx - n_old)
    if new_sample_indices_in_original:
        # mark existing new samples in mask
        new_mask_indices = torch.tensor([i for i, idx in enumerate(selected_indices_np) if idx >= n_old], device=device, dtype=torch.long)
        new_mask[new_mask_indices] = 1.0
    
    n_selected_new = int(new_mask.sum().item())
    
    # Ensure minimum number of new samples using score-based selection
    if n_selected_new < min_new_samples and n_new > 0:
        all_new_indices = set(range(n_old, n_old + n_new))
        unselected_new_indices = list(all_new_indices - selected_set)
        
        if len(unselected_new_indices) > 0:
            n_additional_needed = min_new_samples - n_selected_new
            n_additional = min(n_additional_needed, len(unselected_new_indices))
            
            additional_indices_tensor = torch.tensor(unselected_new_indices, device=device)
            if new_scores_tensor is not None:
                additional_scores = new_scores_tensor[additional_indices_tensor - n_old]
                topk = torch.topk(additional_scores, k=n_additional).indices
                additional_indices = additional_indices_tensor[topk]
            else:
                perm = torch.randperm(len(additional_indices_tensor), device=device)[:n_additional]
                additional_indices = additional_indices_tensor[perm]
            
            additional_x = candidate_x[additional_indices]
            additional_y = candidate_y[additional_indices]
            additional_mask = torch.ones(len(additional_x), dtype=torch.float32, device=device)
            
            selected_x = torch.cat([selected_x, additional_x], dim=0)
            selected_y = torch.cat([selected_y, additional_y], dim=0)
            new_mask = torch.cat([new_mask, additional_mask], dim=0)
            
            selected_set.update(additional_indices.cpu().tolist())
            for idx in additional_indices.cpu().tolist():
                new_sample_indices_in_original.append(idx - n_old)
            
            print(f"Added {len(additional_indices)} additional new samples to meet minimum requirement ({min_new_samples})")
    
    num_new_selected = int(new_mask.sum().item())
    
    # If we exceed buffer capacity after additions, drop old samples first (at random or by score)
    if len(selected_x) > actual_buffer_size:
        excess = len(selected_x) - actual_buffer_size
        if excess > 0:
            old_indices = torch.nonzero(new_mask == 0, as_tuple=False).squeeze(-1)
            if len(old_indices) > 0:
                drop_count = min(excess, len(old_indices))
                drop_old = old_indices[torch.randperm(len(old_indices), device=device)[:drop_count]]
                keep_mask = torch.ones(len(selected_x), dtype=torch.bool, device=device)
                keep_mask[drop_old] = False
                selected_x = selected_x[keep_mask]
                selected_y = selected_y[keep_mask]
                new_mask = new_mask[keep_mask]
                excess = len(selected_x) - actual_buffer_size
        if len(selected_x) > actual_buffer_size:
            keep_indices = torch.randperm(len(selected_x), device=device)[:actual_buffer_size]
            selected_x = selected_x[keep_indices]
            selected_y = selected_y[keep_indices]
            new_mask = new_mask[keep_indices]
    
    num_new_selected = int(new_mask.sum().item())
    
    # Ensure minimum ratio of abnormal samples (for class imbalance handling)
    if dataset != 'nsl':
        unique_labels, counts = torch.unique(selected_y, return_counts=True)
        if len(unique_labels) == 2:
            normal_count = counts[unique_labels == 0].item() if (unique_labels == 0).any() else 0
            abnormal_count = counts[unique_labels == 1].item() if (unique_labels == 1).any() else 0
            total_count = len(selected_y)
            abnormal_ratio = abnormal_count / total_count if total_count > 0 else 0
            
            # Ensure at least 10% abnormal samples (adjustable threshold)
            min_abnormal_ratio = 0.10
            min_abnormal_count = max(1, int(total_count * min_abnormal_ratio))
            
            if abnormal_count < min_abnormal_count:
                # Find abnormal samples in candidates that weren't selected
                abnormal_candidates = torch.where(candidate_y == 1)[0]
                unselected_abnormal = [idx for idx in abnormal_candidates.cpu().numpy() if idx not in selected_set]
                
                if len(unselected_abnormal) > 0:
                    n_needed = min_abnormal_count - abnormal_count
                    n_to_add = min(n_needed, len(unselected_abnormal))
                    
                    # Prioritize by scores if available
                    if new_scores_tensor is not None or old_scores_tensor is not None:
                        abnormal_scores = []
                        for idx in unselected_abnormal:
                            if idx >= n_old:
                                if new_scores_tensor is not None:
                                    abnormal_scores.append((idx, new_scores_tensor[idx - n_old].item()))
                                else:
                                    abnormal_scores.append((idx, 1.0))
                            else:
                                if old_scores_tensor is not None:
                                    abnormal_scores.append((idx, old_scores_tensor[idx].item()))
                                else:
                                    abnormal_scores.append((idx, 1.0))
                        abnormal_scores.sort(key=lambda x: x[1], reverse=True)
                        additional_abnormal_indices = [x[0] for x in abnormal_scores[:n_to_add]]
                    else:
                        perm = torch.randperm(len(unselected_abnormal), device=device)[:n_to_add]
                        additional_abnormal_indices = [unselected_abnormal[i] for i in perm.cpu().numpy()]
                    
                    additional_abnormal_x = candidate_x[additional_abnormal_indices]
                    additional_abnormal_y = candidate_y[additional_abnormal_indices]
                    additional_abnormal_mask = torch.zeros(len(additional_abnormal_x), dtype=torch.float32, device=device)
                    # Mark as new if from new samples
                    for i, idx in enumerate(additional_abnormal_indices):
                        if idx >= n_old:
                            additional_abnormal_mask[i] = 1.0
                            new_sample_indices_in_original.append(idx - n_old)
                    
                    selected_x = torch.cat([selected_x, additional_abnormal_x], dim=0)
                    selected_y = torch.cat([selected_y, additional_abnormal_y], dim=0)
                    new_mask = torch.cat([new_mask, additional_abnormal_mask], dim=0)
                    selected_set.update(additional_abnormal_indices)
                    
                    # If buffer exceeds capacity, drop normal samples first
                    if len(selected_x) > actual_buffer_size:
                        excess = len(selected_x) - actual_buffer_size
                        normal_indices = torch.where(selected_y == 0)[0]
                        if len(normal_indices) > 0:
                            drop_count = min(excess, len(normal_indices))
                            drop_normal = normal_indices[torch.randperm(len(normal_indices), device=device)[:drop_count]]
                            keep_mask = torch.ones(len(selected_x), dtype=torch.bool, device=device)
                            keep_mask[drop_normal] = False
                            selected_x = selected_x[keep_mask]
                            selected_y = selected_y[keep_mask]
                            new_mask = new_mask[keep_mask]
                    
                    print(f"Added {len(additional_abnormal_indices)} abnormal samples to ensure minimum ratio ({min_abnormal_ratio:.1%})")
    
    labeled_indices_current = torch.tensor(new_sample_indices_in_original, dtype=torch.long, device=device) if new_sample_indices_in_original else torch.tensor([], dtype=torch.long, device=device)
    
    # Print final class distribution
    if dataset != 'nsl':
        unique_labels, counts = torch.unique(selected_y, return_counts=True)
        print(f"Final buffer class distribution: ", end="")
        for label, count in zip(unique_labels, counts):
            label_name = "Normal" if label == 0 else "Abnormal"
            print(f"{label_name}={count.item()}", end=", ")
        print()
    
    print(f"Gradient matching: Selected {len(selected_x)} samples ({num_new_selected} new, {len(selected_x) - num_new_selected} old)")
    
    return selected_x, selected_y, labeled_indices_current, new_mask

# MLP function for unsw
class AE_classifier(nn.Module):
    def __init__(self, input_dim):
        super(AE_classifier, self).__init__()
        # Find the nearest power of 2 to input_dim
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))

        # Calculate the dimensions of the 2nd/4th layer and the 3rd layer.
        second_fourth_layer_size = nearest_power_of_2 // 2  # A half
        third_layer_size = nearest_power_of_2 // 4         # A quarter

        # Create encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )

        # Create decoder
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, input_dim),
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, 1),  # 1 neuron for binary classification
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        classify = self.classifier(decode)
        return encode, decode, classify

# Evaluation function for unsw
def evaluate_classifier(model, data_loader, device, get_predict=False, threshold=None, optimize_threshold=True):
    """
    Evaluate classifier model.
    
    Args:
        model: The classifier model
        data_loader: DataLoader for evaluation
        device: Computing device
        get_predict: If True, return predictions; if False, return metrics
        threshold: Classification threshold (if None, will optimize for F1-score)
        optimize_threshold: If True and threshold is None, optimize threshold using labels
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            _, _, classifications = model(inputs)
            probs = classifications.squeeze()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Optimize threshold for F1-score if not provided and optimization is enabled
    if threshold is None and optimize_threshold and len(all_labels) > 0:
        # Check class distribution in labels
        unique_labels, label_counts = np.unique(all_labels, return_counts=True)
        n_normal = label_counts[unique_labels == 0].sum() if len(unique_labels) > 0 and (unique_labels == 0).any() else 0
        n_abnormal = label_counts[unique_labels == 1].sum() if len(unique_labels) > 1 and (unique_labels == 1).any() else 0
        
        # Try different thresholds and find the one with best F1-score
        best_threshold = 0.5
        best_f1 = 0.0
        
        # First, try standard range of thresholds
        for t in np.arange(0.05, 0.95, 0.05):
            preds_t = (all_probs > t).astype(float)
            unique_preds = np.unique(preds_t)
            if len(unique_preds) > 1:  # Ensure both classes are predicted
                try:
                    f1 = f1_score(all_labels, preds_t, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = t
                except:
                    continue
        
        # If no good threshold found and we have abnormal samples, try very low thresholds
        if best_f1 == 0.0 and n_abnormal > 0:
            # Check if all predictions are normal at threshold 0.5
            preds_default = (all_probs > 0.5).astype(float)
            if len(np.unique(preds_default)) == 1 and np.unique(preds_default)[0] == 0:
                # All predicted as normal, try very low thresholds
                for t_low in np.arange(0.001, 0.05, 0.001):
                    preds_low = (all_probs > t_low).astype(float)
                    unique_preds_low = np.unique(preds_low)
                    if len(unique_preds_low) > 1:  # Both classes predicted
                        try:
                            f1_low = f1_score(all_labels, preds_low, zero_division=0)
                            if f1_low > best_f1:
                                best_f1 = f1_low
                                best_threshold = t_low
                        except:
                            continue
                    elif len(unique_preds_low) == 1 and unique_preds_low[0] == 1:
                        # All predicted as abnormal - this might be too low, but record it
                        try:
                            f1_low = f1_score(all_labels, preds_low, zero_division=0)
                            if f1_low > best_f1:
                                best_f1 = f1_low
                                best_threshold = t_low
                        except:
                            continue
        
        threshold = best_threshold
        if not get_predict:  # Only print when evaluating (not when just getting predictions)
            # Check final predictions with optimized threshold
            final_preds = (all_probs > threshold).astype(float)
            final_unique = np.unique(final_preds)
            if len(final_unique) == 1 and n_abnormal > 0:
                print(f"WARNING: Optimized threshold {threshold:.3f} still predicts only one class (F1-score: {best_f1:.4f})")
                print(f"  Label distribution: Normal={n_normal}, Abnormal={n_abnormal}")
                print(f"  Probability range: min={all_probs.min():.4f}, max={all_probs.max():.4f}, mean={all_probs.mean():.4f}")
            else:
                print(f"Optimized threshold: {threshold:.3f} (F1-score: {best_f1:.4f})")
    elif threshold is None:
        threshold = 0.5  # Default threshold
    
    all_preds = (all_probs > threshold).astype(float)

    if not get_predict:
        res = score_detail(all_labels, all_preds, if_print=True)
        return res
    else:
        return all_preds

# Evaluation function for single sample or batch of samples for unsw
def evaluate_inputs(model, inputs, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        _, _, classifications = model(inputs)
        preds = (classifications.squeeze() > threshold).float()
    return preds.cpu().numpy()


#################################################################

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

# 这段代码定义了一个用于数据预处理和标签转换的类 SplitData，它继承自 BaseEstimator 和 TransformerMixin，使其兼容于 sklearn 中的流水线（Pipeline）工作流。这个类的主要功能是：
# 对数据集进行标签处理；
# 根据不同的数据集（NSL 或 UNSW）执行特定的预处理步骤；
# 对特征数据进行归一化处理，使其在训练时具有一致的尺度。
class SplitData(BaseEstimator, TransformerMixin):
    def __init__(self, dataset):
        super(SplitData, self).__init__()
        self.dataset = dataset
    #功能：这是 sklearn 中 TransformerMixin 类要求实现的一个方法，它通常用于拟合（学习）数据的统计特性（如均值、方差等）。
    # 返回值：返回自身，通常用于支持 sklearn 的流水线（pipeline）操作。
    # 作用：该方法在这里并不需要进行任何操作，因为在 SplitData 类中，我们只做了数据转换，未涉及到需要训练的模型参数，因此直接返回 self。
    def fit(self, X, y=None):
        return self
    #功能：这个方法主要用于将输入的原始数据 X 进行处理，提取特征，并返回经过预处理后的特征数据 x_ 和标签 y_。
    # 参数：
    # X：包含数据集的 Pandas DataFrame，通常包含多种特征。
    # labels：是目标列的列名，表示数据集中标签所在的列。
    # one_hot_label：是否使用 One-Hot 编码进行标签转换，虽然该参数在代码中没有使用，但通常它用于将标签转换为独热编码（one-hot encoding）形式。
    def transform(self, X, labels, one_hot_label=True):
        if self.dataset == 'nsl':
            # y = X[labels]：从输入数据 X 中提取标签列。
            # X_ = X.drop(['labels5', 'labels2'], axis=1)：删除 labels5 和 labels2 列，因为这些列是多余的或者不是特征列。
            # y = (y != 'normal')：将 y 中的标签从 normal（正常）和 abnormal（异常）转换成二进制标签。normal 被标记为 0，而 abnormal 被标记为 1，用于二分类任务。
            y = X[labels]
            X_ = X.drop(['labels5', 'labels2'], axis=1)
            # abnormal data is labeled as 1, normal data 0
            y = (y != 'normal')
            y_ = np.asarray(y).astype('float32')

        elif self.dataset == 'unsw':
            # UNSW dataset processing
            y_ = X[labels]
            X_ = X.drop('label', axis=1)

        else:
            raise ValueError("Unsupported dataset type")

        # 使用 MinMaxScaler 对特征进行归一化，将其缩放到 [0, 1] 范围。
        normalize = MinMaxScaler().fit(X_) #初始化 MinMaxScaler，并拟合训练数据的特征（X_）。
        x_ = normalize.transform(X_)    #对训练数据 X_ 进行归一化转换，得到归一化后的特征数据 x_。

        return x_, y_

class AE(nn.Module):
    def __init__(self, input_dim):
        super(AE, self).__init__()

        # 为了优化计算资源和提高模型效率，通常选择将模型的隐藏层尺寸设置为输入维度的 最接近的 2 的幂次方。这样可以更好地适配底层硬件（尤其是 GPU）。
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))

        # 根据计算出的 nearest_power_of_2，构建第二层（second_fourth_layer_size）和第三层（third_layer_size）的尺寸：
        # 第二层是 nearest_power_of_2 的一半（// 2）。
        # 第三层是第二层的一半（// 4）。
        second_fourth_layer_size = nearest_power_of_2 // 2  # A half
        third_layer_size = nearest_power_of_2 // 4         # A quarter

        # 编码器部分由两层全连接层（nn.Linear）组成，负责将输入数据映射到一个较低维度的表示。
        # nn.Linear(input_dim, second_fourth_layer_size)：将输入维度 input_dim 的数据映射到 second_fourth_layer_size 维度，通常用于降低数据的维度。
        # nn.ReLU()：ReLU 激活函数，将负值输出为 0，正值保持不变，使网络引入非线性。
        # nn.Linear(second_fourth_layer_size, third_layer_size)：第二个全连接层，将数据从 second_fourth_layer_size 映射到更低的维度 third_layer_size。
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )

        # 解码器部分也是由两层全连接层组成，负责将编码器生成的低维表示还原为输入数据的近似值。
        # nn.ReLU()：激活函数，确保网络非线性。
        # nn.Linear(third_layer_size, second_fourth_layer_size)：将低维数据从 third_layer_size 映射到 second_fourth_layer_size 维度，开始解码过程。
        # nn.ReLU()：再应用一次 ReLU 激活函数。
        # nn.Linear(second_fourth_layer_size, input_dim)：最后一个全连接层，将数据从 second_fourth_layer_size 映射回原始输入维度 input_dim，完成数据的还原。
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, input_dim),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

class InfoNCELoss(nn.Module):
    def __init__(self, device, temperature=0.1, scale_by_temperature=True):
        super(InfoNCELoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float()
        # compute logits
        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # Calculate the dot product similarity between pairwise samples
        # create mask
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size).to(self.device)
        logits_without_ii = logits * logits_mask

        logits_normal = logits_without_ii[(labels == 0).squeeze()]
        logits_normal_normal = logits_normal[:,(labels == 0).squeeze()]
        logits_normal_abnormal = logits_normal[:,(labels > 0).squeeze()]

        sum_of_vium = torch.sum(torch.exp(logits_normal_abnormal), axis=1, keepdims=True)
        denominator = torch.exp(logits_normal_normal) + sum_of_vium
        log_probs = logits_normal_normal - torch.log(denominator)

        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature

        return loss

def score_detail(y_test,y_test_pred,if_print=True):
    # Confusion matrix
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_test_pred))
    # Accuracy
    print('Accuracy ',accuracy_score(y_test, y_test_pred))
    # Precision
    print('Precision ',precision_score(y_test, y_test_pred))
    # Recall
    print('Recall ',recall_score(y_test, y_test_pred))
    # F1 score
    print('F1 score ',f1_score(y_test,y_test_pred))

    return accuracy_score(y_test, y_test_pred), precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), f1_score(y_test,y_test_pred)
#固定程序中所有随机数生成器的种子，以确保实验的可重复性
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

# Define a small epsilon to avoid log(0)
EPSILON = 1e-10

def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def log_likelihood(params, data):
    mu1, sigma1, mu2, sigma2 = params
    pdf1 = gaussian_pdf(data, mu1, sigma1)
    pdf2 = gaussian_pdf(data, mu2, sigma2)

    # Ensure the values passed to log are above a threshold
    likelihood = 0.5 * pdf1 + 0.5 * pdf2
    likelihood = np.clip(likelihood, a_min=EPSILON, a_max=None)

    # Check for NaN values
    if np.any(np.isnan(likelihood)):
        print("NaN values found in likelihood calculation")
        return np.inf

    return -np.sum(np.log(likelihood))

# Define the batch processing function
def process_batch(data, temp, layer_index, model, batch_size=128, device='cuda'):
    values = []
    model.to(device)
    temp = temp.to(device)

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size].to(device)
        batch_features = F.normalize(model(batch)[layer_index], p=2, dim=1)
        batch_cosine_sim = F.cosine_similarity(batch_features, temp.reshape([-1, temp.shape[0]]), dim=1)
        values.append(batch_cosine_sim)
        del batch, batch_features, batch_cosine_sim  # Free memory
        torch.cuda.empty_cache()  # Clear cache

    return torch.cat(values)

def evaluate(normal_recon_temp, x_train, y_train, x_test, y_test, model, batch_size=128, device='cuda', get_probs=False):
    model.eval()
    # Define dataset and dataloader
    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=False)

    # num_of_layer = 0
    num_of_output = 1

    # values_features_all = []
    # values_features_normal = []
    # values_features_abnormal = []
    values_recon_all = []
    values_recon_normal = []
    values_recon_abnormal = []

    model.to(device)
    # normal_temp = normal_temp.to(device)
    normal_recon_temp = normal_recon_temp.to(device)

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            features, recon_vec = model(inputs)
            # values_features_all.append(F.cosine_similarity(F.normalize(features, p=2, dim=1), normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
            values_recon_all.append(F.cosine_similarity(F.normalize(recon_vec, p=2, dim=1), normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))

            normal_mask = (labels == 0)
            abnormal_mask = (labels == 1)

            if normal_mask.sum() > 0:
                # values_features_normal.append(F.cosine_similarity(F.normalize(features[normal_mask], p=2, dim=1), normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
                values_recon_normal.append(F.cosine_similarity(F.normalize(recon_vec[normal_mask], p=2, dim=1), normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))

            if abnormal_mask.sum() > 0:
                # values_features_abnormal.append(F.cosine_similarity(F.normalize(features[abnormal_mask], p=2, dim=1), normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
                values_recon_abnormal.append(F.cosine_similarity(F.normalize(recon_vec[abnormal_mask], p=2, dim=1), normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))

        values_recon_all = torch.cat(values_recon_all).cpu().numpy()
        # values_recon_all = torch.cat(values_recon_all)
        values_recon_normal = torch.cat(values_recon_normal).cpu().numpy()
        values_recon_abnormal = torch.cat(values_recon_abnormal).cpu().numpy()

    x_test = x_test.to(device)
    # values_features_test = process_batch(x_test, normal_temp, num_of_layer, model, batch_size, device)
    values_recon_test = process_batch(x_test, normal_recon_temp, num_of_output, model, batch_size, device)

    mu3_initial = np.mean(values_recon_normal)
    sigma3_initial = np.std(values_recon_normal)
    mu4_initial = np.mean(values_recon_abnormal)
    sigma4_initial = np.std(values_recon_abnormal)

    # Fit Gaussians to reconstruction similarities
    initial_params = np.array([mu3_initial, sigma3_initial, mu4_initial, sigma4_initial])
    result = opt.minimize(log_likelihood, initial_params, args=(values_recon_all,), method='Nelder-Mead')
    mu3_fit, sigma3_fit, mu4_fit, sigma4_fit = result.x

    if mu3_fit > mu4_fit:
        gaussian3 = Normal(mu3_fit, sigma3_fit)
        gaussian4 = Normal(mu4_fit, sigma4_fit)
    else:
        gaussian4 = Normal(mu3_fit, sigma3_fit)
        gaussian3 = Normal(mu4_fit, sigma4_fit)

    pdf3 = gaussian3.log_prob(values_recon_test.clone().detach()).exp()
    pdf4 = gaussian4.log_prob(values_recon_test.clone().detach()).exp()
    y_test_pred_4 = (pdf4 > pdf3).cpu().numpy().astype("int32")
    # y_test_pro_de = (torch.abs(pdf4 - pdf3)).cpu().detach().numpy().astype("float32")

    if get_probs:
        values_recon_test = values_recon_test.detach()
        return pdf3,pdf4,values_recon_test
    else:
        if not isinstance(y_test, int):
            if y_test.device != torch.device("cpu"):
                y_test = y_test.cpu().numpy()
            result_decoder = score_detail(y_test, y_test_pred_4)

        # y_test_pred_no_vote = torch.where(torch.from_numpy(y_test_pro_en) > torch.from_numpy(y_test_pro_de), torch.from_numpy(y_test_pred_2), torch.from_numpy(y_test_pred_4))
        y_test_pred_no_vote = torch.from_numpy(y_test_pred_4)

        if not isinstance(y_test, int):
            result_final = score_detail(y_test, y_test_pred_no_vote, if_print=True)
            # return result_encoder, result_decoder, result_final
            return result_decoder, result_final
        else:
            return y_test_pred_no_vote

# drift detection
def detect_drift(new_data, control_data, window_size, drift_threshold, method='mmd', sigma=1.0):
    """
    Detect concept drift using MMD (Maximum Mean Discrepancy).
    
    Args:
        new_data: New data samples (tensor or numpy array)
        control_data: Reference/control data samples (tensor or numpy array)
        window_size: Size of sliding window (must be > 0)
        drift_threshold: MMD statistic threshold (typically 0.001-0.01)
        method: Drift detection method (always 'mmd', kept for compatibility)
        sigma: Bandwidth parameter for MMD kernel
    
    Returns:
        Tuple[bool, float]: (drift_detected, statistic)
    """
    # Input validation
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if drift_threshold < 0:
        raise ValueError(f"drift_threshold must be non-negative, got {drift_threshold}")
    
    # Always use MMD for drift detection
    return detect_drift_mmd(new_data, control_data, window_size, drift_threshold, sigma=sigma)


# ============================================================================
# Mask-Based Selection for Ablation Study (only_mmd)
# ============================================================================
# The following functions implement the original mask-optimization-based 
# sample selection from the SSF paper, used for ablation experiments to 
# isolate the contribution of MMD drift detection.

def initialize_mask_tensor(size, initialization, device):
    """
    Initialize a mask tensor with specified range.
    
    Args:
        size: Size of the tensor
        initialization: Initialization type ('0-1', '0-0.5', '0.5-1')
        device: Computing device
    
    Returns:
        Initialized tensor as a parameter
    """
    if initialization == '0-1':
        return torch.nn.Parameter(torch.rand(size, device=device), requires_grad=True)
    elif initialization == '0-0.5':
        return torch.nn.Parameter(torch.rand(size, device=device) * 0.5, requires_grad=True)
    elif initialization == '0.5-1':
        return torch.nn.Parameter(torch.rand(size, device=device) * 0.5 + 0.5, requires_grad=True)
    else:
        raise ValueError("Invalid initialization type. Choose from '0-1', '0-0.5', or '0.5-1'.")


def optimize_old_mask(control_res, treatment_res, device, initialization='0.5-1', 
                      num_bins=10, lr=1.0, steps=100):
    """
    Optimize mask for old (control) samples using KL divergence minimization.
    This is the original mask optimization from SSF paper.
    
    Args:
        control_res: Control sample responses (old samples)
        treatment_res: Treatment sample responses (new samples)
        device: Computing device
        initialization: Mask initialization strategy
        num_bins: Number of histogram bins
        lr: Learning rate for optimization
        steps: Optimization steps
    
    Returns:
        M_c: Optimized mask for control samples
    """
    control_res = torch.tensor(control_res, dtype=torch.float).to(device)
    treatment_res = torch.tensor(treatment_res, dtype=torch.float).to(device)

    M_c = initialize_mask_tensor(control_res.size(0), initialization, device)
    optimizer = torch.optim.SGD([M_c], lr=lr)
    delta = 1e-4

    for step in range(steps):
        with torch.no_grad():
            M_c.clamp_(delta, 1 - delta)

        optimizer.zero_grad()

        bin_edges = torch.linspace(0., 1., num_bins + 1, device=device)
        control_hist = torch.histc(control_res, bins=num_bins, min=0., max=1.)
        treatment_hist = torch.histc(treatment_res, bins=num_bins, min=0., max=1.)

        bin_obs_c = torch.zeros(num_bins, device=device)
        bin_tgt_c = torch.zeros(num_bins, device=device)

        for i in range(num_bins):
            mask_c = (control_res >= bin_edges[i]) & (control_res < bin_edges[i + 1])
            bin_obs_c[i] = torch.sum(M_c * mask_c.float()) / torch.sum(M_c)
            bin_tgt_c[i] = treatment_hist[i] / len(treatment_res)

        bin_obs_c = bin_obs_c / bin_obs_c.sum()
        bin_tgt_c = bin_tgt_c / bin_tgt_c.sum()

        Accuracy_Loss_c = F.kl_div(bin_obs_c.log(), bin_tgt_c, reduction='sum')

        Loss = Accuracy_Loss_c
        Loss.backward()
        optimizer.step()

    return M_c.detach()


def optimize_new_mask(control_res, treatment_res, M_c, device, initialization='0-0.5',
                      num_bins=10, lr=50.0, steps=100):
    """
    Optimize mask for new (treatment) samples using KL divergence minimization.
    This is the original mask optimization from SSF paper.
    
    Args:
        control_res: Control sample responses (old samples)
        treatment_res: Treatment sample responses (new samples)
        M_c: Optimized mask for control samples
        device: Computing device
        initialization: Mask initialization strategy
        num_bins: Number of histogram bins
        lr: Learning rate for optimization
        steps: Optimization steps
    
    Returns:
        M_t: Optimized mask for treatment samples
    """
    control_res = torch.tensor(control_res, dtype=torch.float).to(device)
    treatment_res = torch.tensor(treatment_res, dtype=torch.float).to(device)
    if not isinstance(M_c, torch.Tensor):
        M_c = torch.tensor(M_c, dtype=torch.float).to(device)

    M_t = initialize_mask_tensor(treatment_res.size(0), initialization, device)
    optimizer = torch.optim.SGD([M_t], lr=lr)
    delta = 1e-4

    for step in range(steps):
        with torch.no_grad():
            M_t.clamp_(delta, 1 - delta)

        optimizer.zero_grad()

        bin_edges = torch.linspace(0., 1., num_bins + 1, device=device)
        Drift_Loss_t = 0.

        control_hist = torch.histc(control_res, bins=num_bins, min=0., max=1.)
        treatment_hist = torch.histc(treatment_res, bins=num_bins, min=0., max=1.)

        bin_tgt_t = torch.zeros(num_bins, device=device)
        bin_combined = torch.zeros(num_bins, device=device)
        
        for i in range(num_bins):
            mask_c = (control_res >= bin_edges[i]) & (control_res < bin_edges[i + 1])
            mask_t = (treatment_res >= bin_edges[i]) & (treatment_res < bin_edges[i + 1])

            bin_tgt_t[i] = treatment_hist[i] / len(treatment_res)
            bin_combined[i] = (torch.sum(M_t * mask_t.float()) + torch.sum(M_c * mask_c.float())) / (torch.sum(M_t) + torch.sum(M_c))

        bin_combined = torch.clamp(bin_combined / bin_combined.sum(), min=1e-10)
        bin_combined = bin_combined / bin_combined.sum()
        bin_tgt_t = torch.clamp(bin_tgt_t / bin_tgt_t.sum(), min=1e-10)
        bin_tgt_t = bin_tgt_t / bin_tgt_t.sum()

        Drift_Loss_t = F.kl_div(bin_combined.log(), bin_tgt_t, reduction='sum')

        Loss = Drift_Loss_t
        Loss.backward()
        optimizer.step()

    return M_t.detach()


def select_and_update_representative_samples_mask(
    x_train_this_epoch, y_train_this_epoch, 
    x_test_this_epoch, y_test_this_epoch,
    M_c, M_t, num_labeled_sample, device, buffer_memory_size=None
):
    """
    Select and update representative samples using mask-based method (no drift case).
    This replicates the original SSF selection for ablation studies.
    
    Args:
        x_train_this_epoch: Current buffer samples
        y_train_this_epoch: Current buffer labels
        x_test_this_epoch: New incoming samples
        y_test_this_epoch: New incoming labels
        M_c: Optimized mask for control (old) samples
        M_t: Optimized mask for treatment (new) samples
        num_labeled_sample: Number of labeled samples to select
        device: Computing device
        buffer_memory_size: Buffer capacity (optional)
    
    Returns:
        Updated buffer samples, labels, selected indices, and new sample mask
    """
    M_c_bin = (M_c >= 0.5).float().to(device)
    M_t_bin = (M_t >= 0.5).float().to(device)

    representative_old = x_train_this_epoch[M_c_bin.bool()]
    representative_new = x_test_this_epoch[M_t_bin.bool()]

    print(f"[Mask-based] Selected representative old samples: {representative_old.shape}")
    print(f"[Mask-based] Selected representative new samples: {representative_new.shape}")

    old_indices = torch.arange(len(x_train_this_epoch), device=device)
    representative_old_indices = old_indices[M_c_bin.bool()]

    mask_c = torch.ones(len(x_train_this_epoch), dtype=torch.bool, device=device)
    mask_c[representative_old_indices] = False

    non_representative_old_indices = old_indices[mask_c]
    num_to_remove = num_labeled_sample

    if len(non_representative_old_indices) < num_to_remove:
        print(f"[Mask-based] Not enough non-representative old samples to remove ({len(non_representative_old_indices)}). Removing additional representative samples.")
        additional_remove_needed = num_to_remove - len(non_representative_old_indices)

        remove_indices = non_representative_old_indices

        representative_scores = M_c[M_c_bin.bool()].detach().cpu().numpy()
        sorted_rep_indices = torch.argsort(torch.tensor(representative_scores))[:additional_remove_needed]
        additional_remove_indices = representative_old_indices[sorted_rep_indices]

        remove_indices = torch.cat([remove_indices, additional_remove_indices])
    else:
        remove_indices = non_representative_old_indices[torch.randperm(len(non_representative_old_indices))[:num_to_remove]]

    mask = torch.ones(x_train_this_epoch.size(0), dtype=torch.bool, device=device)
    mask[remove_indices] = False

    x_train_this_epoch = x_train_this_epoch[mask]
    y_train_this_epoch = y_train_this_epoch[mask]

    new_sample_mask = torch.zeros_like(y_train_this_epoch, dtype=torch.float32).to(device)

    if representative_new.shape[0] < num_labeled_sample:
        print(f"[Mask-based] Not enough representative new samples selected ({representative_new.shape[0]}). Selecting additional random samples.")
        additional_samples_needed = num_labeled_sample - representative_new.shape[0]

        selected_indices = set(torch.arange(len(x_test_this_epoch))[M_t_bin.bool().cpu().numpy()])
        available_indices = set(torch.arange(len(x_test_this_epoch)).cpu().numpy()) - selected_indices
        available_indices = torch.tensor(list(available_indices), dtype=torch.long)

        fallback_indices = available_indices[torch.randperm(len(available_indices))[:additional_samples_needed]]
        drift_representative_new = torch.cat([representative_new, x_test_this_epoch[fallback_indices]], dim=0)
        new_labels = torch.cat([y_test_this_epoch[M_t_bin.bool()], y_test_this_epoch[fallback_indices]], dim=0)
        sorted_indices_new = torch.cat([torch.arange(len(representative_new)), fallback_indices], dim=0)
    else:
        scores_new = M_t[M_t_bin.bool()].detach().cpu().numpy()
        sorted_indices_new = torch.argsort(torch.tensor(scores_new), descending=True)[:num_labeled_sample]
        drift_representative_new = representative_new[sorted_indices_new]
        new_labels = y_test_this_epoch[M_t_bin.bool()][sorted_indices_new]

    new_sample_mask = torch.cat([new_sample_mask, torch.ones(len(drift_representative_new), dtype=torch.float32).to(device)])
    x_train_this_epoch = torch.cat([x_train_this_epoch, drift_representative_new], dim=0)
    y_train_this_epoch = torch.cat([y_train_this_epoch, new_labels], dim=0)

    return x_train_this_epoch, y_train_this_epoch, sorted_indices_new, new_sample_mask


def select_and_update_representative_samples_mask_drift(
    x_train_this_epoch, y_train_this_epoch,
    x_test_this_epoch, y_test_this_epoch,
    M_c, M_t, num_labeled_sample, device, buffer_memory_size, model=None, normal_recon_temp=None
):
    """
    Select and update representative samples using mask-based method (drift case).
    This replicates the original SSF selection with pseudo-labeling for ablation studies.
    
    Args:
        x_train_this_epoch: Current buffer samples
        y_train_this_epoch: Current buffer labels
        x_test_this_epoch: New incoming samples
        y_test_this_epoch: New incoming labels
        M_c: Optimized mask for control (old) samples
        M_t: Optimized mask for treatment (new) samples
        num_labeled_sample: Number of labeled samples to select
        device: Computing device
        buffer_memory_size: Buffer capacity
        model: Model for pseudo-labeling (optional)
        normal_recon_temp: Normal reconstruction template (optional, for NSL)
    
    Returns:
        Updated buffer samples, labels, selected indices, and new sample mask
    """
    M_c_bin = (M_c >= 0.5).float().to(device)
    M_t_bin = (M_t >= 0.5).float().to(device)

    representative_old = x_train_this_epoch[M_c_bin.bool()]
    representative_new = x_test_this_epoch[M_t_bin.bool()]

    print(f"[Mask-based] Selected representative old samples: {representative_old.shape}")
    print(f"[Mask-based] Selected representative new samples: {representative_new.shape}")

    old_indices = torch.arange(len(x_train_this_epoch), device=device)
    representative_old_indices = old_indices[M_c_bin.bool()]

    mask_c = torch.ones(len(x_train_this_epoch), dtype=torch.bool, device=device)
    mask_c[representative_old_indices] = False

    non_representative_old_indices = old_indices[mask_c]
    num_to_remove = num_labeled_sample

    # Remove all non-representative samples
    remove_indices = non_representative_old_indices

    if len(non_representative_old_indices) < num_to_remove:
        print(f"[Mask-based] Not enough non-representative old samples to remove ({len(non_representative_old_indices)}). Removing additional representative samples.")
        additional_remove_needed = num_to_remove - len(non_representative_old_indices)

        representative_scores = M_c[M_c_bin.bool()].detach().cpu().numpy()
        sorted_rep_indices = torch.argsort(torch.tensor(representative_scores))[:additional_remove_needed]
        additional_remove_indices = representative_old_indices[sorted_rep_indices]

        remove_indices = torch.cat([remove_indices, additional_remove_indices])

    mask = torch.ones(x_train_this_epoch.size(0), dtype=torch.bool, device=device)
    mask[remove_indices] = False

    x_train_this_epoch = x_train_this_epoch[mask]
    y_train_this_epoch = y_train_this_epoch[mask]

    new_sample_mask = torch.zeros_like(y_train_this_epoch, dtype=torch.float32).to(device)

    if representative_new.shape[0] < num_labeled_sample:
        print(f"[Mask-based] Not enough representative samples selected ({representative_new.shape[0]}). Selecting additional random samples.")
        additional_samples_needed = num_labeled_sample - representative_new.shape[0]

        selected_indices = set(torch.arange(len(x_test_this_epoch))[M_t_bin.bool().cpu().numpy()])
        available_indices = set(torch.arange(len(x_test_this_epoch)).cpu().numpy()) - selected_indices
        available_indices = torch.tensor(list(available_indices), dtype=torch.long)

        fallback_indices = available_indices[torch.randperm(len(available_indices))[:additional_samples_needed]]
        drift_representative_new = torch.cat([representative_new, x_test_this_epoch[fallback_indices]], dim=0)
        new_labels = torch.cat([y_test_this_epoch[M_t_bin.bool()], y_test_this_epoch[fallback_indices]], dim=0)
        sorted_indices_new = torch.cat([torch.arange(len(representative_new)), fallback_indices], dim=0)
    else:
        scores_new = M_t[M_t_bin.bool()].detach().cpu().numpy()
        sorted_indices_new = torch.argsort(torch.tensor(scores_new), descending=True)[:num_labeled_sample]
        drift_representative_new = representative_new[sorted_indices_new]
        new_labels = y_test_this_epoch[M_t_bin.bool()][sorted_indices_new]

    new_sample_mask = torch.cat([new_sample_mask, torch.ones(len(drift_representative_new), dtype=torch.float32).to(device)])
    x_train_this_epoch = torch.cat((x_train_this_epoch, drift_representative_new), dim=0)
    y_train_this_epoch = torch.cat((y_train_this_epoch, new_labels), dim=0)

    # Fill buffer to capacity with pseudo-labeled samples if needed
    if buffer_memory_size and len(x_train_this_epoch) < buffer_memory_size:
        additional_samples_needed = buffer_memory_size - len(x_train_this_epoch)
        print(f"[Mask-based] Buffer memory has extra space for {additional_samples_needed} samples. Adding new samples with pseudo labels.")

        if representative_new.shape[0] > num_labeled_sample:
            remaining_new_samples = representative_new[torch.argsort(torch.tensor(scores_new), descending=True)[num_labeled_sample:]]

            if remaining_new_samples.size(0) >= additional_samples_needed:
                pseudo_labeled_samples = remaining_new_samples[:additional_samples_needed]
                if normal_recon_temp is None and model is not None:
                    from utils1 import evaluate_inputs
                    pseudo_labels = evaluate_inputs(model, pseudo_labeled_samples, device)
                elif normal_recon_temp is not None:
                    from utils1 import evaluate
                    pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch, 
                                            pseudo_labeled_samples, 0, model)
                else:
                    pseudo_labels = torch.zeros(len(pseudo_labeled_samples), device=device)
            else:
                pseudo_labeled_samples = remaining_new_samples
                if normal_recon_temp is None and model is not None:
                    from utils1 import evaluate_inputs
                    pseudo_labels = evaluate_inputs(model, pseudo_labeled_samples, device)
                elif normal_recon_temp is not None:
                    from utils1 import evaluate
                    pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch,
                                            pseudo_labeled_samples, 0, model)
                else:
                    pseudo_labels = torch.zeros(len(pseudo_labeled_samples), device=device)

                random_new_additional_samples_needed = additional_samples_needed - remaining_new_samples.size(0)
                additional_indices = torch.randperm(len(x_test_this_epoch))[:random_new_additional_samples_needed]
                additional_pseudo_labeled_samples = x_test_this_epoch[additional_indices]
                
                if normal_recon_temp is None and model is not None:
                    additional_pseudo_labels = evaluate_inputs(model, additional_pseudo_labeled_samples, device)
                elif normal_recon_temp is not None:
                    additional_pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch,
                                                       additional_pseudo_labeled_samples, 0, model)
                else:
                    additional_pseudo_labels = torch.zeros(len(additional_pseudo_labeled_samples), device=device)

                pseudo_labeled_samples = torch.cat([pseudo_labeled_samples, additional_pseudo_labeled_samples], dim=0)
                if isinstance(pseudo_labels, torch.Tensor):
                    pseudo_labels = torch.cat([pseudo_labels if pseudo_labels.dim() > 0 else pseudo_labels.unsqueeze(0),
                                              additional_pseudo_labels if additional_pseudo_labels.dim() > 0 else additional_pseudo_labels.unsqueeze(0)], dim=0)
                else:
                    pseudo_labels = torch.cat([torch.tensor(pseudo_labels, device=device),
                                              torch.tensor(additional_pseudo_labels, device=device)], dim=0)
        else:
            additional_indices = torch.randperm(len(x_test_this_epoch))[:additional_samples_needed]
            pseudo_labeled_samples = x_test_this_epoch[additional_indices]
            if normal_recon_temp is None and model is not None:
                from utils1 import evaluate_inputs
                pseudo_labels = evaluate_inputs(model, pseudo_labeled_samples, device)
            elif normal_recon_temp is not None:
                from utils1 import evaluate
                pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch,
                                        pseudo_labeled_samples, 0, model)
            else:
                pseudo_labels = torch.zeros(len(pseudo_labeled_samples), device=device)

        x_train_this_epoch = torch.cat((x_train_this_epoch, pseudo_labeled_samples), dim=0)
        if isinstance(pseudo_labels, torch.Tensor):
            y_train_this_epoch = torch.cat((y_train_this_epoch, pseudo_labels.to(device)), dim=0)
        else:
            y_train_this_epoch = torch.cat((y_train_this_epoch, torch.tensor(pseudo_labels, device=device)), dim=0)

        new_sample_mask = torch.cat([new_sample_mask, torch.zeros(len(pseudo_labeled_samples), dtype=torch.float32).to(device)])

    return x_train_this_epoch, y_train_this_epoch, sorted_indices_new, new_sample_mask
