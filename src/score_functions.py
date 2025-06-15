import numpy as np
import torch
from sklearn.model_selection import ParameterGrid


def temperature_scaling(confidence, temperature):
    """
    Apply temperature scaling to a confidence score.
    
    Parameters:
        confidence (float or np.array): Original confidence score(s) in the range [0, 1].
        temperature (float): Temperature parameter. T > 1 reduces confidence; T < 1 increases confidence.
        
    Returns:
        float or np.array: Scaled confidence score(s).
    """
    scaled_confidence = (confidence ** (1 / temperature)) / (confidence ** (1 / temperature) + (1 - confidence) ** (1 / temperature))
    return scaled_confidence

# Define LAC score function
def lac_conformal_score(tsr_confidence, ocr_confidence, temperature = 1.5):
    """
    Conformal score function for Least Ambiguous Classifiers (LAC).
    Uses the minimum of TSR and OCR confidence scores for conformity.
    """
    # Convert tensor to scalar if needed
    if isinstance(tsr_confidence, torch.Tensor):
        tsr_confidence = tsr_confidence.item()

    if isinstance(ocr_confidence, torch.Tensor):
        ocr_confidence = ocr_confidence.item()
    
    # Handle None or invalid values
    if tsr_confidence is None or ocr_confidence is None:
        return 0.0  # or another default value
    
    # Scale the TSR and OCR confidence for reliability
    scaled_tsr_confidence = temperature_scaling(tsr_confidence, temperature)
    scaled_ocr_confidence = temperature_scaling(ocr_confidence, temperature)
    score = min(scaled_tsr_confidence, scaled_ocr_confidence)
    return score, scaled_tsr_confidence, scaled_ocr_confidence

# Define the Adaptive Prediction Sets (APS) Score Function
def aps_conformal_score(tsr_confidence, ocr_confidence, temperature = 1.5):
    """
    Conformal score function for Adaptive Prediction Sets (APS).
    Uses the cumulative confidence of TSR and OCR scores for conformity.
    """
    # Convert tensor to scalar if needed
    if isinstance(tsr_confidence, torch.Tensor):
        tsr_confidence = tsr_confidence.item()
    if isinstance(ocr_confidence, torch.Tensor):
        ocr_confidence = ocr_confidence.item()
    
    # Handle None or invalid values
    if tsr_confidence is None or ocr_confidence is None:
        return 0.0  # or another default value
    
    # Scale the TSR and OCR confidence for reliability
    scaled_tsr_confidence = temperature_scaling(tsr_confidence, temperature)
    scaled_ocr_confidence = temperature_scaling(ocr_confidence, temperature)
    combined_confidences = sorted([scaled_tsr_confidence, scaled_ocr_confidence], reverse=True)
    cumulative_score = sum(combined_confidences)
    return cumulative_score, scaled_tsr_confidence, scaled_ocr_confidence

def nr_cp_conformal_score(tsr_confidence, ocr_confidence, noise_rate=0.1, temperature = 1.5):
    """
    Noisy Robust APS Randomized (NR-CP) conformal score function.
    Incorporates noise robustness into the confidence-based uncertainty quantification.
    
    Parameters:
        tsr_confidence (float): TSR confidence for a cell location.
        ocr_confidence (float): OCR confidence for cell content.
        noise_rate (float): Noise rate for robust APS.
    
    Returns:
        score (float): NR-CP conformal score.
    """

    # Convert tensor to scalar if needed
    if isinstance(tsr_confidence, torch.Tensor):
        tsr_confidence = tsr_confidence.item()
    if isinstance(ocr_confidence, torch.Tensor):
        ocr_confidence = ocr_confidence.item()

    # Handle None or invalid values
    if tsr_confidence is None or ocr_confidence is None:
        return 1.0  # Default value if confidence is missing
    
    scaled_tsr_confidence = temperature_scaling(tsr_confidence, temperature)
    scaled_ocr_confidence = temperature_scaling(ocr_confidence, temperature)
    # Construct probability distribution
    conf_probs = np.array([scaled_tsr_confidence, scaled_ocr_confidence])
    
    # Normalize probabilities to sum to 1 (avoid division by zero)
    if np.sum(conf_probs) > 0:
        conf_probs /= np.sum(conf_probs)
    else:
        conf_probs = np.array([0.5, 0.5])  # Default uniform distribution if both confidences are zero

    # Compute cumulative probabilities in descending order
    sorted_probs = np.sort(conf_probs)[::-1]
    softmax_correct_class = np.cumsum(sorted_probs)

    # Compute cumsum index safely
    cumsum_index = np.searchsorted(softmax_correct_class, softmax_correct_class)

    # Ensure valid indexing
    valid_indices = cumsum_index > 0
    if valid_indices.any():  # Ensure the condition is evaluated correctly
        low = np.zeros_like(softmax_correct_class)
        low[valid_indices] = softmax_correct_class[cumsum_index[valid_indices] - 1]
    else:
        low = np.zeros_like(softmax_correct_class)  # If no valid indices, set to zeros

    # Randomized thresholding
    randomized_threshold = np.random.uniform(low=low[0], high=softmax_correct_class[0])
    noisy_score = randomized_threshold * (1 - noise_rate) + noise_rate * np.mean(conf_probs)

    return noisy_score, scaled_ocr_confidence, scaled_tsr_confidence



def hybrid_spatial_score(tsr_confidence, ocr_confidence, temperature=1.5,
                        row_weight=0.3, col_weight=0.2, text_weight=0.5):
    """Hybrid conformal score with spatial awareness and confidence scaling"""
    try:
        # Handle missing OCR confidence
        if ocr_confidence is None:
            ocr_confidence = tsr_confidence  # Fallback to TSR confidence
        
        # Convert tensors to scalars if needed
        if isinstance(tsr_confidence, torch.Tensor):
            tsr_confidence = tsr_confidence.item()
        if isinstance(ocr_confidence, torch.Tensor):
            ocr_confidence = ocr_confidence.item()

        # Temperature scaling
        scaled_tsr = temperature_scaling(tsr_confidence, temperature)
        scaled_ocr = temperature_scaling(ocr_confidence, temperature)

        # Spatial reliability components
        row_reliability = 1 - (row_weight * (1 - scaled_tsr))
        col_reliability = 1 - (col_weight * (1 - scaled_tsr))
        
        # Content reliability component
        text_reliability = text_weight * scaled_ocr if scaled_ocr > 0 else (1 - text_weight) * scaled_tsr

        # Combined confidence with geometric mean
        structural_conf = (row_reliability * col_reliability) ** 0.5
        content_conf = text_reliability
        combined_conf = (structural_conf * content_conf) ** 0.5
    except Exception as e:
        print(f"An error occurred while computing hybrid score: {str(e)}")
        return 0.0, 0.0, 0.0
    # Return all three required values as a tuple
    return (1 - combined_conf,  # Conformal score
            scaled_tsr,         # Scaled TSR confidence
            scaled_ocr)         # Scaled OCR confidence



def optimize_weights(calibration_data, alpha=0.1):
    """Grid search optimization for hybrid score weights"""
    # Flatten calibration data into individual cells
    all_cells = [cell for table in calibration_data for cell in table]
    
    # Define search space
    param_grid = {
        'row_weight': np.linspace(0.1, 0.5, 5),
        'col_weight': np.linspace(0.05, 0.3, 5),
        'text_weight': np.linspace(0.3, 0.7, 5)
    }
    
    best_params = None
    best_metric = float('inf')
    
    # Cross-validate on calibration data
    for params in ParameterGrid(param_grid):
        scores = []
        for cell in all_cells:
            score, _, _ = hybrid_spatial_score(
                cell['tsr_combined_conf'],
                cell.get('ocr_confidence'),
                **params
            )
            scores.append(score)
        
        # Calculate threshold
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        threshold = np.quantile(scores, q_level, method="higher")
        
        # Calculate validation metrics
        set_sizes = [1 if s <= threshold else 0 for s in scores]
        coverage = np.mean(set_sizes)
        avg_size = np.mean(set_sizes)
        
        # Optimization criterion: minimize set size while maintaining coverage
        if coverage >= (1 - alpha) * 0.95:  # 95% of target coverage
            if avg_size < best_metric:
                best_metric = avg_size
                best_params = params
                
    return best_params or {'row_weight': 0.3, 'col_weight': 0.2, 'text_weight': 0.5}


def tsr_only_score(tsr_confidence, ocr_confidence, temperature = 1.5):
    """
    Conformal score function for Least Ambiguous Classifiers (LAC).
    Uses the minimum of TSR and OCR confidence scores for conformity.
    """
    # Convert tensor to scalar if needed
    if isinstance(tsr_confidence, torch.Tensor):
        tsr_confidence = tsr_confidence.item()

    if isinstance(ocr_confidence, torch.Tensor):
        ocr_confidence = ocr_confidence.item()
    
    # Handle None or invalid values
    if tsr_confidence is None or ocr_confidence is None:
        return 0.0  # or another default value
    
    # Scale the TSR and OCR confidence for reliability
    scaled_tsr_confidence = temperature_scaling(tsr_confidence, temperature)
    scaled_ocr_confidence = temperature_scaling(ocr_confidence, temperature)
    score = scaled_tsr_confidence
    return score, scaled_tsr_confidence, scaled_ocr_confidence

def ocr_only_score(tsr_confidence, ocr_confidence, temperature = 1.5):
    """
    Conformal score function for Least Ambiguous Classifiers (LAC).
    Uses the minimum of TSR and OCR confidence scores for conformity.
    """
    # Convert tensor to scalar if needed
    if isinstance(tsr_confidence, torch.Tensor):
        tsr_confidence = tsr_confidence.item()

    if isinstance(ocr_confidence, torch.Tensor):
        ocr_confidence = ocr_confidence.item()
    
    # Handle None or invalid values
    if tsr_confidence is None or ocr_confidence is None:
        return 0.0  # or another default value
    
    # Scale the TSR and OCR confidence for reliability
    scaled_tsr_confidence = temperature_scaling(tsr_confidence, temperature)
    scaled_ocr_confidence = temperature_scaling(ocr_confidence, temperature)
    score = scaled_ocr_confidence
    return score, scaled_tsr_confidence, scaled_ocr_confidence