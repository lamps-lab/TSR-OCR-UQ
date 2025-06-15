import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import json
import io
from PIL import Image
from paddleocr import PaddleOCR
from transformers import AutoModelForObjectDetection
from pathlib import Path
import base64

# Import your existing functions
try:
    from tsr_ocr import extract_tsr_ocr_confidences
    from score_functions import aps_conformal_score
    from utils import compute_calibration_scores
except ImportError as e:
    st.error(f"Required modules not found: {str(e)}")
    st.error("Make sure tsr_ocr.py, score_functions.py, and utils.py are in the src/ directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Table Extraction & Uncertainty Quantification",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to convert tensors and handle all data types
def safe_convert_value(value):
    """Convert any value to a JSON-serializable Python type"""
    if isinstance(value, torch.Tensor):
        return float(value.item())
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.int32, np.int64)):
        return int(value)
    elif isinstance(value, (np.float32, np.float64)):
        return float(value)
    elif value is None:
        return None
    else:
        return value

def clean_cell_data(cell):
    """Clean a cell dictionary to ensure all values are JSON-serializable"""
    cleaned_cell = {}
    for key, value in cell.items():
        if key == 'bbox' and isinstance(value, (list, tuple)):
            cleaned_cell[key] = [safe_convert_value(x) for x in value]
        else:
            cleaned_cell[key] = safe_convert_value(value)
    return cleaned_cell

# Cache models to avoid reloading
@st.cache_resource
def load_models():
    """Load TSR and OCR models once and cache them"""
    with st.spinner("Loading AI models... This may take a few minutes on first run."):
        tsr_model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition-v1.1-all"
        )
        ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
    return tsr_model, ocr_model

@st.cache_data
def load_calibration_data():
    """Load APS calibration scores from saved numpy file"""
    calibration_file = "/app/data/calibration_data/calibration_scores_aps.npy"
    metadata_file = "/app/data/calibration_data/calibration_metadata.json"
    
    try:
        # Load calibration scores
        cal_scores = np.load(calibration_file)
        
        # Load metadata if available
        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        return cal_scores, metadata
        
    except FileNotFoundError:
        st.error("❌ Calibration data not found!")
        st.error("Please run calibration computation first:")
        st.code("compute_calibration.bat")
        return None, {}
    except Exception as e:
        st.error(f"Error loading calibration data: {e}")
        return None, {}

def compute_uncertainty_scores(extracted_cells, threshold, best_threshold=0.03):
    """Add uncertainty scores and flags using APS conformal score"""
    cleaned_cells = []
    
    for cell in extracted_cells:
        # Clean the cell first
        cleaned_cell = clean_cell_data(cell)
        
        tsr_conf = safe_convert_value(cleaned_cell.get("tsr_combined_conf", 0))
        ocr_conf = safe_convert_value(cleaned_cell.get("ocr_confidence", tsr_conf))
        
        # Compute uncertainty score using APS conformal score
        score, _, _ = aps_conformal_score(tsr_conf, ocr_conf)
        score = safe_convert_value(score)
        uncertainty_score = min(1.0, abs(score - threshold) / threshold)
        
        # Update cleaned cell with computed values
        cleaned_cell["tsr_combined_conf"] = tsr_conf
        cleaned_cell["ocr_confidence"] = ocr_conf
        cleaned_cell["uncertainty_score"] = uncertainty_score
        cleaned_cell["flagged_for_review"] = uncertainty_score > best_threshold
        
        cleaned_cells.append(cleaned_cell)
    
    return cleaned_cells

def create_dataframe(extracted_cells):
    """Convert extracted cells to pandas DataFrame with clean data types"""
    if not extracted_cells:
        return pd.DataFrame()
    
    data = []
    for i, cell in enumerate(extracted_cells):
        # Ensure all values are clean
        ocr_conf = safe_convert_value(cell.get('ocr_confidence', 0))
        tsr_conf = safe_convert_value(cell.get('tsr_combined_conf', 0))
        uncertainty_score = safe_convert_value(cell.get('uncertainty_score', 0))
        
        # Handle bbox safely
        bbox = cell.get('bbox', [0, 0, 0, 0])
        if bbox and len(bbox) >= 4:
            bbox_values = [safe_convert_value(x) for x in bbox[:4]]
        else:
            bbox_values = [0.0, 0.0, 0.0, 0.0]
        
        row = {
            'Cell_ID': i + 1,
            'Text': str(cell.get('text', '')),
            'Start_Row': str(cell.get('start_row', '')),
            'End_Row': str(cell.get('end_row', '')),
            'Start_Col': str(cell.get('start_col', '')),
            'End_Col': str(cell.get('end_col', '')),
            'OCR_Confidence': round(float(ocr_conf), 4),
            'TSR_Confidence': round(float(tsr_conf), 4),
            'Uncertainty_Score': round(float(uncertainty_score), 4),
            'Flagged_for_Review': bool(cell.get('flagged_for_review', False)),
            'BBox_X1': round(float(bbox_values[0]), 2),
            'BBox_Y1': round(float(bbox_values[1]), 2),
            'BBox_X2': round(float(bbox_values[2]), 2),
            'BBox_Y2': round(float(bbox_values[3]), 2)
        }
        data.append(row)
    
    return pd.DataFrame(data)

def reconstruct_table_grid(extracted_cells):
    """Reconstruct the original table structure with 5 columns per cell position"""
    if not extracted_cells:
        return pd.DataFrame(), pd.DataFrame()
    
    # Find table dimensions
    max_row = max(int(cell.get('start_row', 0)) for cell in extracted_cells if cell.get('start_row', '') != '') + 1
    max_col = max(int(cell.get('start_col', 0)) for cell in extracted_cells if cell.get('start_col', '') != '') + 1
    
    # Create column names - 5 columns per original table column
    columns = []
    flag_columns = []
    for col in range(max_col):
        columns.extend([
            f'Text_Col_{col}',
            f'TSR_Col_{col}', 
            f'OCR_Col_{col}',
            f'Uncertainty_Col_{col}',
            f'Flag_Col_{col}'
        ])
        flag_columns.extend([
            f'Flag_Text_Col_{col}',
            f'Flag_TSR_Col_{col}', 
            f'Flag_OCR_Col_{col}',
            f'Flag_Uncertainty_Col_{col}',
            f'Flag_Flag_Col_{col}'
        ])
    
    # Initialize empty dataframes
    display_data = {col: [""] * max_row for col in columns}
    csv_data = {col: [""] * max_row for col in columns}
    flag_data = {col: [False] * max_row for col in flag_columns}
    
    # Fill the grids with cell data
    for cell in extracted_cells:
        try:
            row = int(cell.get('start_row', 0)) if cell.get('start_row', '') != '' else 0
            col = int(cell.get('start_col', 0)) if cell.get('start_col', '') != '' else 0
            
            text = str(cell.get('text', ''))
            tsr_conf = float(cell.get('tsr_combined_conf', 0))
            ocr_conf = float(cell.get('ocr_confidence', 0))
            uncertainty = float(cell.get('uncertainty_score', 0))
            flagged = bool(cell.get('flagged_for_review', False))
            
            # Only fill if there's actual content
            if text.strip():
                # Fill display data
                display_data[f'Text_Col_{col}'][row] = text
                display_data[f'TSR_Col_{col}'][row] = f"{tsr_conf:.3f}"
                display_data[f'OCR_Col_{col}'][row] = f"{ocr_conf:.3f}"
                display_data[f'Uncertainty_Col_{col}'][row] = f"{uncertainty:.3f}"
                display_data[f'Flag_Col_{col}'][row] = "🚩 Review" if flagged else "✅ Good"
                
                # Fill CSV data (same but cleaner format)
                csv_data[f'Text_Col_{col}'][row] = text
                csv_data[f'TSR_Col_{col}'][row] = f"{tsr_conf:.3f}"
                csv_data[f'OCR_Col_{col}'][row] = f"{ocr_conf:.3f}"
                csv_data[f'Uncertainty_Col_{col}'][row] = f"{uncertainty:.3f}"
                csv_data[f'Flag_Col_{col}'][row] = "Review" if flagged else "Good"
                
                # Set flag data for all 5 columns of this cell
                flag_data[f'Flag_Text_Col_{col}'][row] = flagged
                flag_data[f'Flag_TSR_Col_{col}'][row] = flagged
                flag_data[f'Flag_OCR_Col_{col}'][row] = flagged
                flag_data[f'Flag_Uncertainty_Col_{col}'][row] = flagged
                flag_data[f'Flag_Flag_Col_{col}'][row] = flagged
            
        except (ValueError, TypeError) as e:
            # Handle any conversion errors gracefully
            continue
    
    # Convert to DataFrames
    display_df = pd.DataFrame(display_data)
    csv_df = pd.DataFrame(csv_data)
    flag_df = pd.DataFrame(flag_data)
    
    # Set proper row names
    display_df.index = [f'Row_{i}' for i in range(len(display_df.index))]
    csv_df.index = [f'Row_{i}' for i in range(len(csv_df.index))]
    flag_df.index = [f'Row_{i}' for i in range(len(flag_df.index))]
    
    # Store flag information for styling
    display_df._flag_info = flag_df
    
    return display_df, csv_df

def apply_table_styling(df):
    """Apply styling to the reconstructed table based on flags"""
    if not hasattr(df, '_flag_info'):
        return df
    
    flag_df = df._flag_info
    
    def highlight_cells(val):
        """Color cells based on flag status"""
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        
        for row_idx in range(len(df.index)):
            for col_name in df.columns:
                try:
                    # Get corresponding flag column name
                    flag_col_name = f'Flag_{col_name}'
                    
                    if flag_col_name in flag_df.columns:
                        is_flagged = flag_df.loc[df.index[row_idx], flag_col_name]
                        cell_value = df.iloc[row_idx, df.columns.get_loc(col_name)]
                        
                        if cell_value == "":  # Empty cell
                            styles.iloc[row_idx, df.columns.get_loc(col_name)] = ''
                        elif is_flagged:  # Flagged for review
                            styles.iloc[row_idx, df.columns.get_loc(col_name)] = 'background-color: #ffebee; border: 1px solid #f44336'
                        else:  # Good cell with content
                            styles.iloc[row_idx, df.columns.get_loc(col_name)] = 'background-color: #e8f5e8; border: 1px solid #4caf50'
                except (IndexError, TypeError, KeyError):
                    continue
        
        return styles
    
    return df.style.apply(highlight_cells, axis=None)

def filter_reconstructed_table(display_df, csv_df, show_flagged_only=False, show_text_only=False):
    """Filter the reconstructed table based on user preferences"""
    if not hasattr(display_df, '_flag_info'):
        return display_df, csv_df
    
    flag_df = display_df._flag_info
    
    # Apply filters
    filtered_display = display_df.copy()
    filtered_csv = csv_df.copy()
    
    for row_idx in range(len(display_df.index)):
        for col_idx in range(0, len(display_df.columns), 5):  # Step by 5 since each cell has 5 columns
            text_col = display_df.columns[col_idx]  # Text column
            flag_col = f'Flag_Text_Col_{col_idx // 5}'  # Corresponding flag column
            
            try:
                has_text = display_df.iloc[row_idx, col_idx] != ""
                is_flagged = flag_df.loc[display_df.index[row_idx], flag_col] if flag_col in flag_df.columns else False
                
                # Determine if this cell should be shown
                show_cell = True
                
                if show_text_only and not has_text:
                    show_cell = False
                
                if show_flagged_only and not is_flagged:
                    show_cell = False
                
                # If cell should not be shown, clear all 5 columns for this cell
                if not show_cell:
                    for offset in range(5):
                        if col_idx + offset < len(display_df.columns):
                            filtered_display.iloc[row_idx, col_idx + offset] = ""
                            filtered_csv.iloc[row_idx, col_idx + offset] = ""
                            
            except (IndexError, KeyError):
                continue
    
    # Preserve flag info for styling
    filtered_display._flag_info = flag_df
    
    return filtered_display, filtered_csv

def get_table_statistics(display_df):
    """Calculate statistics for the reconstructed table (5-column format)"""
    if not hasattr(display_df, '_flag_info'):
        return 0, 0, 0, 0
    
    flag_df = display_df._flag_info
    
    total_cells = 0
    flagged_cells = 0
    uncertainty_scores = []
    
    # Count cells by looking at every 5th column (the text columns)
    for col_idx in range(0, len(display_df.columns), 5):
        text_col = display_df.columns[col_idx]
        uncertainty_col = display_df.columns[col_idx + 3] if col_idx + 3 < len(display_df.columns) else None
        flag_col = f'Flag_Text_Col_{col_idx // 5}'
        
        for row_idx in range(len(display_df.index)):
            try:
                has_text = display_df.iloc[row_idx, col_idx] != ""
                
                if has_text:
                    total_cells += 1
                    
                    # Check if flagged
                    if flag_col in flag_df.columns:
                        is_flagged = flag_df.loc[display_df.index[row_idx], flag_col]
                        if is_flagged:
                            flagged_cells += 1
                    
                    # Get uncertainty score
                    if uncertainty_col:
                        uncertainty_str = display_df.iloc[row_idx, col_idx + 3]
                        try:
                            uncertainty_scores.append(float(uncertainty_str))
                        except (ValueError, TypeError):
                            pass
                            
            except (IndexError, KeyError):
                continue
    
    cells_with_text = total_cells  # Same as total_cells since we only count non-empty
    avg_uncertainty = np.mean(uncertainty_scores) if uncertainty_scores else 0
    
    return total_cells, flagged_cells, cells_with_text, avg_uncertainty

@st.cache_data
def convert_table_to_csv(csv_df):
    """Convert reconstructed table DataFrame to CSV format"""
    return csv_df.to_csv(index=True).encode('utf-8')

# Helper functions for safe downloading
@st.cache_data
def convert_df_to_csv(df):
    """Convert DataFrame to CSV safely"""
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data  
def convert_df_to_json(df):
    """Convert DataFrame to JSON safely"""
    # Convert DataFrame to records and ensure all values are serializable
    records = df.to_dict('records')
    cleaned_records = []
    for record in records:
        cleaned_record = {}
        for key, value in record.items():
            cleaned_record[key] = safe_convert_value(value)
        cleaned_records.append(cleaned_record)
    
    return json.dumps(cleaned_records, indent=2)

def reconstruct_clean_table(extracted_cells):
    """Reconstruct table with only text content - 1 column per cell position"""
    if not extracted_cells:
        return pd.DataFrame(), pd.DataFrame()
    
    # Find table dimensions
    max_row = max(int(cell.get('start_row', 0)) for cell in extracted_cells if cell.get('start_row', '') != '') + 1
    max_col = max(int(cell.get('start_col', 0)) for cell in extracted_cells if cell.get('start_col', '') != '') + 1
    
    # Create simple column names
    columns = [f'Column_{col}' for col in range(max_col)]
    
    # Initialize empty dataframes
    clean_data = {col: [""] * max_row for col in columns}
    flag_data = {col: [False] * max_row for col in columns}  # For styling info
    
    # Fill the grid with only text content
    for cell in extracted_cells:
        try:
            row = int(cell.get('start_row', 0)) if cell.get('start_row', '') != '' else 0
            col = int(cell.get('start_col', 0)) if cell.get('start_col', '') != '' else 0
            
            text = str(cell.get('text', ''))
            flagged = bool(cell.get('flagged_for_review', False))
            
            # Only fill if there's actual content
            if text.strip():
                clean_data[f'Column_{col}'][row] = text
                flag_data[f'Column_{col}'][row] = flagged
            
        except (ValueError, TypeError) as e:
            # Handle any conversion errors gracefully
            continue
    
    # Convert to DataFrames
    clean_df = pd.DataFrame(clean_data)
    flag_df = pd.DataFrame(flag_data)
    
    # Set proper row names
    clean_df.index = [f'Row_{i}' for i in range(len(clean_df.index))]
    flag_df.index = [f'Row_{i}' for i in range(len(flag_df.index))]
    
    # Store flag information for styling
    clean_df._flag_info = flag_df
    
    return clean_df, clean_df.copy()  # Same for display and CSV

def apply_clean_table_styling(df):
    """Apply styling to the clean text table based on flags"""
    if not hasattr(df, '_flag_info'):
        return df
    
    flag_df = df._flag_info
    
    def highlight_clean_cells(val):
        """Color cells based on flag status for clean format"""
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        
        for row_idx in range(len(df.index)):
            for col_name in df.columns:
                try:
                    is_flagged = flag_df.loc[df.index[row_idx], col_name]
                    cell_value = df.loc[df.index[row_idx], col_name]
                    
                    if cell_value == "":  # Empty cell
                        styles.loc[df.index[row_idx], col_name] = ''
                    elif is_flagged:  # Flagged for review
                        styles.loc[df.index[row_idx], col_name] = 'background-color: #ffebee; border: 1px solid #f44336'
                    else:  # Good cell with content
                        styles.loc[df.index[row_idx], col_name] = 'background-color: #e8f5e8; border: 1px solid #4caf50'
                except (IndexError, TypeError, KeyError):
                    continue
        
        return styles
    
    return df.style.apply(highlight_clean_cells, axis=None)

def filter_clean_table(clean_df, show_flagged_only=False, show_text_only=False):
    """Filter the clean text table based on user preferences"""
    if not hasattr(clean_df, '_flag_info'):
        return clean_df, clean_df
    
    flag_df = clean_df._flag_info
    
    # Apply filters
    filtered_df = clean_df.copy()
    
    for row_idx in range(len(clean_df.index)):
        for col_name in clean_df.columns:
            try:
                has_text = clean_df.loc[clean_df.index[row_idx], col_name] != ""
                is_flagged = flag_df.loc[clean_df.index[row_idx], col_name]
                
                # Determine if this cell should be shown
                show_cell = True
                
                if show_text_only and not has_text:
                    show_cell = False
                
                if show_flagged_only and not is_flagged:
                    show_cell = False
                
                # If cell should not be shown, clear it
                if not show_cell:
                    filtered_df.loc[clean_df.index[row_idx], col_name] = ""
                            
            except (IndexError, KeyError):
                continue
    
    # Preserve flag info for styling
    filtered_df._flag_info = flag_df
    
    return filtered_df, filtered_df.copy()

def get_clean_table_statistics(clean_df):
    """Calculate statistics for the clean text table"""
    if not hasattr(clean_df, '_flag_info'):
        return 0, 0, 0
    
    flag_df = clean_df._flag_info
    
    total_cells = (clean_df != "").sum().sum()
    flagged_cells = ((clean_df != "") & flag_df).sum().sum()
    cells_with_text = total_cells  # Same as total_cells since we only count non-empty
    
    return total_cells, flagged_cells, cells_with_text

@st.cache_data
def convert_clean_table_to_csv(clean_df):
    """Convert clean text table DataFrame to CSV format"""
    return clean_df.to_csv(index=True).encode('utf-8')

def main():
    # Title and description
    st.title("📊 Table Extraction & Uncertainty Quantification")
    st.markdown("""
    Upload a table image to extract its contents with uncertainty quantification using **APS Conformal Prediction**. 
    The app identifies cells that may need human review based on real calibration data.
    """)
    
    # Load calibration data first
    cal_scores, metadata = load_calibration_data()
    
    if cal_scores is None:
        st.stop()
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Configuration")
    
    # Display calibration info
    st.sidebar.success(f"✅ Loaded {len(cal_scores)} calibration scores")
    if metadata:
        st.sidebar.info(f"📊 Processed {metadata.get('successful_extractions', 'N/A')} training images")
    
    # Model parameters
    alpha = st.sidebar.slider("Alpha (Uncertainty Level)", 0.01, 0.5, 0.1, 0.01)
    best_threshold = st.sidebar.slider("Review Threshold", 0.001, 0.1, 0.03, 0.001)
    
    st.sidebar.markdown("**Score Function:** APS Conformal")
    st.sidebar.caption("Uses real calibration data from your training images")
    
    # Calculate threshold from calibration data
    n = len(cal_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    threshold = np.quantile(cal_scores, q_level, method="higher")
    
    st.sidebar.markdown(f"**Calculated Threshold:** {threshold:.4f}")
    
    # Load models
    try:
        tsr_model, ocr_model = load_models()
        st.sidebar.success("✅ AI models loaded")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # File upload
    st.header("📁 Upload Table Image")
    uploaded_file = st.file_uploader(
        "Choose a table image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image containing a table for extraction"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded table image", use_column_width=True)
        
        with col2:
            st.subheader("📊 Processing Information")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Image Size:** {image.size}")
            st.write(f"**Calibration Samples:** {len(cal_scores)}")
            st.write(f"**Score Range:** {cal_scores.min():.3f} - {cal_scores.max():.3f}")
            st.write(f"**Alpha Level:** {alpha}")
            st.write(f"**Threshold:** {threshold:.4f}")
        
        # Process button
        if st.button("🚀 Extract Table Data", type="primary"):
            with st.spinner("Extracting table data using real calibration..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extract table cells
                    extracted_cells = extract_tsr_ocr_confidences(temp_path, tsr_model, ocr_model)
                    
                    if not extracted_cells:
                        st.warning("No table cells were detected in the image.")
                        return
                    
                    # Add uncertainty scores and clean data
                    extracted_cells = compute_uncertainty_scores(
                        extracted_cells, threshold, best_threshold
                    )
                    
                    # Store in session state
                    st.session_state.extracted_cells = extracted_cells
                    st.session_state.processed = True
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    
                    st.success(f"✅ Successfully extracted {len(extracted_cells)} table cells!")
                    
                except Exception as e:
                    st.error(f"Error during extraction: {str(e)}")
                    return
    
    # Display results if available
    if hasattr(st.session_state, 'processed') and st.session_state.processed:
        extracted_cells = st.session_state.extracted_cells
        
        # Filter options
        st.header("🔍 Display & Filter Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_clean_view = st.checkbox("📄 Clean text-only view", value=False, 
                                        help="Show only extracted text without confidence scores")
        with col2:
            show_flagged_only = st.checkbox("🚩 Show only flagged cells", value=False)
        with col3:
            show_text_only = st.checkbox("📝 Show only cells with text", value=True)
        
        # Reconstruct tables based on view mode
        if show_clean_view:
            display_df, csv_df = reconstruct_clean_table(extracted_cells)
            # Use clean table statistics and filtering
            total_cells, flagged_cells, cells_with_text = get_clean_table_statistics(display_df)
            avg_uncertainty = 0  # Not applicable for clean view
        else:
            display_df, csv_df = reconstruct_table_grid(extracted_cells)
            # Use detailed table statistics and filtering
            total_cells, flagged_cells, cells_with_text, avg_uncertainty = get_table_statistics(display_df)
        
        if display_df.empty:
            st.warning("No data to display.")
            return
        
        # Statistics
        st.header("📈 Extraction Statistics")
        if show_clean_view:
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Total Cells", total_cells)
            col2.metric("Flagged for Review", flagged_cells, f"{(flagged_cells/total_cells*100):.1f}%" if total_cells > 0 else "0%")
            col3.metric("Cells with Text", cells_with_text)
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Total Cells", total_cells)
            col2.metric("Flagged for Review", flagged_cells, f"{(flagged_cells/total_cells*100):.1f}%" if total_cells > 0 else "0%")
            col3.metric("Cells with Text", cells_with_text)
            col4.metric("Avg Uncertainty", f"{avg_uncertainty:.3f}")
        
        # Apply filters based on view mode
        if show_clean_view:
            filtered_display_df, filtered_csv_df = filter_clean_table(
                display_df, show_flagged_only, show_text_only
            )
        else:
            filtered_display_df, filtered_csv_df = filter_reconstructed_table(
                display_df, csv_df, show_flagged_only, show_text_only
            )
        
        # Display table
        table_title = "📋 Extracted Table Data (Clean Layout)" if show_clean_view else "📋 Extracted Table Data (Detailed Layout)"
        st.header(table_title)
        
        # Count visible cells after filtering
        if show_clean_view:
            visible_cells = (filtered_display_df != "").sum().sum()
        else:
            visible_cells = 0
            for col_idx in range(0, len(filtered_display_df.columns), 5):
                visible_cells += (filtered_display_df.iloc[:, col_idx] != "").sum()
        
        if visible_cells == 0:
            st.warning("No cells match the current filter criteria.")
        else:
            # Apply styling and display
            if show_clean_view:
                styled_table = apply_clean_table_styling(filtered_display_df)
            else:
                styled_table = apply_table_styling(filtered_display_df)
                
            st.dataframe(styled_table, use_container_width=True)
            
            st.info(f"Showing {visible_cells} of {total_cells} total cells")
            
            # Legend for colors
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🟢 **Green**: Good quality cells")
            with col2:
                st.markdown("🔴 **Red**: Flagged for review")
        
        # Show table format explanation
        with st.expander("📖 Table Format Guide"):
            if show_clean_view:
                st.markdown("""
                **Clean Text-Only Format:**
                - **Column_X**: Contains only the extracted text content
                - Empty cells are left blank
                - Colors indicate quality: 🟢 Good | 🔴 Needs Review
                
                **Perfect for:**
                - Copying to Excel or other applications
                - Clean data analysis without technical metadata
                - Professional presentations
                """)
            else:
                st.markdown("""
                **Detailed Format - Each table cell is represented by 5 columns:**
                - **Text_Col_X**: Extracted text content
                - **TSR_Col_X**: Table Structure Recognition confidence (0-1)
                - **OCR_Col_X**: Optical Character Recognition confidence (0-1)  
                - **Uncertainty_Col_X**: Uncertainty score (0-1)
                - **Flag_Col_X**: Quality flag (✅ Good or 🚩 Review)
                
                **Colors:**
                - 🟢 **Green background**: High confidence, good quality
                - 🔴 **Red background**: Flagged for manual review
                """)
        
        # Download options
        st.header("💾 Download Options")
        
        if show_clean_view:
            # Clean view downloads
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 Clean Text Format")
                st.caption("Pure text content without technical data")
                
                if visible_cells > 0:
                    # Clean CSV download - filtered
                    clean_csv_data = convert_clean_table_to_csv(filtered_csv_df)
                    st.download_button(
                        label="📥 Download Clean Table (Filtered)",
                        data=clean_csv_data,
                        file_name=f"clean_table_filtered_{uploaded_file.name[:-4] if uploaded_file else 'data'}.csv",
                        mime="text/csv",
                        help="Downloads only the visible cells as clean text"
                    )
                    
                    # Clean CSV download - all data
                    all_clean_csv_data = convert_clean_table_to_csv(csv_df)
                    st.download_button(
                        label="📥 Download Clean Table (All)",
                        data=all_clean_csv_data,
                        file_name=f"clean_table_all_{uploaded_file.name[:-4] if uploaded_file else 'data'}.csv",
                        mime="text/csv",
                        help="Downloads all extracted text content"
                    )
            
            with col2:
                st.subheader("📄 Technical Details")
                st.caption("Complete metadata (JSON format)")
                
                # JSON download - original detailed format
                json_data = convert_df_to_json(create_dataframe(extracted_cells))
                st.download_button(
                    label="📥 Download Detailed JSON",
                    data=json_data,
                    file_name=f"detailed_cells_{uploaded_file.name[:-4] if uploaded_file else 'data'}.json",
                    mime="application/json",
                    help="Downloads complete technical information"
                )
        
        else:
            # Detailed view downloads
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Detailed Format")
                st.caption("Table layout with confidence scores")
                
                if visible_cells > 0:
                    # Detailed CSV download - filtered
                    detailed_csv_data = convert_table_to_csv(filtered_csv_df)
                    st.download_button(
                        label="📥 Download Detailed Table (Filtered)",
                        data=detailed_csv_data,
                        file_name=f"detailed_table_filtered_{uploaded_file.name[:-4] if uploaded_file else 'data'}.csv",
                        mime="text/csv",
                        help="Downloads visible cells with all confidence scores"
                    )
                    
                    # Detailed CSV download - all data
                    all_detailed_csv_data = convert_table_to_csv(csv_df)
                    st.download_button(
                        label="📥 Download Detailed Table (All)",
                        data=all_detailed_csv_data,
                        file_name=f"detailed_table_all_{uploaded_file.name[:-4] if uploaded_file else 'data'}.csv",
                        mime="text/csv",
                        help="Downloads complete detailed table data"
                    )
            
            with col2:
                st.subheader("📄 Clean Text Option")
                st.caption("Pure text without technical data")
                
                # Generate clean version for download
                clean_display_df, clean_csv_df = reconstruct_clean_table(extracted_cells)
                clean_csv_data = convert_clean_table_to_csv(clean_csv_df)
                
                st.download_button(
                    label="📥 Download as Clean Text CSV",
                    data=clean_csv_data,
                    file_name=f"clean_text_{uploaded_file.name[:-4] if uploaded_file else 'data'}.csv",
                    mime="text/csv",
                    help="Downloads only extracted text content"
                )
                
                # JSON download
                json_data = convert_df_to_json(create_dataframe(extracted_cells))
                st.download_button(
                    label="📥 Download Complete JSON",
                    data=json_data,
                    file_name=f"complete_data_{uploaded_file.name[:-4] if uploaded_file else 'data'}.json",
                    mime="application/json",
                    help="Downloads complete technical metadata"
                )
    
    # Calibration data info
    if metadata:
        with st.expander("📊 Calibration Data Information"):
            st.json(metadata)

if __name__ == "__main__":
    main()