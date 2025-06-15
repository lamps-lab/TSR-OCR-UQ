# Table Extraction and UQ - Complete Setup Guide

## 📋 Step-by-Step Instructions

### Step 1: Create Project Folder

1. Create a new folder called `table_extraction_and_uq`
2. Navigate to this folder in Command Prompt

```cmd
mkdir table_extraction_and_uq
cd table_extraction_and_uq
```

### Step 2: Copy All Files

You should now have these files in your `table_extraction_and_uq` folder:

```
table_extraction_and_uq/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.bat
├── validate.bat
├── compute_calibration.bat
├── run_app.bat
├── stop_app.bat
├── status.bat
├── STEP_BY_STEP_SETUP.md          ← This file
├── src/
│   ├── compute_calibration_data.py
│   ├── streamlit_app.py
│   ├── validate_setup.py
│   ├── tsr_ocr.py                 ← Copy from your existing project
│   ├── utils.py                   ← Copy from your existing project
│   └── score_functions.py         ← Copy from your existing project
├── data/                          ← Will be created by setup.bat
└── models_cache/                  ← Will be created by setup.bat
```

### Step 3: Copy Your Data

**Copy your data files:**
- Copy `domains_test_images.json` to the main project folder (it will be moved to `data/` by setup.bat)
- Copy all your image folders to a temporary location (you'll move them after setup)

### Step 4: Run Initial Setup

```cmd
setup.bat
```

This will:
- Check Docker installation
- Create `data/` and `models_cache/` directories
- Create domain image folders
- Build the Docker image (takes 5-10 minutes)

### Step 6: Add Your Data

**Add your images to the correct folders:**
```cmd
# Copy your images to these folders:
data/input_images/MatSci/images/     ← All MatSci images here
data/input_images/Biology/images/   ← All Biology images here
data/input_images/CompSci/images/   ← All CompSci images here
data/input_images/ICDAR/images/     ← All ICDAR images here
```

**Add your test images JSON:**
```cmd
# Move domains_test_images.json to: data/
```

### Step 7: Validate Setup (Optional)

```cmd
validate.bat
```

This will:
- Check all required folders exist
- Count images in each domain
- Calculate available calibration images
- Provide recommendations

### Step 8: Compute Calibration Data

```cmd
compute_calibration.bat
```

This will:
- Run validation first
- Ask for confirmation
- Process all training images (excludes test images)
- Compute APS conformal scores
- Save calibration data (takes 10-30 minutes)

**Expected output:**
- `data/calibration_data/calibration_scores_aps.npy`
- `data/calibration_data/calibration_metadata.json`

### Step 9: Run Streamlit App

```cmd
run_app.bat
```

This will:
- Check calibration data exists
- Start the web application at http://localhost:8501
- Show real calibration statistics

### Step 10: Use the Application

1. **Upload Image**: Drag & drop a table image
2. **View Calibration Info**: Check sidebar for real calibration data statistics
3. **Adjust Parameters**: 
   - Alpha (uncertainty level): 0.01 - 0.5
   - Review threshold: 0.001 - 0.1
4. **Extract**: Click "Extract Table Data"
5. **Review**: Check flagged cells (highlighted in red)
6. **Download**: Save results as JSON or CSV

### Step 11: Stop When Done

```cmd
stop_app.bat
```

## 🔧 Troubleshooting

### Check Project Status
```cmd
status.bat
```

### Common Issues

**"Docker not running":**
- Start Docker Desktop
- Wait for whale icon in system tray

**"Missing src files":**
- Copy `tsr_ocr.py`, `utils.py`, `score_functions.py` to `src/` folder

**"No calibration data":**
- Ensure `domains_with_thresholds.json` is in `data/` folder
- Check that images exist in `data/input_images/[domain]/images/` folders
- Run `compute_calibration.bat` again

**"Calibration computation failed":**
- Check Docker has sufficient memory (8GB+ recommended)
- Ensure internet connection for model downloads
- Try processing fewer images first

**"App shows random data":**
- Ensure calibration computation completed successfully
- Check that `data/calibration_data/calibration_scores_aps.npy` exists

## 📊 Expected Results

### Calibration Data:
- **Processing time**: 10-30 minutes
- **Images processed**: 200-600 (excluding test images)
- **App shows**: "Loaded X calibration scores" (not random data)

### Streamlit App Features:
- **Real calibration**: Uses your actual training data
- **APS conformal**: Matches your experimental setup
- **Uncertainty quantification**: Flags cells needing review
- **Export options**: JSON and CSV downloads

## 🎯 Key Benefits

✅ **Real Calibration Data**: Uses your actual training images  
✅ **APS Conformal Prediction**: Matches your experimental setup  
✅ **Professional Interface**: Web app for easy use  
✅ **Uncertainty Quantification**: Flags cells needing review  
✅ **Export Options**: JSON and CSV downloads  
✅ **Production Ready**: Containerized and scalable  

## 🚀 Success Checklist

- [ ] Project folder created with all files
- [ ] Docker Desktop running
- [ ] `setup.bat` completed successfully
- [ ] Your Python scripts copied to `src/` folder
- [ ] Images added to domain folders
- [ ] `domains_with_thresholds.json` in `data/` folder
- [ ] `validate.bat` shows setup is ready
- [ ] `compute_calibration.bat` completed
- [ ] Calibration files created in `data/calibration_data/`
- [ ] `run_app.bat` starts successfully
- [ ] App loads at http://localhost:8501
- [ ] Shows "Loaded X calibration scores" (not "using fallback")
- [ ] Can upload and process images
- [ ] Can download results

When all items are checked, your system is ready for production use! 🎉

## 📝 Notes

- **First run**: Takes longer due to model downloads (stored in `models_cache/`)
- **Data persistence**: All data saved in `data/` folder
- **Container**: Automatically starts/stops as needed
- **Real calibration**: No more random/dummy data!