# TSR-OCR-UQ

# Table Extraction and UQ - Complete Setup Guide
- 📺 [Watch demo video on YouTube](https://www.youtube.com/watch?v=zucvJlah-5U)

## 📋 Step-by-Step Instructions

### Step 1: Create Project Folder

1. Create a new folder called `table_extraction_and_uq` inside the cloned repository.
2. Navigate to this folder in Command Prompt:

```cmd
mkdir table_extraction_and_uq
cd table_extraction_and_uq
```

### Step 2: Copy All Files Into Appropriate Folders

You should now have these files in your `table_extraction_and_uq` folder:

```
table_extraction_and_uq/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.bat (windows)
├── validate.bat (Windows)
├── compute_calibration.bat (Windows)
├── run_app.bat (Windows)
├── stop_app.bat (Windows)
├── status.bat (Windows)
├── STEP_BY_STEP_SETUP.md          ← This file
├── src/
│   ├── compute_calibration_data.py
│   ├── streamlit_app.py
│   ├── validate_setup.py
│   ├── tsr_ocr.py                
│   ├── utils.py                   
│   └── score_functions.py         
├── data/                          ← Will be created by setup.bat/setup.sh
└── models_cache/                  ← Will be created by setup.bat/setup.sh
```

### Step 3: Copy Your Data

**Copy your data files:**
- Ensure `domains_with_thresholds.json` is inside the `data/` folder 

### Step 4: Run Initial Setup

```cmd
setup.bat (Windows) OR ./setup.sh (Mac/Linux)
```

This will:
- Check Docker installation
- Create `data/` and `models_cache/` directories
- Create domain image folders
- Build the Docker image (takes 5-10 minutes)

### Step 5: Add Your Data

**Add your images to the correct folders:**
```cmd
# Copy your images to these folders:
data/input_images/MatSci/images/     ← All MatSci images here
data/input_images/Biology/images/   ← All Biology images here
data/input_images/CompSci/images/   ← All CompSci images here
data/input_images/ICDAR/images/     ← All ICDAR images here
```


### Step 6: Validate Setup (Optional)

```cmd
validate.bat (Windows) OR ./validate.sh (Mac/Linux)
```

This will:
- Check all required folders exist
- Count images in each domain
- Calculate available calibration images
- Provide recommendations

### Step 7: Compute Calibration Data

```cmd
compute_calibration.bat (Windows) OR ./compute_calibration.sh (Mac/Linux)
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

### Step 8: Run Streamlit App for User Interface Table Data Extraction

```cmd
run_app.bat (Windows) OR ./run_app.sh (Mac/Linux)
```

This will:
- Check calibration data exists
- Start the web application at http://localhost:8501
- Show real calibration statistics

### Step 9: Use the Application

1. **Upload Image**: Drag & drop a table image
2. **View Calibration Info**: Check sidebar for real calibration data statistics
3. **Adjust Parameters**: 
   - Alpha (uncertainty level): 0.01 - 0.5
   - Review threshold: 0.001 - 0.1
4. **Extract**: Click "Extract Table Data"
5. **Review**: Check flagged cells (highlighted in red)
6. **Download**: Save results as JSON or CSV

### Step 10: Stop When Done

```cmd
stop_app.bat (Windows) OR ./stop_app.sh (Mac/Linux)
```

## Extract Many Table Images Data
- Ensure you move all table images into the `data/' folder inside `table_extraction_and_uq` folder
- Create a directory to save the extracted images e.g. `output` inside `table_extraction_and_uq` folder
- Next run:
  ```cmd
  docker exec table_extraction_uq_app python3 /app/src/tsr_ocr.py --image_dir app/data/path/to/image/folder --out_dir /path/to/save/extracted/JSON/data
  ```
- To extract a single table image data, run:
  ```cmd
  docker exec table_extraction_uq_app python3 /app/src/tsr_ocr.py --image_path /path/to/table_image.png --out_dir /path/to/save/extracted/JSON/data
  ```

## 🔧 Troubleshooting

### Check Project Status
```cmd
status.bat (Windows) OR ./status.sh (Mac/Linux)
```

### Common Issues

**"Docker not running":**
- Start Docker Desktop
- Wait for whale icon in system tray

**"Missing src files":**
- Copy `tsr_ocr.py`, `utils.py`, `score_functions.py`, and all python files to `src/` folder

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
- [ ] Shows "Loaded X calibration scores"
- [ ] Can upload and process images
- [ ] Can download results


## 📝 Notes

- **First run**: Takes longer due to model downloads (stored in `models_cache/`)
- **Data persistence**: All data saved in `data/` folder
- **Container**: Automatically starts/stops as needed
- **Real calibration**: No more random/dummy data!

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{ajayi-2024,
  title={Uncertainty-Aware Complex Scientific Table Data
Extraction},
  author={Ajayi, Kehinde and He, Yi and Wu, Jian},
  journal={Journal Name},
  year={2024},
  volume={XX},
  number={Y},
  pages={1--15},
  doi={10.xxxx/xxxxx}
}
```

**Paper Link**: [https://doi.org/10.xxxx/xxxxx](https://doi.org/10.xxxx/xxxxx) (coming soon)
