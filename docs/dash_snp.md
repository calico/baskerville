# 🧬 SNP Dashboard Launch Guide

## Quick Start

The SNP Scores Visualization Dashboard is a Streamlit web application that provides interactive visualization of SNP scores with quantile normalization capabilities.

### Prerequisites

Make sure you have the required packages installed:

```bash
pip install streamlit plotly
```

### Launch Methods

#### 1. **Recommended: Launch with scores directory**

```bash
# Launch with your scores directory
dash_snp.py -m /path/scores/directory
```

#### 2. **Launch using streamlit command**

```bash
# Using streamlit run (opens automatically in browser)
streamlit run dash_snp.py -- -s /path/scores/directory

# Without specifying directory (enter path in the app)
streamlit run dash_snp.py
```

### Expected Scores Directory Structure

The dashboard expects a specific directory structure created by `hound_snp_folds`:

```
scores_directory/
├── targets.txt              # Tab-separated target information
├── scores.h5                # HDF5 file with SNP scores
```

If you're working with an ensemble of models, point to the ensemble average scores directory.

### Dashboard Features

Once launched, the dashboard provides:

- **📊 Interactive Table**: SNP-target pairs with sortable columns
- **🎯 Score Types**: Raw scores (logSUM, logD2, etc)
- **📈 Quantile Scores**: Normalized 0-1 scale for cross-comparison
- **🔍 Filtering**: By target groups and individual SNPs
- **📱 Display Modes**: Raw scores, quantiles, or both
- **📊 Visualizations**: Score distributions and top SNPs
- **💾 Export**: Download filtered results as CSV
