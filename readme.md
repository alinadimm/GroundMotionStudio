# GroundMotionStudio

A comprehensive Python toolkit for visualizing, analyzing, and manipulating ground motion records from the PEER NGA Database.

![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

### 📊 Time History Visualization
- Plot acceleration, velocity, and displacement time histories
- Support for all three components (H1, H2, Vertical)
- Customizable time windows
- High-quality export to PNG

### 🔄 Component Rotation
- Rotate horizontal components to any orientation
- Two rotation modes:
  - **Delta mode**: Rotate by a specified angle
  - **Target mode**: Rotate H1 to a target orientation
- Visualize both original (pale) and rotated (bold) components
- Automatic handling of different array lengths through zero-padding

### 🧭 Orientation Tools
- Interactive compass visualization showing component orientations
- RSN station map display with interactive angle measurement
- Automatic extraction of component angles from filenames

### 💾 Data Export
- Export acceleration data to text files
- Support for time-windowed data extraction
- Automatic documentation of rotation parameters
- Organized output with metadata headers

### 🖥️ User-Friendly GUI
- Intuitive Tkinter-based interface
- Browse and select record directories
- Real-time parameter adjustment
- Error handling and user feedback

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Packages
```bash
pip install numpy matplotlib requests pillow
```

### Dependencies
- **numpy**: Numerical computations and array operations
- **matplotlib**: Plotting and visualization
- **tkinter**: GUI framework (included with Python)
- **requests**: HTTP requests for map data
- **PIL (Pillow)**: Image processing

### Supported Data
- PEER NGA Database AT2 format
- Acceleration in units of g
- Time step must be consistent within each file