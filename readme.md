# GroundMotionStudio

A Python toolkit for visualizing, analyzing, and manipulating ground motion records from the PEER NGA Database.

![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)


## 🚀 Quick Start

### Download Standalone Executable (Windows)
**No Python installation required!**

📥 **[Download GroundMotionStudio.exe](https://github.com/alinadimm/GroundMotionStudio/raw/master/GroundMotionStudio.exe)**

Simply download and run the executable - all dependencies are included.

### Windows Security Warning (IMPORTANT)

Windows may show a security warning because this executable is not code-signed. This is **normal** for open-source tools. Here's how to run it safely:

### Method 1: Download Executable (Windows)

**Download:**
1. Go to **[Releases](https://github.com/alinadimm/GroundMotionStudio/releases/latest)**
2. Download `GroundMotionStudio.exe` from **Assets**

**If Windows blocks the executable, try one of these methods:**

#### Option A: Unblock the File (Recommended)
1. Right-click the downloaded `.exe` file → **Properties**
2. At the bottom of the **General** tab, look for **Security** section
3. If you see an **"Unblock"** checkbox, check it
4. Click **OK** to save changes
5. Double-click the `.exe` to run

**Note:** The "Unblock" option only appears if Windows marked the file as downloaded from the internet.

---

#### Option B: Run as Administrator
1. Right-click the `.exe` file → **Run as administrator**
2. If prompted by User Account Control (UAC), click **Yes**
3. If Windows SmartScreen appears:
   - Click **"More info"**
   - Click **"Run anyway"**

---

#### Option C: Change File Permissions (Advanced - Not Recommended)
⚠️ **Warning:** Only use this if Options A and B don't work. Be cautious when changing file permissions.

1. Right-click the `.exe` file → **Properties**
2. Go to the **Security** tab
3. Click **Edit**
4. Select your username or group
5. Check **"Allow"** for **"Read & execute"** permission
6. Click **OK** → **OK**
---

## Screenshots

### Main Interface
![GUI Interface](images/gui.png)
*Simple and intuitive interface for loading and analyzing ground motion records*

### Time History Plots
![Time Histories](images/time-histories.png)
*Acceleration, velocity, and displacement plots with rotation support*

### Component Compass
![Compass View](images/compass.png)
*Visualize original and rotated component orientations*

### Station Map
![Station Map](images/map.png)
*Interactive map with fault locations and manual angle measurement tool*

---
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

### Station Maps
- **Station location maps** are retrieved from [Jack W. Baker's website](https://www.jackwbaker.com/pulse_classification_v2/Maps/)
- Maps are © Jack W. Baker and are fetched dynamically when using the "Show Map" feature
- Map legend and fault information are included in the visualization
- **Attribution is automatically displayed** on map plots as "Map source: jackwbaker.com"

### Usage of Maps
When publishing results that include maps from this tool, please acknowledge:
> Station maps courtesy of Jack W. Baker (https://www.jackwbaker.com)
s
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