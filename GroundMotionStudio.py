import numpy as np
import re
import os
import glob

# Handle matplotlib backend before importing pyplot
import matplotlib
try:
    # Try to set TkAgg backend first
    matplotlib.use('TkAgg')
except ImportError:
    # If TkAgg is not available, use the default backend
    pass

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog



def numpy_cumtrapz(y, dx=1.0, initial=0):
    """
    NumPy-only replacement for scipy.integrate.cumtrapz
    
    Parameters:
    y (array): Input array to integrate
    dx (float): Spacing between samples
    initial (float): Initial value for integration
    
    Returns:
    array: Cumulative integral using trapezoidal rule
    """
    if initial is not None:
        return np.concatenate(([initial], np.cumsum((y[:-1] + y[1:]) / 2.0 * dx)))
    else:
        return np.cumsum((y[:-1] + y[1:]) / 2.0 * dx)

def rotate_components(h1_acc, h2_acc, rotation_angle):
    """
    Rotate two horizontal components by a given angle.
    
    Parameters:
    h1_acc (array): First horizontal component acceleration
    h2_acc (array): Second horizontal component acceleration
    rotation_angle (float): Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
    tuple: (rotated_h1, rotated_h2, new_h1_angle, new_h2_angle)
    """
    theta = np.deg2rad(rotation_angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Rotation matrix transformation
    h1_rotated = h1_acc * cos_theta - h2_acc * sin_theta
    h2_rotated = h1_acc * sin_theta + h2_acc * cos_theta
    
    return h1_rotated, h2_rotated
    
def extract_component_angle(filename):
    """
    Extract component angle from AT2 filename.
    
    Parameters:
    -----------
    filename : str
        The AT2 filename (without path)
    
    Returns:
    --------
    int or None : Angle in degrees, or None for vertical components
    """
    # Check for vertical components first
    vertical_keywords = ['DWN', 'UP', '-UP']
    for keyword in vertical_keywords:
        if keyword in filename.upper():
            return None
    
    # Look for the last 2-3 digit number immediately before .AT2
    # This regex captures exactly 2 or 3 digits at the end
    match = re.search(r'(\d{2,3})\.AT2$', filename, re.IGNORECASE)
    if match:
        angle_str = match.group(1)
        
        # If we got 3 digits, use them directly
        if len(angle_str) == 3:
            angle = int(angle_str)
        # If we got 2 digits, it's a 2-digit angle
        elif len(angle_str) == 2:
            angle = int(angle_str)
        else:
            return None
        
        # Validate it's a reasonable angle (0-360)
        if 0 <= angle <= 360:
            return angle
    
    return None


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
import numpy as np

def plot_rsn_map(rsn_number, crop_tol=245, compass_scale=0.35):
    """
    Display RSN map with cropped margins, legend, compass overlay,
    and interactive manual angle measurement using a directional arrow.

    Parameters
    ----------
    rsn_number : int
        RSN ID of the map to plot
    crop_tol : int, optional
        Brightness tolerance for white-space cropping (0-255)
    compass_scale : float, optional
        Fraction of map's smaller dimension used for compass radius
    """
    import requests, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg

    # URLs
    map_url = f"https://www.jackwbaker.com/pulse_classification_v2/Maps/{rsn_number}.jpg"
    legend_url = "https://www.jackwbaker.com/pulse_classification_v2/Maps/map_legend.png"

    def fetch_image(url, filename):
        try:
            r = requests.get(url, timeout=10)
        except requests.ConnectionError:
            raise RuntimeError("Internet connection not available.")
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(r.content)
            return mpimg.imread(filename)
        elif r.status_code == 404:
            raise RuntimeError(f"RSN {rsn_number} not available at source.")
        else:
            raise RuntimeError(f"Failed to retrieve image from: {url}")

    def crop_horizontal_white_space(img, tol=245):
        """Crop left/right margins where all pixels exceed tol brightness."""
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[..., :3]  # drop alpha
            gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            if gray.max() <= 1:
                gray *= 255
        else:
            gray = img.copy()
        col_mask = (gray < tol).any(axis=0)
        left_idx = np.argmax(col_mask)
        right_idx = len(col_mask) - np.argmax(col_mask[::-1])
        return img[:, left_idx:right_idx]

    def draw_transparent_compass(ax, center, radius, alpha=0.3):
        circle = plt.Circle(center, radius, color='black', fill=False, alpha=alpha)
        ax.add_patch(circle)
        for label, angle in [('N', 0), ('E', 90), ('S', 180), ('W', 270)]:
            rad = np.radians(angle)
            x = center[0] + radius * np.sin(rad)
            y = center[1] - radius * np.cos(rad)
            ax.text(x, y, label, fontsize=12, ha='center', va='center',
                    color='black', alpha=alpha, fontweight='bold')
        for deg in range(0, 360, 30):
            rad = np.radians(deg)
            x1 = center[0] + radius * np.sin(rad)
            y1 = center[1] - radius * np.cos(rad)
            x2 = center[0] + (radius * 0.9) * np.sin(rad)
            y2 = center[1] - (radius * 0.9) * np.cos(rad)
            ax.plot([x1, x2], [y1, y2], color='black', alpha=alpha)

    # Fetch images & crop
    img_map = fetch_image(map_url, 'map_temp.jpg')
    img_legend = fetch_image(legend_url, 'legend_temp.png')
    img_map = crop_horizontal_white_space(img_map, tol=crop_tol)
    img_legend = crop_horizontal_white_space(img_legend, tol=crop_tol)

    # Figure setup
    fig, (ax_map, ax_legend) = plt.subplots(
        2, 1, figsize=(6, 8),
        gridspec_kw={'height_ratios': [8, 1]}
    )

    # Map display
    ax_map.imshow(img_map)
    ax_map.axis('off')
    h_map, w_map = img_map.shape[:2]
    center = (w_map / 1.8, h_map / 2)
    radius = min(w_map, h_map) * compass_scale
    draw_transparent_compass(ax_map, center, radius, alpha=0.25)
    ax_map.text(
        0.99, -0.05,
        "Map source: jackwbaker.com",
        transform=ax_map.transAxes,
        fontsize=9, color='gray', ha='right'
    )
    ax_map.set_title(
        f"RSN {rsn_number}\nClick two points to measure direction from North",
        fontweight='bold'
    )

    ax_legend.imshow(img_legend)
    ax_legend.axis('off')

    # --- Interactive section ---
    manual_points = []
    manual_arrow = None
    manual_box = None
    manual_dots = []

    def onclick(event):
        nonlocal manual_points, manual_arrow, manual_box, manual_dots
        if event.inaxes != ax_map:
            return
        dot = ax_map.plot(event.xdata, event.ydata, 'ro')[0]
        manual_dots.append(dot)
        manual_points.append((event.ydata, event.xdata))
        fig.canvas.draw()
        if len(manual_points) == 2:
            (y1, x1), (y2, x2) = manual_points
            dy_north = -(y2 - y1)
            dx_east = x2 - x1
            angle_manual = (np.degrees(np.arctan2(dx_east, dy_north)) + 360) % 360
            # clean previous drawings
            if manual_arrow: manual_arrow.remove()
            if manual_box: manual_box.remove()
            for dot in manual_dots: dot.remove()
            manual_dots.clear()
            # draw directional arrow
            manual_arrow = ax_map.arrow(
                x1, y1, x2 - x1, y2 - y1,
                head_width=15, head_length=25,
                fc='blue', ec='blue', linewidth=2, alpha=0.85, length_includes_head=True
            )
            # display single angle
            manual_box = ax_map.text(
                0.10, 0.95,
                f"Direction: {angle_manual:.2f}° from North",
                transform=ax_map.transAxes,
                fontsize=12, color='blue',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='k'),
                verticalalignment='top'
            )
            fig.canvas.draw()
            manual_points = []

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()


def parse_at2_file(filename):
    """
    Parse AT2 file from PEER NGA database and extract header metadata and time/acceleration data.
    
    Parameters:
    filename (str): Name of the AT2 file
    
    Returns:
    dict: Dictionary containing:
          - 'dt': time step in seconds
          - 'time': numpy array of time values in seconds
          - 'acceleration': numpy array of acceleration values in g
          - 'npts': number of data points
          - 'event_name': earthquake event name
          - 'year': year of the event
          - 'station_name': recording station name
          - 'filename': base filename of the AT2 file
    """
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Initialize metadata variables
    event_name = "Unknown Event"
    year = ""
    station_name = "Unknown Station"
    
    # Extract metadata from line 2 (index 1)
    # Format: "Event Name, Date, Station Name, Direction"
    if len(lines) > 1:
        header_line = lines[1].strip()
        parts = header_line.split(',')
        
        if len(parts) >= 1:
            event_name = parts[0].strip()
        
        # Extract year from date (format: 8/6/1979)
        if len(parts) >= 2:
            date_str = parts[1].strip()
            if '/' in date_str:
                year = date_str.split('/')[-1]
        
        if len(parts) >= 3:
            station_name = parts[2].strip()
    
    # Find the line with NPTS and DT information
    dt = None
    npts = None
    data_start_line = None
    
    for i, line in enumerate(lines):
        # Look for the line containing NPTS and DT
        if 'NPTS' in line and 'DT' in line:
            # Extract DT value using regex
            dt_match = re.search(r'DT\s*=\s*([0-9.]+)', line)
            if dt_match:
                dt = float(dt_match.group(1))
            
            # Extract NPTS value using regex
            npts_match = re.search(r'NPTS\s*=\s*([0-9]+)', line)
            if npts_match:
                npts = int(npts_match.group(1))
            
            # Data starts from the next line
            data_start_line = i + 1
            break
    
    if dt is None or data_start_line is None:
        raise ValueError("Could not find DT value or data start line in the file")
    
    # Read acceleration data
    acceleration_data = []
    
    for line in lines[data_start_line:]:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Split the line by whitespace and convert to float
        values = line.strip().split()
        for value in values:
            try:
                acceleration_data.append(float(value))
            except ValueError:
                # Skip any non-numeric values
                continue
    
    # Convert to numpy array
    acceleration_array = np.array(acceleration_data)
    
    # Create time array
    time_array = np.arange(len(acceleration_array)) * dt
    
    # Verify the lengths match
    if len(time_array) != len(acceleration_array):
        print(f"Warning: Length mismatch - Time: {len(time_array)}, Acceleration: {len(acceleration_array)}")
    
    # If NPTS was found, verify it matches the actual data length
    if npts is not None and len(acceleration_array) != npts:
        print(f"Warning: Expected {npts} points but found {len(acceleration_array)} points")
    
    # Return dictionary with all data and metadata
    return {
        'dt': dt,
        'time': time_array,
        'acceleration': acceleration_array,
        'npts': len(acceleration_array),
        'event_name': event_name,
        'year': year,
        'station_name': station_name,
        'filename': os.path.basename(filename)
    }

def find_rsn_files(rsn_number, directory="."):
    """
    Find all AT2 files for a given RSN number.
    
    Parameters:
    rsn_number (str): RSN number (e.g., "77", "150")
    directory (str): Directory to search in
    
    Returns:
    dict: Dictionary with keys 'horizontal1', 'horizontal2', 'vertical' and file paths as values
    """
    
    # Pattern to match RSN files
    pattern = f"RSN{rsn_number}_*.AT2"
    files = glob.glob(os.path.join(directory, pattern))
    
    if not files:
        raise FileNotFoundError(f"No AT2 files found for RSN{rsn_number}")
    
    # Sort files to have consistent ordering
    files.sort()
    
    # Classify files into horizontal and vertical components
    horizontal_files = []
    vertical_file = None
    
    for file in files:
        filename = os.path.basename(file)
        # Check if it's a vertical component (contains UP, DWN, DOWN, VERT, etc.)
        if any(keyword in filename.upper() for keyword in ['UP', 'DWN', 'DOWN', 'VERT']):
            vertical_file = file
        else:
            horizontal_files.append(file)
    
    # Organize the files
    file_dict = {}
    
    if len(horizontal_files) >= 1:
        file_dict['horizontal1'] = horizontal_files[0]
    if len(horizontal_files) >= 2:
        file_dict['horizontal2'] = horizontal_files[1]
    if vertical_file:
        file_dict['vertical'] = vertical_file
    
    return file_dict

def calculate_kinematics(acc, dt):
    """
    Calculate velocity and displacement from acceleration using NumPy only.
    
    Parameters:
    acc (array): Acceleration in g
    dt (float): Time step in seconds
    
    Returns:
    tuple: (velocity in cm/s, displacement in cm)
    """
    g = 981  # cm/s^2 (acceleration due to gravity)
    vel = g * numpy_cumtrapz(acc, dx=dt, initial=0)  # Convert g to cm/s^2 and integrate
    dis = numpy_cumtrapz(vel, dx=dt, initial=0)      # Integrate velocity to get displacement
    return vel, dis

def plot_component_compass(rsn_number, directory="./records", save_plot=False, rotation_angle=None, rotation_mode='delta'):
    """
    Plot a compass showing the orientation of horizontal components.
    
    Parameters:
    rsn_number (str): RSN number (e.g., "77", "150")
    directory (str): Directory containing the AT2 files
    save_plot (bool): Whether to save the plot as an image file
    rotation_angle (float): Rotation angle in degrees (None for no rotation)
    rotation_mode (str): 'delta' for rotation amount, 'target' for target angle of H1
    """
    # Find all files for this RSN
    files = find_rsn_files(rsn_number, directory)
    
    # Extract metadata and angles
    event_name = "Unknown Event"
    year = ""
    station_name = "Unknown Station"
    
    h1_angle = None
    h2_angle = None
    
    # Get metadata and angles
    if 'horizontal1' in files:
        try:
            record_data = parse_at2_file(files['horizontal1'])
            event_name = record_data['event_name']
            year = record_data['year']
            station_name = record_data['station_name']
            h1_angle = extract_component_angle(record_data['filename'])
        except Exception as e:
            print(f"Error processing horizontal1: {e}")
    
    if 'horizontal2' in files:
        try:
            record_data = parse_at2_file(files['horizontal2'])
            if event_name == "Unknown Event":
                event_name = record_data['event_name']
                year = record_data['year']
                station_name = record_data['station_name']
            h2_angle = extract_component_angle(record_data['filename'])
        except Exception as e:
            print(f"Error processing horizontal2: {e}")
    
    if h1_angle is None and h2_angle is None:
        raise ValueError("No horizontal component angles found in the filenames")
    
    # Calculate rotated angles if rotation is specified
    h1_angle_rotated = None
    h2_angle_rotated = None
    actual_rotation = 0
    
    if rotation_angle is not None and h1_angle is not None:
        if rotation_mode == 'target':
            # Calculate how much to rotate to reach target angle
            actual_rotation = (rotation_angle - h1_angle + 360) % 360
            if actual_rotation > 180:
                actual_rotation -= 360
        else:  # delta mode
            actual_rotation = rotation_angle
        
        h1_angle_rotated = (h1_angle + actual_rotation + 360) % 360
        if h2_angle is not None:
            h2_angle_rotated = (h2_angle + actual_rotation + 360) % 360
    
    # Create the compass plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    
    # Set theta direction (clockwise from North)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Draw the compass circle and grid
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    
    # Set degree labels
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(['N (0°)', 'NE (45°)', 'E (90°)', 'SE (135°)', 
                        'S (180°)', 'SW (225°)', 'W (270°)', 'NW (315°)'])
    
    # Plot original components (pale if rotated)
    if h1_angle is not None:
        h1_rad = np.deg2rad(h1_angle)
        if rotation_angle is not None:
            # Pale original
            ax.plot([h1_rad, h1_rad], [0, 1], color='lightblue', linewidth=2, 
                   linestyle='--', alpha=0.5, label=f'Original H1: {h1_angle}°')
        else:
            # Normal original
            ax.plot([h1_rad, h1_rad], [0, 1], 'b-', linewidth=3, label=f'H1: {h1_angle}°')
            ax.text(h1_rad, 0.7, f'H1\n{h1_angle}°', ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='blue',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if h2_angle is not None:
        h2_rad = np.deg2rad(h2_angle)
        if rotation_angle is not None:
            # Pale original
            ax.plot([h2_rad, h2_rad], [0, 1], color='lightcoral', linewidth=2, 
                   linestyle='--', alpha=0.5, label=f'Original H2: {h2_angle}°')
        else:
            # Normal original
            ax.plot([h2_rad, h2_rad], [0, 1], 'r-', linewidth=3, label=f'H2: {h2_angle}°')
            ax.text(h2_rad, 0.5, f'H2\n{h2_angle}°', ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='red',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot rotated components if applicable
    if rotation_angle is not None and h1_angle_rotated is not None:
        h1_rot_rad = np.deg2rad(h1_angle_rotated)
        ax.plot([h1_rot_rad, h1_rot_rad], [0, 1], 'b-', linewidth=3, 
               label=f'Rotated H1: {h1_angle_rotated:.1f}°')
        ax.text(h1_rot_rad, 0.7, f'H1\n{h1_angle_rotated:.1f}°', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='blue',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if h2_angle_rotated is not None:
            h2_rot_rad = np.deg2rad(h2_angle_rotated)
            ax.plot([h2_rot_rad, h2_rot_rad], [0, 1], 'r-', linewidth=3, 
                   label=f'Rotated H2: {h2_angle_rotated:.1f}°')
            ax.text(h2_rot_rad, 0.5, f'H2\n{h2_angle_rotated:.1f}°', ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='red',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add title
    if rotation_angle is not None:
        if rotation_mode == 'target':
            title = f"{event_name} {year}, {station_name} RSN{rsn_number}\nRotated to H1={rotation_angle}° (rotation: {actual_rotation:.1f}°)"
        else:
            title = f"{event_name} {year}, {station_name} RSN{rsn_number}\nRotated by {actual_rotation:.1f}°"
    else:
        title = f"{event_name} {year}, {station_name} RSN{rsn_number}\nHorizontal Component Orientations"
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        safe_event = re.sub(r'[<>:"/\\|?*]', '_', event_name)
        safe_station = re.sub(r'[<>:"/\\|?*]', '_', station_name)
        if rotation_angle is not None:
            plot_filename = f"RSN{rsn_number}_{safe_event}_{year}_{safe_station}_compass_rotated.png"
        else:
            plot_filename = f"RSN{rsn_number}_{safe_event}_{year}_{safe_station}_compass.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Compass plot saved as {plot_filename}")
    
    plt.show()

def export_acceleration_data(rsn_number, directory="./records", output_directory=None, xstart=None, xend=None, rotation_angle=None, rotation_mode='delta'):
    """
    Export acceleration data from AT2 files to single-column text files.
    
    Parameters:
    rsn_number (str): RSN number (e.g., "77", "150")
    directory (str): Directory containing the AT2 files
    output_directory (str): Directory where to save the exported files
    xstart (float): Start time in seconds (None for beginning)
    xend (float): End time in seconds (None for full duration)
    rotation_angle (float): Rotation angle in degrees (None for no rotation)
    rotation_mode (str): 'delta' for rotation amount, 'target' for target angle of H1
    
    Returns:
    str: Path to the created output folder
    """
    
    # Find all files for this RSN
    files = find_rsn_files(rsn_number, directory)
    
    # Extract metadata from first available file
    metadata_extracted = False
    event_name = "Unknown Event"
    year = ""
    station_name = "Unknown Station"
    
    for comp_key in files.keys():
        if not metadata_extracted:
            try:
                record_data = parse_at2_file(files[comp_key])
                event_name = record_data['event_name']
                year = record_data['year']
                station_name = record_data['station_name']
                metadata_extracted = True
                break
            except:
                continue
    
    # Read horizontal components for rotation if needed
    h1_data = None
    h2_data = None
    h1_angle_orig = None
    h2_angle_orig = None
    actual_rotation = 0
    
    # Read horizontal components for rotation if needed
    h1_data = None
    h2_data = None
    h1_angle_orig = None
    h2_angle_orig = None
    actual_rotation = 0
    
    if rotation_angle is not None:
        if 'horizontal1' in files:
            h1_record = parse_at2_file(files['horizontal1'])
            h1_data = h1_record['acceleration']
            h1_angle_orig = extract_component_angle(h1_record['filename'])
    
        if 'horizontal2' in files:
            h2_record = parse_at2_file(files['horizontal2'])
            h2_data = h2_record['acceleration']
            h2_angle_orig = extract_component_angle(h2_record['filename'])
    
        # Calculate actual rotation
        if rotation_mode == 'target' and h1_angle_orig is not None:
            actual_rotation = (rotation_angle - h1_angle_orig + 360) % 360
            if actual_rotation > 180:
                actual_rotation -= 360
        else:
            actual_rotation = rotation_angle
    
        # **FIX: Pad arrays to same length before rotation**
        if h1_data is not None and h2_data is not None:
            len_h1 = len(h1_data)
            len_h2 = len(h2_data)
            
            if len_h1 != len_h2:
                max_length = max(len_h1, len_h2)
                print(f"Warning: H1 has {len_h1} points, H2 has {len_h2} points")
                print(f"         Padding shorter array with zeros to match {max_length} points")
                
                if len_h1 < max_length:
                    h1_data = np.pad(h1_data, (0, max_length - len_h1), 
                                   mode='constant', constant_values=0.0)
                if len_h2 < max_length:
                    h2_data = np.pad(h2_data, (0, max_length - len_h2), 
                                   mode='constant', constant_values=0.0)
            
            # Perform rotation with matched-length arrays
            h1_data, h2_data = rotate_components(h1_data, h2_data, actual_rotation)

    
    # Create output directory name
    clean_event_name = re.sub(r'[<>:"/\\|?*]', '_', event_name)
    clean_station_name = re.sub(r'[<>:"/\\|?*]', '_', station_name)
    
    if rotation_angle is not None:
        if rotation_mode == 'target':
            folder_name = f"RSN{rsn_number}_{clean_event_name}_{year}_{clean_station_name}_rotated_to_{rotation_angle}deg"
        else:
            folder_name = f"RSN{rsn_number}_{clean_event_name}_{year}_{clean_station_name}_rotated_{actual_rotation:.1f}deg"
    else:
        folder_name = f"RSN{rsn_number}_{clean_event_name}_{year}_{clean_station_name}"
    
    folder_name = folder_name.replace(' ', '_')
    
    # Set output directory
    if output_directory is None:
        output_directory = os.getcwd()
    
    output_folder_path = os.path.join(output_directory, folder_name)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Component mapping for file naming
    component_names = {
        'horizontal1': 'H1',
        'horizontal2': 'H2',
        'vertical': 'V'
    }
    
    exported_files = []
    
    # Process each component
    for comp_key, file_path in files.items():
        try:
            # Parse the file
            record_data = parse_at2_file(file_path)
            
            # Get full time array and acceleration
            time_full = record_data['time']
            dt = record_data['dt']
            
            # Use rotated data if available
            if comp_key == 'horizontal1' and h1_data is not None:
                acc_full = h1_data
            elif comp_key == 'horizontal2' and h2_data is not None:
                acc_full = h2_data
            else:
                acc_full = record_data['acceleration']
            
            # Determine actual time window
            actual_xstart = xstart if xstart is not None else 0
            actual_xend = xend if xend is not None else time_full[-1]
            
            # Find indices for the time window
            start_idx = np.argmin(np.abs(time_full - actual_xstart))
            end_idx = np.argmin(np.abs(time_full - actual_xend)) + 1
            
            # Extract the windowed data
            time_windowed = time_full[start_idx:end_idx]
            acc_windowed = acc_full[start_idx:end_idx]
            
            # Check if this is full duration or a window
            is_full_duration = (xstart is None and xend is None)
            
            # Extract angle from filename
            angle = extract_component_angle(record_data['filename'])
            
            # Calculate new angle if rotated
            if rotation_angle is not None and angle is not None:
                angle = (angle + actual_rotation + 360) % 360
            
            # Create output filename with angle
            comp_suffix = component_names.get(comp_key, comp_key)
            
            if angle is not None:
                # Horizontal component with angle
                output_filename = f"RSN{rsn_number}_{clean_event_name}_{year}_{clean_station_name}_{comp_suffix}_{angle:.1f}deg.txt"
            else:
                # Vertical component or no angle found
                output_filename = f"RSN{rsn_number}_{clean_event_name}_{year}_{clean_station_name}_{comp_suffix}.txt"
            
            output_file_path = os.path.join(output_folder_path, output_filename)
            
            # Write the data to file
            with open(output_file_path, 'w') as f:
                # Header line with record information and angle
                if angle is not None:
                    f.write(f"{event_name} {year}, {station_name} RSN{rsn_number} ({comp_suffix}, {angle:.1f}° from North)\n")
                else:
                    f.write(f"{event_name} {year}, {station_name} RSN{rsn_number} ({comp_suffix})\n")
                
                # Rotation information
                if rotation_angle is not None and comp_key != 'vertical':
                    if rotation_mode == 'target':
                        f.write(f"Rotated to H1={rotation_angle}° (rotation: {actual_rotation:.1f}°)\n")
                    else:
                        f.write(f"Rotated by {actual_rotation:.1f}°\n")
                else:
                    f.write('single-column acceleration data\n')
                
                f.write(f"dt={dt:.4f}s, npts={len(acc_windowed)}, duration={dt*len(acc_windowed):.3f} seconds\n")
                
                # Time window information
                if is_full_duration:
                    f.write(f"Time window: Full duration as downloaded from PEER ground motion database (0 to {time_full[-1]:.3f}s)\n")
                else:
                    f.write(f"Time window: {actual_xstart:.3f}s to {actual_xend:.3f}s (extracted from full record)\n")
                
                # Blank line
                f.write("\n")
                
                # Acceleration data (one value per line)
                for acc_value in acc_windowed:
                    f.write(f"{acc_value:.10f}\n")
            
            exported_files.append(output_file_path)
            
        except Exception as e:
            print(f"Error processing {comp_key}: {e}")
            continue
    
    if exported_files:
        rotation_info = ""
        if rotation_angle is not None:
            if rotation_mode == 'target':
                rotation_info = f" (rotated to H1={rotation_angle}°)"
            else:
                rotation_info = f" (rotated by {actual_rotation:.1f}°)"
        
        if is_full_duration:
            print(f"Exported {len(exported_files)} acceleration files (full duration){rotation_info} to: {output_folder_path}")
        else:
            print(f"Exported {len(exported_files)} acceleration files (time window {actual_xstart:.3f}s to {actual_xend:.3f}s){rotation_info} to: {output_folder_path}")
        return output_folder_path
    else:
        raise Exception("No files were successfully exported")


# def plot_rsn_record(rsn_number, xstart=0, xend=None, directory="./records", save_plot=False):
#     """
#     Plot acceleration, velocity, and displacement for all components of an RSN record.
    
#     Parameters:
#     rsn_number (str): RSN number (e.g., "77", "150")
#     directory (str): Directory containing the AT2 files
#     save_plot (bool): Whether to save the plot as an image file
#     """
        
#     # Find all files for this RSN
#     files = find_rsn_files(rsn_number, directory)
#     max_acc_h = 0
#     max_vel_h = 0
#     max_dis_h = 0
    
#     # Variables to store metadata (will be extracted from first available file)
#     event_name = "Unknown Event"
#     year = ""
#     station_name = "Unknown Station"
#     metadata_extracted = False
    
#     for comp_key in ['horizontal1', 'horizontal2']:
#         if comp_key in files:
#             try:
#                 record_data = parse_at2_file(files[comp_key])
#                 time = record_data['time']
#                 acc = record_data['acceleration']
#                 dt = record_data['dt']
                
#                 # Extract metadata from first available file
#                 if not metadata_extracted:
#                     event_name = record_data['event_name']
#                     year = record_data['year']
#                     station_name = record_data['station_name']
#                     metadata_extracted = True
                
#                 vel, dis = calculate_kinematics(acc, dt)
#                 # Track maximum absolute values
#                 max_acc_h = max(max_acc_h, np.max(np.abs(acc)))
#                 max_vel_h = max(max_vel_h, np.max(np.abs(vel)))
#                 max_dis_h = max(max_dis_h, np.max(np.abs(dis)))
                
                
#             except Exception as e:
#                 print(f"Warning: Could not process {comp_key} for max calculation: {e}")
    
#     vel_ylim = max_vel_h + 20
#     dis_ylim = max_dis_h + 15
    
#     # Set xend to full duration if not specified
#     if xend is None:
#         # Get time array from first available file to determine duration
#         for comp_key in files.keys():
#             try:
#                 record_data = parse_at2_file(files[comp_key])
#                 xend = record_data['time'][-1]
#                 break
#             except:
#                 continue
    
#     acc_ylim = 1.0
    
#     # Create figure with subplots (3 rows x 3 columns)
#     fig, axes = plt.subplots(3, 3, figsize=(15, 7))
    
#     # Format the title: "Event Name Year, Station Name RSN###"
#     title = f"{event_name} {year}, {station_name} RSN{rsn_number}"
#     fig.suptitle(title, fontsize=16, fontweight='bold')
    
#     # Build column titles with angles
#     col_titles = []
#     component_keys = ['horizontal1', 'horizontal2', 'vertical']
    
#     for comp_key in component_keys:
#         if comp_key in files:
#             try:
#                 angle = extract_component_angle(os.path.basename(files[comp_key]))
#                 if comp_key == 'vertical':
#                     col_titles.append('Vertical Component')
#                 elif angle is not None:
#                     comp_num = '1' if comp_key == 'horizontal1' else '2'
#                     col_titles.append(f'Horizontal Component {comp_num} ({angle}°)')
#                 else:
#                     comp_num = '1' if comp_key == 'horizontal1' else '2'
#                     col_titles.append(f'Horizontal Component {comp_num}')
#             except:
#                 # Fallback to default titles
#                 if comp_key == 'horizontal1':
#                     col_titles.append('Horizontal Component 1')
#                 elif comp_key == 'horizontal2':
#                     col_titles.append('Horizontal Component 2')
#                 else:
#                     col_titles.append('Vertical Component')
#         else:
#             # Add default title for missing component
#             if comp_key == 'horizontal1':
#                 col_titles.append('Horizontal Component 1')
#             elif comp_key == 'horizontal2':
#                 col_titles.append('Horizontal Component 2')
#             else:
#                 col_titles.append('Vertical Component')
    
#     row_titles = ['Acceleration (g)', 'Velocity (cm/s)', 'Displacement (cm)']
    
#     # Set column titles
#     for j, title_text in enumerate(col_titles):
#         axes[0, j].set_title(title_text, fontweight='bold')
    
#     # Set row titles
#     for i, title_text in enumerate(row_titles):
#         axes[i, 0].set_ylabel(title_text, fontweight='bold')
    
#     # Component keys in order
#     component_keys = ['horizontal1', 'horizontal2', 'vertical']

    
#     for j, comp_key in enumerate(component_keys):
#         if comp_key in files:
#             try:
#                 # Parse the file
#                 record_data = parse_at2_file(files[comp_key])
#                 time = record_data['time']
#                 acc = record_data['acceleration']
#                 dt = record_data['dt']
                
#                 vel, dis = calculate_kinematics(acc, dt)
                
#                 # Plot acceleration
#                 if comp_key == 'horizontal2':
#                     axes[0, j].plot(time, acc, 'r-', linewidth=1.2)
#                 elif comp_key == 'vertical':
#                     axes[0, j].plot(time, acc, 'c-', linewidth=1.2)
#                 else:
#                     axes[0, j].plot(time, acc, 'b-', linewidth=1.2)
#                 axes[0, j].grid(True, alpha=0.3)
#                 axes[0, j].set_xlim(xstart, xend)
#                 axes[0, j].set_ylim(-acc_ylim, acc_ylim)

#                 # Plot velocity
#                 axes[1, j].plot(time, vel, 'k-', linewidth=1.0)
#                 axes[1, j].grid(True, alpha=0.3)
#                 axes[1, j].set_xlim(xstart, xend)
#                 axes[1, j].set_ylim(-vel_ylim, vel_ylim)

#                 # Plot displacement
#                 axes[2, j].plot(time, dis, 'k-', linewidth=1.0)
#                 axes[2, j].grid(True, alpha=0.3)
#                 axes[2, j].set_xlim(xstart, xend)
#                 axes[2, j].set_ylim(-dis_ylim, dis_ylim)

#             except Exception as e:
#                 print(f"Warning: Could not process {comp_key}: {e}")
#                 for i in range(3):
#                     axes[i, j].axis('off')
#         else:
#             for i in range(3):
#                 axes[i, j].axis('off')

#     plt.tight_layout(rect=[0, 0, 1, 0.96])

#     if save_plot:
#         safe_event = re.sub(r'[<>:"/\\|?*]', '_', event_name)
#         safe_station = re.sub(r'[<>:"/\\|?*]', '_', station_name)
#         plot_filename = f"RSN{rsn_number}_{safe_event}_{year}_{safe_station}.png"
#         plt.savefig(plot_filename, dpi=300)
#         print(f"Plot saved as {plot_filename}")
#     plt.show()

def apply_rotation_to_record(rsn_number, rotation_angle, rotation_mode='delta', directory="./records"):
    """
    Apply rotation to horizontal components of an RSN record.

    Parameters:
    rsn_number (str): RSN number (e.g., "77", "150")
    rotation_angle (float): Rotation angle in degrees (None for no rotation)
    rotation_mode (str): 'delta' for rotation amount, 'target' for target angle of H1
    directory (str): Directory containing the AT2 files

    Returns:
    dict: Dictionary containing:
        - 'h1_original': Original H1 acceleration data (padded if needed)
        - 'h2_original': Original H2 acceleration data (padded if needed)
        - 'h1_rotated': Rotated H1 acceleration data
        - 'h2_rotated': Rotated H2 acceleration data
        - 'h1_angle_original': Original H1 angle
        - 'h2_angle_original': Original H2 angle
        - 'h1_angle_rotated': Rotated H1 angle
        - 'h2_angle_rotated': Rotated H2 angle
        - 'actual_rotation': Actual rotation applied in degrees
        - 'time': Time array (extended to match longest component)
        - 'dt': Time step
        - None if rotation cannot be applied
    """
    if rotation_angle is None:
        return None

    # Find all files for this RSN
    files = find_rsn_files(rsn_number, directory)

    # Check if both horizontal components exist
    if 'horizontal1' not in files or 'horizontal2' not in files:
        print("Warning: Both horizontal components required for rotation")
        return None

    try:
        # Parse horizontal component files
        h1_record = parse_at2_file(files['horizontal1'])
        h2_record = parse_at2_file(files['horizontal2'])

        # Extract original angles
        h1_angle_orig = extract_component_angle(h1_record['filename'])
        h2_angle_orig = extract_component_angle(h2_record['filename'])

        if h1_angle_orig is None:
            print("Warning: Could not extract H1 angle from filename")
            return None

        # Calculate actual rotation to apply
        if rotation_mode == 'target':
            # Calculate how much to rotate to reach target angle
            actual_rotation = (rotation_angle - h1_angle_orig + 360) % 360
            if actual_rotation > 180:
                actual_rotation -= 360
        else:  # delta mode
            actual_rotation = rotation_angle

        # Calculate rotated angles
        h1_angle_rotated = (h1_angle_orig + actual_rotation + 360) % 360
        h2_angle_rotated = (h2_angle_orig + actual_rotation + 360) % 360 if h2_angle_orig is not None else None

        # Get original acceleration data
        h1_acc_orig = h1_record['acceleration']
        h2_acc_orig = h2_record['acceleration']
        dt = h1_record['dt']

        # Check if arrays have different lengths
        len_h1 = len(h1_acc_orig)
        len_h2 = len(h2_acc_orig)

        if len_h1 != len_h2:
            max_length = max(len_h1, len_h2)
            print(f"Warning: H1 has {len_h1} points, H2 has {len_h2} points")
            print(f"         Padding shorter array with zeros to match {max_length} points")

            # Pad the shorter array with zeros
            if len_h1 < max_length:
                h1_acc_orig = np.pad(h1_acc_orig, (0, max_length - len_h1), 
                                    mode='constant', constant_values=0.0)
            if len_h2 < max_length:
                h2_acc_orig = np.pad(h2_acc_orig, (0, max_length - len_h2), 
                                    mode='constant', constant_values=0.0)

            # Extend time array to match the longer component
            time_array = np.arange(max_length) * dt
        else:
            time_array = h1_record['time']

        # Apply rotation
        h1_acc_rotated, h2_acc_rotated = rotate_components(h1_acc_orig, h2_acc_orig, actual_rotation)

        return {
            'h1_original': h1_acc_orig,
            'h2_original': h2_acc_orig,
            'h1_rotated': h1_acc_rotated,
            'h2_rotated': h2_acc_rotated,
            'h1_angle_original': h1_angle_orig,
            'h2_angle_original': h2_angle_orig,
            'h1_angle_rotated': h1_angle_rotated,
            'h2_angle_rotated': h2_angle_rotated,
            'actual_rotation': actual_rotation,
            'time': time_array,
            'dt': dt
        }

    except Exception as e:
        print(f"Error applying rotation: {e}")
        import traceback
        traceback.print_exc()
        return None




def plot_rsn_record(rsn_number, xstart=0, xend=None, directory="./records", save_plot=False,
                   rotation_angle=None, rotation_mode='delta'):
    """
    Plot acceleration, velocity, and displacement for all components of an RSN record.

    Parameters:
    rsn_number (str): RSN number (e.g., "77", "150")
    xstart (float): Start time for x-axis
    xend (float): End time for x-axis (None for full duration)
    directory (str): Directory containing the AT2 files
    save_plot (bool): Whether to save the plot as an image file
    rotation_angle (float): Rotation angle in degrees (None for no rotation)
    rotation_mode (str): 'delta' for rotation amount, 'target' for target angle of H1
    """

    # Apply rotation if requested
    rotation_data = None
    if rotation_angle is not None:
        rotation_data = apply_rotation_to_record(rsn_number, rotation_angle, rotation_mode, directory)
        if rotation_data is None:
            print("Warning: Could not apply rotation, proceeding without rotation")

    # Find all files for this RSN
    files = find_rsn_files(rsn_number, directory)
    max_acc_h = 0
    max_vel_h = 0
    max_dis_h = 0

    # Variables to store metadata (will be extracted from first available file)
    event_name = "Unknown Event"
    year = ""
    station_name = "Unknown Station"
    metadata_extracted = False

    # Calculate max values for scaling
    for comp_key in ['horizontal1', 'horizontal2']:
        if comp_key in files:
            try:
                # Use rotated data if available
                if rotation_data is not None:
                    if comp_key == 'horizontal1':
                        acc = rotation_data['h1_rotated']
                        dt = rotation_data['dt']
                    else:
                        acc = rotation_data['h2_rotated']
                        dt = rotation_data['dt']
                    
                    # Extract metadata from first file for reference
                    if not metadata_extracted:
                        record_data = parse_at2_file(files[comp_key])
                        event_name = record_data['event_name']
                        year = record_data['year']
                        station_name = record_data['station_name']
                        metadata_extracted = True
                else:
                    record_data = parse_at2_file(files[comp_key])
                    acc = record_data['acceleration']
                    dt = record_data['dt']
                    
                    # Extract metadata
                    if not metadata_extracted:
                        event_name = record_data['event_name']
                        year = record_data['year']
                        station_name = record_data['station_name']
                        metadata_extracted = True

                vel, dis = calculate_kinematics(acc, dt)

                # Track maximum absolute values
                max_acc_h = max(max_acc_h, np.max(np.abs(acc)))
                max_vel_h = max(max_vel_h, np.max(np.abs(vel)))
                max_dis_h = max(max_dis_h, np.max(np.abs(dis)))

            except Exception as e:
                print(f"Warning: Could not process {comp_key} for max calculation: {e}")

    vel_ylim = max_vel_h + 20
    dis_ylim = max_dis_h + 15

    # Set xend to full duration if not specified
    if xend is None:
        if rotation_data is not None:
            xend = rotation_data['time'][-1]
        else:
            for comp_key in files.keys():
                try:
                    record_data = parse_at2_file(files[comp_key])
                    xend = record_data['time'][-1]
                    break
                except:
                    continue

    acc_ylim = 1.0

    # Create figure with subplots (3 rows x 3 columns)
    fig, axes = plt.subplots(3, 3, figsize=(15, 7))

    # Add main title
    if rotation_data is not None:
        if rotation_mode == 'target':
            main_title = f"{event_name} {year}, {station_name} RSN{rsn_number} - Rotated to H1={rotation_angle}° (Δ={rotation_data['actual_rotation']:.1f}°)"
        else:
            main_title = f"{event_name} {year}, {station_name} RSN{rsn_number} - Rotated by {rotation_data['actual_rotation']:.1f}°"
    else:
        main_title = f"{event_name} {year}, {station_name} RSN{rsn_number}"
    
    fig.suptitle(main_title, fontsize=14, fontweight='bold')
    # Component keys in order
    component_keys = ['horizontal1', 'horizontal2', 'vertical']
    # Define column titles and row titles
    col_titles = []
    for comp_key in component_keys:
        if comp_key in files:
            try:
                angle = extract_component_angle(os.path.basename(files[comp_key]))
                if comp_key == 'vertical':
                    col_titles.append('Vertical Component')
                elif angle is not None:
                    comp_num = '1' if comp_key == 'horizontal1' else '2'
                    if rotation_data is not None:
                        col_titles.append(f'H{comp_num} (angle as recorded = {angle}°)')
                    else:
                        col_titles.append(f'H{comp_num} Component (angle as recorded = {angle}°)')
                else:
                    comp_num = '1' if comp_key == 'horizontal1' else '2'
                    col_titles.append(f'H{comp_num} Component')
            except:
                # Fallback to default titles
                if comp_key == 'horizontal1':
                    col_titles.append('Horizontal Component 1')
                elif comp_key == 'horizontal2':
                    col_titles.append('Horizontal Component 2')
                else:
                    col_titles.append('Vertical Component')
        else:
            # Add default title for missing component
            if comp_key == 'horizontal1':
                col_titles.append('Horizontal Component 1')
            elif comp_key == 'horizontal2':
                col_titles.append('Horizontal Component 2')
            else:
                col_titles.append('Vertical Component')
    row_titles = ['Acceleration (g)', 'Velocity (cm/s)', 'Displacement (cm)']

    # Set column titles
    for j, title_text in enumerate(col_titles):
        axes[0, j].set_title(title_text, fontweight='bold')

    # Set row titles
    for i, title_text in enumerate(row_titles):
        axes[i, 0].set_ylabel(title_text, fontweight='bold')



    for j, comp_key in enumerate(component_keys):
        if comp_key in files:
            try:
                # Determine if we should plot rotated data
                plot_rotated = False
                
                if rotation_data is not None and comp_key != 'vertical':
                    # **USE PADDED DATA FROM rotation_data**
                    plot_rotated = True
                    
                    if comp_key == 'horizontal1':
                        acc_original = rotation_data['h1_original']
                        acc_to_plot = rotation_data['h1_rotated']
                    else:  # horizontal2
                        acc_original = rotation_data['h2_original']
                        acc_to_plot = rotation_data['h2_rotated']
                    
                    time_original = rotation_data['time']
                    time_to_plot = rotation_data['time']
                    dt = rotation_data['dt']
                    
                    pga_orig = np.abs(acc_original).max()
                    pga_to_plot = np.abs(acc_to_plot).max()
                else:
                    # Parse the file normally for vertical or non-rotated case
                    record_data = parse_at2_file(files[comp_key])
                    time_to_plot = record_data['time']
                    acc_to_plot = record_data['acceleration']
                    dt = record_data['dt']
                    pga_to_plot = np.abs(acc_to_plot).max()

                # Calculate kinematics for both original and rotated (if applicable)
                if plot_rotated:
                    vel_original, dis_original = calculate_kinematics(acc_original, dt)
                    pgv_orig = np.abs(vel_original).max()
                    pgd_orig = np.abs(dis_original).max()
                vel_to_plot, dis_to_plot = calculate_kinematics(acc_to_plot, dt)
                
                # Calculate PGV for display (not in legend)
                pgv_to_plot = np.abs(vel_to_plot).max()
                pgd_to_plot = np.abs(dis_to_plot).max()
                # Define colors
                if comp_key == 'horizontal2':
                    bold_color = 'r-'
                    pale_color = 'grey'
                elif comp_key == 'vertical':
                    bold_color = 'c-'
                    pale_color = 'grey'
                else:  # horizontal1
                    bold_color = 'b-'
                    pale_color = 'grey'

                # Plot ACCELERATION
                if plot_rotated:
                    # Plot original in pale color first
                    axes[0, j].plot(time_original, acc_original, color=pale_color, linewidth=1.0,
                                   alpha=0.5, linestyle='--', label='Original')
                    # Plot rotated in bold color on top
                    axes[0, j].plot(time_to_plot, acc_to_plot, bold_color, linewidth=1.5, label='Rotated')
                    axes[0, j].legend(loc='upper right', fontsize=8)
                    axes[0, j].text(0.98, 0.05, f'PGA: Original={pga_orig:.3f} | Rotated={pga_to_plot:.3f}', 
                                    transform=axes[0, j].transAxes, fontsize=9, ha='right', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                else:
                    axes[0, j].plot(time_to_plot, acc_to_plot, bold_color, linewidth=1.2)
                    axes[0, j].text(0.98, 0.05, f'PGA={pga_to_plot:.3f}', 
                                    transform=axes[0, j].transAxes, fontsize=9, ha='right', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                axes[0, j].grid(True, alpha=0.3)
                axes[0, j].set_xlim(xstart, xend)
                axes[0, j].set_ylim(-acc_ylim, acc_ylim)
                # Add PGV values as text annotation in top-right corner

                # Plot VELOCITY
                if plot_rotated:
                    axes[1, j].plot(time_original, vel_original, color=pale_color, linewidth=1.0,
                                   alpha=0.5, linestyle='--', label='Original')
                    axes[1, j].plot(time_to_plot, vel_to_plot, 'k-', linewidth=1.2, label='Rotated')
                    axes[1, j].legend(loc='upper right', fontsize=8)
                    axes[1, j].text(0.98, 0.05, f'PGV: Original={pgv_orig:.1f} | Rotated={pgv_to_plot:.1f}', 
                                    transform=axes[1, j].transAxes, fontsize=9, ha='right', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                else:
                    axes[1, j].plot(time_to_plot, vel_to_plot, 'k-', linewidth=1.0)
                    
                    axes[1, j].text(0.98, 0.05, f'PGV={pgv_to_plot:.1f}', 
                                    transform=axes[1, j].transAxes, fontsize=9, ha='right', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                axes[1, j].grid(True, alpha=0.3)
                axes[1, j].set_xlim(xstart, xend)
                axes[1, j].set_ylim(-vel_ylim, vel_ylim)

                # Plot DISPLACEMENT
                if plot_rotated:
                    axes[2, j].plot(time_original, dis_original, color=pale_color, linewidth=1.0,
                                   alpha=0.5, linestyle='--', label='Original')
                    axes[2, j].plot(time_to_plot, dis_to_plot, 'k-', linewidth=1.2, label='Rotated')
                    axes[2, j].legend(loc='upper right', fontsize=8)
                    axes[2, j].text(0.98, 0.05, f'PGD: Original={pgd_orig:.1f} | Rotated={pgd_to_plot:.1f}', 
                                    transform=axes[2, j].transAxes, fontsize=9, ha='right', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                else:
                    axes[2, j].plot(time_to_plot, dis_to_plot, 'k-', linewidth=1.0)
                    axes[2, j].text(0.98, 0.05, f'PGD={pgd_to_plot:.1f}', 
                                    transform=axes[2, j].transAxes, fontsize=9, ha='right', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                axes[2, j].grid(True, alpha=0.3)
                axes[2, j].set_xlim(xstart, xend)
                axes[2, j].set_ylim(-dis_ylim, dis_ylim)

            except Exception as e:
                print(f"Warning: Could not process {comp_key}: {e}")
                import traceback
                traceback.print_exc()
                for i in range(3):
                    axes[i, j].axis('off')
        else:
            for i in range(3):
                axes[i, j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_plot:
        safe_event = re.sub(r'[<>:"/\\|?*]', '_', event_name)
        safe_station = re.sub(r'[<>:"/\\|?*]', '_', station_name)
        if rotation_data is not None:
            plot_filename = f"RSN{rsn_number}_{safe_event}_{year}_{safe_station}_rotated.png"
        else:
            plot_filename = f"RSN{rsn_number}_{safe_event}_{year}_{safe_station}.png"
        plt.savefig(plot_filename, dpi=300)
        print(f"Plot saved as {plot_filename}")
    plt.show()




class GroundMotionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ground Motion Studio")
        self.root.geometry("550x290")

        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # RSN input
        ttk.Label(main_frame, text="RSN Number:").grid(row=0, column=0, sticky="e", padx=(0, 5), pady=5)
        self.rsn_entry = ttk.Entry(main_frame, width=15)
        self.rsn_entry.grid(row=0, column=1, sticky="w", pady=5)

        # Directory input
        ttk.Label(main_frame, text="Records Directory:").grid(row=1, column=0, sticky="e", padx=(0, 5), pady=5)
        self.dir_entry = ttk.Entry(main_frame, width=40)
        self.dir_entry.grid(row=1, column=1, sticky="w", pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_directory).grid(row=1, column=2, padx=(5, 0), pady=5)

        # Time range inputs
        ttk.Label(main_frame, text="X Start (sec):").grid(row=2, column=0, sticky="e", padx=(0, 5), pady=5)
        self.xstart_entry = ttk.Entry(main_frame, width=15)
        self.xstart_entry.grid(row=2, column=1, sticky="w", pady=5)
        self.xstart_entry.insert(0, "0")

        ttk.Label(main_frame, text="X End (sec):").grid(row=3, column=0, sticky="e", padx=(0, 5), pady=5)
        self.xend_entry = ttk.Entry(main_frame, width=15)
        self.xend_entry.grid(row=3, column=1, sticky="w", pady=5)
        ttk.Label(main_frame, text="(leave empty for full duration)", font=('TkDefaultFont', 8, 'italic')).grid(row=3, column=2, sticky="w", padx=(5, 0))

        # Rotation angle inputs
        rotation_frame = ttk.LabelFrame(main_frame, text="Rotation (optional)", padding="5")
        rotation_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky="ew")
        
        ttk.Label(rotation_frame, text="Angle (deg):").grid(row=0, column=0, sticky="e", padx=(0, 5))
        self.rotation_entry = ttk.Entry(rotation_frame, width=15)
        self.rotation_entry.grid(row=0, column=1, sticky="w")
        
        ttk.Label(rotation_frame, text="Mode:").grid(row=0, column=2, sticky="e", padx=(10, 5))
        self.rotation_mode = ttk.Combobox(rotation_frame, values=["Delta (rotate by)", "Target (H1 to)"], width=18, state="readonly")
        self.rotation_mode.grid(row=0, column=3, sticky="w")
        self.rotation_mode.set("Delta (rotate by)")
        
        ttk.Label(rotation_frame, text="(leave angle empty for no rotation)", font=('TkDefaultFont', 8, 'italic')).grid(row=1, column=0, columnspan=4, pady=(5,0))

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="Plot Time Histories", command=self.plot_record).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Export Acceleration", command=self.export_acceleration).pack(side=tk.LEFT, padx=(5, 5))
        ttk.Button(button_frame, text="Show Compass", command=self.show_compass).pack(side=tk.LEFT, padx=(5, 5))
        ttk.Button(button_frame, text="Show Map", command=self.show_map).pack(side=tk.LEFT, padx=(5, 0))

        # Set default directory
        self.dir_entry.insert(0, "./records")

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, directory)
    
    def get_rotation_params(self):
        """Get rotation angle and mode from GUI inputs"""
        rotation_angle = None
        rotation_mode = 'delta'
        
        rotation_str = self.rotation_entry.get().strip()
        if rotation_str:
            try:
                rotation_angle = float(rotation_str)
                mode_str = self.rotation_mode.get()
                rotation_mode = 'target' if 'Target' in mode_str else 'delta'
            except ValueError:
                messagebox.showwarning("Warning", "Invalid rotation angle, ignoring rotation.")
                rotation_angle = None
        
        return rotation_angle, rotation_mode

    def plot_record(self):
        rsn = self.rsn_entry.get().strip()
        directory = self.dir_entry.get().strip() or "./records"
        
        try:
            xstart = float(self.xstart_entry.get()) if self.xstart_entry.get() else 0
        except ValueError:
            xstart = 0
            
        try:
            xend = float(self.xend_entry.get()) if self.xend_entry.get() else None
        except ValueError:
            xend = None
        
        # Get rotation parameters
        rotation_angle, rotation_mode = self.get_rotation_params()
    
        if not rsn:
            messagebox.showerror("Error", "Please enter an RSN number.")
            return
    
        try:
            # Pass rotation parameters to the plotting function
            plot_rsn_record(rsn, xstart=xstart, xend=xend, directory=directory,
                           rotation_angle=rotation_angle, rotation_mode=rotation_mode)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot record:\n{str(e)}")


    def export_acceleration(self):
        rsn = self.rsn_entry.get().strip()
        directory = self.dir_entry.get().strip() or "./records"
        
        # Get time window values
        try:
            xstart = float(self.xstart_entry.get()) if self.xstart_entry.get() else None
        except ValueError:
            xstart = None
            
        try:
            xend = float(self.xend_entry.get()) if self.xend_entry.get() else None
        except ValueError:
            xend = None
        
        # Get rotation parameters
        rotation_angle, rotation_mode = self.get_rotation_params()
        
        if not rsn:
            messagebox.showerror("Error", "Please enter an RSN number.")
            return

        try:
            folder_path = export_acceleration_data(rsn, directory=directory, xstart=xstart, xend=xend, 
                                                   rotation_angle=rotation_angle, rotation_mode=rotation_mode)
            
            # Create more informative message
            rotation_info = ""
            if rotation_angle is not None:
                if rotation_mode == 'target':
                    rotation_info = f", rotated to H1={rotation_angle}°"
                else:
                    rotation_info = f", rotated by {rotation_angle}°"
            
            if xstart is None and xend is None:
                msg = f"Acceleration data (full duration{rotation_info}) exported to:\n{folder_path}"
            else:
                time_info = f"{xstart if xstart is not None else 0}s to {xend if xend is not None else 'end'}s"
                msg = f"Acceleration data (time window: {time_info}{rotation_info}) exported to:\n{folder_path}"
            
            messagebox.showinfo("Export Complete", msg)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export acceleration data:\n{str(e)}")

    def show_compass(self):
        rsn = self.rsn_entry.get().strip()
        directory = self.dir_entry.get().strip() or "./records"
        
        # Get rotation parameters
        rotation_angle, rotation_mode = self.get_rotation_params()
        
        if not rsn:
            messagebox.showerror("Error", "Please enter an RSN number.")
            return

        try:
            plot_component_compass(rsn, directory=directory, rotation_angle=rotation_angle, rotation_mode=rotation_mode)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show compass:\n{str(e)}")

    def show_map(self):
        rsn = self.rsn_entry.get().strip()
        if not rsn:
            messagebox.showerror("Error", "Please enter an RSN number.")
            return
        try:
            rsn_int = int(rsn)
            plot_rsn_map(rsn_int, crop_tol=245, compass_scale=0.35)
        except ValueError:
            messagebox.showerror("Error", "RSN must be a number.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show RSN map:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GroundMotionGUI(root)
    root.mainloop()
