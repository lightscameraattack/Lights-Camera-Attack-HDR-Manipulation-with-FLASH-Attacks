# Modular HDR Imaging Pipeline

This project provides a comprehensive High Dynamic Range (HDR) imaging pipeline capable of processing bracketed exposure images into a single high-quality output. The system is designed with a modular architecture, allowing users to select and combine different algorithms for image alignment, merging, and tone mapping. A graphical user interface (GUI) is included to facilitate interactive processing and real-time visualization of results.

## Features

*   **Modular Architecture**: Users can independently select algorithms for each stage of the HDR pipeline.
*   **Alignment Methods**: Includes Median Threshold Bitmap (MTB) for fast alignment and Homography-based alignment for complex transformations.
*   **Merging Algorithms**: Supports Robertson and Debevec methods for recovering camera response functions, as well as Mertens exposure fusion which does not require exposure times.
*   **Tone Mapping**: Offers multiple operators including Mantiuk, Reinhard, Drago, and a custom local tone mapping implementation.
*   **Single Image Processing**: innovative auto-bracketing feature allows HDR-like processing from a single input image.
*   **Graphical User Interface**: A user-friendly interface for loading images, adjusting parameters, and viewing results immediately.

## Prerequisites

*   Python 3.6 or higher
*   pip (Python package installer)

## Installation

1.  Open your terminal or command prompt.
2.  Navigate to the project directory.
3.  Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Graphical User Interface

The primary way to use this application is through the GUI. To launch the application, execute the following command:

```bash
python3 hdr_gui.py
```

### Application Workflow

1.  **Import Images**: Click the "Import Images" button to select your source files. The application supports standard image formats such as JPEG and PNG.
2.  **Configuration**:
    *   **Source Images**: View your loaded images in the list.
    *   **Exposure Settings**: For standard HDR, ensure exposure times are correct. For single images, the "Auto Bracket" mode will engage automatically.
    *   **Pipeline Configuration**: Select your desired methods for Alignment, Merging, and Tone Mapping using the radio buttons.
3.  **Processing**: The application processes the image automatically when settings are changed. The result is displayed in the main preview window.
4.  **Saving**: Click "Save Result" to export the final processed image to your computer.

## Project Structure

*   `hdr_gui.py`: The main entry point for the graphical user interface.
*   `hdr_pipeline_organized.py`: The core logic containing the implementation of HDR algorithms and file handling.
*   `hdr_pipeline_gui.py`: A wrapper script connecting the core pipeline to the GUI.
*   `tests/`: A directory containing default test images used for demonstration.

## License

This software is provided for educational and research purposes.
