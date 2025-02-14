# kim-lab-tools

## Overview

This repository contains various python-based tools that represent modular refactors of pre-existing code or new code for a variety of tasks.

## Installation

```bash
pip install kim-lab-tools
```

## Usage

### ROI Area Analysis

```bash
python scripts/roi_area_analysis.py /path/to/roi/directory # using default settings
python scripts/roi_area_analysis.py /path/to/roi/directory --max-workers 4 # using 4 worker threads
python scripts/roi_area_analysis.py /path/to/roi/directory --use-gpu # using GPU if available
python scripts/roi_area_analysis.py /path/to/roi/directory --output-dir /path/to/output/directory # save output to a directory
```
