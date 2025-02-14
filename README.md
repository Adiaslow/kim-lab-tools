# Kim Lab Tools

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
![pytest](https://github.com/Adiaslow/kim-lab-tools/actions/workflows/pytest.yml/badge.svg)
[![Pylint](https://github.com/Adiaslow/kim-lab-tools/actions/workflows/pylint.yml/badge.svg)](https://github.com/Adiaslow/kim-lab-tools/actions/workflows/pylint.yml)
[![GitHub last commit](https://img.shields.io/github/last-commit/Adiaslow/kim-lab-tools.svg)](https://github.com/Adiaslow/kim-lab-tools/commits/main)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/Adiaslow/kim-lab-tools.svg)](https://github.com/Adiaslow/kim-lab-tools)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
