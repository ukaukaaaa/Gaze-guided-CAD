# NBME Lung AI Anchoring — Reading Platform Code

Experimental reading platform used in the study of AI-induced attentional anchoring in 3D CT lung nodule reading. This repository contains the PyQt-based reading UI and gaze calibration utilities. **Statistical analysis code is not included.**

## Repository contents

```
ui/
├── ui-ai.py                       # Reading UI for Exp2 (concurrent AI assistance)
├── ui-gazect.py                   # Reading UI for Exp1 (no AI) and Exp3 (gaze-guided second look)
├── calibrate.py                   # Eye-tracker calibration entry point
├── CalibrationGraphicsPygame.py   # Pygame calibration graphics for EyeLink/Tobii
├── utils.py                       # Shared helpers
├── ui-2560x1080.ui                # Qt Designer layout (2560×1080 target)
└── run_nu.bat                     # Windows launch script
```

## Hardware and software requirements

- Display: 2560×1080 or compatible high-resolution monitor
- Eye tracker: EyeLink or Tobii-compatible device (platform assumes 60 Hz sampling or higher)
- OS: Windows 10/11 (eye-tracker SDK dependencies)
- Python: 3.9+

## Installation

```bash
pip install -r requirements.txt
```

Install the eye-tracker SDK separately (SR Research EyeLink Developer Kit or Tobii Pro SDK).

## Data

The CT volumes, AI predictions, reader annotations, and gaze recordings associated with this platform are deposited separately in the companion data repository (see paper for DOI).

## Citation

Please cite the associated paper when using this code.

## License

MIT License (see `LICENSE`).
