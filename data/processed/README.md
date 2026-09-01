# Processed Data

This folder contains generated analytical outputs from the project.

## Dashboard data

`dashboard/` contains the files required by the Streamlit dashboard.

These files are intentionally kept separate from the larger analytical artifacts so that the dashboard has a clear and stable data dependency.

## Other processed outputs

The parent `processed/` folder contains intermediate and full-resolution analytical artifacts used during development and validation.

These files are kept locally and are excluded from the public repository when they are large or not required by the dashboard.

## Rebuilding outputs

The numbered scripts in `notebooks/` document the analysis in execution order. The final forecasting and decision layers use the validated outputs produced by those scripts.
