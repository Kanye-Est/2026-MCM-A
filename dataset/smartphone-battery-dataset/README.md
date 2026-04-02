# Smartphone Battery Consumption Dataset

This repository provides a real-world smartphone battery consumption dataset collected from multiple devices
(e.g., iPhone, HUAWEI, REDMI) for battery consumption modeling and time-to-empty (TTE) prediction.

## Files
- `data.xlsx`: main dataset file
  - `devices`: device-level static features (model, OS, nominal capacity/energy, SOH, cycle count, usage months, charging habits)
  - `sessions`: scenario/session-level metadata (scenario name, network mode, GPS status, start/end SOC, etc.)
  - `samples`: time-series samples (SOC, temperature, brightness, CPU usage, RSSI, traffic, estimated power, etc.)

## License
This repository is released under **CC0 1.0**.

## Citation
If you use this dataset, please cite:
Author, *Smartphone Battery Consumption Dataset*, 2026. GitHub: https://github.com/zhangxiwen952-star/smartphone-battery-dataset
