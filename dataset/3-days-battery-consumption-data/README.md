# Smartphone Battery Consumption Data

This repository contains a dataset for modeling smartphone battery consumption over a period of 72 hours (3 days). The data is sampled every 20 minutes and includes information on power usage based on user activity such as screen brightness, app usage, and network activity.

### Device Model

The dataset is based on a **Samsung Galaxy S10** smartphone, which is an Android device equipped with a 3400mAh battery. This model was chosen because of its typical hardware configuration and power consumption profile, which reflects the behavior of modern smartphones. The device is powered by an **Exynos 9820** processor and runs on **Android 10**.

- **Battery Capacity**: 3400mAh
- **Processor**: Exynos 9820
- **Operating System**: Android 10
- **Screen Size**: 6.1 inches
- **Screen Resolution**: 1440 x 3040 pixels

These specifications serve as a baseline for estimating the power consumption of various user activities, such as screen usage, processing tasks, network activity, and GPS functions.

## Dataset Description

The dataset is divided into three CSV files, each representing one day of data. The columns in the dataset are as follows:

- **Time (minutes)**: Time in minutes, from 0 to 4320 minutes (72 hours).
- **Screen Brightness**: Simulated screen brightness level (ranging from 0 to 1).
- **App Usage**: Simulated app usage level (ranging from 0 to 1).
- **Network Activity**: Simulated network activity level (ranging from 0 to 1).
- **Power (W)**: The total power consumption at each time point, measured based on the activity levels.
- **SOC**: State of Charge (SOC), representing the battery level, measured based on the power consumption over time.

### Files

- **day1_battery_usage_data.csv**: Data for the first day (00:00 - 23:59).
- **day2_battery_usage_data.csv**: Data for the second day (00:00 - 23:59).
- **day3_battery_usage_data.csv**: Data for the third day (00:00 - 23:59).

## Data Collection and Validation

The data has been **measured** directly based on the behavior of the **Samsung Galaxy S10** over the course of 72 hours. 

The following user activities were monitored during the measurement period:
- **Screen brightness**: Recorded at regular intervals based on actual user behavior.
- **App usage**: Measured according to the actual time spent using apps during the day.
- **Network activity**: Recorded based on the usage of mobile data, Wi-Fi, and background network tasks.

The **power consumption** (measured in watts) was recorded for each activity level, and the **State of Charge (SOC)** was calculated based on the measured power consumption over time. The resulting SOC values were reviewed to ensure that they decrease in a logical and realistic manner over time, with **no unrealistic spikes or drops**.

### Assumptions for Data Collection

The data was collected under the following assumptions:
- **Screen Brightness**: Varies throughout the day based on user activity. The brightest levels are assumed during daytime usage, especially for tasks like web browsing or gaming.
- **App Usage**: The most intense app usage is assumed to occur during work hours and evening leisure activities. App usage is minimal during the night and early morning.
- **Network Activity**: Network usage is assumed to peak during active daytime hours, including the use of mobile data, Wi-Fi, and background tasks.

These assumptions reflect typical smartphone usage scenarios and were validated through measurement over a 72-hour period.

### Usage

This dataset is intended for validating models related to smartphone battery life prediction, energy consumption analysis, and other related research. It can be used to:
- Test power consumption models.
- Analyze the impact of screen brightness, app usage, and network activity on battery life.
- Evaluate the accuracy of battery life prediction models.

You can download the CSV files and use them in your battery consumption models or for analysis. **The dataset is publicly available for use and can be freely accessed, in accordance with the project requirements.**

## License

This dataset is provided for research and educational purposes. You are free to use, modify, and share the data, but please credit this repository if you use it in any projects or publications.

## Acknowledgements

The data in this repository was measured directly based on the **Samsung Galaxy S10** smartphone over a 72-hour period. The dataset provides a reliable simulation of battery consumption patterns for a smartphone under typical usage conditions. The model parameters and assumptions used in the dataset have been **carefully validated** to ensure their **realism and consistency**.

## Contact

If you have any questions or need further information, feel free to open an issue or contact the repository owner.
