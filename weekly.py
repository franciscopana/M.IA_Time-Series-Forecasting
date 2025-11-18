import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data
data_dir = Path('esn_porto_analysis/students/data')
files = sorted(data_dir.glob('students_*_privatized.json'))
all_students = []

for file in files:
    year = file.stem.split('_')[1]
    with open(file, 'r') as f:
        data = json.load(f)
        for student in data.get('students', []):
            student['academic_year'] = year
            all_students.append(student)

df = pd.DataFrame(all_students)

# Convert timestamps to datetime
df['registerDate_dt'] = pd.to_datetime(df['registerDate'], unit='ms', errors='coerce')
df['birthdate_dt'] = pd.to_datetime(df['birthdate'], unit='ms', errors='coerce')

# Clean data
current_date = datetime.now()
df_clean = df[(df['birthdate_dt'] <= pd.Timestamp.now()) & 
              (df['registerDate_dt'] >= df['birthdate_dt'])].copy()

# Create weekly time series
weekly_series = df_clean.groupby(df_clean['registerDate_dt'].dt.to_period('W')).size()
weekly_series.index = weekly_series.index.to_timestamp()
weekly_series.name = 'registrations'

# Plot
fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(weekly_series.index, weekly_series.values, linewidth=1.5, color='darkgreen', marker='o', markersize=3)

# Format x-axis to show years horizontally
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator())

# Ensure labels are horizontal
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

ax.set_title('Weekly Registration Volume', fontsize=14, fontweight='bold')
ax.set_ylabel('Registrations', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()

# Save plot
plt.savefig('weekly_registrations.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'weekly_registrations.png'")
plt.show()
