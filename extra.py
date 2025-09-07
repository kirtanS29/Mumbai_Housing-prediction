import pandas as pd

# Load the CSV file
df = pd.read_csv('mumbai.csv')

# Count occurrences of each location
location_counts = df['Location'].value_counts()

# Filter locations with count greater than 1 and print them
locations_gt_1 = location_counts[location_counts > 6]
print(locations_gt_1.index.tolist())
