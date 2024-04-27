import random

# Set the total number of IDs
total_ids = 1288  # 0 to 1287 inclusive

# Generate a list of all IDs
ids = list(range(total_ids))

# Shuffle the list of IDs to ensure random distribution
random.shuffle(ids)

# Calculate the split index for 80-20 distribution
split_index = int(0.8 * total_ids)

# Split the IDs into training and testing datasets
train_ids = ids[:split_index]
test_ids = ids[split_index:]

# Function to write IDs to a file
def write_ids_to_file(file_path, ids):
    with open(file_path, 'w') as file:
        for id in ids:
            file.write(f"{id}\n")

# Write training and testing datasets to respective files
path = '../dataset/'

write_ids_to_file(f'{path}train_clusters.txt', train_ids)
write_ids_to_file(f'{path}valid_clusters.txt', test_ids)

print("Training and testing IDs have been written to train.txt and test.txt respectively.")
