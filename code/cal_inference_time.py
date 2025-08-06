import re

log_file = "test_console.log"  # Replace with your actual log file name

embed_times = []
extract_times = []

# Regex patterns
embed_pattern = re.compile(r"Embed time:\s*([\d\.]+)\s*seconds")
extract_pattern = re.compile(r"Extract time:\s*([\d\.]+)\s*seconds")

# Read file and extract times
with open(log_file, "r") as f:
    for line in f:
        embed_match = embed_pattern.search(line)
        extract_match = extract_pattern.search(line)
        
        if embed_match:
            embed_times.append(float(embed_match.group(1)))
        if extract_match:
            extract_times.append(float(extract_match.group(1)))

# Skip the first image
if len(embed_times) > 1 and len(extract_times) > 1:
    embed_times = embed_times[1:]
    extract_times = extract_times[1:]

# Calculate number of images after skipping
num_images = len(embed_times)

# Compute averages
avg_embed = sum(embed_times) / num_images if num_images else 0
avg_extract = sum(extract_times) / num_images if num_images else 0

# Print results
print(f"Number of images (after skipping first): {num_images}")
print(f"Average Embed time: {avg_embed:.6f} seconds")
print(f"Average Extract time: {avg_extract:.6f} seconds")
