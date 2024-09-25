import os
import shutil
import re


# Function to process the text file
def process_file(text_file, source_folder, destination_folder):
    # Open the text file
    with open(text_file, "r") as file:
        # Read through each line
        for line in file:
            # Split the line into words and numbers (assuming whitespace separation)
            parts = line.split(":")

            # Make sure there are enough parts (first word and second number)
            if len(parts) >= 2:
                first_number = re.findall(r"\d+", parts[0])[0]  # Extract the first word
                second_number = float(
                    parts[1].strip()
                )  # Extract the second number and convert it to float

                # Check if the second number is 1.0
                if second_number == 1.0:
                    # Construct the filename from the first word
                    file_name_1 = "AOI_1_RIO_img" + str(first_number) + ".tif"

                    # Check if a file with this name exists in the source folder
                    for file in os.listdir(source_folder + "/mask"):
                        if file.startswith(file_name_1):
                            # Full path of the file in the source folder
                            source_path = os.path.join(source_folder + "/mask", file)

                            # Full path for where to save the file in the destination folder
                            destination_path = os.path.join(
                                destination_folder + "/mask", file
                            )

                            # Copy the file to the destination folder
                            shutil.copy(source_path, destination_path)
                            print(f"Copied {file} to {destination_folder}")


# Example usage
text_file = "/home/filip/satmae_experiments/jaccard_results/image_results_iou_201.txt"  # Path to your text file
source_folder = "/home/filip/SpaceNetV1/"  # Path to the folder containing files
destination_folder = "/home/filip/SpaceNetV1_10pc"  # Path to the destination folder

# Make sure the destination folder exists
os.makedirs(destination_folder + "/mask", exist_ok=True)

# Call the function to process the file
process_file(text_file, source_folder, destination_folder)
