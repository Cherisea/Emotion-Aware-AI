"""
    A collection of supporting functions
"""

import matplotlib.pyplot as plt
import re
import os
import pandas as pd
import numpy as np

def verify_images(csv_file, audit=True):
    """
        Opens the saved images from the CSV file and allows the user to accept or reject them.
        If rejected, removes the corresponding row from the CSV file.

        csv_file: csv file where each row consists of an integer label column and image pixel column
        audit: view images without modifying the file if False
    """
    df = pd.read_csv(csv_file)
    rows = len(df.index)

    for index, row in df.iterrows():
        pixel_values = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
        image = pixel_values.reshape(48, 48)  # Reshape to 48x48
        
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title(f"Image {index}/{rows}")
        plt.show()

        if audit:
            decision = input("Accept this image? (y/n): ").strip().lower()
            if decision == "n":
                df.drop(index, inplace=True)  # Remove the row

    # Save the updated CSV
    if audit:
        df.to_csv(csv_file, index=False)
        print(f"\nSummary: {len(df.index)} out of {rows} were kept.")

def concat_csv(file_dir, output_file):
    """
        Check if there are five/two mini-batch files and combine them for double check 
    """
    # Match file names like "boredom_ggl_1.csv"
    pattern = re.compile(r"\b[a-z]+_[a-z]+_([0-9]+)\.csv\b")
    total_rows = 0

    df_new = pd.DataFrame(columns=["emotion", "pixels"])
    for csv in os.listdir(file_dir):
        if pattern.fullmatch(csv):
            full_path = os.path.join(file_dir, csv)
            df = pd.read_csv(full_path)
            total_rows += len(df.index)
            df_new = pd.concat([df_new, df], ignore_index=True)
            os.remove(full_path)

    df_new.to_csv(os.path.join(file_dir, output_file), index=False)
    print(f"A total of {total_rows} images has been saved to {output_file}")