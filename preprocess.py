
import os
import re

DATA_DIR = "./"
DIRS = ["train", "dev", "test"]

def preprocess_line(line):
    # Regex replacements
    line = re.sub(r"wzjwz\d+", "giảng viên", line)
    line = line.replace("doubledot", ":")
    line = line.replace("colonlove", "yêu thích")
    line = line.replace("colonsmile", "vui vẻ")
    line = line.replace("colonsad", "buồn")
    line = line.replace("colonp", ":p")
    line = line.replace("colonb", ":b")
    line = line.replace("colond", ":d")
    line = line.replace("colonright", ")")
    line = line.replace("colonleft", "(")
    return line

def process_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    with open(filepath, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(preprocess_line(line))

def main():
    print("Preprocessing files...")
    for d in DIRS:
        path = os.path.join(DATA_DIR, d, "sents.txt")
        if os.path.exists(path):
            print(f"Processing {path}")
            process_file(path)
    print("Done.")

if __name__ == "__main__":
    main()
