import csv
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(BASE_DIR, 'movement_labels.csv')

def main():
    if not os.path.exists(CSV_FILE):
        print(f"Error: Could not find the file at {CSV_FILE}")
        return

    m_values = []
    n_values = []

    # 1. Read the CSV and separate the data
    with open(CSV_FILE, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            action = row.get('action')
            if not action or action not in ['M', 'N']:
                continue
            
            green_pct = float(row['green_pct'])
            
            if action == 'M':
                m_values.append(green_pct)
            elif action == 'N':
                n_values.append(green_pct)

    print(f"Total 'Move' (M) images: {len(m_values)}")
    print(f"Total 'No Move' (N) images: {len(n_values)}")

    # 2. Find the Averages
    if m_values:
        print(f"Average Green % when moving: {sum(m_values)/len(m_values):.2f}%")
        print(f"Lowest Green % you still moved on: {min(m_values):.2f}%")
    
    if n_values:
        print(f"Average Green % when stopping: {sum(n_values)/len(n_values):.2f}%")
        print(f"Highest Green % you stopped at: {max(n_values):.2f}%")

    # 3. Test a range of Thresholds to find the best one
    print("\n--- Testing Thresholds ---")
    best_threshold = 0
    best_accuracy = 0

    # Test every whole number percentage from 0 to 100
    for test_thresh in range(0, 101):
        correct_m = sum(1 for val in m_values if val >= test_thresh)
        correct_n = sum(1 for val in n_values if val < test_thresh)
        
        total_correct = correct_m + correct_n
        total_images = len(m_values) + len(n_values)
        
        if total_images == 0:
            continue
            
        accuracy = (total_correct / total_images) * 100
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = test_thresh

    print(f"\n⭐ BEST THRESHOLD: {best_threshold}%")
    print(f"This rule correctly predicts your brain's choice {best_accuracy:.1f}% of the time!")

if __name__ == "__main__":
    main()