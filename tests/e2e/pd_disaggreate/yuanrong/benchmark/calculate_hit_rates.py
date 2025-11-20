import re
import sys

def calculate_hit_rates(file_path):
    # Initialize statistical variables
    sum_total_tokens = 0
    sum_hbm_hit = 0
    sum_ddr_hit = 0
    line_count = 0

    # Regular expression to match numbers
    # Matching format: Total tokens 8212, HBM hit tokens: 4096, External hit tokens: 0
    pattern = re.compile(r"Total tokens\s+(\d+),.*?HBM hit tokens:\s+(\d+),.*?External hit tokens:\s+(\d+)")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Simple filtering, only process lines containing specific keywords
                if "Total tokens" not in line:
                    continue

                match = pattern.search(line)
                if match:
                    total = int(match.group(1))
                    hbm = int(match.group(2))
                    load = int(match.group(3))

                    sum_total_tokens += total
                    sum_hbm_hit += hbm
                    sum_ddr_hit += load
                    line_count += 1

        if sum_total_tokens == 0:
            print("No valid Token data found.")
            return

        # Calculate hit rates
        # Note: Due to Block alignment in vLLM, hit + load may be slightly larger than total, calculated by actual values here
        hbm_rate = (sum_hbm_hit / sum_total_tokens) * 100
        ddr_rate = (sum_ddr_hit / sum_total_tokens) * 100
        total_hit_rate = ((sum_hbm_hit + sum_ddr_hit) / sum_total_tokens) * 100

        print("-" * 40)
        print(f"Log Analysis Result ({file_path})")
        print("-" * 40)
        print(f"Processed Lines (Requests) : {line_count}")
        print(f"Total Token Count          : {sum_total_tokens}")
        print(f"HBM Hit (GPU) Count        : {sum_hbm_hit}")
        print(f"DDR Hit (CPU) Count        : {sum_ddr_hit}")
        print("-" * 40)
        print(f"HBM Hit Rate (GPU)         : {hbm_rate:.2f}%")
        print(f"DDR Hit Rate (CPU)         : {ddr_rate:.2f}%")
        print(f"Overall Hit Rate           : {total_hit_rate:.2f}%")
        print("-" * 40)

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calc_hit_rate.py <log_file_name>")
    else:
        calculate_hit_rates(sys.argv[1])
