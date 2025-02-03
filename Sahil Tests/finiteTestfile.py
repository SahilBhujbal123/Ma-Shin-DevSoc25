# finite_program.py
import time

def main():
    print("Starting finite program...")
    start_time = time.time()
    array = []
    # Simulate a finite task
    total = 0
    for i in range(10000000):  # Adjust the range for longer/shorter execution
        total += i
        if i%10000 == 0:
            array.append(i)

    print(f"Finite program completed. Total: {total}")
    print(f"Execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()