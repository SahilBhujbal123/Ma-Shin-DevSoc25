# infinite_program.py
import time

def main():
    print("Starting infinite program...")
    counter = 0

    while True:  # Infinite loop
        counter += 1
        print(f"Processing iteration {counter}...")
        time.sleep(1)  # Simulate work by sleeping for 1 second

if __name__ == "__main__":
    main()