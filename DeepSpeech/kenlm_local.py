import random

def generate_digit_sequences(num_sequences, max_sequence_length=4):
    digit_words = ["yes", "no"]
    with open("digit_sequences.txt", "w") as f:
        for _ in range(num_sequences):
            sequence_length = random.randint(1, max_sequence_length)
            sequence = [random.choice(digit_words) for _ in range(sequence_length)]
            f.write(" ".join(sequence) + "\n")

if __name__ == "__main__":
    num_sequences = 1000
    generate_digit_sequences(num_sequences)
    print(f"Generated {num_sequences} digit sequences in digit_sequences.txt")
