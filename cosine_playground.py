import numpy as np

# =========================
# COSINE SIMILARITY BASICS
# =========================

def cosine_similarity(a, b):
    """
    Cosine similarity:
        +1  = same direction
         0  = unrelated / perpendicular
        -1  = opposite direction
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def print_case(name, a, b):
    print("\n" + "=" * 50)
    print(name)
    print("A:", a)
    print("B:", b)
    print("cosine similarity:", cosine_similarity(a, b))


# =========================
# PLAYGROUND CASES
# =========================

# 1. Same direction
a = [1, 2, 3]
b = [1, 2, 3]
print_case("Same vector", a, b)

# 2. Same direction but bigger magnitude
a = [1, 2, 3]
b = [10, 20, 30]
print_case("Same direction, different size", a, b)

# 3. Opposite direction
a = [1, 2, 3]
b = [-1, -2, -3]
print_case("Opposite direction", a, b)

# 4. Partially similar
a = [1, 2, 3]
b = [1, 2, 0]
print_case("Partially similar", a, b)

# 5. Perpendicular in 2D
a = [1, 0]
b = [0, 1]
print_case("Perpendicular", a, b)


# =========================
# RANDOM VECTOR SIMULATION
# =========================

print("\n" + "#" * 50)
print("RANDOM VECTOR SIMULATION")
print("#" * 50)

DIM = 8              # vector size
NOISE_AMOUNT = 0.25  # change this
SCALE = 1.0          # change this

base = np.random.randn(DIM)

# Similar vector = base + noise
similar = base + np.random.randn(DIM) * NOISE_AMOUNT

# Scaled vector = same direction, bigger/smaller magnitude
scaled = base * SCALE

# Random unrelated vector
random_vec = np.random.randn(DIM)

# Opposite vector
opposite = -base

print_case("Base vs similar noisy copy", base, similar)
print_case("Base vs scaled copy", base, scaled)
print_case("Base vs random vector", base, random_vec)
print_case("Base vs opposite vector", base, opposite)


# =========================
# COLLAPSE SIMULATION
# =========================

print("\n" + "#" * 50)
print("VECTOR COLLAPSE SIMULATION")
print("#" * 50)

NUM_VECTORS = 10
DIM = 16

collapse_strength = 0.80
# 0.00 = fully random/distinct vectors
# 1.00 = all vectors collapse toward same shared direction

shared_direction = np.random.randn(DIM)
shared_direction = shared_direction / np.linalg.norm(shared_direction)

vectors = []

for i in range(NUM_VECTORS):
    unique_part = np.random.randn(DIM)
    unique_part = unique_part / np.linalg.norm(unique_part)

    v = (
        collapse_strength * shared_direction
        + (1.0 - collapse_strength) * unique_part
    )

    vectors.append(v)

# Compute average pairwise cosine similarity
similarities = []

for i in range(NUM_VECTORS):
    for j in range(i + 1, NUM_VECTORS):
        sim = cosine_similarity(vectors[i], vectors[j])
        similarities.append(sim)

print("collapse_strength:", collapse_strength)
print("average cosine similarity:", np.mean(similarities))
print("min cosine similarity:", np.min(similarities))
print("max cosine similarity:", np.max(similarities))