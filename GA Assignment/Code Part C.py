import random

LOCATIONS = ["A", "B", "C", "D", "E", "F", "G"]
DIST_MATRIX = [
    [0, 4, 8, 6, 7, 9, 5],  # A (Depot)
    [4, 0, 5, 7, 3, 8, 6],  # B
    [8, 5, 0, 4, 6, 3, 7],  # C
    [6, 7, 4, 0, 5, 6, 8],  # D
    [7, 3, 6, 5, 0, 4, 6],  # E
    [9, 8, 3, 6, 4, 0, 5],  # F
    [5, 6, 7, 8, 6, 5, 0],  # G
]
IDX = {location: i for i, location in enumerate(LOCATIONS)}
DELIVERY_STOPS = ["B", "C", "D", "E", "F", "G"]


def route_distance(chromosome):
    """Calculate total round-trip distance starting and ending at depot A."""
    full_route = ["A"] + chromosome + ["A"]
    return sum(
        DIST_MATRIX[IDX[full_route[i]]][IDX[full_route[i + 1]]]
        for i in range(len(full_route) - 1)
    )


def tournament_selection(population, fitnesses, rng, tournament_size=3):
    """Select the shortest route among k randomly sampled candidates."""
    contestants = rng.sample(range(len(population)), tournament_size)
    best_index = min(contestants, key=lambda i: fitnesses[i])
    return population[best_index][:]


def order_crossover(parent1, parent2, rng):
    """Order Crossover (OX1) preserving valid permutations without duplicates."""
    start, end = sorted(rng.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start : end + 1] = parent1[start : end + 1]
    remaining_stops = iter([stop for stop in parent2 if stop not in child])
    return [next(remaining_stops) if stop is None else stop for stop in child]


def swap_mutation(chromosome, mutation_rate, rng):
    """Randomly swap two delivery locations with probability mutation_rate."""
    mutated_route = chromosome[:]
    if rng.random() < mutation_rate:
        i, j = rng.sample(range(len(mutated_route)), 2)
        mutated_route[i], mutated_route[j] = mutated_route[j], mutated_route[i]
    return mutated_route


def run_ga(seed=42, pop_size=50, generations=80, mutation_rate=0.2, n_elites=2):
    rng = random.Random(seed)

    # Step 1: Initialize population with valid random permutations of B-G
    population = [rng.sample(DELIVERY_STOPS, len(DELIVERY_STOPS)) for _ in range(pop_size)]

    for _ in range(generations):
        # Step 2: Evaluate fitness (total round-trip distance) for every route
        fitnesses = [route_distance(route) for route in population]

        # Step 3: Elitism - copy the shortest routes unchanged into next generation
        ranked_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])
        next_population = [population[i][:] for i in ranked_indices[:n_elites]]

        # Step 4: Selection, Crossover, and Mutation to generate remaining offspring
        while len(next_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, rng)
            parent2 = tournament_selection(population, fitnesses, rng)
            child = order_crossover(parent1, parent2, rng)
            child = swap_mutation(child, mutation_rate, rng)
            next_population.append(child)

        population = next_population

    # Step 5: Identify the shortest route in the final generation
    final_fitnesses = [route_distance(route) for route in population]
    best_index = min(range(pop_size), key=lambda i: final_fitnesses[i])
    return ["A"] + population[best_index] + ["A"], final_fitnesses[best_index]


if __name__ == "__main__":
    best_route, best_distance = run_ga(seed=42, pop_size=50, generations=80)
    print(f"Best Route: {' -> '.join(best_route)}")
    print(f"Total Distance: {best_distance} km")