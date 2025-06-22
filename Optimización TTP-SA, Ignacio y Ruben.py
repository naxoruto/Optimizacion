import numpy as np
import math
import random
import concurrent.futures
import argparse
import multiprocessing
from pathlib import Path
from numba import njit
import matplotlib.pyplot as plt
import time

def parse_ttp_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    problem_info = {
        "cities": 0,
        "items": 0,
        "capacity": 0,
        "min_speed": 0.0,
        "max_speed": 0.0,
        "renting_ratio": 0.0,
        "city_coordinates": [],
        "items_info": [],
    }

    coord_section = False
    item_section = False

    for line in lines:
        if line.startswith("DIMENSION"):
            problem_info["cities"] = int(line.split()[-1])
        elif line.startswith("NUMBER OF ITEMS"):
            problem_info["items"] = int(line.split()[-1])
        elif line.startswith("CAPACITY OF KNAPSACK"):
            problem_info["capacity"] = int(line.split()[-1])
        elif line.startswith("MIN SPEED"):
            problem_info["min_speed"] = float(line.split()[-1])
        elif line.startswith("MAX SPEED"):
            problem_info["max_speed"] = float(line.split()[-1])
        elif line.startswith("RENTING RATIO"):
            problem_info["renting_ratio"] = float(line.split()[-1])
        elif line.startswith("NODE_COORD_SECTION"):
            coord_section = True
            continue
        elif line.startswith("ITEMS SECTION"):
            coord_section = False
            item_section = True
            continue
        elif coord_section:
            parts = line.strip().split()
            if len(parts) == 3:
                problem_info["city_coordinates"].append((int(parts[0]), float(parts[1]), float(parts[2])))
        elif item_section:
            parts = line.strip().split()
            if len(parts) == 4:
                problem_info["items_info"].append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))

    return problem_info

@njit
def evaluate_solution_components(tour, picked, item_city, item_value, item_weight, dist_matrix, capacity, min_speed, max_speed, R):
    total_value = 0
    total_weight = 0
    time_total = 0.0
    n_items = len(picked)
    for i in range(len(tour) - 1):
        city = tour[i]
        next_city = tour[i + 1]
        for idx in range(n_items):
            if picked[idx] and item_city[idx] == city:
                total_value += item_value[idx]
                total_weight += item_weight[idx]
        v = max(min_speed, max_speed - (total_weight / capacity) * (max_speed - min_speed))
        dist = dist_matrix[city, next_city]
        time_total += dist / v
    penalty = R * time_total
    fitness = total_value - penalty
    return fitness, total_value, penalty

@njit
def evaluate_solution_jit(tour, picked, item_city, item_value, item_weight, dist_matrix, capacity, min_speed, max_speed, R):
    total_value = 0
    total_weight = 0
    time_total = 0.0
    n_items = len(picked)
    n_steps = len(tour) - 1
    for i in range(n_steps):
        city = tour[i]
        next_city = tour[i + 1]
        for idx in range(n_items):
            if picked[idx] and item_city[idx] == city:
                total_value += item_value[idx]
                total_weight += item_weight[idx]
        v = max(min_speed, max_speed - (total_weight / capacity) * (max_speed - min_speed))
        dist = dist_matrix[city, next_city]
        time_total += dist / v
    return total_value - R * time_total

def intelligent_initial_solution(cities, items, capacity, items_by_city, items_info):
    tour = list(range(cities))
    random.shuffle(tour)
    tour.append(tour[0])
    picked = [False] * items
    total_weight = 0
    for city in tour[:-1]:
        for item_id, ratio in items_by_city[city]:
            value, weight, city = items_info[item_id]
            if total_weight + weight <= capacity:
                picked[item_id] = True
                total_weight += weight
    return tour, picked

def generate_smart_neighbor(tour, picked, items):
    new_tour = tour[:-1]
    i, j = sorted(random.sample(range(1, len(new_tour)-1), 2))
    new_tour[i:j] = reversed(new_tour[i:j])
    new_tour.append(new_tour[0])
    new_picked = picked[:]
    for _ in range(3):
        idx = random.randint(0, items - 1)
        new_picked[idx] = not new_picked[idx]
    return new_tour, new_picked

def get_best_neighbor(current_tour, current_picked, n_neighbors, items, item_city, item_value, item_weight, dist_matrix, capacity, min_speed, max_speed, R):
    neighbors = [generate_smart_neighbor(current_tour, current_picked, items) for _ in range(n_neighbors)]
    best_score = -np.inf
    best_neighbor = None
    for tour, picked in neighbors:
        score = evaluate_solution_jit(np.array(tour), np.array(picked), item_city, item_value, item_weight, dist_matrix, capacity, min_speed, max_speed, R)
        if score > best_score:
            best_score = score
            best_neighbor = (tour, picked)
    return best_neighbor[0], best_neighbor[1], best_score

def parallel_simulated_annealing(ttp, dist_matrix, item_city, item_value, item_weight, items_by_city, seed, params):
    random.seed(seed)
    np.random.seed(seed)

    cities = ttp["cities"]
    items = ttp["items"]
    capacity = ttp["capacity"]
    min_speed = ttp["min_speed"]
    max_speed = ttp["max_speed"]
    R = ttp["renting_ratio"]

    current_tour, current_picked = intelligent_initial_solution(cities, items, capacity, items_by_city, list(zip(item_value, item_weight, item_city)))
    best_tour, best_picked = current_tour, current_picked
    best_score = evaluate_solution_jit(np.array(best_tour), np.array(best_picked), item_city, item_value, item_weight, dist_matrix, capacity, min_speed, max_speed, R)
    T = params["T0"]

    for _ in range(params["max_iter"]):
        if T < params["Tmin"]:
            break
        neighbor_tour, neighbor_picked, neighbor_score = get_best_neighbor(
            current_tour, current_picked, params["n_neighbors"], items,
            item_city, item_value, item_weight, dist_matrix, capacity, min_speed, max_speed, R
        )
        delta = neighbor_score - best_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_tour, current_picked = neighbor_tour, neighbor_picked
            if neighbor_score > best_score:
                best_score = neighbor_score
                best_tour, best_picked = neighbor_tour, neighbor_picked
        T *= params["alpha"]
    return best_score, best_tour

def plot_tour(tour, coordinates, title="Ruta óptima"):
    coords = np.array(coordinates)[[i for i in tour]]
    plt.figure(figsize=(10, 6))
    plt.plot(coords[:, 0], coords[:, 1], 'o-', markersize=4)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main(file_path, parallel_runs=8, return_tour=False, fast_test=False, profile=False, return_components=False):
    start_time = time.time()

    ttp = parse_ttp_file(file_path)
    coords = sorted(ttp["city_coordinates"], key=lambda x: x[0])
    parsed_coords = [(x, y) for _, x, y in coords]
    positions = np.array(parsed_coords)
    dist_matrix = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    items_info = np.array(ttp["items_info"])

    item_value = np.array([v for _, v, _, _ in items_info])
    item_weight = np.array([w for _, _, w, _ in items_info])
    item_city = np.array([c for _, _, _, c in items_info]) - 1

    item_ratios = [(item_id - 1, value / weight if weight != 0 else 0, city - 1)
                   for item_id, value, weight, city in items_info]
    items_by_city = [[] for _ in range(ttp["cities"])]
    for item_idx, ratio, city in item_ratios:
        items_by_city[city].append((item_idx, ratio))
    for city_items_list in items_by_city:
        city_items_list.sort(key=lambda x: -x[1])

    params = {
        "T0": 6000,
        "alpha": 0.998,
        "Tmin": 0.01,
        "max_iter": 3000,
        "n_neighbors": 30
    }

    if fast_test:
        parallel_runs = 2
        params["max_iter"] = 50

    seeds = [random.randint(0, 1_000_000) for _ in range(parallel_runs)]
    with multiprocessing.Pool(processes=parallel_runs) as pool:
        results = pool.starmap(
            parallel_simulated_annealing,
            [(ttp, dist_matrix, item_city, item_value, item_weight, items_by_city, seed, params) for seed in seeds]
        )

    best_score, best_tour = max(results, key=lambda x: x[0])
    elapsed_time = time.time() - start_time

    if return_tour or return_components:
        return best_score, best_tour, parsed_coords, item_city, item_value, item_weight, dist_matrix, ttp

    print(f"Mejor puntaje final entre {parallel_runs} corridas: {best_score:.2f}")
    print(f"⏱️ Tiempo de ejecución: {elapsed_time:.2f} segundos")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Ruta al archivo TTP")
    parser.add_argument("--parallel-runs", type=int, default=8)
    parser.add_argument("--fast-test", action="store_true", help="Usar configuración liviana para pruebas rápidas")
    parser.add_argument("--show-profit", action="store_true", help="Mostrar desglose de profit y penalización")
    parser.add_argument("--profile", action="store_true", help="Medir tiempo de ejecución")
    parser.add_argument("--plot", action="store_true", help="Mostrar visualización de la mejor ruta")

    args = parser.parse_args()

    if args.show_profit:
        start_time = time.time()
        result = main(args.file, args.parallel_runs, return_tour=True, fast_test=args.fast_test, profile=args.profile, return_components=True)
        elapsed_time = time.time() - start_time
        print(f"⏱️ Tiempo de ejecución: estimado en {elapsed_time:.2f} segundos")
        score, tour, coords, item_city, item_value, item_weight, dist_matrix, ttp = result

        from numpy import array
        fitness, profit, penalty = evaluate_solution_components(
            array(tour),
            array([True] * len(item_value)),
            item_city, item_value, item_weight,
            dist_matrix,
            ttp["capacity"], ttp["min_speed"], ttp["max_speed"], ttp["renting_ratio"]
        )
        print(f"🏆 Fitness final: {fitness:.2f}")
        print(f"💰 Profit total (valor ítems): {profit:.2f}")
        print(f"🕓 Penalización por tiempo: {penalty:.2f}")
        if args.plot:
            plot_tour(tour, coords)
    else:
        main(args.file, args.parallel_runs, fast_test=args.fast_test, profile=args.profile)
