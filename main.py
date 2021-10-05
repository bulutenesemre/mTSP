import sys
import json
import numpy as np
import itertools
from itertools import chain, combinations
from sys import maxsize
from more_itertools import set_partitions

node_represents = 2 ** 31


class Vehicle():
    """
    Args:
        features: Array of feature of the vehicle
        nodes_combinations : Number of combinatio of nodes
    Returns:
        Instance of vehicle class
    """

    def __init__(self, features, subsets, jobs, matrix):
        self.id = features['id']
        self.start_index = features['start_index']
        self.capacity = np.sum(features['capacity'])
        self.time_distance = list()
        self.order_of_destination = list()
        self.min_distance_subsets(subsets, jobs, matrix)

    def min_distance_subsets(self, subsets, jobs, matrix):
        """
        Determine the minimum distance time for all given subsets
        Args:
            subsets: all subsets of the job destinations
            jobs: all jobs list
            matrix: time distance matrix

        Returns:
            None
        """
        for destinations in subsets:

            time_distance, order_of_destinations = self.minimum_distance(
                destinations, jobs, matrix)

            self.time_distance.append(time_distance)
            self.order_of_destination.append(order_of_destinations)

    def minimum_distance(self, destinations, jobs, matrix):
        """
            Finds the minimum distances with given distances for the vehicle.

        Args:
            destinations: destinations of vehicles
            jobs: all job list
            matrix: time distance matrix

        Returns:
            time_distance
            order_of_destinations
        """
        destination_number = len(destinations)
        # If 0 return the same as node represents value
        if destination_number == 0:
            return node_represents, list()

        # Create all delivery information using destination_number
        all_job_ids = np.zeros(destination_number, dtype=np.int32)
        all_job_locations = np.zeros(destination_number, dtype=np.int32)
        all_service_times = np.zeros(destination_number, dtype=np.int32)
        total_service_times = 0
        total_cargo = 0

        min_order_destinations = -1

        for idx, destination in enumerate(destinations):
            job = jobs[destination]
            job_id, job_location_index, job_deliveries, job_service_time = job.values()
            all_job_ids[idx] = job_id
            all_job_locations[idx] = job_location_index
            all_service_times[idx] = job_service_time

            total_service_times += job_service_time
            total_cargo += np.sum(job_deliveries)

            if total_cargo > self.capacity:
                node_represents, list()
            min_path = node_represents
            all_permutations = list(itertools.permutations(destinations))

            for permutation in all_permutations:
                total_delivery_duration = matrix[self.start_index,
                                                 permutation[0]]
                for i in range(len(permutation) - 1):
                    total_delivery_duration += matrix[permutation[i],
                                                      permutation[i + 1]]
                if min_path > total_delivery_duration:
                    min_path = total_delivery_duration
                    min_order_destinations = permutation

            min_path += total_service_times

            return min_path, min_order_destinations


class MTSP:
    """
    Brute Force SOlver problem for Multiple Travelling Salesman Problem

    Args:
        input_path : Relative json file path

    Returns:
        Class instance
    """

    def __init__(self, input_path):
        vehicles, jobs, matrix = self.read_json_file(input_path)
        self.vehicles = vehicles
        self.jobs = jobs
        self.matrix = matrix
        self.number_of_vehicles = len(vehicles)
        self.number_of_jobs = len(jobs)
        self.vehicle_list = list()
        self.destination_combinations = list(set_partitions(np.arange(self.number_of_jobs),
                                                            self.number_of_vehicles))
        self.number_of_combinations = len(self.destination_combinations)

        self.all_destination_subsets = self.generate_all_subsets()

        # Collect all vehicle properties in once
        for vehicle_properties in vehicles:
            self.vehicle_list.append(Vehicle(
                vehicle_properties, self.all_destination_subsets, self.jobs, self.matrix))

    def read_json_file(self, path):
        """Reads given json file and return objects

        Args:
            path: Relative json file path

        Returns:
            vehicles: object in vehicles array
            jobs: object in jobs array
            matrix: duration matrix array
        """
        with open(path, 'r') as input_file:
            data = input_file.read()

        data_obj = json.loads(data)
        return data_obj['vehicles'], data_obj['jobs'], np.asarray(data_obj['matrix'])

    def generate_all_subsets(self):
        """

        Returns:
            Generation of all subsets
        """
        my_list = np.arange(self.number_of_jobs)
        itr = chain.from_iterable(combinations(my_list, n)
                                  for n in range(len(my_list) + 1))

        return [list(el) for el in itr]

    def find_best_route(self, vehicles, jobs, matrix, combinations):
        """
        Finds the best route with checking possible combination
        Args:
            vehicles: all vehicles
            jobs: all jobs
            matrix: all destinations
            all_combinations: all destination combinations

        Returns:
            json: output json object
        """

        output_dict = dict()
        min_routes = dict()
        min_distance = node_represents

        for combination in combinations:
            total_time_distance = 0
            total_destination_order_list = list()
            routes = dict()

            for i in range(len(combination)):
                all_subset_index = self.all_destination_subsets.index(
                    combination[i])
                vehicle = vehicles[i]
                vehicle_id = vehicle.id
                vehicle_time = vehicle.time_distance[all_subset_index]
                vehicle_destination_order = vehicle.order_of_destination[all_subset_index]

                total_time_distance += vehicle_time
                total_destination_order_list.append(vehicle_destination_order)
                routes[vehicle_id] = {
                    "jobs": vehicle_destination_order, "delivery_duration": vehicle_time}
            if total_time_distance < min_distance:
                min_distance = total_time_distance
                min_routes = routes

        output_dict["total_delivery_duration"] = min_distance
        output_dict["routes"] = min_routes
        return output_dict

    def calculate(self):
        output_dict = self.find_best_route(
            self.vehicle_list, self.jobs, self.matrix, self.destination_combinations)
        json_output = json.dumps(output_dict, cls=NpEncoder, indent=4)
        return json_output

class NpEncoder(json.JSONEncoder):
    #This class fixing type error of Numpy and json incompatibility
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


if __name__ == '__main__':
    input_data = "data_input.json"
    output_data = "data_output.json"
    bf = MTSP(input_data)
    solution = bf.calculate()
    f = open(output_data, "w")
    f.write(solution)
    f.close()
