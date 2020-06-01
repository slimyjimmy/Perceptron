import random
import numpy
from numpy.linalg import norm

class Perceptron:
    def generate_points(self, number_of_points):
        return_list = []
        for i in range(number_of_points):
            x1 = random.uniform(-1, 1)
            x2 = random.uniform(-1, 1)
            height_of_x_in_position_of_x1 = self.x[2] * (x1 / self.x[1])
            point_is_above_funcion = True if height_of_x_in_position_of_x1 < x2 else False
            point = Point(x1, x2, point_is_above_funcion)
            return_list.append(point)
        return return_list

    def __init__(self, N, Nt):
        self.weights = [0, 0, 0]
        self.x = [1, random.uniform(-1, 1), random.uniform(-1, 1)]
        self.N = N
        self.points = self.generate_points(N)
        self.Nt = Nt
    
    def converge_random_point(self):
        random_point = self.points[random.randint(0, len(self.points) - 1)]
        result = numpy.sign(self.weights[0] * random_point.x0 + self.weights[1] * random_point.x1 + self.weights[2] * random_point.x2)
        expected_result = 1 if random_point.y else -1
        if result != expected_result:
            d = -1 if result < 0 else 1
            self.weights[0] += d * random_point.x0
            self.weights[1] += d * random_point.x1
            self.weights[2] += d * random_point.x2
            return False
        return True

    def find_g(self):
        number_of_converged_points = 0
        number_of_weight_changes = 0
        iteration_counter = 0
        while number_of_converged_points < self.N:
            iteration_counter += 1
            if not self.converge_random_point():
                number_of_weight_changes += 1
            number_of_converged_points += 1
        return number_of_weight_changes


    def test_g(self):
        number_of_test_points = self.Nt
        number_of_correctly_tested_points = 0
        test_points = self.generate_points(number_of_test_points)
        for i in test_points:
            expected_result = 1 if i.y else -1
            result = numpy.sign(self.weights[0] * i.x0 + self.weights[1] * i.x1 + self.weights[2] * i.x2)
            if expected_result == result:
                number_of_correctly_tested_points += 1
        return number_of_correctly_tested_points / number_of_test_points



class Point:
    def __init__(self, x1, x2, y):
        self.x0 = 1
        self.x1 = x1
        self.x2 = x2
        self.y = y

def main():

    # train
    total_number_of_weight_changes = 0
    sum_of_accuracy_averages = 0.0
    number_of_runs = 1000
    for i in range(number_of_runs):
        perceptron = Perceptron(10, 1000)
        total_number_of_weight_changes += perceptron.find_g()
        # test
        sum_of_accuracy_averages += perceptron.test_g()
    average_number_of_weight_changes = total_number_of_weight_changes / number_of_runs
    accuracy_average = sum_of_accuracy_averages / number_of_runs
    print("The average number of iterations it took to converge was: " + str(average_number_of_weight_changes))
    print("The average accuracy tested on test points was: " + str(accuracy_average * 100) + "%")



if __name__ == "__main__":
    main()
