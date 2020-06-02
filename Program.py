import random

"""Class Perceptron"""
class Perceptron():
    def __init__(self, training_points, g):
        self.training_points = training_points
        self.g = g

    def train_g(self):
        no_more_misclassified_points = False
        iteration_counter = 0

        while not no_more_misclassified_points:

            iteration_counter += 1 

            misclassified_points = []
            for training_point in self.training_points:
                if evaluate(training_point.x, self.g) != training_point.y:
                    misclassified_points.append(training_point)
            if len(misclassified_points) == 0:
                no_more_misclassified_points = True
                continue
            
            random_misclassified_point = misclassified_points[random.randint(0, len(misclassified_points) - 1)]

            for i in range(len(self.g)):
                self.g[i] += (random_misclassified_point.y - (not random_misclassified_point.y)) * random_misclassified_point.x[i]

        return iteration_counter

    def test_g(self, test_points):
        number_of_correctly_classified_points = 0
        for test_point in test_points:
            if evaluate(test_point.x, self.g) == test_point.y:
                number_of_correctly_classified_points += 1
        return number_of_correctly_classified_points / len(test_points)



"""Class Point"""
class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y



"""Functions"""
def generate_points(number_of_training_points, target_function):
    generated_points = []
    for i in range(number_of_training_points):
        x = [1, random.uniform(-1, 1), random.uniform(-1, 1)]
        y = evaluate(x, target_function)
        generated_points.append(Point(x, y))

    return generated_points


def evaluate(input, function):
    result = function[0] * input[0] + function[1] * input[1] + function[2] * input[2]
    return True if result > 0 else False


"""MAIN"""
def main():
    number_of_runs = 1000
    total_number_of_iterations = 0
    total_number_of_correctly_classified_points = 0
    number_of_training_points = 100
    number_of_test_points = 100

    for i in range(number_of_runs):
        default_g = [0, 0, 0]
        target_function = [0, random.uniform(-1, 1), random.uniform(-1, 1)]
        training_points = generate_points(number_of_training_points, target_function)
        perceptron = Perceptron(training_points, default_g)
        total_number_of_iterations += perceptron.train_g()
        test_points = generate_points(number_of_test_points, target_function)
        total_number_of_correctly_classified_points += perceptron.test_g(test_points)
    
    print("It took the perceptron the following number of iterations to converge on average: " + str(total_number_of_iterations / number_of_runs))
    print("The average error of g was: " + str(1 - (total_number_of_correctly_classified_points / number_of_runs)))


if __name__ == "__main__":
    main()
