import random
import matplotlib.pyplot as plt


def generate_data_around_line(m, b, start, end, num_pts, sd):
    above_line = []
    below_line = []
    for _ in range(num_pts):
        x1 = random.random() * (end - start) + start
        noise1 = random.gauss(b + 2 * sd, sd)
        x2 = random.random() * (end - start) + start
        noise2 = random.gauss(b - 2 * sd, sd)
        above_line.append((x1, m * x1 + noise1))
        below_line.append((x2, m * x2 + noise2))
    return above_line, below_line


if __name__ == '__main__':
    g1, g2 = generate_data_around_line(3, 1, 5, 10, 200, 1.5)
    with open("data_linear.txt", mode="w") as f:
        f.write(str(200))
        for point in g1:
            f.write("\n{0:.3f} {1:.3f}".format(point[0], point[1]))
        for point in g2:
            f.write("\n{0:.3f} {1:.3f}".format(point[0], point[1]))

    # plt.scatter([pt[0] for pt in g1], [pt[1] for pt in g1], color="red")
    # plt.scatter([pt[0] for pt in g2], [pt[1] for pt in g2], color="green")
    # plt.show()