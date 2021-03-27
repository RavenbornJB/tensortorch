import random
import matplotlib.pyplot as plt


def generate_data_around_line(m, b, start, end, num_pts, sd):
    above_line = []
    below_line = []
    for _ in range(num_pts):
        x1 = random.random() * (end - start) + start
        noise1 = random.gauss(b + 3 * sd, sd)
        x2 = random.random() * (end - start) + start
        noise2 = random.gauss(b - 3 * sd, sd)
        above_line.append((x1, m * x1 + noise1))
        below_line.append((x2, m * x2 + noise2))
    return above_line, below_line

def generate_data_around_4_points(p1, p2, p3, p4, num_pts, sd):
    class1 = []
    class2 = []
    for _ in range(num_pts):
        for pt in [p1, p2]:
            x = pt[0] + random.gauss(0, sd)
            y = pt[1] + random.gauss(0, sd)
            class1.append((x, y))
        for pt in [p3, p4]:
            x = pt[0] + random.gauss(0, sd)
            y = pt[1] + random.gauss(0, sd)
            class2.append((x, y))
    return class1, class2

if __name__ == '__main__':
    num_points = 500
    # g1, g2 = generate_data_around_line(1.5, 0, -1, 1, num_points, 0.5)
    g1, g2 = generate_data_around_4_points((-1, -1), (1, 1), (1, -1), (-1, 1), num_points, 0.5)
    # with open("data_clumped_4_point_test.txt", mode="w") as f:
    #     f.write(str(num_points))
    #     for point in g1:
    #         f.write("\n{0:.3f} {1:.3f}".format(point[0], point[1]))
    #     for point in g2:
    #         f.write("\n{0:.3f} {1:.3f}".format(point[0], point[1]))

    plt.scatter([pt[0] for pt in g1], [pt[1] for pt in g1], color="red")
    plt.scatter([pt[0] for pt in g2], [pt[1] for pt in g2], color="green")
    plt.show()