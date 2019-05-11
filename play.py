

acids = ["A", "B", "C"]
test = ''.join(np.random.choice(acids,) for _ in range(10))


point = []
all_points = []
for acid in test:
    if acid is not "B":
        point.append(acid)
    else:
        if len(point) is not 0:
            all_points.append(point)
        point = []

if len(point) is not 0:
    all_points.append(point)
import shapely.geometry
