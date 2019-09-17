choices = [
    'Forward',
    'Curl clockwise'
]

while len(results) <= pop_size:
    if num_segments == 0:
        orientations = [np.random.choice(choices)
                        for _ in range(np.random.randint(1, 50))]
    else:
        orientations = [np.random.choice(choices)
                        for _ in range(num_segments)]

    results.append(evaluate(orientations, True))
