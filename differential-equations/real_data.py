cremona = [916, 957, 1061, 1302, 1344, 1565, 1792, 1881]
piacenza = [602, 633, 664, 679, 710, 853, 1012, 1073]
lodi = [482, 559, 658, 739, 811, 853, 928, 963, 1025, 1123, 1133, 1276, 1320, 1362]
bergamo = [2136, 2368, 2864, 3416, 3760]

cremona_population = 360000
piacenza_population = 287000
lodi_population = 230000
bergamo_population = 1108000

cremona = [x / cremona_population for x in cremona]
piacenza = [x / piacenza_population for x in piacenza]
lodi = [x / lodi_population for x in lodi]
bergamo = [x / bergamo_population for x in bergamo]


def get_data_dict(province, num_days=4, time_unit=0.40):
    traj = {}

    if not num_days:
        num_days = len(province)

    for i in range(num_days):
        traj[i * time_unit] = [1 - province[i], province[i], 0.0]

    return traj
