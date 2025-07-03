# iris.csv 파일을 열어서 데이터를 한 번에 읽고 저장
all_data = list()

with open("iris.csv", "r") as file:
    header = file.readline().strip().split(",")
    for line in file:
        values = line.strip().split(",")
        all_data.append(dict(zip(header, values)))

# 1
species_count = {}

for row in all_data:
    species = row["species"]
    species_count[species] = species_count.get(species, 0) + 1

with open("Results.txt", "w") as file:
    file.write(
        ", ".join(f"{species}={count}" for species, count in species_count.items())
    )
    file.write("\n")  # 구분용 줄바꿈

"""
setosa_filt = 0
verisicolor_filt = 0
virginica_filt = 0

for line in all_data:
    if line["species"] == "setosa":
        setosa_filt += 1
    elif line["species"] == "verisicolor":
        verisicolor_filt += 1
    else:
        viriginica_filt += 1
    
with open("Result.txt", "w") as file:
    file.write(f"setosa : {setosa_filt}, verisicolor : {verisicolor_filt}, virginica : {virginica_filt}")

"""


"""
species_count = {"setosa": 0, "versicolor": 0, "virginica": 0}
for row in all_data:
    species = row["species"]
    if species in species_count:
        species_count[species] += 1
with open("Results.txt", "w") as file:
    file.write(
        ", ".join(f"{species}={count}" for species, count in species_count.items())
    )
"""


# 2
filtered_species_count = {species: 0 for species in species_count}  # 모든 0을 초기화
# ? filtered_species_count = {"setosa": 0, "versicolor": 0, "virginica": 0}
for row in all_data:
    sepal_length = float(row["sepal_length"])
    petal_length = float(row["petal_length"])
    species = row["species"]

    if sepal_length >= 4.8 and petal_length > 2:
        filtered_species_count[species] += 1

with open("Results.txt", "a") as file:
    file.write(
        ", ".join(
            f"{species}={count}" for species, count in filtered_species_count.items()
        )
    )
    file.write("\n")


# 3
filtered_rows = []

for row in all_data:
    sepal_width = float(row["sepal_width"])
    petal_length = float(row["petal_length"])
    petal_width = float(row["petal_width"])

    if sepal_width > 3 and petal_length < 6 and petal_width >= 0.8:
        filtered_rows.append(row)

with open("Results.txt", "a") as file:

    file.write("\t".join(header) + "\n")
    for row in filtered_rows:
        file.write("\t".join(row[col] for col in header) + "\n")
