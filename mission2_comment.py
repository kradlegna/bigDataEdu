# 1. iris.csv 파일을 열어서 데이터를 한 번에 읽고 저장
all_data = ()  # 꽃 정보를 담을 리스트 만들기

with open("iris.csv", "r") as file:  # iris파일을 열어서 file이라는 이름으로 사용
    header = (
        file.readline().strip().split(",")
    )  # 첫 줄 읽어서 공백은 없애고 쉼표로 나눠서 제목 리스트 만들어
    for line in file:  # 두 번째 줄부터 마지막 줄까지 한 줄씩 반복해서 읽어
        values = line.strip().split(",")  # 읽은 줄은 공백 없애고 쉼표로 기준으로 나눔
        all_data.append(
            dict(zip(header, values))
        )  # 헤더와 값들을 짝 지어서 딕셔너리로 저장해

# 2. 모든 꽃 종류가 몇 개씩 있는지 세기
species_count = {}  # 종 이름별로 개수를 세기 위해 빈 딕셔너리를 만들어

for row in all_data:
    species = row["species"]  # 각 줄에서 species, 어떤 종류인지만 꺼내와
    species_count[species] = (
        species_count.get(species, 0) + 1
    )  # 키가 딕셔너리에 아직 없으면 0을 주고 +1, 있으면 그 값에 +1

with open("Results.txt", "w") as file:  # 첫 번째는 "w"로 씀 (초기화)
    file.write(
        ", ".join(
            f"{species}={count}" for species, count in species_count.items()
        )  # item이 딕셔너리를 [(키,값),(키,값),]이렇게 쌍으로 나오게
    )
    file.write("\n")  # 구분용 줄바꿈


# 3. 조건1: sepal_length ≥ 4.8 and petal_length > 2 만족하는 종 개수
filtered_species_count = {
    species: 0 for species in species_count
}  # species_count에 이미 모든 종의 목록 들어있어서 그 값들을 0으로 초기화

for row in all_data:
    sepal_length = float(row["sepal_length"])
    petal_length = float(row["petal_length"])
    species = row["species"]

    if sepal_length >= 4.8 and petal_length > 2:
        filtered_species_count[species] += 1

with open("Results.txt", "a") as file:  # 이후는 반드시 "a" 모드 (이어쓰기)
    file.write(
        ", ".join(
            f"{species}={count}" for species, count in filtered_species_count.items()
        )
    )
    file.write("\n")

# 4.조건2 sepal_width > 3 and petal_length < 6 and petal_width >= 0.8:
filtered_rows = []  # 전체 데이터를 나중에 출력할꺼니까 리스트

for row in all_data:
    sepal_width = float(
        row["sepal_width"]
    )  # 딕셔너리에서 꺼낸 값이 문자열이라 숫자형으로 바꿔
    petal_length = float(row["petal_length"])
    petal_width = float(row["petal_width"])

    if sepal_width > 3 and petal_length < 6 and petal_width >= 0.8:
        filtered_rows.append(row)  # 조건에 맞는 줄만 리스트에 넣기

with open("Results.txt", "a") as file:  # 이어쓰기 모드

    file.write("\t".join(header) + "\n")  # 헤더 출력
    for row in filtered_rows:  # 조건에 맞는 줄만 하나씩 꺼내
        file.write(
            "\t".join(row[col] for col in header) + "\n"
        )  # header에 들어있는 열이름(col)을 순서대로 꺼내서
