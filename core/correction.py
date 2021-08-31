
# 국가 보정
def nationCorrection(value):
    # 국가명 파일 로드
    f = open("res/nationality.txt", 'r')
    nationality = []
    while True:
        line = f.readline()
        if not line: break
        nationality.append(line)
    f.close()

    # 글자수 체크
    if len(value) != 3: return value

    # 국가명 확인
    for nation in nationality:
        if nation == value:
            return value

    strFront = value[0:2]
    strBack = value[1:]
    strMiddle = value[0] + value[2]
    if strFront == 'KO': return 'KOR'
    if strBack == 'OR': return 'KOR'
    if strMiddle == 'KR': return 'KOR'

    count, resultNation = 0, ''

    # 앞에 두자리 맞으면 비슷한 국가 출력
    for nation in nationality:
        if len(nation) != 3: continue
        if count > 1: return nation  # 오탐일 경우 재 탐색하는 기능 여부에 따라 수정
        if strFront == nation[0:2]:
            count += 1
            resultNation = nation

    if count == 1: return resultNation
    count, resultNation = 0, ''

    # 뒤의 두자리 맞으면 비슷한 국가 출력
    for nation in nationality:
        if len(nation) != 3: continue
        if count > 1: return nation   # 오탐일 경우 재 탐색하는 기능 여부에 따라 수정
        if strBack == nation[1:]:
            count += 1
            resultNation = nation

    if count == 1: return resultNation
    count, resultNation = 0, ''

    # 중간만 틀렸을 때 비슷한 국가 출력
    for nation in nationality:
        nation = nation[0] + nation[2]
        if len(nation) != 3: continue
        if count > 1: return nation  # 오탐일 경우 재 탐색하는 기능 여부에 따라 수정
        if strMiddle == nation:
            count += 1
            resultNation = nation

    if count == 1: return resultNation
    return value


# mrz 영어, 숫자 보정
def mrzCorrection(value, flag):
    if flag == 'en2dg':
        return value.replace('O', '0').replace('Q', '0').replace('U', '0').replace('D', '0')\
            .replace('I', '1').replace('Z', '2').replace('B', '3').replace('A', '4').replace('S', '5')
    else:
        return value.replace('0', 'O').replace('1', 'I').replace('2', 'Z').replace('3', 'B')\
            .replace('4', 'A').replace('8', 'B')


# 성별 보정
def sexCorrection(value):
    return value.replace('P', 'F').replace('T', 'F').replace('N', 'M')


# 여권 타입 보정
def typeCorrection(value):
    return value.replace('FM', 'PM').replace('PN', 'PM')