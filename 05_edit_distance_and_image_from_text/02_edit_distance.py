def editDistDP(str1, str2): 
    m = len(str1)
    n = len(str2)
    mat = [[0 for x in range(n + 1)] for x in range(m + 1)] 
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0: 
                mat[i][j] = j
            elif j == 0: 
                mat[i][j] = i
            elif str1[i-1] == str2[j-1]: 
                mat[i][j] = mat[i-1][j-1] 
            else: 
                mat[i][j] = 1 + min(mat[i][j-1],
                                   mat[i-1][j],
                                   mat[i-1][j-1])
  
    return mat[m][n] 

stringArray = ["sunday","saturday","sund","sunway"]

THRESHOLD = 2
stringDict = {}
for stringComp in stringArray :
    tempstringComp = [x for x in stringArray if stringComp != x]
    stringMinDistCount = {}
    for stringToBeComp in tempstringComp : 
        edit_distance = editDistDP(stringComp, stringToBeComp)
        if edit_distance <= THRESHOLD : 
            stringMinDistCount[stringToBeComp] = edit_distance
    if stringMinDistCount :
        stringDict[stringComp] = stringMinDistCount


for key in stringDict : 
    print ("String \"{}\" with edit distance less than {} are {}".format(key,THRESHOLD,list(stringDict[key].keys())))

    
