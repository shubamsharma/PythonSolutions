
# Check if combination be dereived from parent string

from collections import Counter

def checkPer(check1, tocompare):
    
    if(len(check1) != len(tocompare)):
        return(False)
    else:
        tempc = Counter(check1)
        tempto = Counter(tocompare)

        if(tempc == tempto):
            return(True)
        else:
            return(False)
    return(True)
s = 'cab'
o = 'ccc'

output = checkPer(s,o)
print(output)

# Print triangle - Showing iterator 

no_of_elements = 6
output = ([ m for n in range(m)] for m in range(no_of_elements+1) if m%2==0 )
print(next(output))