from functools import reduce

def is_digit(n):
    try:
        int(n)
        return(True)
    except ValueError:
        return(False)

def validate_array(func):
    def inner(A):
        print("Recieved array ", A)
        validate_arr = all([is_digit(s) == True for s in A])
        if not validate_arr :
            print("Invalid input, returning 0 and 0")
            return(0,0)
        return func(A)
    return inner

@validate_array
def min_max_no(listA) :
    maxNo = reduce(lambda x, y: x if x > y else y, listA)
    minNo = reduce(lambda x, y: x if x < y else y, listA)
    print("Max - {}, Min {}".format(maxNo,minNo))
    return( maxNo, minNo)

#A = [-1, 3, 7, 99, 99]
#x,y = min_max_no(A)
#print(x,y)