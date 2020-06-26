import collections
from collections import Counter

#S = [1,8,2,1,4,1,2,9,1,8,4,2,4]

S = [2,1,3,1,4,1,3]

init_dict = dict(Counter(S))

d = dict()

for i in range(len(S)-1):
    m = list()
    temp_dict = dict.fromkeys(init_dict,0)
    m.append(S[i])
    temp_dict[S[i]] += 1
    for j in (range(i,len(S)-1)):
        if(j+2>len(S)-1):
            if m[-1] <= S[j+1]: 
                m.append(S[j+1])
                temp_dict[S[j+1]] += 1
        else :          
            if S[j+1] >= m[-1]: 
                if S[j+2] <= S[j+1]:
                    if S[j+2] < m[-1]:
                        m.append(S[j+1])
                        temp_dict[S[j+1]] += 1
                    else : 
                        next
                else:
                    m.append(S[j+1])
                    temp_dict[S[j+1]] += 1
    if len(m) > 1:
        d[i]= [m,temp_dict]


#final_count = len(init_dict.keys())
for final_loop in range(len(d)-1):
    sub_df = { key :  init_dict[key] - d[final_loop][1].get(key,0) for key in init_dict}
    if all(value == 1 for value in sub_df.values()):
        print(d[final_loop][0])




    




