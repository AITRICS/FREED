export PATH="bin:$PATH" # for docking use

CUDA_LAUNCH_BLOCKING=1 python run_rl.py --name='docking_ac4_gcn_ecfp_parp1_rand92_alr5e-4_mcv4std_141' \
                       --load=0 --train=1 --has_feature=1 --is_conditional=0 \
                       --name_full_load='' \
                       --min_action=1 --max_action=4 \
                       --graph_emb=1 --gcn_aggregate='sum' --gcn_type='GCN' \
                       --reward_type='crystal' \
                       --seed=141 --adv_rew=0 --intr_rew=0 --intr_rew_ratio=5e-1 --target='parp1' \
                       --update_after=3000 --start_steps=4000  --update_freq=256  --init_alpha=1. \
                       --desc='ecfp' \
                       --rl_model='sac' \
                       --target_entropy=3. \
                       --active_learning='mc' \
                       --gpu_id=7 --emb_size=64 --tau=1e-1 --batch_size=256 


# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
# def get_set(S):
#     char1 = set()
#     char2 = set()
#     for s in S:
#         if s not in char2:
#             if s not in char1:
#                 char1.add(s)
#             else:
#                 char2.add(s)
#                 char1.remove(s)
#     return char1


# def solution(S):
#     # write your code in Python 3.6
#     res = 0
#     for i in range(len(S)-1):
#         for j in range(i+1, len(S)+1):
#             res += len(get_set(S[i:j]))

#     res += 1

#     return res

# def solution(S):
#     ret = [0] * (len(S) + 1)
#     chars = [[-1,-1]]*26
#     for i, c in enumerate(S):
#         char_idx = ord(c)-ord('A')
#         first, second = chars[char_idx]
#         ret[i+1] = 1 + ret[i] + (i-1-second) - (second-first)
#         chars[char_idx]=[second, i]
#     return sum(ret)
        

# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
# def check(a):
#     ret = True
#     for i in range(len(a)-1):
#         if a[i]>= a[i+1]:
#             ret = False
#     return ret

# def solution(A):
#     # write your code in Python 3.6
#     res = 0
#     max_size = 0
#     for i in range(len(A)):
#         for j in range(i+1, len(A)+1):
            
#             if check(A[i:j]):
#                 # res = max(res, j-i)
#                 if j-i > max_size:
#                     max_size = j-i
#                     res = i
#     return res

# def solution(A):
#     res = 0
#     maxi = 1
#     leng = 1
#     for i in range(1, len(A)):
#         if A[i] > A[i-1]:
#             leng += 1
#         else:
#             if maxi < leng:
#                 maxi = leng
#                 res = i
#     if maxi < leng:
#         maxi = leng
#         res = len(A)
#     print(res)
#     return res

# def solution(A):
#     # write your code in Python 3.6
#     res = 0
#     max_size = 0
#     for i in range(len(A)):
#         for j in range(i+1, len(A)):
#             if A[j-1] >= A[j]:
#                 break
#             else:
#                 if j-i+1 > max_size:
#                     max_size = j-i+1
#                     res = i                    
#     return res
        
