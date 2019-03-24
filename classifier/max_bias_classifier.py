import os
import random

file_dir = 'bias_result'

f_max = []
f_avg = []
t_max = []
t_avg = []

for file in os.listdir(file_dir):
    file_name = os.path.join(file_dir, file)
    with open(file_name, 'r') as f:
        m, a = map(float, f.read().strip().split("\t"))
        if m == 0.0:
            continue
        if file.__contains__("fake"):
            f_max.append(m)
            f_avg.append(a)
        else:
            t_max.append(m)
            t_avg.append(a)

def cal_f1():
    random.shuffle(f_max)
    random.shuffle(f_avg)
    random.shuffle(t_max)
    random.shuffle(t_avg)

    f_max_train = f_max[:1000]
    f_max_test = f_max[1000:]
    f_avg_train = f_avg[:1000]
    f_avg_test = f_avg[1000:]
    t_max_train = t_max[:1000]
    t_max_test = t_max[1000:]
    t_avg_train = t_avg[:1000]
    t_avg_test = t_avg[1000:]

    f_max_train = sorted(f_max_train)
    f_avg_train = sorted(f_avg_train)
    f_max_test = sorted(f_max_test)
    f_avg_test = sorted(f_avg_test)

    t_max_train = sorted(t_max_train)
    t_avg_train = sorted(t_avg_train)
    t_max_test = sorted(t_max_test)
    t_avg_test = sorted(t_avg_test)
    classifier(f_max_train, f_max_test, t_max_train, t_max_test)
    print('------------')
    classifier(f_avg_train, f_avg_test, t_avg_train, t_avg_test)

def classifier(f_max_train, f_max_test, t_max_train, t_max_test):
    tmp = 0.0
    index_f = 0
    index_t = 0

    max_f1 = 0.0
    max_p = 0.0
    max_r = 0.0

    p_val = 0.0
    r_val = 0.0
    f1_val = 0.0

    for i in range(len(f_max_train)):
        index_f = i
        for j in range(len(t_max_train)):
            if t_max_train[j] >= f_max_train[i]:
                index_t = j
                break
        tp = index_t + 1
        fn = len(t_max_train) - index_t + 1
        fp = index_f + 1
        tn = len(f_max_train) - index_f + 1

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)

        if f1 > max_f1:
            max_f1 = f1
            f1_val = f_max_train[i]
        if p > max_p:
            max_p = p
            p_val = f_max_train[i]
        if r > max_r:
            max_r = r
            r_val = f_max_train[i]

    print(max_f1, f1_val)
    print(max_p, p_val)
    print(max_r, r_val)

    tp = 1
    fn = 1
    fp = 1
    tn = 1
    for num in f_max_test:
        if num <= f1_val:
            fp += 1
    tn = len(f_max_test) - fp + 1
    for num in t_max_test:
        if num <= f1_val:
            tp += 1
    fn = len(t_max_test) - tp + 1

    test_p = tp / (tp + fp)
    test_r = tp / (tp + fn)
    test_f1 = 2 * test_p * test_r / (test_p + test_r)

    print(f_max_test)
    print(t_max_test)
    print(test_p, test_r, test_f1)


cal_f1()