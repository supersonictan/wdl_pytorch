

if __name__ == '__main__':
    seg_str = '南:NN 柱:NN 赫:NR 李圣经:NR'
    seg_str = seg_str
    seg_list = seg_str.split(' ')
    term_pos_dic = [{item.split(':')[0], item.split(':')[1]} for item in seg_list]

    print(term_pos_dic)