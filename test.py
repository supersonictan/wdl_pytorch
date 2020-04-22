# coding: UTF-8
#! /usr/bin/python


import re,math
import json
import numpy as np


# pat_company = re.compile('(:.*?)(,|$)')
pat_company2 = re.compile('(,|^)(.*?:)')

seg = '电视:NN 连续剧:NN 伪:VV 奘:NN 者:NN'

if __name__ == '__main__':
    # text1 = '笑:OPTIONAL_LOW,功:OPTIONAL_LOW,峥:OPTIONAL_LOW,武林:OPTIONAL_LOW'
    # text = '验光:OPTIONAL_LOW,仪:OPTIONAL_LOW,的:OPTIONAL_HIGH,使用:OPTIONAL_LOW,方法:QUESTION'
    # new = re.sub(pat_company2, ' ', text)
    # print(new)

    # pos_term_list = seg.split(' ')
    # pos_dic = {item.split(':')[0]: item.split(':')[1] for item in pos_term_list}
    #
    # print(pos_dic)

    qt_term = '钩针:OPTIONAL_LOW,编织:OPTIONAL_LOW,宝宝:OPTIONAL_LOW,围巾:OPTIONAL_LOW'
    qt_term_list = qt_term.split(',')
    qt_dic = {item.split(':')[0]: item.split(':')[1] for item in qt_term_list}

    print(qt_dic)