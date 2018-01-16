# -*- coding:utf-8 -*-
import sys


# 编辑距离计算
def levenshtein(first, second):
    if len(first) > len(second):
        first, second = second, first
    if len(first) == 0:
        return len(second)
    if len(second) == 0:
        return len(first)

    # print first,second;
    temp = 0
    if second.startswith(first):
        # print "Start";
        temp = 0.1;

    first_length = len(first) + 1
    second_length = len(second) + 1
    distance_matrix = [list(range(second_length)) for x in range(first_length)]
    # print distance_matrix
    for i in range(1, first_length):
        for j in range(1, second_length):
            deletion = distance_matrix[i - 1][j] + 1
            insertion = distance_matrix[i][j - 1] + 1
            substitution = distance_matrix[i - 1][j - 1]
            if first[i - 1] != second[j - 1]:
                substitution += 1
            if i > 1 and j > 1 and first[i - 1] == second[j - 2] and first[i - 2] == second[j - 1]:
                exchange = distance_matrix[i - 2][j - 2] + 1;
            else:
                exchange = 100;
            distance_matrix[i][j] = min(insertion, deletion, substitution, exchange)
            # print distance_matrix
    return distance_matrix[first_length - 1][second_length - 1] - temp;


def InputLegitimacy(str):  # input legitimacy judgment
    for char in str:
        if char >= 'a' and char <= 'z':
            continue
        else:
            return 0
    return 1


def OutputProcess(result, outputlen):
    for item in result:
        print(item)
    inputlen = len(result)
    output = {}
    split_sym_count = 0
    for item in result:
        if item == "'":
            split_sym_count += 1
    for i in range(inputlen - split_sym_count):
        for j in range(outputlen):
            output[(i, j)] = 0
    targetnum = 0
    placeholder = 0
    for i in range(inputlen):
        if result[i] == "'":
            targetnum += 1
            placeholder += 1
            continue
        output[(i - placeholder, targetnum)] = 1
        # print((i - placeholder, targetnum))
        # print(output[(i - placeholder, targetnum)])
    return output


def CutPy(line):
    # Init
    lineSplit = line.strip().split()
    pinyin = lineSplit[1]
    input = lineSplit[0]
    if InputLegitimacy(input) == 0 or len(input) > 20:
        return "error", "error", "error"
    count = 1
    begin = []
    begin.append(pinyin[0])
    beginpos = []
    beginpos.append(0)
    pinyin_bake = ""
    pyarray = []
    single = ""

    # get split pinyin
    listTemp = pinyin.split("'")
    listPy = []
    for item in listTemp:
        if len(item) > 0:
            listPy.append(item)
            # print item;
    # get the split first char
    for i in range(len(pinyin)):
        if pinyin[i] == "'":
            if len(single) > 0:
                pyarray.append(single)
                single = ""
            if i == (len(pinyin) - 1) or pinyin[i + 1] == "'":
                return "error", "error", "error"
            begin.append(pinyin[i + 1])
            beginpos.append(i + 1 - count)
            count += 1
        elif pinyin[i] < "a" or pinyin[i] > "z":
            # print "INPUT ERROR!";
            print("EndJP")
            return "error", "error", "error"
        else:
            pinyin_bake = pinyin_bake + (pinyin[i])
            single = single + (pinyin[i])
        if i == (len(pinyin) - 1):
            pyarray.append(single)
    return input, pinyin, count


def DieCi(string):
    lastchar = ""
    for char in string:
        if lastchar == char:
            return 1
        lastchar = char
    return 0


def EditDistance(splitresult, topword):
    # 计算切分结果与注音结果之间的编辑距离
    score = 0
    splitlist = splitresult.split("'")
    toplist = topword.split("'")
    if len(splitlist) != len(toplist):
        #		print splitlist,toplist
        return 100
    for i in range(len(splitlist)):

        temp = levenshtein(splitlist[i], toplist[i]);
        if temp == 0:
            temp -= 0.5
        score += temp
    # print splitresult,"\t",score
    return score


def Itetator(beginnum, input, userpinyin, splitnum, splitresult, scorelist, resultlist):
    '''
	通过迭代不断在上一次切分基础上对剩余字段进行处理
	beginnum 上一次切分的结尾位置，剩余字段从 beginnum+1 开始
	input 用户输入
	userpinyin 注音结果
	splitnum 注音结果一共分成了几段-1，即有多少个\'
	splitresult 用于保存当前已经完成切分的字符串
	'''
    splitnum -= 1;
    if beginnum != 0:
        splitresult += "'"
    if beginnum == (len(input) - 1) or splitnum <= 0:
        # 剩余字段不足以被切分为制定个数，或者待切分个数为0
        splitresult += input[beginnum:len(input)]
        score = EditDistance(splitresult, userpinyin)
        scorelist.append(score)
        resultlist.append(splitresult)
    else:
        endpos = beginnum + 7 if beginnum + 7 < len(input) else len(input);
        for i in range(beginnum + 1, endpos):  # 切分窗口[beginnum,i],i最大值为endpos
            if (len(input) - beginnum) <= (splitnum - 1):  # 剩余部分不足以被拆分为指定段
                continue
            else:
                # print input[beginnum:i]
                splitresult += input[beginnum:i]
                beginnum = i
                Itetator(beginnum, input, userpinyin, splitnum, splitresult, scorelist, resultlist);


def MultipleOutput(list, userpinyin, targetlen):
    '''
		对有多个匹配的情况，增加了首字符匹配，叠词识别，全拼输入识别过程
	'''
    initialchar = ""
    qiefentopcharlist = []
    Disscorelist = []
    pinyinList = userpinyin.split("'")
    for item in pinyinList:
        initialchar += item[0]
    for result in list:
        qiefentopchar = ""
        Result = result.split("'")
        for yinjie in Result:
            if len(yinjie) == 0:
                continue
            qiefentopchar += yinjie[0]
        qiefentopcharlist.append(qiefentopchar)
    for topchar in qiefentopcharlist:
        if len(topchar) == len(initialchar):
            Disscorelist.append(levenshtein(topchar, initialchar))
    MinScore = min(Disscorelist)
    ResultList = []
    for i in range(len(Disscorelist)):
        if Disscorelist[i] == MinScore:
            ResultList.append(list[i])
    if len(ResultList) == 1:
        return OutputProcess(ResultList[0], targetlen)
        # print((str(ResultList[0]) + "\t"), end=' ')
    else:
        Step2Result = []
        for words in ResultList:
            if DieCi(words) == 1:
                Step2Result.append(words)
        if len(Step2Result) == 1:
            return OutputProcess(Step2Result[0], targetlen)
        else:
            CompareCountList = [0 for item in ResultList]
            for i in range(len(ResultList)):
                SplitResult = ResultList[i].split("'")
                for fragment in SplitResult:
                    if fragment in pinyinList:
                        CompareCountList[i] += 1
            MaxCompareScore = max(CompareCountList)
            for i in range(len(CompareCountList)):
                if CompareCountList[i] == MaxCompareScore:
                    return OutputProcess(ResultList[i], targetlen)
                    # print ResultList[i],"\t",


def calc_segment(userinput, userpinyin):
    linenum = 0
    scorelist = []
    resultlist = []
    inputcount = 1
    for i in range(len(userpinyin)):
        if userpinyin[i] == "'":
            inputcount += 1
    result = ""
    Itetator(0, userinput, userpinyin, inputcount, result, scorelist, resultlist)  # 切分
    if len(scorelist) == 0:
        print("Wrong Split!")
        return
    minscore = min(scorelist)  # 找打分结果最小值
    list = []
    for i in range(len(scorelist)):
        if scorelist[i] == minscore:
            list.append(resultlist[i])
    print(list)
    if (len(list) == 1):
        return OutputProcess(list[0], inputcount)
    else:
        return MultipleOutput(list, userpinyin, inputcount)

#
#if __name__ == '__main__':
#    print(calc_segment("chaor", "cho'ren"))
