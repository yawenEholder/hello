#coding=utf-8
from math import sqrt
#import sys
import random
from PIL import Image,ImageDraw,ImageFont
import pylab as pl
import numpy as np
from sklearn import cluster
#import matplotlib.pyplot as plt
#import imp
#imp.reload(sys)
#sys.setdefaultencoding('utf-8')
'''reload(sys)
sys.setdefaultencoding("utf-8")'''

def pearson(v1, v2):
    #定义紧密度
    #  简单求和
    sum1 = sum(v1)
    sum2 = sum(v2)
    sum1Sq = sum([v*v for v in v1])
    sum2Sq = sum([v*v for v in v2])
    pSum = sum({v1[i] * v2[i] for i in range(len(v1))})
    #Calculate Pearson Score
    num = pSum - (sum1 * sum2/len(v1))
    #print sum1, sum2, sum1Sq, sum2Sq, pSum, num
    den = sqrt(sum1Sq - pow(sum1, 2)/len(v1)) * (sum2Sq - pow(sum2, 2)/len(v1))
    if den == 0:
        return 0
    # 返回1.0减去皮尔逊相关度之后的结果，这样做的目的是为了让相似度越大的两个元素之间距离变得更小
    return 1.0 - num/den

def cosdis(v1, v2):
    #余弦值越接近1，越相似cos((x1,y1),(x2,y2))=x1x2+y1y2/(sqrt(x12+y12)*sqrt(x22+y22))
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(v1, v2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return 1.0-dot_product / ((normA * normB) ** 0.5)#使相似度越大的距离越小
    
def tanimoto(v1,v2):
    #Tanimoto系数：代表交集与并集之间的比率
    c1,c2,shr=0,0,0
    for i in range(len(v1)):
        if v1[i]!=0:
            c1+=1
        if v2[i]!=0:
            c2+=1
        if v1[i]!=0 and v2[i]!=0:
            shr+=1
    return 1.0-(float(shr)/(c1+c2-shr))
class bicluster:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance
def hcluster(rows, distance=pearson):
    distances = {}
    currentclustid = -1
    #最开始聚类的是数据集中的行
    clust = [bicluster(rows[i], id=i) for i in range(len(rows))]
    while len(clust)>1:
        lowestpair=(0,1)
        for item in range(len(clust[0].vec)):
            (clust[0].vec)[item] = int((clust[0].vec)[item])
        for item in range(len(clust[1].vec)):
            (clust[1].vec)[item] = int((clust[1].vec)[item])
        #print clust[0].vec, type(clust[0].vec)
        #print clust[1].vec, type(clust[1].vec)
        closest=distance(clust[0].vec, clust[1].vec)
        #遍历每一个配对，寻找最小距离
        for i in range(len(clust)):
            for j in range(i+1, len(clust)):
                #用distances来缓存距离的计算值
                for item in range(len(clust[i].vec)):
                    (clust[i].vec)[item] = int((clust[i].vec)[item])
                for item in range(len(clust[j].vec)):
                    (clust[j].vec)[item] = int((clust[j].vec)[item])
                distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)

                d = distances[(clust[i].id, clust[j].id)]
                if d < closest:
                    closest = d
                    lowestpair = (i, j)
        # 计算两个聚类的平均值
        mergevec = [
            (clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i])/2.0 for i in range(len(clust[0].vec))]
        # 建立新的聚类
        newcluster = bicluster(mergevec, left=clust[lowestpair[0]], right=clust[lowestpair[1]], distance=closest, id=currentclustid)
        # 不在原始集合中的聚类，其id为负数
        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)
    return clust[0]
def kcluster(rows,distance=cosdis, k=4):
    #确定每个点的最大值和最小值
    ranges=[(min([row[i] for row in rows]),max([row[i] for row in rows])) for i in range(len(rows[0]))]
    # 随机创建k个中心点
    #print (float(ranges[i][1]) - float(ranges[i][0]))
    ra = random.random()
    print('random', ra)
    clusters=[[ra*(float(ranges[i][1])-float(ranges[i][0]))+float(ranges[i][0]) for i in range(len(rows[0]))] for j in range(k)]
    lastmatches=None
    for t in range(100):
        print('Iteration %d' % t)
        bestmatches=[[] for i in range(k)]
        #在每一行中寻找距离最近的中心点
        for j in range(len(rows)):
            row=[]
            for item in rows[j]:
                row.append(int(item))
            #row=rows[j]
            bestmatch=0
            for i in range(k):
                #print 'i', i
                #print 'row', row
                #print 'clusters[i]', clusters[i]
                d=distance(clusters[i], row)
                #print 'd', d
                if d<distance(clusters[bestmatch],row):
                    #print 'bestmatch', distance(clusters[bestmatch], row)
                    bestmatch=i
            bestmatches[bestmatch].append(j)
            #print bestmatches
        #如果结果与上一次相同，整个过程结束
        if bestmatches==lastmatches:
            break
        lastmatches=bestmatches
        #把中心点移到所有成员的平均位置处
        for i in range(k):
            avgs = [0.0]*len(rows[0])
            if len(bestmatches[i])>0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        #print type(rows[rowid][m]), rows[rowid][m]
                        #print type(avgs[m]), avgs[m]
                        avgs[m]+=float(rows[rowid][m])
                for j in range(len(avgs)):
                    avgs[j]/=len(bestmatches[i])
                clusters[i]=avgs
                clusters[i]=avgs
    return ra, bestmatches
def printclust(clust, labels=None,n=0):
    for i in range(n): print ' ',
    if clust.id < 0:
        print('-')
    else:
        if labels == None:
            print(clust.id)
        else:
            print(labels[clust.id])
    if clust.left!=None:
        printclust(clust.left, labels=labels, n=n+1)
    if clust.right!=None:
        printclust(clust.right, labels=labels, n=n+1)
def getheight(clust):
    #这是一个叶节点吗？若是，高度为1
    if clust.left == None and clust.right == None:
        return 1
    #否则，高度为每个分支的高度之和
    return getheight(clust.left)+getheight(clust.right)
def getdepth(clust):
    #一个叶节点的距离是0.0
    if clust.left==None and clust.right==None:
        return 0
    #一个枝节点的距离的能够与左右两侧分支中距离较大者，加上该枝节点自身距离
    return max(getdepth(clust.left),getdepth(clust.right))+clust.distance
def drawnode(draw,clust,x,y,scaling,labels):
    font1 = ImageFont.truetype('simsun.ttc', 24)
    if clust.id<0:
        h1=getheight(clust.left)*20
        h2=getheight(clust.right)*20
        top=y-(h1+h2)/2
        bottom=y+(h1+h2)/2
        #线的长度
        l1=clust.distance*scaling
        #聚类到其子节点的垂直线
        draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))
        #连接左侧节点的水平线
        draw.line((x,top+h1/2,x+l1,top+h1/2),fill=(255,0,0))
        #连接右侧节点的水平线
        draw.line((x,bottom-h2/2,x+l1,bottom-h2/2),fill=(255,0,0))
        #调用函数绘制左右节点
        drawnode(draw,clust.left,x+l1,top+h1/2,scaling,labels)
        drawnode(draw,clust.right,x+l1,bottom-h2/2,scaling,labels)
    else:
        #如果这是一个叶节点，则绘制节点的标签
        print(labels[clust.id])
        draw.text((x+5,y-7),labels[clust.id],(0,0,0),font=font1)

def drawdendrogram(clust,labels,jpeg='hcluseters.jpg'):
    #高度和宽度
    h=getheight(clust)*20
    w=1200
    depth=getdepth(clust)
    #由于宽度是固定的，因此我们需要对距离值作相应的调整
    scaling=float(w-150)/depth
    #新建一张白色背景的图片
    img=Image.new('RGB',(w,h),(255,255,255))
    draw=ImageDraw.Draw(img)
    draw.line((0,h/2,10,h/2),fill=(255,0,0))
    #画第一个节点
    drawnode(draw,clust,10,(h/2),scaling,labels)
    img.save(jpeg,'JPEG')

def scaledown(data,distance=cosdis,rate=0.01):
    # 接受一个数据向量作为参数，并返回一个只包含两列的向量，即数据项在二维图上的横坐标和纵坐标
    n=len(data)
    #将字符转化为整数进行计算
    for i in range(n):
        m=len(data[i])
        for j in range(m):
            data[i][j]=int(data[i][j])
    print(data[2])
    print(type(data[2]))
    # 对每一对数据项之间的真实距离
    realdist=[[distance(data[i],data[j]) for j in range(n)] for i in range(0,n)]
    outersum=0.0
    #随机初始化节点在二维空间中的起始位置
    loc=[[random.random(),random.random()] for i in range(n)]
    fakedist=[[0.0 for j in range(n)] for i in range(n)]
    lasterror=None
    for m in range(0,1000):
        #寻找投影后的距离
        for i in range(n):
            for j in range(n):
                fakedist[i][j]=sqrt(sum([pow(loc[i][x]-loc[j][x],2) for x in range(len(loc[i]))]))
        # 移动节点
        grad=[[0.0,0.0] for i in range(n)]
        totalerror=0
        for k in range(n):
            for j in range(n):
                if j==k:
                    continue
                #误差值等于目标距离与当前距离之间chazhi的百分比
                if realdist[j][k]!=0:
                    errorterm=(fakedist[j][k]-realdist[j][k])/realdist[j][k]
                else:
                    errorterm==fakedist[j][k]
                #每一个节点都需要根据误差的多少，按比例移离或移向其他节点
                grad[k][0]+=((loc[k][0]-loc[j][0])/fakedist[j][k]*errorterm)
                grad[k][1]+=((loc[k][1]-loc[j][1])/fakedist[j][k]*errorterm)
                #记录总的误差值
                totalerror+=abs(errorterm)
        print(totalerror)
        #如果节点移动之后的情况变得更糟，则程序结束
        if lasterror and lasterror<totalerror:
            break
        lasterror=totalerror
        #根据rate参数与grad值相承的结果，移动每一个节点
        for k in range(n):
            loc[k][0]-=rate*grad[k][0]
            loc[k][1]-=rate*grad[k][1]
    return loc
def draw2d(data,labels,jpeg='mds2d.jpg'):
    img=Image.new('RGB',(5000,5000),(255,255,255))
    draw=ImageDraw.Draw(img)
    print(len(data))
    for i in range(len(data)):
        x=(data[i][0]+0.5)*100
        y=(data[i][1]+0.5)*100
        draw.text((x,y),labels[i],(0,0,0))
        print(labels[i],(x,y))
    img.save(jpeg,'JPEG')
def pltdraw(data,labels,clust):
    color =  ['or', 'ob', 'og', 'oy' ,'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(len(data)):
        x=(data[i][0]+0.5)*100
        y=(data[i][1]+0.5)*100
        for j in range(10):
            if labels[i] in clust[j]:
                print(labels[i],(x,y),color[j])
                pl.plot(x,y,color[j])
    pl.show()
    print ('a new function')
def test():
    blognames, words, data = readfile('test.txt')
    clust=hcluster(data)
    #printclust(clust, labels=blognames)
    drawdendrogram(clust,blognames,jpeg='hcla3.jpg')
    '''ra, kclust=kcluster(data, k=2)
    print(kclust)
    print ([blognames[r] for r in kclust[0]])
    print ([blognames[r] for r in kclust[1]])
    #print ([blognames[r] for r in kclust[2]])
    f = open('kclu.txt', 'a')
    f.write('\n')
    f.write(str(ra))
    f.write('   ')
    f.write(str(kclust))'''
def test2():
    mblogids,words,data=readfile('matrix_a.txt')
    d = 11
    ra,kclust=kcluster(data,k=d)
    print(kclust)
    f=open('kclu.txt','a')
    f.write('\n')
    f.write('test2:k='+str(d)+'\n')
    f.write(str(ra))
    f.write('\n')
    for i in range(d):
        print([mblogids[r] for r in kclust[i]])
        for r in kclust[i]:
            f.write(mblogids[r]+'  ')
        f.write('\n')
    #f.write(str(kclust))
    f.close()
def test3():
    blognames, words, data = readfile('matrix2.txt')
    clust = hcluster(data)
    printclust(clust, labels=blognames)
def test4():
    blogname,words,data=readfile('matrix_cla3.txt')
    coords=scaledown(data)
    draw2d(coords,blogname,jpeg='cla1.jpg')
def test5():
    blogname, words, data = readfile('matrix_a.txt')
    coords = scaledown(data)
    ra, kclust = kcluster(data, k=10)
    clust = []
    for i in range(10):
        # print([blogname[r] for r in kclust[i]])
        clust.append([])
        for r in kclust[i]:
            clust[i].append(blogname[r])
    print(clust)
    print (coords)
    pltdraw(coords,blogname,clust)
def testkmeans(d):
    keywords, weiboid, data = readfile('test.txt')
    print(keywords)
    print(weiboid)
    print(data)
    data2 = []
    for line in data:
        data2.append([int(item) for item in line])
    arr = np.array(data2)
    print(arr.dtype)
    kmeans = cluster.KMeans(d)
    s = kmeans.fit(arr)
    print(s)
    kclust = kmeans.labels_
    center = kmeans.cluster_centers_#中心店
    number = kmeans.inertia_#评估簇的数目是否合适，距离越小说明簇越好
    print(number)
    print(kclust)
    print(type(kmeans.labels_))#使用kmeans
    print(kclust.shape, kclust.dtype)
    wordlist = []
    for i in range(len(keywords)):
        wordlist.append([keywords[i],kclust[i]])
    print(wordlist)
    f = open('kclu.txt', 'a')
    f.write('\n')
    f.write('test_Kmeans_of_sklearn: k=' + str(d) +'  '+str(number)+ '\n')
    f.write('\n')
    for i in range(d):
        for item in wordlist:
            if item[1] == i:
                print item[0],
                f.write(item[0]+'  ')
        f.write('\n')
    f.close()
    '''c = []
    #ax1 = fig.add_subplot(5,5,1)
    for i in range(5, 60):
        clf = cluster.KMeans(i)
        s = clf.fit(arr)
        #print(i, clf.inertia_)
        c.append([i,clf.inertia_])
    d=[]
    for j in range(1,len(c)):
        d.append([j,c[j-1][1]-c[j][1]])
        print(c[j][1],c[j-1][1],c[j-1][1]-c[j][1])
    plt.plot([x[0] for x in d],[x[1] for x in d],'ko--')
    plt.show()'''
#testkmeans(3)
'''test2()
test2()
test2()
test2()
test5()'''
'''testkmeans(3)
testkmeans(4)
testkmeans(4)
testkmeans(5)
testkmeans(5)
testkmeans(6)
testkmeans(6)
testkmeans(6)
testkmeans(7)
testkmeans(7)
testkmeans(7)
testkmeans(8)
testkmeans(8)
testkmeans(8)
testkmeans(9)
testkmeans(9)
testkmeans(9)'''

'''row,col,data = readfile('matrix_a.txt')
#print(row)
#print(col)
#print(data)

for i in range(len(data)):
    sum = 0
    for item in data[i]:
        sum +=int(item)
    #print(sum)
    if sum == 0:
        print(i)'''

test()









'''10.23
层次聚类遇到问题：'latin-1' codec can't encode characters in position 9-13: ordinal not in range(256)
问题分析：编码解码问题
解决办法：增加一种字体
font1 = ImageFont.truetype('simsun.ttc', 24)宋体字
draw.text((x+5,y-7),labels[clust.id].encode('utf-8').decode('utf-8'),(0,0,0),font=font1)
decode的作用是将其他编码的字符串转换成unicode编码，如str1.decode('gb2312')，表示将gb2312编码的字符串str1转换成unicode编码。
encode的作用是将unicode编码转换成其他编码的字符串，如str2.encode('gb2312')，表示将unicode编码的字符串str2转换成gb2312编码。
'''