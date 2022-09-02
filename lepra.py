## Likelihood estimator and population genetics statistics for RADSeq assembly, "Lepra".
## This script is designed to obtain basic information of the RADseq assembly, including the distributions of the RAD loci length, number of SNPs, pairwise differences (pi), Watterson's theta, and Tajima's D.
## Also, to evaluate the fitness of the current assembly parameters, this script calculates a multinomial likelihood estimator based on binomial distribution/Watterson's expected distribution of #SNPs/locus distribution, KL divergence, p-value of chi-squared test.
## Command usage: please see --help
## Author: Kung-Ping Lin, last update: 20210928, current version: 4.0.

import argparse
import sys
import glob
import matplotlib.pylab as plt
import numpy as np
import itertools
import os
import copy
import scipy.stats as ss
from scipy.special import comb
import math
import random
from collections import Counter
from scipy.stats import entropy

## Argument parser
parser = argparse.ArgumentParser(description='Lepra arguments.')

# Select the input format for getting RAD loci length
inLociLenFor = parser.add_mutually_exclusive_group(required=True)
inLociLenFor.add_argument('-f', '--fasta', type=str, help='Path for the input FASTA file for getting RAD loci length; cannot be used with -t.', metavar='FASTA_PATH')
inLociLenFor.add_argument('-t', '--tsv', type=str, help='Path for the input .tsv file with RAD loci name in the first column and length in the last column for getting RAD loci length; cannot be used with -f.', metavar='TSV_PATH')

# Select the input format for reading SNPs
inSNPFor = parser.add_mutually_exclusive_group(required=True)
inSNPFor.add_argument('-v', '--vcf', type=str, help='Path for the VCF file for SNP reading; cannot be used with -p.', metavar='VCF_PATH')
inSNPFor.add_argument('-p', '--plink', type=str, help='Path for the PLINK prefix for SNP reading; cannot be used with -v.', metavar='PLINK_PREFIX_PATH')

# Output format and options
parser.add_argument('-o', '--out', type=str, help='Specified output path and prefix.', required=True, metavar='OUTPUT_PATH')
parser.add_argument('-l', '--per-locus', action='store_true', help='Whether to output files for per-locus statistics.')

# Whether to calculate the fitness of assembly parameter and its associated parameters.
parser.add_argument('-L', '--fitness', type=int, nargs=3, default=[-1, 100000, 20], metavar=('FW_LEN', 'N_SAMPLE_LOCI', 'REP'), help='Specifying the forward length, number of sampled RAD loci, and number of repeat for assembly parameter fitness caluculation. If the user does not specify this flag or a non-positive value is given for the forward read, fitness calculation will not be executed.')

args = parser.parse_args()
fastaPath = args.fasta
tsvPath = args.tsv
vcfPath = args.vcf
plinkPath = args.plink
outPath = args.out
perLocus = args.per_locus
fitValues = args.fitness
print(' '.join(sys.argv))

## Functions
def getLenSeq(seqList, loci):
# return three length from a list of sequences. 1) full length with all of the Ns. 2) length that excludes fully N sites. 3) length that excludes both fully and partial N sites.
    if any([len(e) != len(seqList[0]) for e in seqList]):
        print('ERROR: Length are not equal in the same RAD loci. Loci name: ' + loci + ', seqList = ' + str(seqList))
        return 0

    fullLength = len(seqList[0])
    noFullNLength = 0
    noNLength = 0
    for n in range(len(seqList[0])):
        nucList = []
        for i in range(len(seqList)):
            nucList.append(seqList[i][n])   # nucList stores the single nucleotide from every haplotype
        if 'N' in nucList or 'n' in nucList or '-' in nucList or '0' in nucList:
        # If at least one missing genotype is present in the given nucleotide position
            if not all([n == 'N' or n == 'n' or n == '-' or n == '0' for n in nucList]):
            # If this site is not completely missing
                noFullNLength += 1
        else:
            noFullNLength += 1
            noNLength += 1
    return [fullLength, noFullNLength, noNLength]

def getLenDicTsv(tsv, noN = False):
# return a dictionary contains the length (w/ and w/o Ns) of every loci, the key is the loci name and the value is a list with two elements: the assembled length of the loci with and without uncalled bases (Ns).
    readTsv = open(tsv, 'r')
    lociDic = {}
    for row in readTsv:
        if row[0] != '#':
            if noN:
                lociDic[row.split()[0]] = [int(row.split()[-1]), [], []]
            else:
                lociDic[row.split()[0]] = [int(row.split()[1]), [], []]
    readTsv.close()
    return lociDic

def getLenDicFasta(fasta):
# return a dictionary same as getLenDicTsv but from a raw samples.fa from Stacks.
# Three types of length will be returned as the first element of lociDic: 1) full length with all of the Ns 2) length that excludes fully N sites 3) length that excludes both fully and partial N sites.
    readFa = open(fasta, 'r')
    lociDic = {}
    lastLoci = ''
    for row in readFa:
        if row[0] == '>':
            loci = row.split()[0].split('_Sample_')[0][1:]
            if loci != lastLoci:
                if lastLoci == '':
                    seqList = []
                else:
                    threeLen = getLenSeq(seqList, lastLoci)
                    lociDic[lastLoci] = [threeLen, [], []]
                    seqList = []
                lastLoci = loci
        else:
            if lastLoci == '':
                continue
            seqList.append(row.rstrip('\n'))
    threeLen = getLenSeq(seqList, lastLoci)
    lociDic[lastLoci] = [threeLen, [], []]
    readFa.close()
    return lociDic

def getHaps(hapstats, lociDic1):
# This function is abandoned
# modify the lociDic, append in the haplotype seqeunces for the loci with variant using .hapstats file
    readHap = open(hapstats, 'r')
    for row in readHap:
        if row[0] != '#':
            rs = row[:-1].split()
            loci = 'CLocus_' + rs[0]
            if rs[-1][1:] != rs[4]:
                rawHaps = rs[-1].split(';')
                haps = []
                for rawHap in rawHaps:
                    seq = rawHap.split(':')[0]
                    rep = int(rawHap.split(':')[1])
                    for i in range(rep):
                        haps.append(seq)
                lociDic1[loci].append(haps)
    return lociDic1

def getHapsPlink(plinkPrefix, lociDic):
# modify the lociDic, append the haplotype sequences for the loci with variant using plink files from Stacks. Also, append a list of the positions of these SNPs.
# lociDic = {'loci1': [[lengths], [pos1, pos2 ... ], [hap_seq1, hap_seq2, ... ]], 'loci2': ... }
    # read plink map and get the information of positions
    plinkMap = plinkPrefix + '.map'
    readMap = open(plinkMap, 'r')
    for row in readMap:
        if row[0] == '#':
            continue
        rs = row.rstrip('\n').split()
        loci, pos = rs[1].split('_')
        loci = 'CLocus_' + loci
        pos = int(pos)
        lociDic[loci][1].append(pos)
    readMap.close()

    # read plink ped file and get the haplotype sequences
    plinkPed = plinkPrefix + '.ped'
    readPed = open(plinkPed, 'r')
    nRow = 1    # counter for the number of rows
    for row in readPed:
        if row[0] != '#':
            rs = row.rstrip('\n').split()
            genos = rs[6:]
            nSnp = 0    # the counter for the number of SNPs read
            for loci in lociDic:
                if len(lociDic[loci][2]) < nRow*2:
                    lociDic[loci][2].append('')
                    lociDic[loci][2].append('')
                for n in range(len(lociDic[loci][1])):
                    lociDic[loci][2][(nRow - 1)*2] += genos[nSnp*2]
                    lociDic[loci][2][(nRow - 1)*2 + 1] += genos[nSnp*2 + 1]
                    nSnp += 1
            nRow += 1
    readPed.close()

def getHapsPlinkList(plinkList, lociDic1):
# modify the lociDic, append the haplotype sequences for the loci with variant using plink .list file.
# return another lociDic where only forward end loci (87 bp) were included. For the comparison between theoretical and empirical nSNP distributions.
    lociDic3 = copy.deepcopy(lociDic1)
    fwLen = 87
    
    tInds = 0
    readPlink = open(plinkList, 'r')
    for row in readPlink:
        if row[0] == '#':
            continue
        geno = row.split()[2]
        if geno == '00':
            break
        else:
            tInds += len(row.split()[3:])/2
    
    readPlink = open(plinkList, 'r')
    selInd = 0
    for row in readPlink:
        if row[0] == '#':
            continue
        rs = row[:-1].split()
        loci = 'CLocus_' + rs[1].split('_')[0]
        pos = int(rs[1].split('_')[1])
        geno = rs[2]
        if len(rs) > 3:
            inds = int(len(rs[3:])/2)
        else:
            inds = 0
        if geno == '00':
            selInd = 0
        else:
            if len(lociDic1[loci]) == 1:
                lociDic1[loci].append([])
                if pos < fwLen:
                    lociDic3[loci].append([])
            if inds == 0:
                continue
            else:
                for i in range(inds):
                    for nu in geno:
                        if len(lociDic1[loci][1]) < tInds*2:
                            lociDic1[loci][1].append(nu)
                            if pos < fwLen:
                                lociDic3[loci][1].append(nu)
                        else:
                            lociDic1[loci][1][selInd] += nu
                            if pos < fwLen:
                                lociDic3[loci][1][selInd] += nu
                            selInd += 1
    return [lociDic1, lociDic3]

def pairDiff(hapList):
# return the total raw pi from a list of haplotypes.
    tNum = 0
    tDenom = 0
    for x in range(len(hapList[0])):
        snpList = []
        for y in range(len(hapList)):
            if hapList[y][x] != 'N' and hapList[y][x] != 'n' and hapList[y][x] != '-' and hapList[y][x] != '0':
                snpList.append(hapList[y][x])
        tPairDiff = 0
        for c in itertools.combinations(snpList, 2):
            if c[0] != c[1]:
                tPairDiff += 1
        tNum += tPairDiff        

        denom = comb(len(snpList), 2)
        tDenom += denom
    return [tNum, tDenom]
    
def getPi(lociDic):
# return a list of per site pi values
    piList = []
    for loci in lociDic:
        if len(lociDic[loci][1]) == 0:
            piList.append(0.0)
        elif len(lociDic[loci][1]) >= 1:
            tNum, tDenom = pairDiff(lociDic[loci][2])
            tDenom += (lociDic[loci][0][1] - len(lociDic[loci][2][0])) * comb(len(lociDic[loci][2]), 2)
            psPi = float(tNum)/tDenom
            piList.append(psPi)
    return piList

def an(n):
    a = 0.0
    for i in range(1, n):
        a += 1.0/float(i)
    return a

def WatersonTheta(S, n):
    return S/an(n)

def getTheta(lociDic):
# return a list of per site theta values
    thetaList = []
    for loci in lociDic:
        if len(lociDic[loci][1]) == 0:
            thetaList.append(0.0)
        if len(lociDic[loci][1]) >= 1:
            S = 0
            deNom = 0
            hapList = lociDic[loci][2]
            for n in range(len(hapList[0])):
                snpList = []
                for i in range(len(hapList)):
                    nu =  hapList[i][n]
                    if nu != 'N' and nu != 'n' and nu != '-' and nu != '0':
                        snpList.append(hapList[i][n])
                if snpList.count(snpList[0]) != len(snpList):
                    S += 1
                deNom += an(len(snpList))
            deNom += (lociDic[loci][0][1] - len(hapList[0])) * an(len(hapList))
            theta = float(S) / deNom
            thetaList.append(theta)
    return thetaList

def getLenList(lociDic):
# return a list of loci lengths
    fullLenList = []
    noFullNLenList = []
    noNLenList = []
    for loci in lociDic:
        fullLenList.append(lociDic[loci][0][0])
        noFullNLenList.append(lociDic[loci][0][1])
        noNLenList.append(lociDic[loci][0][2])
    return [fullLenList, noFullNLenList, noNLenList]

def getNSnpList(lociDic, fwLen = -1):
# return a list of per-locus SNP number, with or without specifying a forward length.
    nSnpList = []
    for loci in lociDic:
        snps = 0
        pos = lociDic[loci][1]
        hapList = lociDic[loci][2]
        if fwLen == -1:
            for n in range(len(hapList[0])):
                snpList = [i[n] for i in hapList if (i[n] != 'N' and i[n] != 'n' and i[n] != '-' and i[n] != '0')]
                if snpList.count(snpList[0]) < len(snpList):
                    snps += 1
        else:
            for n in range(len(hapList[0])):
                snpList = [i[n] for i in hapList if (i[n] != 'N' and i[n] != 'n' and i[n] != '-' and i[n] != '0')]
                if snpList.count(snpList[0]) < len(snpList) and pos[n] <= fwLen:
                    snps += 1
        nSnpList.append(snps)
    return nSnpList

def nSnpListComparison(nSnpList3, fwLen = 87, nSeq = 100000, repeats = 20):
    totalLen = len(nSnpList3) * fwLen
    snpFreq = float(sum(nSnpList3))/totalLen
    expFreqs = []
    expCounts = []
    for x in range(fwLen + 1):
        expFreq = ss.binom.pmf(x, fwLen, snpFreq)
        expFreqs.append(expFreq)
        expCounts.append(expFreq*nSeq)

    obsCounts = np.histogram(nSnpList3, bins=list(range(fwLen + 2)))[0]
    obsFreqs = np.array([float(x) for x in obsCounts])/sum(obsCounts)

    chiScore, pChisquare = ss.chisquare(f_obs=obsCounts, f_exp=expCounts)
    print('p-value_of_Chi_square_goodness_of_fit = ' + str(pChisquare))
    KLdivergence = entropy(obsFreqs, qk=expFreqs, base=None)
    print('KL_divergence = ' + str(KLdivergence))

    LList = []
    for r in range(repeats):
        selList = random.sample(nSnpList3, nSeq)
        totalLen = nSeq * fwLen
        snpFreq = float(sum(selList))/totalLen

        expFreqs = []
        expCounts = []
        for x in range(fwLen + 1):
            expFreq = ss.binom.pmf(x, fwLen, snpFreq)
            expFreqs.append(expFreq)
            expCounts.append(expFreq*nSeq)

        obsCounts = np.histogram(selList, bins=list(range(fwLen + 2)))[0]
        ll = logMultinomialLikelihood(obsCounts, expFreqs)
        LList.append(str(ll))
#   print 'Chi-square_test_statistics = ' + ' '.join(chiList) + '\n'
    print('Log_likelihood_of_polynomial_fit = ' + ' '.join(LList) + '\n')

def logMultinomialLikelihood(observed, model, base = math.exp(1)):
    inds = 0
    for i in observed:
        inds += i
    factorialDenoTerm = 0
    logTerms = 0
    for i in range(len(observed)):
        if model[i] == 0:
            if observed[i] == 0:
                logTerms += 0
            else:
                print('ERROR: impossible expected spectrum.')
        else:
            logTerms += observed[i] * math.log(model[i], base)
        factorialDenoTerm += math.log(math.factorial(observed[i]))
    return math.log(math.factorial(inds), base) - factorialDenoTerm + logTerms

def simpleBarplot(values):
    l5barList = values[0:5] + [sum(values[5:])]
    xPos = list(range(6))
    plt.bar(xPos, l5barList, color='#808588', edgecolor='#FFFFFF')
    plt.xlabel('Number of SNPs')
    plt.ylabel('Counts')
    plt.xticks(xPos, [0, 1, 2, 3, 4, '>5'])
    plt.savefig('expected_binomial.png')
    plt.close()

def getTajimaD(nSnpList, piList, lenList, n):
    DList = []
    a1 = 0.0
    for i in range(1, n):
        a1 += 1.0/float(i)
    a2 = 0.0
    for i in range(1, n):
        a2 += 1.0/(float(i)**2)
    b1 = (n + 1.0)/(3.0*(n - 1.0))
    b2 = 2.0*(n**2 + n + 3.0)/(9*n*(n-1))
    c1 = b1 - (1.0/a1)
    c2 = b2 - ((n + 2.0)/(a1*n)) + (a2/(a1**2))
    e1 = c1/a1
    e2 = c2/(a1**2 + a2)
    for i in range(len(lenList)):
        S = float(nSnpList[i])
        if S == 0.0:
            continue
        else:
            D = (piList[i] - S/a1)/((e1*S + e2*S*(S - 1.0))**0.5)
            DList.append(D)
    return DList

'''
testLociDic = {'loci1':[[12, 9, 5], [2, 4, 7, 10], ['TAGT', 'GACN', 'NAGN', 'GNNT']]}
piList = getPi(testLociDic)
thetaList = getTheta(testLociDic)
print(piList)
print(thetaList)
'''

## Function calls
print('Reading loci length ...')
if tsvPath != None:
    lociDic = getLenDicTsv(tsvPath)
else:
    lociDic = getLenDicFasta(fastaPath)

print('Reading SNPs genotypes ...')
if plinkPath != None:
    getHapsPlink(plinkPath, lociDic)
# else read it from vcf (not done yet)

print('\nCalculating population genetics summary statistics ...')
piList = getPi(lociDic)
thetaList = getTheta(lociDic)
lenList1, lenList2, lenList3 = getLenList(lociDic)
nSnpList = getNSnpList(lociDic)
print('Mean of Pi = ' + str(np.mean(piList)))
print('Mean of theta = ' + str(np.mean(thetaList)))
print('Mean of loci length (excluding fully missing sites) = ' + str(np.mean(lenList2)))
print('Mean of per-locus SNP number = ' + str(np.mean(nSnpList)))

if perLocus:
    print('\nWriting per-locus summary statistics ...')
    writePi = open(outPath + '.loci_pi.tsv', 'w')
    writeTheta = open(outPath + '.loci_theta.tsv', 'w')
    writeLen = open(outPath + '.loci_length.tsv', 'w')
    writeSNPs = open(outPath + '.loci_SNP.tsv', 'w')
    for i, locus in enumerate(list(lociDic.keys())):
        writePi.write(locus + '\t' + str(piList[i]) + '\n')
        writeTheta.write(locus + '\t' + str(thetaList[i]) + '\n')
        writeLen.write(locus + '\t' + str(lenList1[i]) + '\t' + str(lenList2[i]) + '\t' + str(lenList3[i]) + '\n')
        writeSNPs.write(locus + '\t' + str(nSnpList[i]) + '\n')

if fitValues[0] > 0:
    print('\nCalculating fitness of assembly parameters ...')
    fwLen, nRead, repeat = fitValues
    bList = []
    for loci in lociDic:
        length = lociDic[loci][0][1]
        if length < fwLen:
            bList.append(loci)
    for loci in bList:
        del lociDic[loci]
    print(str(len(bList)) + ' loci drop because they are shorter than the given forward length ' + str(fwLen))
    trimSnpList = getNSnpList(lociDic, fwLen)
    nSnpListComparison(trimSnpList, fwLen, nRead, repeat)

'''
print('Plink list: ' + sys.argv[1])
print('Loci length tsv: ' + sys.argv[2])
print('Out prefix : ' + sys.argv[3] + '\n')
lociDic2, lociDic3 = getHapsPlink(sys.argv[1], getLenDic(sys.argv[2]))
piList = getPi(lociDic2)
thetaList = getTheta(lociDic2)
lenList = getLenList(lociDic2)
nSnpList = getNSnpList(lociDic2)
nSnpList3 = getNSnpList(lociDic3)
outPrefix = sys.argv[3]
nSnpListComparison(nSnpList3)

## Plots
plt.rcParams.update({'font.size':18})

plt.figure(figsize = (6.4, 4.8))
plt.hist(piList, bins = 100, color='#446AF5', edgecolor='#FFFFFF')
plt.xlabel('Pairwise differences (per site)')
plt.ylabel('Counts')
#plt.yticks(range(4000, 16001, 4000))
plt.xlim(0, 0.01)
#plt.xticks([0.00, 0.01, 0.02, 0.03])
#plt.title('Distribution of pairwise differences of scaffolds/contigs')
plt.axvline(x=np.mean(piList), color='#808080', linestyle='--')
plt.tight_layout()
plt.savefig(outPrefix + '_pairwise_differences.png', dpi=200)
plt.close()

plt.figure(figsize = (6.4, 4.8))
plt.hist(thetaList, bins = 100, color='#446AF5', edgecolor='#FFFFFF')
plt.xlabel('Watterson\'s estimator (per site)')
plt.ylabel('Counts')
#plt.yticks(range(1000, 4001, 1000))
plt.xlim(0, 0.01)
#plt.xticks([0.000, 0.005, 0.010, 0.015, 0.020])
#plt.title('Distribution of Watterson\'s theta of scaffolds/contigs')
plt.axvline(x=np.mean(thetaList), color='#808080', linestyle='--')
plt.tight_layout()
plt.savefig(outPrefix + '_Watterson_theta.png', dpi=200)
plt.close()

plt.figure(figsize = (6.4, 4.8))
plt.hist(lenList, bins = 60, color='#808588', edgecolor='#FFFFFF', align='mid')
#plt.xlim(0, 300)
plt.xlabel('Length of RAD locus')
plt.ylabel('Counts (K)')
plt.yticks(np.arange(2500, 15001, 2500), np.arange(2.5, 15.1, 2.5))
#plt.title('Distribution of lengths of scaffolds/contigs')
plt.axvline(x=np.mean(lenList), color='#2A70E8', linestyle='--', lw=3)
plt.tight_layout()
plt.savefig(outPrefix + '_scaffold_lengths.png', dpi=200)
plt.close()

plt.figure(figsize = (6.4, 4.8))
nSnpDict = Counter(nSnpList)
snpKeys = list(nSnpDict.keys())
snpKeys.sort()
barList = [nSnpDict[k] for k in snpKeys]
l5barList = barList[0:5] + [sum(barList[5:])]
xPos = list(range(6))
plt.bar(xPos, l5barList, color='#808588', edgecolor='#FFFFFF')
plt.xlabel('Number of SNP')
plt.ylabel('Counts (K)')
plt.xticks(xPos, [0, 1, 2, 3, 4, '>5'])
#plt.xlim(0, 20)
plt.yticks(np.arange(25000, 150001, 25000), np.arange(25, 151, 25))
#plt.title('Distribution of segregating sites on scaffolds/contigs')
plt.axvline(x=np.mean(nSnpList), color='#2A70E8', linestyle='--', lw=4)
plt.tight_layout()
plt.savefig(outPrefix + '_scaffold_SNPs_number.png', dpi=200)
plt.close()

print('Pi_mean = ' + str(np.mean(piList)))
print('Pi_variance = ' + str(np.var(piList)))
print('Length_mean = ' + str(np.mean(lenList)))
print('Length_variance = ' + str(np.var(lenList)))
print('#SNPs_mean = ' + str(np.mean(nSnpList)))
print('#SNPs_variance = ' + str(np.var(nSnpList)))
print('Theta_mean = ' + str(np.mean(thetaList)))
print('Theta_variance = ' + str(np.var(thetaList)))

print('Pis = ' + ' '.join([str(x) for x in piList]))
print('Lengths = ' + ' '.join([str(x) for x in lenList]))
print('#SNPs = ' + ' '.join([str(x) for x in nSnpList]))
print('thetas = ' + ' '.join([str(x) for x in thetaList]))
'''
