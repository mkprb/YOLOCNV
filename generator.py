import pandas as pd
import numpy as np
from Bio import SeqIO
import random
import math
import os
import sys

def generate(ID,dest,FASTA,bowtie_files,pathOut):
	ID = str(ID)
	np.random.seed()

	wgsim_path='wgsim'
	bedtools_path = "bedtools"
	samtools_path = "samtools"

	cnv_count = 5
	cnv_size = 1000
	dup_len = 10000
	del_len = 10000
	read_len = 100
	cov = np.random.choice([5,10,20,30])
	single_ends = True


	new_chr = []
	chrs = []
	for r in SeqIO.parse(open(FASTA),"fasta"):
		total = pd.DataFrame()
		df_del = pd.DataFrame(columns = ['start','end','type','n'])
		df_dup = pd.DataFrame(columns = ['start','end','type','n'])

		chrs.append(r.id)
		dup_lengths = []
		del_lengths = []
		while len(dup_lengths) < cnv_count:
			x = round(np.random.normal(dup_len, cnv_size, 1)[0])
			if x > 100:
				dup_lengths.append(x)
		while len(del_lengths) < cnv_count:
			x = round(np.random.normal(del_len, cnv_size, 1)[0])
			if x > 100:
				del_lengths.append(x)
		dup_start = list(np.random.randint(100,len(str(r.seq)), size=(1, cnv_count))[0])
		del_start = list(np.random.randint(100,len(str(r.seq)), size=(1, cnv_count))[0])
		dup_ends = list(map(int,[a + b for a, b in zip(dup_start, dup_lengths)]))
		del_ends = list(map(int,[a + b for a, b in zip(del_start, del_lengths)]))
		dups = pd.DataFrame({'start':dup_start,'end':dup_ends,'type':1,
							 'n':np.random.choice([2,3,4,5,6,7,8,9,10], cnv_count, p=[0.5, 0.2, 0.15, 0.06, 0.05,0.01,0.01,0.01,0.01])
							})
		dels = pd.DataFrame({'start':del_start,'end':del_ends,'type':2, 'n':0})
		df_dup = df_dup.append(dups)
		df_del = df_del.append(dels)
		total = df_dup.append(df_del).sort_values(by=['start']).reset_index(drop = True)
		total = total.drop(total[total.end > total.shift(-1).start].index).reset_index(drop = True)
		total2 = total
		chr_len = len(str(r.seq))
		chro = str(r.seq)

		new = pd.DataFrame()
		new = new.append(pd.DataFrame([[1,total.start[0]-1,0,1]], columns = ['start','end','type','n']))
		for i,e in total.iterrows():   
			if i == len(total)-1:
				new = new.append(pd.DataFrame([[total.end[i]+1,chr_len,0,1]], columns = ['start','end','type','n']))
			else:
				new = new.append(pd.DataFrame([[total.end[i]+1,total.start[i+1]-1,0,1]], columns = ['start','end','type','n']))

		total = total.append(new).sort_values(by=['start']).reset_index(drop = True)
		total2['chr']=r.id
		total2 = total2[['chr','start','end','type','n']]

		chro2 = str()

		for i,e in total.iterrows():
			if e[2] == 0:
				chro2 += chro[e[0]:e[1]]
			elif e[2] == 2:
				pass
			elif e[2] == 1:
				for v in range(0,e[3]):
					chro2 += chro[e[0]:e[1]]

		new_chr.append(chro2)
		total2['start']/=chr_len
		total2['end']/=chr_len

	for i in range(len(chrs)):
		path = pathOut + chrs[i] + "_" + "_CNV.fa"
		os.makedirs(os.path.dirname(path), exist_ok=True)
		os.makedirs(os.path.dirname(pathOut+"Labels/"), exist_ok=True)
		total2.to_csv(pathOut+"Labels/Labels_"+ID,sep = ' ',index=False,header = False)
		#print(total2)
		out = open(path,"w+")
		out.write(">" + chrs[i] + "\n" + new_chr[i] + "\n")
	out.close()



	chr_lens = {}
	if single_ends:
		for r in SeqIO.parse(open(path),"fasta"):
			chr_lens[r.id] = len(str(r.seq))
		for chr in chr_lens:
			reads = round(chr_lens[chr]/(2*int(read_len)))*int(cov)
			os.system(wgsim_path + " -N " + str(reads) + " -1 " + str(read_len) + " " + pathOut + chr + "_" + "_CNV.fa " + pathOut + chr + ".fq /dev/null > stdout")
		for chr in chr_lens:
			os.system("cat " + pathOut + chr + ".fq >> " + pathOut + "_" + str(cov) + ".fq")
			os.remove(pathOut + chr + ".fq")
	else:
		for r in SeqIO.parse(open(path),"fasta"):
			chr_lens[r.id] = len(str(r.seq))
		for chr in chr_lens:
			reads = round(chr_lens[chr]/(2*int(read_len)))*int(cov)
			os.system(wgsim_path + " -N " + str(reads) + " -1 " + str(read_len) + " -2 " + str(read_len) + " " + pathOut + chr + "_" + "_CNV.fa " + pathOut + chr + "_1.fq " + pathOut + chr + "_2.fq > stdout")
		for chr in chr_lens:
			os.system("cat " + pathOut + chr + "_1.fq >> " + pathOut + "_" + str(cov) + "_1.fq")
			os.system("cat " + pathOut + chr + "_2.fq >> " + pathOut + "_" + str(cov) + "_2.fq")
			os.remove(pathOut + chr + "_1.fq")
			os.remove(pathOut + chr + "_2.fq")
	os.remove(path)


	mapq=[0,35]

	for i in mapq:
		if single_ends:
			#os.system("bwa mem -t 4 -w 0 " + FASTA +" "+ pathOut + "_" + str(cov) + ".fq" +"|samtools view -Shb -q " + str(i) + "|samtools sort - > " +  pathOut + "total" + str(i) + ".bam")
			os.system("bowtie2 -x " + bowtie_files + " -U "+ pathOut + "_" + str(cov) + ".fq --local -p 2" +"|samtools view -Shb -q " + str(i) + "|samtools sort - > " +  pathOut + "total" + str(i) + ".bam")
			os.system("bedtools genomecov -d -ibam " +  pathOut + "total" + str(i) + ".bam >"+ pathOut + "cov" + str(i) + ".bam")
			os.remove(pathOut + "total" + str(i) + ".bam")

	os.remove(pathOut + "_" + str(cov) + ".fq")

	window = 50

	for i in mapq:
		if single_ends:
			d1=pd.read_csv(pathOut + "cov" + str(i) + ".bam",delimiter='\t',usecols = [2]).to_numpy()
			os.remove(pathOut + "cov" + str(i) + ".bam")
			a = np.append(d1,np.zeros((window-len(d1)%window,1)),axis=0)
			a = np.std(np.reshape(a,(int(len(a)/window),window)),axis = 1).round(5)
			a = np.reshape(a,(len(a),1))
			a = np.reshape(np.repeat(a,window,1).flatten()[:len(d1)],(len(d1),1))
			d1 = np.append(d1,a,axis = 1)
			if i==0:
				d = d1
			else:
				d = np.concatenate((d,d-d1),axis=1)

	os.makedirs(os.path.dirname(pathOut+"Dataset/"), exist_ok=True)
	np.savetxt(pathOut+"Dataset/Dataset_"+ID, d, delimiter = ' ', fmt = '%.5g')