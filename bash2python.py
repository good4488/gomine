import os
import codecs
import subprocess as sp
import math
import time


# 1st authorized by Baekjun Kim, Molecular Simulation Group.
# 2016-06-29 - 
# function from OS : do_something, do_qsub, do_grep, do_mkdir, do_rmdir, do_chmod, do_touch, do_cp, do_mv, do_del, do_cat
# How to import : 'from bash2python import *'
# 더 섬세한 에러처리 필요, timeout 주의
# sleep function -> python time.h


def make_qsub(filename, node_type, node_num, core_per_node, content):
    ''' make qsub file, type(str, str, int, int, str) '''
    with open(filename, 'w', encoding='UTF-8') as f:
        f.write('#PBS -l nodes=%d:ppn=%d:%s\n'%(node_num,core_per_node,node_type))
        f.write('#PBS -q %s\n'%node_type)
        f.write('cd $PBS_O_WORKDIR\n\n')
        f.write(content+'\n')
    

def do_something(sents):
    ''' same as bash script, type(str)'''
    proc = sp.Popen(sents, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        out, err = proc.communicate(timeout=3)
        print(out.decode('UTF-8'))
    except TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()


def do_cat(sents):
    proc = sp.Popen(['cat', sents], stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        out, err = proc.communicate(timeout=3)
        print(out.decode('UTF-8'))
    except TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()


def do_qsub(qsubname):
    ''' same with linux qsub xxxx.qsub, do_qsub('xxxx.qsub')'''
    proc = sp.Popen(['qsub', qsubname], stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        out, err = proc.communicate(timeout=3)
        print(out.decode('UTF-8'))
    except TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()

def do_grep(sents):
    proc = sp.Popen(['grep', sents], stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        out, err = proc.communicate(timeout=5)
        return out.decode('UTF-8')
    except TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()

def do_mkdir(sents):
    proc = sp.Popen(['mkdir', sents], stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        out, err = proc.communicate(timeout=3)
        print(out.decode('UTF-8'))
    except TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()


def do_chmod(sents):
    proc = sp.Popen(['chmod', sents], stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        out, err = proc.communicate(timeout=3)
        print(out.decode('UTF-8'))
    except TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()


def do_touch(sents):
    proc = sp.Popen(['touch', sents], stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        out, err = proc.communicate(timeout=3)
        print(out.decode('UTF-8'))
    except TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()


def do_cp(sents):
    proc = sp.Popen(['cp', sents], stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        out, err = proc.communicate(timeout=10)
        print(out.decode('UTF-8'))
    except TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()


def do_mv(sents):
    proc = sp.Popen(['mv', sents], stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        out, err = proc.communicate(timeout=10)
        print(out.decode('UTF-8'))
    except TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()


def do_rmdir(sents):
    proc = sp.Popen(['rmdir', sents], stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        out, err = proc.communicate(timeout=10)
        print(out.decode('UTF-8'))
    except TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()


def do_del(sents):
    proc = sp.Popen(['del', sents], stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        out, err = proc.communicate(timeout=10)
        print(out.decode('UTF-8'))
    except TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()



