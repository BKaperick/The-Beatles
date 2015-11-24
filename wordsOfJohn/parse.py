import re
from os import listdir

def parseFile(filename):
    file = open(filename + '.txt', 'r+')
    
    lines = []

    for line in file.readlines():
        line = re.sub(r'\([^)]*\)', '', line)
        line = ''.join(i for i in line if not i.isdigit())
        line = re.sub('\s+', ' ', line)
        line = line.replace("- -", "$")
        line = line.lower()
        line = line.replace("“", "")
        line = line.replace(":", "")
        line = line.replace("”", "")
        line = line.replace('"', "")
        line = line.replace(".", "")
        line = line.replace(",", "")
        line = line.replace("?", "")
        lines.append(line)

    file.close()

    newfile = open('PARSED-' + filename + '.txt', 'w')

    for l in lines:
        newfile.write(l)

    newfile.close()

def parseInterview(filename = ""):
    file = open(filename + "interview.txt", 'r')
    new = open(filename + "interviewcut.txt", "w")
    for line in file.readlines():
        if line[0] == 'L':
            new.write(line[7:])
    file.close()
    new.close()
    parseFile(filename + 'interviewcut')


def combineGood():
    total = open("total.txt", "w")
    files = [file for file in listdir() if file[:7] == "PARSED-"]
    for filename in files:
        file = open(filename, "r")
        lines = file.readlines()
        for line in lines:
            total.write(line)
        file.close()
    total.close()
    
