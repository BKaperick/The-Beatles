#Python library for dealing with random elements
from random import *

#Generator, takes a string and parameter k.  To use, it is assigned to a variable, (see textGen below) and every time
#"next(textGen)" is called, the next k characters from the string are returned.  So for the initial string "Markov" and
#k=2, subsequent "next" calls return "Ma", "ar", "rk", "ko", "ov".
def nextK(string, k):
        index = 0
        while index < len(string):
            yield string[index:index+k]
            index += 1

#This is also a generator which works much like the one above, except its modified to return the next k words, not characters
def nextWord(string,k):
    words = string.split()          #str.split() function will convert a string with whitespace to a list with each whitespace
                                    #separated chunk (a word) as a new element.
                                    #E.g. "A Walk in the Woods" -> ["A", "Walk", "in", "the", "Woods"]
    for i in range(0,len(words)):
        out = words[i:i + k]
        outString = ""
        for o in out:
            outString += o + " "
        yield outString


#If words is true, then the script will sort the input text with the nextWord generator, else it'll use the nextK generator.
words = True
#This is the (integer) size of chunks considered.  Either two words (if words=True) or two letters (if words=False)
#Generally, coherence of the output improves as this parameter increases to a certain point.  If it is two high, then it mostly
#ends up just quoting from the input document, which is not desired.
param = 1


#This block of commands reads the training data ("in.txt" in the current directory) and removes the annoying punctuation that
#distracts from the meaning of the text.  (The in.txt I uploaded is the text of Alice in Wonderland)
studyText = ""
file = open("in.txt","r+")
for line in file.readlines():
    line = line.lower()
    line = line.replace(".","")
    line = line.replace(",","")
    line = line.replace("\n"," ")
    line = line.replace("\'","'")
    line = line.replace('"','')
    studyText += line
file.close()
    


#Decides which generator to assign to textGen based on the value of "words" above
if words:
    textGen = nextWord(studyText, param)
else:
    textGen = nextK(studyText, param)

#This is the dictionary which will store the data.  The keys are all the unique "chunks" that appear in the text.  The values are lists
#containing all chunks that immediately follow the key chunk in the text
#For example:  consider the text "morocco more or less" with chunks based on letters with chunk size 3.
#Then, the dictionary keys and values are
#knowledge = {
#                "mor" : ["oro", "ore"],
#                "oro" : ["roc"],
#                "roc" : ["occ"],
#                "occ" : ["cco"],
#                "cco" : [" mo"],
#                " mo" : ["ore"],
#                "ore" : ["re "],
#                "re " : ["e o"],
#                "e o" : [" or"],
#                " or" : ["or "],
#                "or " : ["r l"],
#                "r l" : [" le"],
#                " le" : ["les"],
#                "les" : ["ess"],
#                "ess" : [""]
#           }
#
knowledge = {}

#Calls the first two "chunks" retrieved from the generator.
#From the above example, these two values become "Ma" and "ar", respectively.
prev = next(textGen)
bit = next(textGen)

#Loops until the generator has no more elements in it.  In the above example, this would iterate over "rk", "ko", and "ov", then break.
while True:
    try:
        #loads all chunks to the next chunk
        if prev in knowledge.keys():
            knowledge[prev].append(bit)
        else:
            knowledge[prev] = [bit]
        prev = bit
        bit = next(textGen)
    except StopIteration:
        break

#Using the above example as an example ("morocco more or less"), this function takes the parameter length, which is the number of chunks that will be printed.
#The function randomly chooses a "seed" chunk to begin the output.  Say this seed was "mor".  Then, a random element is pulled from the list this seed chunk
#corresponds to in the knowledge dictionary.  Here, the choice() function randomly pulls an element from knowledge["mor"] = ["oro", "ore"].  This  continues
#until the maximum length is reached.
def gibberish(length):
    output = ""
    seed = choice(list(knowledge.keys()))

    output += seed
    last = seed

    while len(output) < length:
        last = choice(knowledge[last])
        output += last
    return output  
        
