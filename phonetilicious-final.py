
# coding: utf-8

# In[1]:

from topia.termextract import extract
extractor = extract.TermExtractor()	
import pattern.en

using_topia_tagger = True

def extract_words(sentence):
    # tokenized sentence. returns a list of [word, part of speech, word_singular]
    if(using_topia_tagger):
        tok = extractor.tagger(sentence)
        if not tok:
            return []
        # hack for fixing the bug which sometimes makes extractor's first word of the senten lower case
        first = sentence.strip()[:1] # first letter
        if (first.upper() == first):
            tok[0][0] = tok[0][0][:1].upper() + tok[0][0][1:]
        else:
            tok[0][0] = tok[0][0][:1] + tok[0][0][1:]
    else:
        # use nltk's tagger maxent_treebank_pos_tagger
        text = word_tokenize(sentence)
        tok = pos_tag(text)
    """
    # pattern's tagger:
    pattern.en.tag(text)
    """
        
    #if DEBUG:
        #print tok 
        
    # nothing to extract
    if len(tok) == 0:
        return []
    
    
    formatted = map(lambda w: {"word_original": w[0], "word_new": w[0], "ignore": False, "pos": w[1], "word_lower": justlowerletters(w[0])}, tok)
    

    return formatted
    #sentence = justlowerletters(sentence)
    #return sentence.split(" ")
    
# given string, returns ascii string, lowercase
def justlowerletters(string):
    string = string.encode('ascii','ignore')
    #string = str(string)
    return str.lower(string)
    #return sub(r'\s+',' ',sub(r'[^a-z\s]', '', str.lower(string)))


# In[2]:

# given sentence, returns list of formatted words
# TODO: consider using a tokenizer
from nltk.corpus import stopwords 
stop = map(lambda string: string.encode('ascii','ignore'), stopwords.words('english'))


# In[3]:

from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict as cmu
from re import sub, split
from random import randrange
from numpy import *
from nltk import word_tokenize, sent_tokenize, pos_tag    
from random import shuffle
import re

DEBUG = True

prondict = cmu.dict() 
Phonemes = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]
Consonants = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"]
Vowels =  ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]

num_phonemes = len(Phonemes)
max_synonyms = 100

# TODO
# consonance_only
# assonance_only
# look at first phoneme of each word
# look at first phoneme of each syllable
# look at first phoneme of stressed syllables
# swap out synonyms in same part of speech only
# only swap out Nouns, Adjectives, Verbs, Adverbs
# ignore most common words
# keep punctuation

#alliteration

# HEURISTIC. Efficient. Runs in n*m time. 

# given a sentence (list of Word Objects), returns phonetlicious score
def phonetilicious_score(loWO, flatten=False, normalize=False):
    if not loWO:
        return 0
    
    phonevectors = map(lambda WO: WO["phonevector"], loWO)
    
    if flatten:
        # set each word's phoneme count to 0 or 1.
        phonevectors = map(lambda v: map(lambda p: 0 if p == 0 else 1, v),  phonevectors)
    
    # sum up phonevectors for each word
    vector_sum = reduce(lambda v,w: v + w, phonevectors)
    
    score = 0.0

    phone_count = 0
    for n in vector_sum:
        if n > 0:
            score += n * n * n  # recurring phones get cubed points
            phone_count += n
    
    if normalize:
        score /= phone_count
        
    # count repeated words
    # no, this isn't the place to do this, because by now we've already maximized one example for each phoneme
    # do this earlier
    
    words = map(lambda WO: WO["word_lower"], loWO)
    word_doubles = 0
    for word in words:
        extras = words.count(word) - 1
        if extras > 0:
            word_doubles += extras
    score /= (0.2 * (word_doubles+1)) #severely penalized for extra words
    
    return score 
    
"""
Word Object looks like:
,
WO = {    'word_original': 'Truly',
          'word_lower': 'truly'
        'pos': 'RB',
        'ignore': true,
        'phonevector': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])}
"""
# given a phone ph, and list of Word Objects, returns the Word Object with max occurance of ph
# try to avoid words on blacklist (word_lower)
def maxphone(ph, loWO, blacklist=[]):
    if blacklist: 
        new_loWO = [WO for WO in loWO if WO["word_lower"] not in blacklist]
        if len(new_loWO) == 0:
            new_loWO = loWO
    else:
        new_loWO = loWO
    
    result = max(new_loWO, key = lambda x: x['phonevector'][Phonemes.index(ph)])
    
    return result

        

# format new word with the capitalization of the old word
def format_capitalization(new_word, old_word):
    if old_word.capitalize() == old_word:
        return new_word.capitalize()
    elif old_word.upper() == old_word:
        return new_word.upper()
    elif old_word.lower() == old_word:
        return new_word.lower()
    else:
        # if it's none of these, then it's something weirder
        return new_word
    
    
def pos_topia2wordnet(pos):
    x = pos[:1]
    if x == "N":
        return "n"
    elif x == "V":
        return "v"
    elif x == "J":
        return "a"
    elif x == "R":
        return "r"
    else:
        return x.lower()
    
import random

def algo(text, 
         alliteration_only = False, 
         number_of_examples = 4, 
         force_phoneme = [], 
         flatten = False, 
         split_by = "sentences", # "sentences" or "punctuation"
         normalize = False, 
         try_to_avoid_dups = False, 
         word_additions = [],
         swap_probability = 1.0,
         hyper_profanity = False,
         ignore_numbers = True,
         debug = False):
    
    DEBUG = debug
    
    out = []
    
    if split_by == "punctuation":
        sentences = re.split(r'( *[\.\?\;\,\:\—\n!][\'"\)\]]* *)', text)
    elif split_by == "lines":
        sentences = re.split(r'(\n)', text)
    else:
        sentences = sent_tokenize(text)
    
    if hyper_profanity:
        word_additions += PROFANITY
    
    if DEBUG:
        print sentences
        
    for sentence in sentences: 
        
        # should be tokenized with part of speech, and commonness
        words = extract_words(sentence)
        
        if not words:
            continue #empty sentence
        
        if DEBUG:
            print words
        # which words are we keeping?
        
        # just nouns, verbs, adjectives, adverbs
        pos = ["N","V","J","R"]
        
        for w in words:
            w["ignore"] = (w['word_lower'] in stop) or not (w['pos'][:1] in pos)
            w["phonevector"] = phonevector(phones(w["word_lower"]), alliteration_only)
        
        #if DEBUG: 
            #print words
        
        
        # for each word, find a list of related Word Objects
        lolorWO = map(lambda w: 
                      [w] if w["ignore"] else
                      map(lambda rw: {
                            "word_lower": rw, 
                            "word_original": w['word_original'],
                            "word_new": format_capitalization(rw, w['word_original']),
                            "pos": w['pos'],
                            "phonevector": phonevector(phones(rw), alliteration_only)
                        }, possible_words(w["word_lower"],w['pos'], word_additions, swap_probability, ignore_numbers)),
                    words) 
        
        #if DEBUG: 
            #print lolorWO
            
        def generate_sentence_with_maxphone(p,lolorWO):
            sentence = []
            blacklist = []
            for loWO in lolorWO:
                mp = maxphone(p, loWO, blacklist)
                if try_to_avoid_dups:
                    blacklist.append(mp["word_lower"])
                sentence.append(mp)
            return sentence
            #sentence = map(lambda loWO: maxphone(p, loWO), lolorWO)
            
        
        if len(force_phoneme) == 0:
            phonemes_to_maximize = Phonemes
        else:
            phonemes_to_maximize = force_phoneme
        loS = map(lambda p: generate_sentence_with_maxphone(p, lolorWO), phonemes_to_maximize)
        
        #if DEBUG:
            #print loS
        
        
        
        
        sentences_with_score = map(lambda s: {'sentence': sentencify(s, sentence), 'score': phonetilicious_score(s, flatten, normalize)}, loS)
        
        indexes = unique(sentences_with_score, return_index=True)[1]
        sentences_unique = [sentences_with_score[index] for index in sorted(indexes)]
        # unique array, still sorted by score. 
        # http://stackoverflow.com/questions/12926898/numpy-unique-without-sort
        
        best_sentences = sorted(sentences_unique, key=lambda s: s['score'], reverse=True)
        
        if DEBUG:
            for s in best_sentences:
                print s
                
        # trim down number of examples
        best_sentences = best_sentences[:number_of_examples]
                
        if DEBUG:
            print "\nBEST SENTENCE:"
            print best_sentences[0]
            print "\n"
        
        for s in best_sentences:
            out.append(s['sentence'])
        
    return out

# punctuation = (';', ':', ',', '.', '!', '?','...','"',"'") 

# given a list of Word Objects, returns sentence
def sentencify(loWO, original_sentence):
    
    """
    # remove unnecessary space around punctuation
    i = 0
    while i<len(words):
        if i>0 and (words[i] in PUNCTUATION):
            words[i-1] = words[i-1] + words[i]
            words.pop(i)
        else:
            i+=1
    """

    
    # this is wrong, 
    # because if our replacements are [B->C, C->D] 
    # this will incorrectly go  A B C => A C C => A D C  
    # instead of                A B C => A C D
    x = 0
    for WO in loWO:
        original_sentence = original_sentence[:x] + original_sentence[x:].replace(WO['word_original'], WO['word_new'], 1)
        x += len(WO['word_new'])
    
    words = original_sentence.split(" ")
        
    # fix an; "a egg"  -> "an egg"
    i = 0
    for (i,word) in enumerate(words):
        if (i+1) < len(words):
            vowel_word = words[i+1][:1].lower() in ["a","e","i","o","u"]
            if word.lower() == "a" and vowel_word:
                words[i] = format_capitalization("an",word)
            elif word.lower() == "an" and not vowel_word:
                words[i] = format_capitalization("a",word)
    
    return " ".join(words)

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# In[12]:

#from pattern.en.wordlist import PROFANITY
PROFANITY = []
with open("profanity.txt") as f:
    for line in f:
        PROFANITY.append(line.strip())
f.close()


# In[13]:

# given a word, return list of related words
# pos = part of speech
# word_additions = extra words to choose from (ex: profanity)
# addition_probably = probably that we add the word_additions to the new word list (0 to 1)
def possible_words(word, pos, word_additions, swap_probability=1.0, ignore_numbers = True):
    
    if(swap_probability <= random.random()):
        return [word]
    
    if ignore_numbers and (word.strip().isdigit()):
        return [word]
    
    pos_wordnet = pos_topia2wordnet(pos)
   
    # synonyms
    #lemma = lemmatizer.lemmatize(word, pos) 
    syn_sets = wn.synsets(word,pos_wordnet)

    all_syn_sets = syn_sets
    
    #add hypernyms
    for syn_set in syn_sets:
        all_syn_sets += syn_set.hypernyms()
        #all_syn_sets += syn_set.hyponyms()
        
    words = []

    # TODO: pick synsets based on part of speech
    for syn_set in all_syn_sets:
        for w in syn_set.lemma_names():
            if w not in words:
                if w and ("_" not in w) and (w is not word):  # for now, ignore multi-word synonyms
                    words.append(w)
            elif len(words) >= max_synonyms:    
                break
        if len(words) >= max_synonyms:    
            break
    if not words:
        words = [word]
        
    def format_word(w):
        w = justlowerletters(w)
        if pos[:1]=="V":
            w = conjugate_verb(w, pos)
        if pos[:1]=="N":
            w = conjugate_noun(w, pos)
        return w

    
    if word_additions:
        # add extra words. (ex. profanity)
        words = word_additions[:]
    
    # mix it up
    shuffle(words)
    
    # conjugate, lowercase, etc
    words = map(format_word, words)
    return words
    
# given a word, return list of phones
def phones(word):
    if word in prondict:
        return prondict[word][0]
    else: # word is not in cmudict, return empty list
        return []
    
# given a list of phones, return phone vector.
def phonevector(lop, alliteration_only):
    vector = [0]*num_phonemes

    for p in lop:
        p = p[:2] # ignore stress markers
        vector[Phonemes.index(p)] += 1 #= 1 to flatten phonemes
        if alliteration_only: 
            # only look at the first letter
            break
        
    return array(vector)


# In[14]:

from pattern.en import conjugate, pluralize, singularize

def conjugate_noun(noun, pos):
    if pos=="NNS" or pos =="NNPS":
        return str(pluralize(noun))
    elif pos=="NN" or pos =="NNP":
        return str(singularize(noun))
    else:
        return noun
    
# conjugate verb into new pos
def conjugate_verb(verb, pos):
    if pos[:1]=="V":
        # verb which isn't be, do, have, or will/would/can/could
        # we need to conjugate
        if pos[2:3]=="B":
            conj = conjugate(verb, tense = "infinitive")
        elif pos[2:3]=="D":
            conj = conjugate(verb, tense = "past")
        elif pos[2:3]=="G":
            conj = conjugate(verb, aspect = "progressive")
        elif pos[2:3]=="I":
            conj = conjugate(verb, tense = "infinitive")
        elif pos[2:3]=="N":
            conj = conjugate(verb, tense = "past", aspect="progressive")
        elif pos[2:3]=="Z":
            conj = conjugate(verb, tense = "present", person = 3, number = "singular")
        else:
            conj = verb
    return str(conj)


# In[ ]:

f = open("taoteching.txt","r")
tao = f.read()
f.close()

def tao_rmx(text):
    rmx = algo(text,
              alliteration_only = True, 
              number_of_examples = 1, 
              split_by = "lines",
              force_phoneme = [], # Consonants, Vowels, [], ["K"]
              word_additions = [],
              swap_probability = 0.7,
              hyper_profanity = True,
              ignore_numbers = True,
              flatten = False, 
              normalize = False, 
              try_to_avoid_dups = True,
              debug = True)
    linguistic_nectar = "\n".join(rmx)
    return linguistic_nectar


body = tao_rmx(tao)

f = open("taoteching-profanity-remixK.txt","w+")


# In[ ]:

f.write(body)
f.close()


# In[ ]:




# In[ ]:




# In[17]:

from pattern.en import conjugate, pluralize, singularize

text = """
National Novel Generation Month - based on an idea I tweeted on a whim. This is the 2014 edition, see here for 2013.
The Goal
Spend the month of November writing code that generates a novel of 50k+ words 

The Rules

The only rule is that you share at least one novel and also your source code at the end.

The source code does not have to be licensed in a particular way, so long as you share it. The code itself does not need to be on GitHub, either. I'm just using this repo as a place to organize the community.

The "novel" is defined however you want. It could be 50,000 repetitions of the word "meow". It could literally grab a random novel from Project Gutenberg. It doesn't matter, as long as it's 50k+ words.

Please try to respect copyright. I'm not going to police it, as ultimately it's on your head if you want to just copy/paste a Stephen King novel or whatever, but the most useful/interesting implementations are going to be ones that don't engender lawsuits.

This activity starts at 12:01am GMT on Nov 1st and ends at 12:01am GMT Dec 1st.

How to Participate

Open an issue on this repo and declare your intent to participate. You may continually update the issue as you work over the course of the month. Feel free to post dev diaries, sample output, etc.

Also feel free to comment on other participants' issues.

Resources

There's an open issue where you can add resources (libraries, corpuses, APIs, techniques, etc).

There are already a ton of resources on the old resources thread for the 2013 edition.

You might want to check out corpora, a repository of public domain lists of things: animals, foods, names, occupations, countries, etc.

That's It

So yeah. Have fun with this!

"""
rmx = algo(text,
      alliteration_only = True, 
      number_of_examples = 1, 
      split_by = "sentence",
      force_phoneme = Consonants, #Consonants,#, Vowels, [], ["K"]
      word_additions = [],
      hyper_profanity = False,
      flatten = False, 
      normalize = False, 
      try_to_avoid_dups = True,
      debug = True)
output = " ".join(rmx)
print output


# In[ ]:




# In[ ]:

def build_latex(book):
    latex = ''
    
    for part in book["parts"]:
        latex+=ur'\part*{' + part["title"] + "}\n\n"
        for chapter in part["chapters"]:
            latex+=ur'\chapter*{' + chapter["title"] + "}\n\n" + chapter["body"] + "\n\n"
    latex += ur'\end{document}'
    print latex
    return latex

from tex import latex2pdf 

"""
f = open("alice-in-wonderland.txt","r")
book_title = ""
book_author = ""
chapters = []
inside_a_chapter = False
for line in f:
    line = line.strip()
    if line[:27]=="End of Project Gutenberg's ":
        chapters.append(chapter)
        break
    elif line[0:8] == 'CHAPTER ':
        if inside_a_chapter: 
            chapters.append(chapter)
        chapter = {"title": line, "body": ""}
    else:
        chapter["body"] += line

alice_in_curiosity_country = {"title": "Alice's Adventures in Curiosity Country", "chapters": chapters}
"""


def alice_rmx(text):
    rmx = algo(text,
              alliteration_only = True, 
              number_of_examples = 1, 
              split_by = "punctuation",
              force_phoneme = Consonants, # Consonants, Vowels, [], ["K"]
              word_additions = [],
              hyper_profanity = False,
              flatten = False, 
              normalize = False, 
              try_to_avoid_dups = True,
              debug = False)
    linguistic_nectar = "".join(rmx)
    return linguistic_nectar


# In[ ]:

f = open("gettysburg.txt","r")
gettysburg_body = f.read()
gettysburg_body = gettysburg_body.replace("—","--")
gettysburg_body = "itinerary excellence patterns."
f.close()


def getty_rmx(text):
    rmx = algo(text,
              alliteration_only = True, 
              number_of_examples = 1, 
              split_by = "sentences",
              force_phoneme = [], # Consonants, Vowels, [], ["K"]
              word_additions = [],
              hyper_profanity = False,
              flatten = False, 
              normalize = False, 
              try_to_avoid_dups = True,
              debug = True)
    linguistic_nectar = " ".join(rmx)
    return linguistic_nectar

gettysburg_address = {
    "title": "The Gettysburg Address", 
    "chapters": [{
        "title": getty_rmx("The Gettysburg Address"), 
        "body" : getty_rmx(gettysburg_body)}
    ]
}

print gettysburg_address['chapters'][0]['body']


# In[ ]:


    
    
def censor(w):
    x = w[0]
    for i in range(len(w)-1):
        x += random.choice(["#","@","$","%","&","*","1"])
    x += w[-1]
    return x


# In[ ]:

import inspect
source = inspect.getsourcelines(alice_rmx)[0]
code = "".join(map(lambda x: str(x), source))
print code



# In[ ]:

book = {"parts": [
       #alice_in_curiosity_country,
        gettysburg_address
]}
latex = build_latex(book)
f = open("book4.tex","w")
f.write(latex)
f.close()


# In[ ]:

from tex import latex2pdf 
pdf = latex2pdf(latex) 
f = open("book.pdf", 'w+')
f.write(pdf)
f.close()


# In[ ]:



