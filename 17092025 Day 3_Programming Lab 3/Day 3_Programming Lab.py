#!/usr/bin/env python
# coding: utf-8

# # Day 3 Programming Lab

# The following exercises will not count towards your course mark, but they provide you with an opportunity to receive feedback on your programming skills in advance of you completing your summative assignments.
# 
# The goal of the following exercises is to make you apply the concepts and general methods seen in Day 3 of the Bootcamp and develop Python scripts to perform NLP tasks.

# # Exercises

# Create a for loop to count the characters of the following text
# 
# text = "NLTK is really cool!"

# In[139]:


# Provide your solution here


# Create a function to count the number of words that start with a given letter. Use the function to print the number of words that start with the letter "c" using the text variable. Fell free to look online and use the following
# 
# * Regular expressions link: https://www.w3schools.com/python/python_regex.asp
# * Use the re.search() method
# * Use the split method to create the tokens
# * You can implement the function without using a regex library

# In[2]:


text = """Two households, both alike in dignity, In fair Verona, 
          where we lay our scene, From ancient grudge break to new mutiny, 
          Where civil blood makes civil hands unclean."""

# Provide your solution here


# Use the NLKT *word_tokenize* method to count the words of the following text
# 
# text = "NLTK is really cool!"

# In[3]:


# Please provide your solution here


# Use the NLKT gutenberg.words method to count the unique words of the following text of the gutenberg Corpus
# 
# book = shakespeare-macbeth.tx

# In[4]:


# Please provide your solution here


# ## Word Frequency

# In[5]:


# Run and analyse the following commands to create a dictionary containg:
# words as keys and a counter showing how many times a word appeared in a text frequency

text = """London is a London most most"""

# Create an empty dictionary
word_freq = dict()

# Transform the corpus to a string and then
# split the corpus variable to words

corpus_word = str(text).split()
for index in range(len(corpus_word)):
    if corpus_word[index] not in word_freq:
        word_freq[corpus_word[index]] = 1
    else:
        word_freq[corpus_word[index]] += 1
print(word_freq)


# Use the "word_tokenize" method to trasnform a text to a list of words.

# In[6]:


import nltk
nltk.download('punkt')

text = 'Analytic Tools for Data Science'

# Provide your solution here


# Download the Guteneberg coprus.
# 
# Print the list of books provided with the gutenber corpus.
# 
# * The Project Gutenberg electronic text archive, which contains some 25,000 free electronic books, hosted at http://www.gutenberg.org/.

# In[7]:


# Provide your solution here


# What is the total number of words written in the shakespeare-caesar.txt file?

# In[8]:


# Provide your solution here


# Transform the macbeth text in an NLTK Text object. Use the concordance method to print the **concordances** of the word "caesar" using the caesar text.

# In[9]:


# Provide your solution here


# Use "sents" method to find the number of sentences in shakespeare-caesar.txt

# In[10]:


# Provide your solution here


# Run the following command to download and import the books corpus

# In[11]:


import nltk
nltk.download('book')
from nltk.book import *


# What is text1 about? Run the following command.
# 

# In[12]:


text1


# Use the concordance method to find the concordance of the words "Monwtrous" and "size" in the Moby Dick text.
# 
# Hint: pass the two words as a list to the concordance method

# In[13]:


# Provide your solution here


# Use the collocations method to print the collocations for text5

# In[14]:


# Provide your solution here


# Run the following command to create a dispersion plot diagram for the word whale using the Moby Dick text.

# In[50]:


text1.dispersion_plot(['whale'])


# Create a dispersion plot diagram using the words JOIN, chat and Player using the chat corpus.
# 
# Hint: Create a list containg the three words then pass the list to the dispersion_plot method

# In[ ]:


# Provide your solution here


#  Print all the words in the Moby Dick text longer than 16 characters

# In[ ]:


# Provide your solution here


# Run the following command to download the stopwords in English

# In[ ]:


stopwords = nltk.corpus.stopwords.words('english')


# Print the first 10 stopwords in English

# In[ ]:


# Provide your solution here


# Let us use the Pirates of the Carribean dataset, to find how many times the word "crew" exists in our data. To make analysis easier, we can clean our dataset from any stopwords. To run this task there are a series of steps that you will need to perform including:
# 
# Importing the webtext, that includes the pirates.txt file
# 
# Create a frequency distribution dictionary, excluding the stopwords

# In[ ]:


# Provide your solution here


# Create the sentence tokenizer.
# 
# Use the following text to extract a list of sentences using the sent_tokenize method.

# In[18]:


import nltk
nltk.download('punkt') # Install Punkt corpus (required by the sent_tokenize)

text = """Athens is the capital and largest city of Greece. It dominates the Attica region and is one of the world's oldest cities, with its recorded history spanning over 3,400 years. Classical Athens was a powerful city-state. """


from nltk.tokenize import sent_tokenize

# Please provide your solution here


# How many sentences are in the text?

# In[19]:


# Please provide your solution here


# Create the word tokens for each sentence.
# 
# Use the the word_tokenize method to extract the words (tokens) of the first sentence (Athens is the capital and largest city of Greece.).

# In[20]:


from nltk.tokenize import word_tokenize
# Please provide your solution here


# How many words are in the first sentence?

# In[21]:


# Please provide your solution here


# Let's examine the part of speech for the first sentence. Use the Part of Speech tagger "pos_tag" method to print the tag for each word (token) in the first sentence.

# In[22]:


nltk.download('averaged_perceptron_tagger')
# Please provide your solution here


# Click on the following link to understand the meanings of some of the tags: https://www.guru99.com/pos-tagging-chunking-nltk.html
#     

# Remove the stop words from the first sentence and store the words into a new list called filtered_sentence.

# In[45]:


# Please provide your solution here


# ## Python Regular Expressions

# Regular expressions are a powerful language for matching text patterns.
# This page gives a basic introduction to regular expressions themselves
# sufficient for our Python exercises and shows how regular expressions
# work in Python. The Python "re" module provides regular expression
# support.
# 
# In Python a regular expression search is typically written as:
# 
# ``` python
# import re
# match = re.search(pat, str)
# ```

# The re.search() method takes a regular expression pattern and a string
# and searches for that pattern within the string. If the search is
# successful, search() returns a match object or None otherwise.
# Therefore, the search is usually immediately followed by an if-statement
# to test if the search succeeded, as shown in the following example which
# searches for the pattern 'word:' followed by a 3 letter word (details
# below):

# In[108]:


# Please analyse and run the following commands

import re
str = 'an example word:cat!!'
match = re.search(r'word:\w\w\w', str)
if match:
    print('found', match.group())
else:
    print('did not find')


# The code `match = re.search(pat, str)` stores the search result in a
# variable named `match`. Then the if-statement tests the match -- if true
# the search succeeded and `match.group()` is the matching text (e.g.
# 'word:cat'). Otherwise if the match is false (None to be more specific),
# then the search did not succeed, and there is no matching text.
# 
# The `'r'` at the start of the pattern string designates a python "raw" string
# which passes through backslashes without change which is very handy for regular
# expressions. I recommend that you always write pattern strings with the `'r'`
# just as a habit.

# Repeat the above commands using the pattern "I love Python" as the string and "I \w\w\w\w" as the pattern

# In[117]:


# Please provide you solution here


# ### Basic Examples
# 

# The power of regular expressions is that they can specify patterns, not
# just fixed characters. Here are the most basic patterns which match
# single chars:
# 
# -   `a`, `X`, `9`, `<` -- ordinary characters just match themselves exactly.
#     The meta-characters which do not match themselves because they have
#     special meanings are: `. ^ $ * + ? { [ ] \ | ( )`
#     (details below)
# -   `.` (a period) -- matches any single character except newline '\\n'
# -   `\w` -- (lowercase w) matches a "word" character: a letter or digit
#     or underscore `[a-zA-Z0-9\_]`. Note that although "word" is the
#     mnemonic for this, it only matches a single word char, not a
#     whole word. `\W` (upper case W) matches any non-word character.
# -   `\b` -- boundary between word and non-word
# -   `\s` -- (lowercase s) matches a single whitespace character -- space,
#     newline, return, tab, form `[ \n\r\t\f]`. `\S` (upper case S)
#     matches any non-whitespace character.
# -   `\t`, `\n`, `\r` -- tab, newline, return
# -   `\d` -- decimal digit `[0-9]` (some older regex utilities do not
#     support but `\d`, but they all support `\w` and `\s`)
# -   `^` = start, `$` = end -- match the start or end of the string
# -   `\` -- inhibit the "specialness" of a character. So, for example,
#     use `\.` to match a period or `\\` to match a slash. If you are
#     unsure if a character has special meaning, such as `@`, you can put
#     a slash in front of it, `\@`, to make sure it is treated just as
#     a character.

# The basic rules of regular expression search for a pattern within a
# string are:
# 
# -   The search proceeds through the string from start to end, stopping
#     at the first match found
# -   All of the pattern must be matched, but not all of the string
# -   If `match = re.search(pat, str)` is successful, match is not `None`
#     and in particular `match.group()` is the matching text

# ### Leftmost & Largest
# 

# First the search finds the leftmost match for the pattern, and second it
# tries to use up as much of the string as possible -- i.e. `+` and `*` go as
# far as possible (the `+` and `*` are said to be "greedy"). 

# In[131]:


import re

## i+ = one or more i's, as many as possible.
#match = re.search(r'pi+', 'piiig')

## Finds the first/leftmost solution, and within it drives the +
## as far as possible (aka 'leftmost and largest').
## In this example, note that it does not get to the second set of i's.

# Please uncomment the following command, study and analyse the result
#match = re.search(r'i+', 'piigiiii')

## \s* = zero or more whitespace chars
## Here look for 3 digits, possibly separated by whitespace.

# Please uncomment the following command, study and analyse the result
#match = re.search(r'\d\s*\d\s*\d', 'xx1 2   3xx')

# Please uncomment the following command, study and analyse the result
#match = re.search(r'\d\s*\d\s*\d', 'xx12  3xx')


# Please uncomment the following command, study and analyse the result
#match = re.search(r'\d\s*\d\s*\d', 'xx123xx')

## ^ = matches the start of string, so this fails:

# Please uncomment the following command, study and analyse the result
#match = re.search(r'^b\w+', 'foobar')

## but without the ^ it succeeds:

# Please uncomment the following command, study and analyse the result
#match = re.search(r'b\w+', 'foobar')

print(match)


# # Emails Example
# 
# Suppose you want to find the email address inside the string 
# ' purple alice-b@google.com monkey dishwasher'. We'll use this as a running example
# to demonstrate more regular expression features. Here's an attempt using
# the pattern r'\\w+@\\w+':

# In[132]:


# Please analyse and run the following commands

import re
str = 'purple alice-b@google.com monkey dishwasher'
match = re.search(r'\w+@\w+', str)
if match:
    print(match.group())


# ## Group Extraction

# The "group" feature of a regular expression allows you to pick out parts of the matching text. Suppose for the emails problem that we want to extract the username and host separately. To do this, add parenthesis () around the username and host in the pattern, like this: r'([\w.-]+)@([\w.-]+)'. In this case, the parenthesis do not change what the pattern will match, instead they establish logical "groups" inside of the match text. On a successful search, match.group(1) is the match text corresponding to the 1st left parenthesis, and match.group(2) is the text corresponding to the 2nd left parenthesis. The plain match.group() is still the whole match text as usual.
# 
# 

# In[133]:


# Please analyse and run the following commands

str = 'purple alice-b@google.com monkey dishwasher'
match = re.search('([\w.-]+)@([\w.-]+)', str)
if match:
    print(match.group())   ## 'alice-b@google.com' (the whole match)
    print(match.group(1))  ## 'alice-b' (the username, group 1)
    print(match.group(2))  ## 'google.com' (the host, group 2)


# A common workflow with regular expressions is that you write a pattern
# for the thing you are looking for, adding parenthesis groups to extract
# the parts you want.

# ## findall

# `findall()` is probably the single most powerful function in the `re`
# module. Above we used `re.search()` to find the first match for a pattern.
# `findall()` finds *all* the matches and returns them as a list of
# strings, with each string representing one match.

# In[135]:


## Suppose we have a text with many email addresses
str = 'purple alice@google.com, blah monkey bob@abc.com blah dishwasher'

## Here re.findall() returns a list of all the found email strings
emails = re.findall(r'[\w\.-]+@[\w\.-]+', str) ## ['alice@google.com', 'bob@abc.com']
for email in emails:
    # do something with each found email string
    print(email)

