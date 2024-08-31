# PDF Search Engine in Python
A python project that allows for a quick **search through any PDF and ranks the results** based on both the number of occurrences of the search query on the page, as well as the 'page rank' of the page (how many times have other pages referenced the current page). The works through a console menu.

![image](https://github.com/user-attachments/assets/f546da60-0095-4577-82a2-eea0f0d5104e)


## Technologies used
The project was built in Python 3.
The pages of the PDF were organized in a **graph**, and a **trie** was used for word searching.

### External libraries

- *PyMuPDF* or now *Fitz* - used for parsing the PDF into 1 txt file per page, also used for extracting pages and highlighting the query within the 'first 10 results' request
- *difflib* - finding similar words, for the 'did you mean' feature, using Levenshtein distance
- Collections -> *defaultdict*  - used for an alternative to a dictionary
- *os, re* - used os and re for writing, creating files and regular expressions

## Features
The app supports 1 word search, multiple word searches, logical operators (and, or and binary not), it handles searching for a phrase, autocomplete and a 'did you mean' feature which gets triggered if there are no search results. As an additional feature, a

