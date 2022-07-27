import fire
import pandas as pd
from tools import extract_acronyms
from tqdm import tqdm
import wikipedia
from textsearch import TextSearch
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')

def create_lexicon(file_name:str='rnr-examples.csv', column:str="text", top_n_acronyms:int=200):
    """ Create a lexicon based on the Acronyms extracted from a document.
    """
    
    # Load the data file
    df = pd.read_table(file_name, sep=",", header=0, encoding='utf-8')

    # Extract the Top Acronyms across all the Documents
    # Truncated for Time
    acronyms = extract_acronyms(" ".join(df[column]))
    top_acronyms = sorted(acronyms.items(), key = lambda x:x[1], reverse=True)[:top_n_acronyms]


    # Define a set of screening keywords for quicke fire-disambiguation
    kws = ["bank", "regulat", "financial",  "securities", "directive",  "fund", "board", "standard","money","finance"]

    # Create a text tagger based on the above keywords
    tagger = TextSearch("sensitive", dict)
    tagger.add(kws)

    # Get Definitions for some of the most widely used Acronyms
    acronym_wiki = []

    for i, (acronym, _) in enumerate(tqdm(top_acronyms)):
        print(acronym)

        pages = wikipedia.search(acronym)
        # Pull in several pages worth, and attempt to find a good match
        for page in pages:
            try:
                summary = wikipedia.page(page).summary
                
                # tag the summary with keywords
                tags = tagger.findall(summary)

                acronym_wiki.append({
                    "acronym":acronym,
                    "summary":summary,
                    "tags":tags
                })
                
            except:
                # If there are no pages, we can ignore this
                pass

        # Do a progressive partial dump
        if i%10 ==0:
            lexicon = pd.DataFrame(acronym_wiki)
            lexicon.to_csv("./data/acronym_lexicon_partial.csv")            
            
    lexicon = pd.DataFrame(acronym_wiki)
    lexicon.to_csv("./data/acronym_lexicon.csv")            

if __name__ == '__main__':
  fire.Fire(create_lexicon)
