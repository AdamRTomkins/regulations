import streamlit as st
import pandas as pd
import numpy as np

from streamlit_shap import st_shap
import shap

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier


@st.cache()
def load_data():
    df = pd.read_table('rnr-examples.csv', sep=",", header=0, encoding='utf-8')
    return df

@st.cache()
def load_taxonomy():
    return pd.read_csv("data/financial_taxonomy.csv")


def df_filter(column, name, dataframe):

    values = ["All"]
    values.extend(set(dataframe[column].to_list()))
    filter_vals = st.multiselect(name, values, ["All"])

    if "All" in filter_vals:
        return dataframe
    else:
        return dataframe[dataframe[column].isin(filter_vals)]

st.sidebar.header("Regulatory Classifcation Data Explorer")

with st.spinner("Loading Data"):
    df = load_data()
    taxonomy = load_taxonomy()

def draw_data_viewer():
    
    st.header("Explore Data")

    with st.sidebar:
        st.session_state["fdata"] = df_filter("label", "Classification", df)

    for record in st.session_state["fdata"].to_dict('records')[:20]:
        with st.expander(record["text"].split(".")[0]):
            st.text(record["text"])

        

def draw_taxonomy():

    st.header("Taxonomic Creation")

    st.info("Here, we look at how we can derive knowledge from the given data. We propose a simple architecture to extract acronyms, and build a taxonomy of financial concepts. ")

    st.image("data/images/acronym_extraction.png")
    st.image("data/images/lexicon_creation.png")

    # Allow Taxonomy filters
    with st.sidebar:
        with st.form("Filter Taxonomy"):
            filtered_taxonomy = df_filter("semantic_type", "Semantic Type", taxonomy)
            filtered_taxonomy = df_filter("acronym", "Entity", filtered_taxonomy)

            apply_filter = st.form_submit_button("Apply")
            if apply_filter:
                st.session_state["ftaxonomy"] = filtered_taxonomy
    
    # View Taxonomy Records
    for record in st.session_state["ftaxonomy"].to_dict('records')[:10]:
        with st.expander(f"{record['acronym']} :  {record['semantic_type']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"__{record['acronym']}__")

            with col2:
                st.write(f"__{record['semantic_type']}__")
            st.write(record["summary"])

    st.header("Knowledge Representations")

    
    if st.button("Draw Taxonomy Graph"):
        graph()

def draw_introduction():
    with open('README.txt') as f:
        lines = f.readlines()

    st.markdown("\n".join(lines))

def draw_notes():
    with open('Notes.md') as f:
        lines = f.readlines()
    for l in lines:
        st.write(l)

def draw_readme():
    with open('README.md') as f:
        lines = f.readlines()
    for l in lines:
        st.write(l)

def graph():
    from streamlit_agraph import agraph, Node, Edge, Config

    nodes = []
    edges = []

    semantic_types = set(st.session_state.ftaxonomy["semantic_type"].to_list())
    acronyms = set(st.session_state.ftaxonomy["acronym"].to_list())
    
    nodes.append( Node(id="root", 
                        label="Taxonomy", 
                        size=400, 
                        ) 
                    ) # includes **kwargs


    for semantic_type in semantic_types:

        nodes.append( Node(id=semantic_type, 
                        label=semantic_type, 
                        size=400, 
                        ) 
                    ) # includes **kwargs

    for acronym in acronyms:

        nodes.append( Node(id=acronym, 
                        label=acronym, 
                        size=400, 
                        ) 
                    ) # includes **kwargs
                    

    for semantic_type in semantic_types:

        edges.append( Edge(source="root", 
                                label="subclass", 
                                target=semantic_type, 
                                type="CURVE_SMOOTH") 
                            ) # includes **kwargs


    for record in st.session_state.ftaxonomy.to_dict('records'):

        edges.append( Edge(source=record["acronym"], 
                        label="has_semantic_type", 
                        target=record["semantic_type"], 
                        type="CURVE_SMOOTH") 
                    ) # includes **kwargs

    config = Config(width=500, 
                    height=500, 
                    directed=True,
                    nodeHighlightBehavior=True, 
                    highlightColor="#F7A7A6", # or "blue"
                    collapsible=True,
                    node={'labelProperty':'label'},
                    link={'labelProperty': 'label', 'renderLabel': True}
                    # **kwargs e.g. node_size=1000 or node_color="blue"
                    ) 

    return_value = agraph(nodes=nodes, 
                        edges=edges, 
                        config=config)

@st.experimental_memo
def load_model(X, y):

    # Some validation
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import recall_score
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    from sklearn.ensemble import RandomForestClassifier

    clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf',  RandomForestClassifier()),
    ])

    with st.spinner("Fitting Model"):
        clf.fit(X, y)
    
    return clf


pages = {
    "Introduction":draw_introduction,
    "Report":draw_readme,
    "Project Notes":draw_notes,
    "Data Explorer":draw_data_viewer,
    "Taxonomy Understanding":draw_taxonomy,
}



if "data" not in st.session_state:
    st.session_state["data"] = df.copy()

if "fdata" not in st.session_state:
    st.session_state["fdata"] = df.copy()


if "taxonomy" not in st.session_state:
    st.session_state["taxonomy"] = taxonomy.copy()

if "ftaxonomy" not in st.session_state:
    st.session_state["ftaxonomy"] = taxonomy.copy()

with st.sidebar:
    page = pages[st.selectbox("Page", pages.keys())]

page()