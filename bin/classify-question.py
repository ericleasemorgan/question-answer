#!/usr/bin/env python

# see: https://haystack.deepset.ai/tutorials/14_query_classifier

# configure
QUERIES = [

    'Arya Stark father',
    'Who was the father of Arya Stark',
    'Lord Eddard was the father of Arya Stark',
    'Who is/was Jove?',
    'What did they eat?',
    'Who killed Hector?',
    'covid-19 cure vaccine',
    'How do you make scrabbled eggs?'
    
]

# require
from haystack.nodes import SklearnQueryClassifier
from pprint         import pprint
import pandas       as pd
import warnings

# initialize
warnings.filterwarnings( 'ignore' )
classifier  = SklearnQueryClassifier()
results     = { 'query':[], 'type':[] }

# process each query
for query in QUERIES :

    # do the work
    type = classifier.run( query=query )[ 1 ]

    # update the results
    results[ 'query' ].append( query )
    results[ 'type' ].append( 'question' if type == 'output_1' else 'keyword' )

# done
results = pd.DataFrame.from_dict( results )
print( results )
exit()
