#!/usr/bin/env python

# question-and-answer.py - given a corpus (study carrel) and a question, output a list of answers
# see: https://haystack.deepset.ai/tutorials/03_basic_qa_pipeline_without_elasticsearch

# Eric Lease Morgan <emorgan@nd.edu>
# (c) University of Notre Dame; distribured under a GNU Public License

# December 31, 2022 - first investigations; very cool
# January   2, 2023 - added smarter documents


# configure
MAXIMUM       = 12
THRESHOLD     = .3
SPLITLENGTH   = 100
RESPECT       = False
INDEXES       = './indexes'
MODEL         = 'deepset/roberta-base-squad2'
EMBEDDEDMODEL = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
MODELFORMAT   = 'sentence_transformers'
DB            = '##CARREL##.db'
INDEX         = '##CARREL##.faiss'
CONFIG        = '##CARREL##.json'
URL           = 'sqlite:///##DB##'

# require
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes           import EmbeddingRetriever
from haystack.nodes           import FARMReader
from haystack.pipelines       import ExtractiveQAPipeline
from haystack.utils           import convert_files_to_docs
from pathlib                  import Path
import rdr
import re
import sys
from pprint import pprint
from haystack.nodes import PreProcessor
from haystack.nodes import RAGenerator, DensePassageRetriever


# get input
if len( sys.argv ) != 3 : sys.exit( 'Usage: ' + sys.argv[ 0 ] + " <carrel> <question>" )
carrel   = sys.argv[ 1 ]
question = sys.argv[ 2 ]


def carrel2documents( carrel ) :

	'''Given a study carrel, output a list of documents amenable to an FAISSDocumentStore'''
	
	# configure and initialize
	PATTERN   = '*.txt'
	documents = []
	corpus    = rdr.configuration( 'localLibrary' )/carrel/rdr.TXT
	
	# process each plain text file in the given carrel
	for file in corpus.glob( PATTERN ) :
		
		# slurp up the given file
		with open( file ) as handle : content = handle.read()
		
		# normalize the content and file name
		content = content.replace( '-\n', '' )
		content = content.replace( '- ', '' )
		content = content.replace( " 's", "'s" )
		name    = file.name
		
		# create a document and update the list of documents
		document = { 'content':content, 'meta':{ 'name':name } }
		documents.append( document )
				
	# done
	return( documents )


# initialize index file names
indexes = Path( INDEXES )
db      = indexes/( DB.replace( '##CARREL##', carrel ) )
index   = indexes/( INDEX.replace( '##CARREL##', carrel ) )
config  = indexes/( CONFIG.replace( '##CARREL##', carrel ) )

# check to see if the files exist
if not db.exists() :

	# index, and create a data store
	sys.stderr.write( 'Indexing. Please be patient; if you had to read whole books, it would take you a long time too.\n' )
	documents    = carrel2documents( carrel )
	preprocessor = PreProcessor( split_length=SPLITLENGTH, split_respect_sentence_boundary=RESPECT )
	documents    = preprocessor.process( documents )



	# Initialize FAISS document store.
	# Set `return_embedding` to `True`, so generator doesn't have to perform re-embedding
	store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True, sql_url=URL.replace( '##DB##', str( db ) ))

	# Initialize DPR Retriever to encode documents, encode question and query documents
	retriever = DensePassageRetriever(
		document_store=store,
		query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
		passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
		use_gpu=False,
		embed_title=True,
	)

	# Initialize RAG Generator
	generator = RAGenerator(
		model_name_or_path="facebook/rag-token-nq",
		use_gpu=False,
		top_k=1,
		max_length=200,
		min_length=2,
		embed_title=True,
		num_beams=2,
	)

	# Delete existing documents in documents store
	store.delete_documents()

	# Write documents to document store
	store.write_documents(documents)

	# Add documents embeddings to index
	store.update_embeddings(retriever=retriever)

	store.write_documents( documents )
	store.save( index_path=index )

	
else :

	# load the previously created index
	sys.stderr.write( 'Loading index.\n' )
	store  = FAISSDocumentStore( faiss_index_path=index )
	store.load( index_path=index, config_path=config )
	retriever = DensePassageRetriever(
		document_store=store,
		query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
		passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
		use_gpu=False,
		embed_title=True,
	)

	# Initialize RAG Generator
	generator = RAGenerator(
		model_name_or_path="facebook/rag-token-nq",
		use_gpu=False,
		top_k=1,
		max_length=200,
		min_length=2,
		embed_title=True,
		num_beams=2,
	)


QUESTIONS = [
    "How did Hector die?",
    "What did the men eat?",
    "Who were the suitors?"
]


# Or alternatively use the Pipeline class
from haystack.pipelines import GenerativeQAPipeline
from haystack.utils import print_answers

pipe = GenerativeQAPipeline(generator=generator, retriever=retriever)
for question in QUESTIONS:
    res = pipe.run(query=question, params={"Generator": {"top_k": 10}, "Retriever": {"top_k": 5}})
    pprint( res )
    
