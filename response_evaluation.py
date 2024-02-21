import logging
import sys

from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.llms.openai import OpenAI

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# create llm
llm = OpenAI(model="gpt-4", temperature=0.0)

PERSIST_DIR = "./storage"
# build index
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
vector_index = load_index_from_storage(storage_context)

# define evaluator
evaluator = FaithfulnessEvaluator(llm=llm)

# query index
query_engine = vector_index.as_query_engine()
response = query_engine.query(
    "What battles took place in New York City in the American Revolution?"
)
print(response)

eval_result = evaluator.evaluate_response(response=response)
print(str(eval_result))
