import discord
from discord.ext import commands
from llm.llm_rag_2 import LLM_RAG
from vectordb.vector_db import VectorDB
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

TOKEN = os.getenv('TOKEN')
CHANNEL_ID = os.getenv('CHANNEL_ID')
 
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print('Bot: {}'.format(bot.user))

@bot.command()
async def ping(ctx,*,text):
    result = llm.query(text)
    await ctx.send(result)

# 현재 스크립트가 위치한 디렉토리 경로를 가져옵니다.
current_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_directory)

vector_db_path = './FAISS'
print('=== Initialize ...... ===')
notice_vdb = VectorDB()
notice_vdb.load_local(vector_db_path + '/NOTICE')
school_vdb = VectorDB()
school_vdb.load_local(vector_db_path + '/SCHOOL_INFO')

llm = LLM_RAG(trace=True)
llm.set_retriver(data_type='notice', retriever=notice_vdb.get_retriever())
llm.set_retriver(data_type='school_info', retriever=school_vdb.get_retriever())
llm.set_chain()

bot.run(TOKEN)
