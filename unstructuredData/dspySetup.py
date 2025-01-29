# Signature tells us what the model should take as input and give as output.
# Modules define how it should do the above task. ex - Predict, ChainOfThought, etc.
# We can define our own custom modules as well.

import dspy
import os
from dotenv import load_dotenv

load_dotenv()


lm = dspy.LM(os.getenv("AZURE_AI_MODEL_NAME"), api_key=os.getenv("AZURE_AI_API_KEY"), api_base=os.getenv("AZURE_AI_API_BASE"), api_version=os.getenv("AZURE_AI_API_VERSION"))
dspy.configure(lm=lm)


# ********Option 1******************
question = "Who won the Champions Trophy 2017 and by how many runs and who scored the most number of runs in that match and took the most number of wickets?"

# *********Predict******************
# Predict: Question -> LM -> Answer

# **************Option 1******************
predict= dspy.Predict("question -> answer")
prediction = predict(question=question) 
print("Prediction: ",prediction)
print("Predicted Answer: ",prediction.answer)
print(lm.inspect_history(n=1))

# ********Option 2******************
class QA(dspy.Signature):
  question = dspy.InputField()
  answer = dspy.OutputField()

predict = dspy.Predict(QA)
prediction = predict(question=question)
print("Predicted Answer: ",prediction.answer)
print(lm.inspect_history(n=1))

# ********Option 3******************
#The desc and doc string matter, basically it is given as a hint to the model

class QA(dspy.Signature):
  """Given a question, generate the answer."""
  question = dspy.InputField(desc="The question to answer.")
  answer = dspy.OutputField(desc="Keep it short and sweet.")

predict = dspy.Predict(QA)
prediction = predict(question=question)
print("Predicted Answer: ",prediction.answer)
print(lm.inspect_history(n=1))


# ********Chain of Thought******************
# It is used when we want model to think and then give the answer, usually useful when having multiple questions linked to each other.
# Question -> LM -> Thought(Reasoning) + Answer

answer = dspy.ChainOfThought(QA)
prediction = answer(question=question)
print("Predicted Answer: ",prediction.answer)
# print("Rationale:",prediction.rationale)
lm.inspect_history(n=1)


# *********Custom Module******************
# Question -> LM -> Thought(Reasoning) + Question -> Answer
multi_step_question = "Who scored the most runs in the Champions Trophy 2017? and Who took the most wickets in that match? Also, tell me which state of the country they belong to? And what is the famous food of that state?"

class DoubleChainOfThoughtModule(dspy.Module):
  def __init__(self):
    self.cot1 = dspy.ChainOfThought("question -> step_by_step_thought")
    self.cot2 = dspy.ChainOfThought("question, thought -> concise_answer")

  def forward(self, question):
    thought = self.cot1(question=question).step_by_step_thought
    answer = self.cot2(question=question, thought=thought).concise_answer
    return dspy.Prediction(thought=thought, answer=answer)
  

doubleCot = DoubleChainOfThoughtModule()
prediction = doubleCot(question = multi_step_question)
print("Prediction: ",prediction)
print("Predicted Answer: ",prediction.answer)
print("Thought: ",prediction.thought)
lm.inspect_history(n=2)

# *********Typed Answer******************
# We should use pydantic to get output in different data types like float, int, bool, json, etc.
# Note - It might not work always but most of the time it works.

# Example 1
from pydantic import BaseModel, Field

class AnswerConfidence(BaseModel):
  answer: str = Field("Answer 1-5 Words")
  confidence: float = Field("Confidence in the answer between 0-1")

class QAWithConfidence(dspy.Signature):
  """
  Given a question, generate the answer with confidence value.
  """
  question = dspy.InputField()
  answer: AnswerConfidence = dspy.OutputField()


question = "Who was the top scorer from India in the Champions Trophy 2017 final match?"

predict = dspy.TypedChainOfThought(QAWithConfidence)
output = predict(question=question)
print("Predictions: ",output)
print("Predicted Answer 1: ",output.answer)
print("Predicted Answer 2 : ",output.answer.answer)
print("Confidence: ",output.answer.confidence)

# Example 2
class Answer(BaseModel):
  country: str = Field("Country Name")
  year: int = Field("Year of the event")

class QAList(dspy.Signature):
  """
  Given a question, generate answer in JSON readable format.
  """
  question = dspy.InputField()
  answer_list: list[Answer] = dspy.OutputField()

question = "Which countries have won the ICC World Cup since 2000 and in which year?"

predict = dspy.TypedChainOfThought(QAList)
answer = predict(question=question)
print("Predictions: ",answer)
print("Predicted Answer 1: ",answer.answer_list)
lm.inspect_history(n=1)


# *********RAG******************
# ColBERTv2 is a retrieval server that stores wikipedia articles and can be used to retrieve relevant articles for a given query. till 2017.
# Take it from official website
