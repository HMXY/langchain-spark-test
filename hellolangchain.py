from SparkApiLangChain import Spark
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain.utilities import BingSearchAPIWrapper

spark = Spark(max_tokens=2048, temperature=0.9)
prompt = PromptTemplate(
    input_variables=["products"],
    template="what is a good name for a company that makes {products}"
)

# chain = LLMChain(llm=spark, prompt=prompt)
# print(chain.run("water"))


human_message_prompt = HumanMessagePromptTemplate(prompt=prompt)
chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
chain_chat = LLMChain(llm=spark, prompt=chat_prompt_template)
# print(chain_chat.run("colorful socks"))

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
# tools = load_tools(['BingSearchAPIWrapper', 'llm-math'], llm=spark)

# Finally, let's initialize an agent with the tools, the language model,
# and the type of agent we want to use.
# agent = initialize_agent(tools, spark, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")

search = BingSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
        return_direct=True
    )
]

from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish


class FakeAgent(BaseSingleActionAgent):
    """Fake Custom Agent."""

    @property
    def input_keys(self):
        return ["input"]

    def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")

    async def aplan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")


agent = FakeAgent()
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
agent_executor.run("How many people live in canada as of 2023?")