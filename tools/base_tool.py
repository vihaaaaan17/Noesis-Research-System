"""
tools/base_tool.py
---------------------------------------------------------------------
The BaseTool class that every tool must inherit from.
 
Why a base class?
  * Forces every tool to have a name + description (LLM needs these)
  * Forces every tool to implement run() with a standard signature
  * Lets us treat all tools the same way - store in a list, loop over
    them, pass them around without knowing what type they are
 
The description field is THE most important thing in a tool.
The LLM reads the description to decide:
  1. Whether to use this tool at all
  2. What kind of input to pass it
So write descriptions like you're explaining the tool to a smart person
who has never seen your code.
---------------------------------------------------------------------
"""

from abc import ABC, abstractmethod

class BaseTool(ABC):
    """
    Abstract base class for all tools.
 
    Every tool you create must:
      1. Inherit from BaseTool
      2. Set self.name and self.description in __init__
      3. Implement the run(input_text) method
 
    The run() method always takes a plain string and returns a plain
    string - this keeps the interface simple and LLM-friendly.
    """

    def __init__(self , name :str , description:str):
        """
        Parameters
        ----------
        name : str
            Short identifier for the tool. Used by the LLM to call it.
            e.g. "calculator", "web_search", "read_file"
 
        description : str
            Plain English explanation of what this tool does, what
            input it expects, and what it returns.
            The LLM uses this to decide when and how to use the tool.
        """
        self.name = name 
        self.description = description


    @abstractmethod
    def run(self,input_text:str)->str:
        """
        Execute the tool with the given input.
 
        Parameters
        ----------
        input_text : str
            The input the LLM decided to pass to this tool.
            Always a string - even if it represents a number or JSON.
 
        Returns
        -------
        str
            The result of running the tool.
            Always a string - gets injected back into the LLM's context.
        """
        pass


    def __repr__(self):
        return f"Tool(name={self.name!r})"
        

    def to_dict(self)->dict:
        """
        Return a dict representation of the tool.
        Used when building the tool list we show to the LLM in the
        system prompt - so it knows what tools exist.
        """
        return {
            "name":        self.name,
            "description": self.description,
        }