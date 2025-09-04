import json
from datetime import date, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from mistralai import Mistral

# --- Step 1: Define Pydantic Models for Tool Inputs ---

# Math Tool Input: Validates a mathematical expression
class MathInput(BaseModel):
    expression: str = Field(
        ..., 
        description="A mathematical expression to evaluate (e.g., '5 + 3 * 2')"
    )

# Unit Conversion Tool Input: Validates unit conversion parameters
class LengthUnit(str, Enum):
    km = "km"
    m = "m"
    cm = "cm"
    miles = "miles"
    yards = "yards"
    feet = "feet"
    inches = "inches"

class UnitConversionInput(BaseModel):
    value: float = Field(..., description="The numerical value to convert")
    from_unit: LengthUnit = Field(..., description="The unit to convert from")
    to_unit: LengthUnit = Field(..., description="The unit to convert to")

# Date Tool Input: Validates date operation parameters
class DateOperation(str, Enum):
    get_current = "get_current"
    add_days = "add_days"
    subtract_days = "subtract_days"
    diff_days = "diff_days"

class DateInput(BaseModel):
    operation: DateOperation = Field(..., description="The date operation to perform")
    base_date: date = Field(None, description="Base date for operations (optional)")
    days: int = Field(None, description="Number of days for add/subtract (optional)")
    second_date: date = Field(None, description="Second date for difference (optional)")

# Text Analysis Tool Input: Validates text analysis parameters
class TextAnalysisInput(BaseModel):
    text: str = Field(..., description="The text to analyze")
    character: str = Field(..., description="The character to count in the text")
    case_sensitive: bool = Field(False, description="Whether the search should be case sensitive")

# --- Step 2: Implement Tool Functions ---

def math_tool(input: MathInput) -> float:
    """Evaluates a mathematical expression."""
    # Note: eval() is used for simplicity; in production, use safer alternatives
    try:
        result = eval(input.expression)
        return result
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")

# Conversion factors to meters (base unit)
conversion_to_m = {
    "km": 1000,
    "m": 1,
    "cm": 0.01,
    "miles": 1609.34,
    "yards": 0.9144,
    "feet": 0.3048,
    "inches": 0.0254,
}

def unit_conversion_tool(input: UnitConversionInput) -> float:
    """Converts a value between length units."""
    if input.from_unit not in conversion_to_m or input.to_unit not in conversion_to_m:
        raise ValueError("Invalid unit specified")
    value_in_m = input.value * conversion_to_m[input.from_unit]
    converted_value = value_in_m / conversion_to_m[input.to_unit]
    return converted_value

def date_tool(input: DateInput) -> str:
    """Performs date operations."""
    if input.operation == "get_current":
        return str(date.today())
    elif input.operation == "add_days":
        if input.base_date is None or input.days is None:
            raise ValueError("base_date and days are required for add_days")
        new_date = input.base_date + timedelta(days=input.days)
        return str(new_date)
    elif input.operation == "subtract_days":
        if input.base_date is None or input.days is None:
            raise ValueError("base_date and days are required for subtract_days")
        new_date = input.base_date - timedelta(days=input.days)
        return str(new_date)
    elif input.operation == "diff_days":
        if input.base_date is None or input.second_date is None:
            raise ValueError("base_date and second_date are required for diff_days")
        diff = (input.second_date - input.base_date).days
        return str(diff)
    else:
        raise ValueError("Invalid date operation")

def text_analysis_tool(input: TextAnalysisInput) -> str:
    """Counts occurrences of a character in text."""
    text = input.text if input.case_sensitive else input.text.lower()
    character = input.character if input.case_sensitive else input.character.lower()
    
    count = text.count(character)
    
    return f"The character '{input.character}' appears {count} times in '{input.text}'"

# --- Step 3: Define Tool Specifications for Mistral AI ---

tools = [
    {
        "type": "function",
        "function": {
            "name": "math",
            "description": "Evaluates a mathematical expression (e.g., '5 + 3 * 2')",
            "parameters": MathInput.schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_conversion",
            "description": "Converts a value between length units (e.g., km to miles)",
            "parameters": UnitConversionInput.schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "date_tool",
            "description": "Performs date operations (e.g., add days, get current date)",
            "parameters": DateInput.schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_analysis",
            "description": "Analyzes text for specific character occurrences",
            "parameters": TextAnalysisInput.schema(),
        },
    },
]

# --- Step 4: Set Up Mistral AI Client ---

# Replace 'your_api_key' with your actual Mistral AI API key
client = Mistral(api_key= "")

# --- Step 5: Helper Function to Execute Tools ---

def call_tool(tool_name: str, params: dict) -> str:
    """Executes the appropriate tool based on name and parameters."""
    try:
        if tool_name == "math":
            input_model = MathInput(**params)
            return str(math_tool(input_model))
        elif tool_name == "unit_conversion":
            input_model = UnitConversionInput(**params)
            return str(unit_conversion_tool(input_model))
        elif tool_name == "date_tool":
            input_model = DateInput(**params)
            return date_tool(input_model)
        elif tool_name == "text_analysis":
            input_model = TextAnalysisInput(**params)
            return str(text_analysis_tool(input_model))
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    except Exception as e:
        return f"Error: {str(e)}"

# --- Step 6: Conversation Loop ---

def handle_conversation(user_query: str) -> None:
    """Handles the conversation with the agent, including tool calls."""
    print(f"User Query: {user_query}")
    messages = [{"role": "user", "content": user_query}]

    while True:
        # Send the message to Mistral AI and get a response
        response = client.chat.complete(
            model="mistral-small-2506",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        if assistant_message.tool_calls:
            # Handle each tool call in the response
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                params = json.loads(tool_call.function.arguments)
                print(f"Agent decided to use tool: {tool_name} with params: {params}")
                
                result = call_tool(tool_name, params)
                print(f"Tool result: {result}")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            print(f"Agent Response: {assistant_message.content}")
            break

# --- Step 7: Demonstrate the Agent with Example Queries ---

if __name__ == "__main__":
    print("=== Welcome to the Agent Tools Demo ===\n")

    # Example 1: Using the Math Tool
    print("--- Example 1: Mathematical Calculation ---")
    handle_conversation("What is 698552 multiplied by 659 subtract 25574 divided by 2 is an eighth of the people who attended the show. How many people attended the show?")

    # Example 2: Using the Unit Conversion Tool
    print("\n--- Example 2: Unit Conversion ---")
    handle_conversation("Convert 10 kilometers to miles.")

    # Example 3: Using the Date Tool (Add Days)
    print("\n--- Example 3: Date Operation (Add Days) ---")
    handle_conversation("What is the date 5 days from now?")

    # Example 4: Using the Date Tool (Difference in Days)
    print("\n--- Example 4: Date Operation (Difference) ---")
    handle_conversation("How many days are between 2023-01-01 and 2023-12-31?")

    # Example 5: Direct Response (No Tool Needed)
    print("\n--- Example 5: Direct Response ---")
    handle_conversation("Hello, how are you?")

    # Example 6: Text Analysis Tool
    print("\n--- Example 6: Text Analysis (Character Count) ---")
    handle_conversation("How many r's are in strawberry?")