"""
Dummy tools used to test the harness
"""

from agent_harness.tool_registry import tool


@tool(
    name="RetrieveDataA",
    description="Retrieves Data A using a password.",
    parameters={
        "type": "object",
        "properties": {
            "password": {"type": "string", "description": "Password required to access Data A."}
        },
        "required": ["password"],
    },
)
def RetrieveDataA(password: str) -> str:
    """
    Tool used to retrieve Data A

    Args:
        password: A password used to access Data A

    Returns:
        Data A
    """

    if password == "123A":
        return "345B"

    else:
        return "Wrong Password"


@tool(
    name="RetrieveDataB",
    description="Retrieves Data B using a password.",
    parameters={
        "type": "object",
        "properties": {
            "password": {"type": "string", "description": "Password required to access Data B."}
        },
        "required": ["password"],
    },
)
def RetrieveDataB(password: str) -> str:
    """
    Tool used to retrieve Data B

    Args:
        password: A password used to access Data B

    Returns:
        Data B
    """

    if password == "345B":
        return "Hello World"

    else:
        return "Wrong Password"
