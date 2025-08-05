tools=[
    {
        "type": "function",
        "function": {
            "name": "read_txt_file",
            "description": "Read the contents of a .txt file from the filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The full path to the text file"
                    }
                },
                "required": ["path"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_txt_file",
            "description": "Write content to a .txt file on the filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The full path to the text file"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["path", "content"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_auto_update_interval",
            "description": "Set or disable the periodic auto-update interval for the agent's system prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "interval_seconds": {
                        "type": "integer",
                        "description": "Interval in seconds. Set to 0 to disable auto-update."
                    }
                },
                "required": ["interval_seconds"],
                "additionalProperties": False
            }
        }
    }
]



def calculate_square(number):
    """Calculate the square of a number."""
    return number * number

