import os
import importlib
import inspect

def get_tools():
    """Dynamically discovers and returns a list of all available tools."""
    tools_dir = os.path.dirname(__file__)
    discovered_tools = []

    for item in os.listdir(tools_dir):
        item_path = os.path.join(tools_dir, item)
        
        # Look for directories that are not __pycache__
        if os.path.isdir(item_path) and item != "__pycache__":
            # Assume the module name is the same as the directory name
            # and the file inside is also named the same
            module_name = f"tools.{item}.{item}"
            try:
                module = importlib.import_module(module_name)
                
                # Iterate through members of the module to find decorated tools
                for name, obj in inspect.getmembers(module):
                    # Check if it's a tool by checking for the 'name' and 'description' attributes
                    # which are present on langchain tools
                    if hasattr(obj, "name") and hasattr(obj, "description"):
                        discovered_tools.append(obj)
            except (ImportError, AttributeError) as e:
                # Some modules might not follow the naming convention, skip them
                continue
                
    return discovered_tools

