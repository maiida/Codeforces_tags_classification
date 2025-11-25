import ast

def parse_tags(tags_str):
    """
    Safely parses a string representation of a list of tags.
    Handles "['math', 'trees']" -> ['math', 'trees'].
    Returns an empty list if parsing fails or input is invalid.
    """
    if not isinstance(tags_str, str):
        if isinstance(tags_str, list):
            return tags_str
        return []
    
    try:
        tags = ast.literal_eval(tags_str)
        if isinstance(tags, list):
            return tags
        return []
    except (ValueError, SyntaxError):
        return []