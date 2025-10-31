import re
import unicodedata
from urllib.parse import urlparse

__all__ = ('is_url', 'sanitize_identifier')


def capitalize(input_string):
    if not input_string:
        return ''
    return input_string[0].upper() + input_string[1:]


def is_url(text):
    try:
        result = urlparse(text)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join(c for c in nfkd_form if not unicodedata.combining(c))


def sanitize_name_python_keywords(name: str) -> str:
    import keyword

    if name in keyword.kwlist:
        return f'{name}_'
    return name


def sanitize_parameter_field_name(name: str) -> str:
    """Sanitize parameter or field names to be valid Python identifiers.

    - Replace spaces and hyphens with underscores
    - Remove other invalid characters
    - Ensure it doesn't start with a digit
    """
    if not name:
        raise ValueError('Name cannot be empty')

    sanitized = sanitize_name_python_keywords(name)
    sanitized = re.sub(r'[-\s]+', '_', remove_accents(sanitized))
    sanitized = re.sub(r'[^A-Za-z0-9_]', '', sanitized)

    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized
    return sanitized


def sanitize_identifier(name: str) -> str:
    """Convert a string into a valid Python identifier.

    - Replace spaces and hyphens with underscores
    - Remove other invalid characters
    - Ensure it doesn't start with a digit
    - Convert to PascalCase for class names
    """
    if not name:
        return 'UnnamedType'

    # Replace spaces and hyphens with underscores, then split
    parts = re.sub(r'[^A-Za-z0-9]+', '_', remove_accents(name)).split('_')

    # Capitalize each part and join (PascalCase)
    if len(parts) == 1:
        sanitized = parts[0]
    else:
        sanitized = ''.join(capitalize(part) for part in parts if part)

    # Remove any remaining invalid characters
    sanitized = re.sub(r'[^A-Za-z0-9_]', '', sanitized)

    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized

    # If empty after sanitization, return a default
    return sanitized or 'UnnamedType'
