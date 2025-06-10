import re


def format_key_suffix(key_suffix):
    if key_suffix:
        if key_suffix[0] != "_":
            key_suffix = f"_{key_suffix}"
    else:
        key_suffix = ""
    return key_suffix


COMMENTS_REGEX = r"""
^ # Begin of line.
(?:
  # A) Capturing group n°1: Full-line comment followed by empty lines.
  (
    [ \t]*           # Optional spaces or tabs.
    \#[^\r\n]*\r?\n  # The comment and the new line.
    (?:[ \t]*\r?\n)* # Optional empty lines (perhaps with spaces/tabs).
  )
|
  # B) Statement and optional comment at the end.
  (?:
    ( # Capturing group n°2 : The statement
      (?:
        # Multi-line strings with \"\"\" or '''.
        # Capturing group n°3 : The triple quotes.
        (['\"]{3})[\s\S]*?\3
      |
        # Double-quoted string "It's ok".
        \"(?: \\. | [^\"] )*\"
      |
        # Single-quoted string 'I\'ll say "Hello!"'.
        '(?: \\. | [^'] )*'
      |
        # Any chars, except spaces, hashtag, quotes and new lines.
        [^ \t#\"'\r\n]+
      |
        # Horizontal spaces, but not followed by a comment, because
        # we want the spaces in front of the comment to be matched
        # together with the optional comment we want to get rid of.
        [ \t]+(?![ \t]*\#)
      )+
    )
    # Capturing group n°4: An optional comment at the end of a statement.
    (
      [ \t]*\#[^\r\n]*
    )?
  )+
)
"""


def strip_comments_from_string(string):
    return re.sub(
        COMMENTS_REGEX,
        "\\2",
        string,
        0,
        re.MULTILINE | re.VERBOSE | re.UNICODE,
    )
