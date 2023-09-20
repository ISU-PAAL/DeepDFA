import re


def tokenise(s):
    """Tokenise according to IVDetect.

    Tests:
    s = "FooBar fooBar foo bar_blub23/x~y'z"
    """
    spec_char = re.compile(r"[^a-zA-Z0-9\s]")
    camelcase = re.compile(r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    spec_split = re.split(spec_char, s)
    space_split = " ".join(spec_split).split()

    def camel_case_split(identifier):
        return [i.group(0) for i in re.finditer(camelcase, identifier)]

    camel_split = [i for j in [camel_case_split(i) for i in space_split] for i in j]
    remove_single = [i for i in camel_split if len(i) > 1]
    return " ".join(remove_single)


def tokenise_lines(s):
    r"""Tokenise according to IVDetect by splitlines.

    Example:
    s = "line1a line1b\nline2a asdf\nf f f f f\na"
    """
    slines = s.splitlines()
    lines = []
    for sline in slines:
        tokline = tokenise(sline)
        if len(tokline) > 0:
            lines.append(tokline)
    return lines
