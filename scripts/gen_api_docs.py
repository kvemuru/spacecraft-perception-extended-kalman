"""Generate API reference markdown from module docstrings."""

import inspect
import pkgutil
import importlib
import kalman


def extract_classes(mod):
    classes = []
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ != mod.__name__:
            continue
        doc = inspect.getdoc(obj) or ""
        methods = []
        for mname, mobj in inspect.getmembers(obj, inspect.isfunction):
            if mname.startswith("_"):
                continue
            mdoc = inspect.getdoc(mobj) or ""
            sig = str(inspect.signature(mobj))
            methods.append((mname, sig, mdoc))
        classes.append((name, doc, methods))
    return classes


def extract_functions(mod):
    funcs = []
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if obj.__module__ != mod.__name__:
            continue
        if name.startswith("_"):
            continue
        doc = inspect.getdoc(obj) or ""
        sig = str(inspect.signature(obj))
        funcs.append((name, sig, doc))
    return funcs


def main():
    lines = ["# API Reference\n"]
    seen = set()
    for importer, modname, ispkg in pkgutil.walk_packages(
            kalman.__path__, kalman.__name__ + "."):
        if modname in seen:
            continue
        seen.add(modname)
        if "tests" in modname:
            continue
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        mod_doc = inspect.getdoc(mod) or ""
        lines.append(f"\n## `{modname}`\n")
        if mod_doc:
            lines.append(f"{mod_doc}\n")
        classes = extract_classes(mod)
        for cname, cdoc, methods in classes:
            lines.append(f"### `{cname}`\n")
            if cdoc:
                lines.append(f"{cdoc}\n")
            for mname, sig, mdoc in methods:
                lines.append(f"- **`{mname}{sig}`**")
                if mdoc:
                    lines.append(f"  - {mdoc.split(chr(10))[0]}")
            lines.append("")
        funcs = extract_functions(mod)
        if funcs:
            lines.append("#### Functions\n")
            for fname, sig, fdoc in funcs:
                lines.append(f"- **`{fname}{sig}`**")
                if fdoc:
                    lines.append(f"  - {fdoc.split(chr(10))[0]}")
            lines.append("")
    out = "\n".join(lines)
    with open("API_REFERENCE.md", "w") as f:
        f.write(out)
    print(f"Generated API_REFERENCE.md ({len(out)} chars)")


if __name__ == "__main__":
    main()
