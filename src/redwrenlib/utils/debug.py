
# utils/errors.py

#- Imports -----------------------------------------------------------------------------------------

import os
import sys
import inspect
from typing import Optional

from colorama import init, Fore


#- Private Methods ---------------------------------------------------------------------------------

# Initialise colorama
init()


#- Public Methods ----------------------------------------------------------------------------------

# Print where this procedure was called, with optional message.
def alert(prompt: str = "") -> None:
    # Immediate caller frame.
    # 0 is this function, 1 is what called it

    stack = inspect.stack()
    if len(stack) > 2:
        caller = stack[1]
        caller_file = os.path.abspath(caller.filename)
        caller_line = caller.lineno

        # Attempt to find the project root by looking for common repo/project markers.
        def _find_project_root(start_path: str) -> Optional[str]:
            cur = os.path.dirname(start_path)

            while True:
                candidate = os.path.join(cur, "redwrenlib")
                if os.path.exists(candidate):
                    return cur

                parent = os.path.dirname(cur)
                if parent == cur:
                    return None

                cur = parent

        project_root = _find_project_root(caller_file)

        if project_root:
            rel_path = os.path.relpath(caller_file, project_root)
        else:
            # fallback to current working directory if no project root found
            try:
                rel_path = os.path.relpath(caller_file, os.getcwd())
            except Exception:
                rel_path = caller_file

        caller_info: str = Fore.YELLOW + f"{rel_path}:{caller_line}" + Fore.RESET

    else:
        caller_info: str = Fore.YELLOW + "<unknown>" + Fore.RESET
    
    print(Fore.RED + f"[Error] {caller_info} {prompt}", file=sys.stderr)

