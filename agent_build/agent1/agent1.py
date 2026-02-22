import asyncio
import os
import sys


if __package__ in {None, ""}:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    from agent_build.agent1.runtime import main
else:
    from .runtime import main


if __name__ == "__main__":
    asyncio.run(main())
