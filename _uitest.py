import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("ui.py", default_timeout=60)
at.run()
excs = list(at.exception)
if excs:
    print("❌ RUNTIME EXCEPTIONS:")
    for e in excs:
        print("  -", e.type, e.value)
else:
    print("✅ NO runtime exceptions")
print("radio options:", [list(r.options) for r in at.radio])
print("sidebar markdown blocks:", len(at.sidebar.markdown))

# touched 2026-06-03
