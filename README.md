# MarketMind AI (Streamlit)
Bounded plan–act–observe research agent with evidence binding, QC, repair loop and a four-way human approval gate. No agent framework.

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Enter your OpenAI key on the first screen (session memory only) or set `OPENAI_API_KEY` and click "Use environment key".

Layout: `marketmind/{config,llm,schemas,state,corpus,evidence,tools,agents,report}.py`, `app.py` (UI). Runs are saved to `runs/<id>/` (git-ignored).
Corpus: 11 fictional documents with planted contradiction (D01/D02), promotional claim (D03), gap (Quillbase enterprise pricing) and injection (D11).
Verify model names, prices (`config.py`) and API patterns against current OpenAI docs and record them in `docs/api_verification.md`.
Not included: evaluation harness, 18-case runner, screenshots, tech docs. Presets in the Mission tab map to T-01/04/05/06/08/09/14/15.
